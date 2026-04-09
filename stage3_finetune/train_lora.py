#!/usr/bin/env python3
"""
Stage 3 — LoRA fine-tuning for Z-Image (base, non-distilled)
=============================================================

Architecture
------------
- Z-Image uses FlowMatchEulerDiscreteScheduler (flow matching, shift=6.0)
- VAE: 16-channel, scaling_factor=0.3611, shift_factor=0.1159
- Transformer: 6.15B params, 30 layers, dim=3840
- LoRA targets: to_q / to_k / to_v / to_out.0  (attention projections)
  Optional: add w1 / w2 / w3 (FFN) for higher-capacity fine-tuning
  Frozen:   QK norm weights (most stable vs Turbo, ~0.999 cosine)

Training loss (flow matching)
-----------------------------
  z0   = VAE_encode(image)              normalised latent
  eps  ~ N(0, I)                        noise
  t    ~ U[0, 1000)                     integer timestep
  sigmas = scheduler.sigmas[t]          noise weight
  zt   = (1 - sigma) * z0 + sigma * eps  noisy latent
  pred = transformer(zt, t, text_emb)   predicted velocity
  loss = MSE(pred, eps - z0)            flow matching target

LoRA → Turbo transfer
---------------------
  LoRA adapters trained on Z-Image can be loaded into Z-Image-Turbo
  by targeting the same layer indices. Layers 7–14 (cosine ~0.920 with
  Turbo) are most transferable; layers 0–3 and 26–29 (cosine ~0.85)
  are most risky. The --transfer_to_turbo flag tests this automatically.

Usage
-----
  # Default: fine-tune on DRDELATV/SHORT_NSFW, rank 16, 500 steps
  python3 stage3_finetune/train_lora.py

  # Custom dataset + config
  python3 stage3_finetune/train_lora.py \\
      --dataset DRDELATV/SHORT_NSFW \\
      --image_col image --text_col text \\
      --rank 16 --alpha 16 \\
      --target_modules to_q to_k to_v to_out.0 \\
      --steps 500 --batch_size 1 --grad_accum 4 --lr 1e-4 \\
      --res 512 \\
      --save_every 100 \\
      --output_dir stage3_finetune/checkpoints/run01

  # With FFN (higher capacity, more VRAM)
  python3 stage3_finetune/train_lora.py \\
      --target_modules to_q to_k to_v to_out.0 w1 w2 w3 \\
      --rank 8

  # Transfer adapter to Turbo and evaluate
  python3 stage3_finetune/train_lora.py --transfer_to_turbo \\
      --lora_path stage3_finetune/checkpoints/run01/final_adapter
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

MODEL_BASE = (
    "/scratch/hf-cache/huggingface/hub/"
    "models--Tongyi-MAI--Z-Image/snapshots/"
    "04cc4abb7c5069926f75c9bfde9ef43d49423021"
)
MODEL_TURBO = (
    "/scratch/hf-cache/huggingface/hub/"
    "models--Tongyi-MAI--Z-Image-Turbo/snapshots/"
    "f332072aa78be7aecdf3ee76d5c247082da564a6"
)


def _load_env():
    env = _ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

# ── VAE helpers ────────────────────────────────────────────────────────────────
VAE_SCALE  = 0.3611
VAE_SHIFT  = 0.1159


def encode_image(vae: AutoencoderKL, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode a batch of images to normalised latents."""
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    return (latents - VAE_SHIFT) * VAE_SCALE


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    latents = latents / VAE_SCALE + VAE_SHIFT
    with torch.no_grad():
        return vae.decode(latents).sample


# ── text encoding ──────────────────────────────────────────────────────────────

def encode_text(pipe: ZImagePipeline, prompts: list[str],
                device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    """
    Encode a list of prompts using the pipeline's text encoder.
    Returns a list of (seq_len, 2560) tensors — one per prompt.
    This matches the list[tensor] format the ZImageTransformer expects for cap_feats.
    """
    result = pipe.encode_prompt(
        prompts[0] if len(prompts) == 1 else prompts,
        device=device,
        do_classifier_free_guidance=False,
    )
    # encode_prompt returns (list[tensor], list[tensor]) for (pos, neg)
    pos_feats: list[torch.Tensor] = result[0]
    # Ensure dtype
    return [f.to(dtype=dtype) for f in pos_feats]


# ── flow matching loss ─────────────────────────────────────────────────────────

def get_noisy_latents(scheduler: FlowMatchEulerDiscreteScheduler,
                      latents: torch.Tensor, noise: torch.Tensor,
                      timesteps: torch.Tensor) -> torch.Tensor:
    """
    Apply flow matching forward process.
    sigmas[t] ∈ [0,1]:  0 = clean, 1 = pure noise
    x_t = (1 - sigma) * x_0 + sigma * noise
    """
    sigmas = scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
    # timesteps are integer indices into the sigma schedule
    s = sigmas[timesteps].view(-1, 1, 1, 1)
    return (1.0 - s) * latents + s * noise


def flow_matching_loss(pred: torch.Tensor, noise: torch.Tensor,
                       latents: torch.Tensor) -> torch.Tensor:
    """
    Flow matching target: model predicts the velocity field
        v = noise - x_0
    so the target is (noise - x_0).
    """
    target = noise - latents
    return F.mse_loss(pred.float(), target.float())


# ── dataset ────────────────────────────────────────────────────────────────────

PATCH_MULTIPLE = 16   # VAE×8 + transformer 2× patch


def _round_to_mult(x: int, mult: int) -> int:
    return (x // mult) * mult


def make_image_transform(min_side: int = 512, patch_mult: int = PATCH_MULTIPLE,
                         pre_cropped: bool = False):
    """
    Return a callable that converts a PIL Image to a normalised float tensor.

    pre_cropped=True  (finetune_dataset):  image already at correct size,
                                           just convert + normalise.
    pre_cropped=False (HF streaming):      resize shortest side to min_side,
                                           floor-round both dims to patch_mult,
                                           CenterCrop, then convert + normalise.
    """
    to_tensor  = transforms.ToTensor()
    normalise  = transforms.Normalize([0.5], [0.5])

    if pre_cropped:
        def transform(img: Image.Image) -> torch.Tensor:
            return normalise(to_tensor(img.convert("RGB")))
    else:
        def transform(img: Image.Image) -> torch.Tensor:
            img = img.convert("RGB")
            w, h = img.size
            scale = min_side / min(w, h)
            new_w = max(min_side, int(w * scale))
            new_h = max(min_side, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            crop_w = _round_to_mult(new_w, patch_mult)
            crop_h = _round_to_mult(new_h, patch_mult)
            x0 = (new_w - crop_w) // 2
            y0 = (new_h - crop_h) // 2
            img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            return normalise(to_tensor(img))

    return transform


def build_dataloader(dataset_id: str, image_col: str, text_col: str,
                     res: int, batch_size: int, hf_token: str | None,
                     max_samples: int | None = None):
    """Load an HF image-text dataset and return an infinite shuffled iterator.

    Supports both HuggingFace Hub IDs and local directories in imagefolder format:
      - HF Hub:  dataset_id = 'DRDELATV/SHORT_NSFW'
      - Local:   dataset_id = '/path/to/finetune_dataset'   (contains metadata.jsonl)
    """
    import os
    is_local = os.path.isdir(dataset_id)
    if is_local:
        # Local finetune_dataset — output of build_dataset.py
        ds = load_dataset("imagefolder", data_dir=dataset_id, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_id, split="train", streaming=True, token=hf_token)
    # Local images are already the correct crop; HF streaming needs full pipeline.
    transform = make_image_transform(min_side=res, pre_cropped=is_local)

    def collate(rows):
        imgs, texts = [], []
        for row in rows:
            img = row[image_col]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            imgs.append(transform(img.convert("RGB")))
            texts.append(str(row[text_col]))
        # Variable-size images (different aspect ratios) only stack at batch_size=1.
        # For batch_size>1 all images in the batch must have the same dimensions.
        return torch.stack(imgs), texts

    # Buffer and batch from streaming dataset
    buf: list = []
    samples_seen = 0

    def infinite_iter():
        nonlocal buf, samples_seen
        while True:
            for row in ds:
                buf.append(row)
                if len(buf) >= batch_size:
                    import random
                    random.shuffle(buf)
                    yield collate(buf[:batch_size])
                    buf = buf[batch_size:]
                    samples_seen += batch_size
                if max_samples and samples_seen >= max_samples:
                    buf = []
                    samples_seen = 0
                    break  # restart streaming

    return infinite_iter()


# ── LoRA setup ─────────────────────────────────────────────────────────────────

def apply_lora(transformer, rank: int, alpha: int,
               target_modules: list[str], dropout: float = 0.0):
    """Freeze all params then attach LoRA adapters to target_modules."""
    transformer.requires_grad_(False)
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(transformer, lora_cfg)


# ── Turbo transfer evaluation ──────────────────────────────────────────────────

def eval_transfer_to_turbo(lora_transformer, output_dir: Path,
                            test_prompts: list[str], res: int, seed: int):
    """
    Load Z-Image-Turbo transformer, inject the LoRA adapter weights from the
    fine-tuned Z-Image transformer (matching layer names), and generate test images.
    Saves side-by-side originals vs transferred panels.
    """
    from diffusers import ZImagePipeline
    from peft import get_peft_model, LoraConfig
    from PIL import ImageDraw

    print("\n=== LoRA → Turbo transfer evaluation ===")
    pipe_turbo = ZImagePipeline.from_pretrained(
        MODEL_TURBO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")

    # Extract LoRA state dict from the trained Z-Image transformer
    lora_state = {
        k: v for k, v in lora_transformer.state_dict().items()
        if "lora_" in k
    }
    print(f"  LoRA tensors to transfer: {len(lora_state)}")

    # Inject into Turbo transformer (same architecture, matching names)
    turbo_T = pipe_turbo.transformer
    missing, unexpected = turbo_T.load_state_dict(lora_state, strict=False)
    print(f"  Loaded: missing={len(missing)}  unexpected={len(unexpected)}")

    # Generate
    out_dir = output_dir / "turbo_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(test_prompts):
        gen = torch.Generator("cuda").manual_seed(seed + i)
        with torch.no_grad():
            img = pipe_turbo(
                prompt, height=res, width=res,
                num_inference_steps=9, guidance_scale=0.0,
                generator=gen,
            ).images[0]
        img.save(out_dir / f"{i:03d}.jpg", quality=90)
        print(f"  [{i+1}/{len(test_prompts)}] {prompt[:70]}")

    print(f"  Transfer images → {out_dir}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    scratch = os.environ.get("TRAINING_SCRATCH", "/tmp")
    default_out = Path(scratch) / "lora_run01"

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Z-Image")
    parser.add_argument("--dataset",         default="DRDELATV/SHORT_NSFW")
    parser.add_argument("--image_col",       default="image")
    parser.add_argument("--text_col",        default="text_en")
    parser.add_argument("--rank",            type=int,   default=16)
    parser.add_argument("--alpha",           type=int,   default=16)
    parser.add_argument("--target_modules",  nargs="+",  default=["to_q","to_k","to_v","to_out.0"])
    parser.add_argument("--lora_dropout",    type=float, default=0.0)
    parser.add_argument("--steps",           type=int,   default=500)
    parser.add_argument("--batch_size",      type=int,   default=1)
    parser.add_argument("--grad_accum",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--res",             type=int,   default=512)
    parser.add_argument("--save_every",      type=int,   default=100)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--output_dir",      type=Path,  default=default_out)
    parser.add_argument("--transfer_to_turbo", action="store_true",
                        help="After training, evaluate adapter transfer to Z-Image-Turbo")
    parser.add_argument("--lora_path",       type=Path,  default=None,
                        help="Load existing LoRA adapter instead of training")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    # ── load pipeline ──────────────────────────────────────────────────────────
    print("Loading Z-Image pipeline …")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=dtype, low_cpu_mem_usage=False,
    ).to(device)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()

    scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
    scheduler.set_timesteps(1000)   # full sigma schedule for training sampling

    # ── apply LoRA ─────────────────────────────────────────────────────────────
    if args.lora_path:
        print(f"Loading existing LoRA from {args.lora_path} …")
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
    else:
        print(f"Applying LoRA: rank={args.rank}  alpha={args.alpha}  "
              f"targets={args.target_modules}")
        pipe.transformer = apply_lora(
            pipe.transformer, args.rank, args.alpha,
            args.target_modules, args.lora_dropout,
        )

    pipe.transformer.print_trainable_parameters()

    if args.lora_path:
        print("Skipping training (--lora_path provided). Moving to evaluation.")
    else:
        # ── optimizer ─────────────────────────────────────────────────────────
        trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.lr * 0.1,
        )

        # ── dataset ────────────────────────────────────────────────────────────
        print(f"\nLoading dataset {args.dataset!r} …")
        dataloader = build_dataloader(
            args.dataset, args.image_col, args.text_col,
            args.res, args.batch_size,
            hf_token=os.environ.get("HF_TOKEN"),
        )

        # ── training loop ──────────────────────────────────────────────────────
        print(f"\nTraining for {args.steps} steps  "
              f"(effective batch = {args.batch_size * args.grad_accum})\n")
        pipe.transformer.train()

        global_step   = 0
        accum_loss    = 0.0
        accum_count   = 0
        t0            = time.time()
        optimizer.zero_grad()

        for pixel_values, texts in dataloader:
            if global_step >= args.steps:
                break

            pixel_values = pixel_values.to(device=device, dtype=dtype)

            # ── encode image & text ────────────────────────────────────────────
            latents   = encode_image(pipe.vae, pixel_values)          # (B, 16, H/8, W/8)
            with torch.no_grad():
                cap_feats = encode_text(pipe, texts, device, dtype)   # list of (seq, 2560)

            # ── sample timestep & noise ────────────────────────────────────────
            B  = latents.shape[0]
            t_idx = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (B,), device=device)
            noise    = torch.randn_like(latents)
            noisy_z  = get_noisy_latents(scheduler, latents, noise, t_idx)

            # ── forward pass ───────────────────────────────────────────────────
            # ZImageTransformer expects x and cap_feats as Python lists (one tensor per batch item)
            noisy_z_list = list(noisy_z.unsqueeze(2).unbind(0))    # list of (1, C, H, W)

            out = pipe.transformer(
                x=noisy_z_list,
                t=t_idx.float(),
                cap_feats=cap_feats,
                return_dict=False,
            )
            # out[0] is a list of (1, C, H, W) tensors; stack back to (B, C, H, W)
            pred = torch.stack(out[0]).squeeze(2)

            loss = flow_matching_loss(pred, noise, latents) / args.grad_accum
            loss.backward()
            accum_loss  += loss.item() * args.grad_accum
            accum_count += 1

            # ── gradient step ──────────────────────────────────────────────────
            if accum_count == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler_lr.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss     = accum_loss / accum_count
                elapsed      = time.time() - t0
                steps_per_s  = global_step / elapsed
                eta          = (args.steps - global_step) / max(steps_per_s, 1e-6)

                print(
                    f"step {global_step:5d}/{args.steps}"
                    f"  loss={avg_loss:.5f}"
                    f"  lr={scheduler_lr.get_last_lr()[0]:.2e}"
                    f"  {steps_per_s:.2f}it/s"
                    f"  ETA {eta/60:.1f}min",
                    flush=True,
                )
                accum_loss  = 0.0
                accum_count = 0

                # ── checkpoint ────────────────────────────────────────────────
                if global_step % args.save_every == 0:
                    ckpt = args.output_dir / f"checkpoint_{global_step:05d}"
                    pipe.transformer.save_pretrained(ckpt)
                    print(f"  → saved {ckpt}")

        # ── final save ─────────────────────────────────────────────────────────
        final_dir = args.output_dir / "final_adapter"
        pipe.transformer.save_pretrained(final_dir)
        print(f"\nFinal adapter → {final_dir}")

    # ── Turbo transfer ─────────────────────────────────────────────────────────
    if args.transfer_to_turbo:
        test_prompts = [
            "nude woman in studio lighting, photography",
            "a golden retriever puppy on a sunny lawn",
            "cyberpunk cityscape at night, neon lights",
        ]
        eval_transfer_to_turbo(
            pipe.transformer, args.output_dir, test_prompts, args.res, args.seed,
        )


if __name__ == "__main__":
    main()
