#!/usr/bin/env python3
"""
Stage 3 — Full fine-tuning for Z-Image (base, non-distilled)
=============================================================

Why full fine-tune instead of LoRA
-----------------------------------
At inference time a LoRA adapter requires the PEFT runtime hooks present in memory
even though the adapter weights are small.  For deployment under tight VRAM budgets
it is preferable to bake the learned delta directly into the model weights so the
adapter overhead disappears entirely.  Full fine-tuning produces a plain
``ZImageTransformer2DModel`` checkpoint with no third-party runtime dependency.

Architecture
------------
- Transformer : ZImageTransformer2DModel  6.15B params, 30 S3-DiT layers, dim=3840
- VAE          : 16-channel, scaling_factor=0.3611, shift_factor=0.1159  (frozen)
- Text encoder : Qwen3 cap_embedder, cap_feat_dim=2560                   (frozen)
- Scheduler    : FlowMatchEulerDiscreteScheduler shift=6.0, 1000 steps

Training strategy
-----------------
Only the transformer is trained.  VAE and text encoder are frozen and cast to
bfloat16 for encode-only inference.

To mitigate catastrophic forgetting on a small dataset (188 samples):
  - Low learning rate  (2e-5 default, vs 1e-4 for LoRA)
  - Strong weight decay (5e-2)
  - Cosine LR with warm-up
  - Gradient clipping  (1.0)
  - Gradient checkpointing when --grad_ckpt is set (saves ~40% VRAM)

Training loss (flow matching — identical to train_lora.py)
----------------------------------------------------------
  z0   = VAE_encode(image)              normalised latent
  eps  ~ N(0, I)                        noise
  t    ~ U[0, 1000)                     integer timestep
  sigma = scheduler.sigmas[t]           noise weight ∈ [0,1]
  zt   = (1-sigma)*z0 + sigma*eps       noisy latent
  pred = transformer(zt, t, text_emb)   predicted velocity
  loss = MSE(pred, eps - z0)            flow matching target

Saving
------
Only the *transformer* is saved (not the full pipeline) as a diffusers
``ZImageTransformer2DModel`` directory so it can be hot-swapped into any
``ZImagePipeline`` via ``pipe.transformer = ZImageTransformer2DModel.from_pretrained(...)``.

Usage
-----
  # Default: DRDELATV/SHORT_NSFW, 2000 steps, lr=2e-5, batch_size=1, grad_accum=4
  python3 stage3_finetune/train_fullft.py

  # Custom config
  python3 stage3_finetune/train_fullft.py \\
      --dataset DRDELATV/SHORT_NSFW \\
      --image_col image --text_col text \\
      --steps 2000 --batch_size 1 --grad_accum 4 --lr 2e-5 \\
      --warmup_steps 50 --weight_decay 0.05 \\
      --res 512 --save_every 500 \\
      --grad_ckpt \\
      --output_dir /scratch/training/fullft_run01

  # Resume from checkpoint
  python3 stage3_finetune/train_fullft.py \\
      --resume /scratch/training/fullft_run01/checkpoint_00500
"""

import copy
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
from PIL import Image
from torchvision import transforms

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

MODEL_BASE = (
    "/scratch/hf-cache/huggingface/hub/"
    "models--Tongyi-MAI--Z-Image/snapshots/"
    "04cc4abb7c5069926f75c9bfde9ef43d49423021"
)

# ── env ────────────────────────────────────────────────────────────────────────

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
VAE_SCALE = 0.3611
VAE_SHIFT  = 0.1159


def encode_image(vae: AutoencoderKL, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode a batch of images to normalised latents (frozen, no_grad)."""
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    return (latents - VAE_SHIFT) * VAE_SCALE


# ── text encoding ──────────────────────────────────────────────────────────────

def encode_text(pipe: ZImagePipeline, prompts: list[str],
                device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    """
    Encode prompts using the pipeline's (frozen) text encoder.
    Returns list[tensor(seq_len, 2560)] — one per prompt.
    This matches the list[tensor] format ZImageTransformer expects for cap_feats.
    """
    with torch.no_grad():
        result = pipe.encode_prompt(
            prompts[0] if len(prompts) == 1 else prompts,
            device=device,
            do_classifier_free_guidance=False,
        )
    pos_feats: list[torch.Tensor] = result[0]
    return [f.to(dtype=dtype) for f in pos_feats]


# ── flow matching ──────────────────────────────────────────────────────────────

def get_noisy_latents(scheduler: FlowMatchEulerDiscreteScheduler,
                      latents: torch.Tensor, noise: torch.Tensor,
                      timesteps: torch.Tensor) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
    s = sigmas[timesteps].view(-1, 1, 1, 1)
    return (1.0 - s) * latents + s * noise


def flow_matching_loss(pred: torch.Tensor, noise: torch.Tensor,
                       latents: torch.Tensor,
                       sigmas: torch.Tensor,
                       min_snr_gamma: float = 5.0) -> torch.Tensor:
    target = noise - latents
    # Reduce over spatial dims per-sample so all resolutions contribute equally.
    diff = (pred.float() - target.float()) ** 2
    per_sample = diff.mean(dim=list(range(1, diff.ndim)))   # (B,)
    # Min-SNR-γ weighting: downweight near-pure-noise timesteps.
    # SNR = (1-σ)²/σ²; weight = min(SNR, γ) — caps clean-timestep dominance
    # and strongly downweights σ≈1 steps where loss explodes.
    snr = (1.0 - sigmas) ** 2 / (sigmas ** 2 + 1e-8)
    weight = snr.clamp(max=min_snr_gamma)          # (B,)
    return (weight * per_sample).mean()


# ── dataset ────────────────────────────────────────────────────────────────────

PATCH_MULTIPLE = 16   # VAE×8 + transformer 2× patch


def _round_to_mult(x: int, mult: int) -> int:
    return (x // mult) * mult


def make_image_transform(min_side: int = 512, patch_mult: int = PATCH_MULTIPLE,
                         pre_cropped: bool = False, flip_aug: bool = False):
    """
    Return a callable that converts a PIL Image to a normalised float tensor.

    pre_cropped=True  (finetune_dataset):  image already at correct size,
                                           optionally flip, then convert + normalise.
    pre_cropped=False (HF streaming):      resize shortest side to min_side,
                                           floor-round both dims to patch_mult,
                                           CenterCrop, then convert + normalise.
    flip_aug: randomly flip horizontally 50% of the time.
    """
    import random as _random
    to_tensor  = transforms.ToTensor()
    normalise  = transforms.Normalize([0.5], [0.5])

    def _maybe_flip(img: Image.Image) -> Image.Image:
        if flip_aug and _random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    if pre_cropped:
        def transform(img: Image.Image) -> torch.Tensor:
            return normalise(to_tensor(_maybe_flip(img.convert("RGB"))))
    else:
        def transform(img: Image.Image) -> torch.Tensor:
            img = _maybe_flip(img.convert("RGB"))
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
                     flip_aug: bool = False):
    """Load dataset from HF Hub ID or local imagefolder directory."""
    import os
    is_local = os.path.isdir(dataset_id)
    if is_local:
        ds = load_dataset("imagefolder", data_dir=dataset_id, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_id, split="train", streaming=True, token=hf_token)
    transform = make_image_transform(min_side=res, pre_cropped=is_local, flip_aug=flip_aug)

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

    buf: list = []

    def infinite_iter():
        nonlocal buf
        while True:
            for row in ds:
                buf.append(row)
                if len(buf) >= batch_size:
                    import random
                    random.shuffle(buf)
                    yield collate(buf[:batch_size])
                    buf = buf[batch_size:]

    return infinite_iter()


# ── LR schedule with warm-up ───────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int,
                  min_lr_ratio: float = 0.1):
    """Linear warm-up followed by cosine annealing."""
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    scratch = os.environ.get("TRAINING_SCRATCH", "/tmp")
    default_out = Path(scratch) / "fullft_nsfw_run01"

    parser = argparse.ArgumentParser(description="Full fine-tuning for Z-Image")
    parser.add_argument("--model_path",   default=None,
                        help="Pipeline path (default: Z-Image base). Pass Turbo path here.")
    parser.add_argument("--dataset",       default="DownFlow/nsfw")
    parser.add_argument("--text_col",      default="text_en")
    parser.add_argument("--image_col",     default="image")
    parser.add_argument("--steps",         type=int,   default=2000)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--grad_accum",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=5e-6)
    parser.add_argument("--warmup_steps",  type=int,   default=200)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--res",           type=int,   default=512)
    parser.add_argument("--save_every",    type=int,   default=500)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--grad_ckpt",     action="store_true",
                        help="Enable gradient checkpointing (~40%% VRAM reduction)")
    parser.add_argument("--flip_aug",      action="store_true",
                        help="Random horizontal flip augmentation (doubles effective dataset size)")
    parser.add_argument("--caption_dropout", type=float, default=0.1,
                        help="Fraction of steps to drop captions (enables CFG-amplified style at inference)")
    parser.add_argument("--ema_decay",     type=float, default=0.9999,
                        help="EMA decay for model weights. 0 = disabled")
    parser.add_argument("--timestep_bias", type=float, default=1.2,
                        help="Logit-normal sigma for timestep sampling. >1 biases toward low-noise (style) steps")
    parser.add_argument("--resume",        type=Path,  default=None,
                        help="Resume from a saved transformer checkpoint directory")
    parser.add_argument("--resume_step",   type=int,   default=0,
                        help="Global step the checkpoint was saved at (fast-forwards LR schedule)")
    parser.add_argument("--output_dir",    type=Path,  default=default_out)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    # ── load pipeline ──────────────────────────────────────────────────────────
    model_path = args.model_path or MODEL_BASE
    print(f"Loading pipeline from {model_path} …")
    pipe = ZImagePipeline.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=False,
    ).to(device)

    # Freeze VAE and text encoder — only train the transformer
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    # Freeze text encoder components if present as named attributes
    for name in ("text_encoder", "tokenizer", "cap_embedder"):
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
            module.eval()

    scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
    scheduler.set_timesteps(1000)

    # ── optionally resume from checkpoint ─────────────────────────────────────
    if args.resume is not None:
        from diffusers import ZImageTransformer2DModel  # type: ignore[attr-defined]
        print(f"Resuming transformer from {args.resume} …")
        pipe.transformer = ZImageTransformer2DModel.from_pretrained(
            args.resume, torch_dtype=dtype,
        ).to(device)

    # ── make transformer trainable ────────────────────────────────────────────
    pipe.transformer.requires_grad_(True)
    pipe.transformer.train()

    if args.grad_ckpt:
        pipe.transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.")

    n_params = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e9:.3f}B")

    # ── optimizer ─────────────────────────────────────────────────────────────
    # Separate weight decay: apply to weight matrices, not biases or norms
    decay_params, no_decay_params = [], []
    for name, param in pipe.transformer.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    lr_sched = get_scheduler(optimizer, args.warmup_steps, args.steps)

    # ── EMA ────────────────────────────────────────────────────────────────────
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(pipe.transformer)
        ema_model.requires_grad_(False)
        print(f"EMA enabled  (decay={args.ema_decay})")

    # Fast-forward LR schedule to resume_step so the LR is correct immediately.
    if args.resume_step > 0:
        for _ in range(args.resume_step):
            lr_sched.step()
        print(f"LR schedule fast-forwarded to step {args.resume_step}: "
              f"lr={lr_sched.get_last_lr()[0]:.3e}")

    # ── dataset ────────────────────────────────────────────────────────────────
    print(f"Loading dataset {args.dataset!r} …")
    dataloader = build_dataloader(
        args.dataset, args.image_col, args.text_col,
        args.res, args.batch_size,
        hf_token=os.environ.get("HF_TOKEN"),
        flip_aug=args.flip_aug,
    )

    # ── training loop ──────────────────────────────────────────────────────────
    eff_batch = args.batch_size * args.grad_accum
    print(f"\nFull fine-tune: {args.steps} steps  "
          f"(effective batch={eff_batch}, lr={args.lr:.1e}, "
          f"caption_dropout={args.caption_dropout}, flip_aug={args.flip_aug})\n")

    global_step = args.resume_step
    accum_loss  = 0.0
    accum_count = 0
    t0 = time.time()
    optimizer.zero_grad()

    for pixel_values, texts in dataloader:
        if global_step >= args.steps:
            break

        pixel_values = pixel_values.to(device=device, dtype=dtype)

        # encode image & text (text encoder frozen → no_grad inside)
        latents   = encode_image(pipe.vae, pixel_values)
        # Caption dropout: zero out text conditioning to train unconditional mode.
        # At inference, CFG will then push hard toward the fine-tuned style.
        texts_input = [
            "" if (torch.rand(1).item() < args.caption_dropout) else t
            for t in texts
        ]
        cap_feats = encode_text(pipe, texts_input, device, dtype)

        # sample timestep & add noise
        # Logit-normal with timestep_bias sigma: higher values push toward
        # low-noise steps (σ→0) where style/texture decisions are made.
        B     = latents.shape[0]
        u     = torch.sigmoid(torch.randn(B, device=device) * args.timestep_bias)
        t_idx = (u * scheduler.config.num_train_timesteps).long().clamp(
                    0, scheduler.config.num_train_timesteps - 1)
        sigmas_t = scheduler.sigmas.to(device=device, dtype=dtype)[t_idx]  # (B,)
        noise   = torch.randn_like(latents)
        noisy_z = get_noisy_latents(scheduler, latents, noise, t_idx)

        # forward — transformer expects list[tensor] not batched tensor
        noisy_z_list = list(noisy_z.unsqueeze(2).unbind(0))   # list of (1, C, H, W)
        out = pipe.transformer(
            x=noisy_z_list,
            t=t_idx.float(),
            cap_feats=cap_feats,
            return_dict=False,
        )
        pred = torch.stack(out[0]).squeeze(2)                  # (B, C, H, W)

        loss = flow_matching_loss(pred, noise, latents, sigmas_t) / args.grad_accum
        loss.backward()
        accum_loss  += loss.item() * args.grad_accum
        accum_count += 1

        if accum_count == args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipe.transformer.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()

            # EMA update
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p_model in zip(ema_model.parameters(),
                                              pipe.transformer.parameters()):
                        p_ema.data.mul_(args.ema_decay).add_(
                            p_model.data, alpha=1.0 - args.ema_decay
                        )

            global_step += 1
            elapsed     = time.time() - t0
            sps         = global_step / elapsed
            eta         = (args.steps - global_step) / max(sps, 1e-6)
            avg_loss    = accum_loss / accum_count

            print(
                f"step {global_step:5d}/{args.steps}"
                f"  loss={avg_loss:.5f}"
                f"  lr={lr_sched.get_last_lr()[0]:.2e}"
                f"  {sps:.2f}it/s"
                f"  ETA {eta/60:.1f}min",
                flush=True,
            )
            accum_loss  = 0.0
            accum_count = 0

            if global_step % args.save_every == 0:
                ckpt = args.output_dir / f"checkpoint_{global_step:05d}"
                pipe.transformer.save_pretrained(ckpt)
                print(f"  → checkpoint saved: {ckpt}")

    # ── final save ─────────────────────────────────────────────────────────────
    # Save EMA weights as the final model if EMA was enabled — EMA gives
    # smoother, more generalised weights than the raw last-step parameters.
    final_dir = args.output_dir / "final_transformer"
    save_model = ema_model if ema_model is not None else pipe.transformer
    save_model.save_pretrained(final_dir)
    ema_note = " (EMA weights)" if ema_model is not None else ""
    print(f"\nFinal transformer{ema_note} → {final_dir}")
    print(
        f"Load with:\n"
        f"  from diffusers import ZImagePipeline, ZImageTransformer2DModel\n"
        f"  pipe = ZImagePipeline.from_pretrained(MODEL_BASE, torch_dtype=torch.bfloat16)\n"
        f"  pipe.transformer = ZImageTransformer2DModel.from_pretrained(\n"
        f"      '{final_dir}', torch_dtype=torch.bfloat16)\n"
        f"  pipe = pipe.to('cuda')"
    )


if __name__ == "__main__":
    main()
