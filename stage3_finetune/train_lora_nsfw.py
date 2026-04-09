#!/usr/bin/env python3
"""
Stage 3 — LoRA fine-tuning on HF streaming datasets (e.g. DownFlow/nsfw)
=========================================================================

Based on train_fullft.py (dataset / flow-matching loop) but trains only a
LoRA adapter rather than all transformer parameters.

Key differences vs. train_fullft.py
------------------------------------
- Injects a PEFT LoRA adapter; base weights stay frozen.
- Saves adapter_checkpoint_XXXXX/ dirs, not full-transformer dirs.
- Supports --resume_adapter to continue from an existing adapter (e.g. the
  Fuli artist adapter).
- --n_items  cap how many dataset items to load into RAM and cycle through.
  Set to 20 for a fast "style-transfer" style run.

Usage
-----
  # Continue from artist adapter, 20 nsfw images, 2000 steps
  python3 stage3_finetune/train_lora_nsfw.py \\
      --model_path /scratch/hf-cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6 \\
      --resume_adapter /scratch/training/lora_artist_turbo_run01/final_adapter \\
      --dataset DownFlow/nsfw \\
      --n_items 20 \\
      --steps 2000 --lr 5e-5 --grad_accum 4 \\
      --output_dir /scratch/training/lora_nsfw_run01

  # From scratch (no resume)
  python3 stage3_finetune/train_lora_nsfw.py \\
      --dataset DownFlow/nsfw --n_items 20 --steps 2000
"""

import argparse
import copy
import io
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
from PIL import Image
from torchvision import transforms

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

MODEL_TURBO = (
    "/scratch/hf-cache/huggingface/hub/"
    "models--Tongyi-MAI--Z-Image-Turbo/snapshots/"
    "f332072aa78be7aecdf3ee76d5c247082da564a6"
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

# ── constants ──────────────────────────────────────────────────────────────────
VAE_SCALE     = 0.3611
VAE_SHIFT     = 0.1159
PATCH_MULTIPLE = 16


# ── image helpers ──────────────────────────────────────────────────────────────

def _round_to_mult(x: int, mult: int) -> int:
    return (x // mult) * mult


def make_transform(min_side: int = 512, flip_aug: bool = False):
    to_tensor = transforms.ToTensor()
    normalise = transforms.Normalize([0.5], [0.5])

    def tf(img: Image.Image) -> torch.Tensor:
        if flip_aug and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.convert("RGB")
        w, h = img.size
        scale = min_side / min(w, h)
        nw = max(min_side, int(w * scale))
        nh = max(min_side, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        cw = _round_to_mult(nw, PATCH_MULTIPLE)
        ch = _round_to_mult(nh, PATCH_MULTIPLE)
        x0 = (nw - cw) // 2
        y0 = (nh - ch) // 2
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        return normalise(to_tensor(img))

    return tf


# ── dataset: load N items into RAM, then cycle ────────────────────────────────

def load_n_items(dataset_id: str, image_col: str, text_col: str,
                 n_items: int, res: int, flip_aug: bool,
                 hf_token: str | None) -> list[tuple[torch.Tensor, str]]:
    """Stream up to n_items rows from the HF dataset into RAM as (tensor, caption) pairs."""
    from datasets import load_dataset

    is_local = os.path.isdir(dataset_id)
    if is_local:
        ds = load_dataset("imagefolder", data_dir=dataset_id, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_id, split="train", streaming=True, token=hf_token)

    tf = make_transform(min_side=res, flip_aug=False)  # augment at runtime, not here
    records: list[tuple[torch.Tensor, str]] = []

    print(f"  Streaming {n_items} items from '{dataset_id}' ...")
    for row in ds:
        img = row[image_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        try:
            tensor = tf(img)
        except Exception as e:
            print(f"  [skip] image transform failed: {e}")
            continue
        caption = str(row[text_col])
        records.append((tensor, caption))
        if len(records) >= n_items:
            break

    print(f"  Loaded {len(records)} items.")
    return records


def infinite_cycle(records: list[tuple[torch.Tensor, str]], batch_size: int,
                   flip_aug: bool):
    """Yield (pixel_values, [caption]) batches, shuffling each epoch."""
    idx = list(range(len(records)))
    while True:
        random.shuffle(idx)
        buf_imgs, buf_texts = [], []
        for i in idx:
            tensor, cap = records[i]
            if flip_aug and random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[-1])
            buf_imgs.append(tensor)
            buf_texts.append(cap)
            if len(buf_imgs) == batch_size:
                yield torch.stack(buf_imgs), buf_texts
                buf_imgs, buf_texts = [], []


# ── encoding ──────────────────────────────────────────────────────────────────

def encode_image(vae: AutoencoderKL, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    return (latents - VAE_SHIFT) * VAE_SCALE


def encode_text(pipe: ZImagePipeline, prompts: list[str],
                device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    with torch.no_grad():
        result = pipe.encode_prompt(
            prompts[0] if len(prompts) == 1 else prompts,
            device=device,
            do_classifier_free_guidance=False,
        )
    return [f.to(dtype=dtype) for f in result[0]]


# ── flow matching ─────────────────────────────────────────────────────────────

def get_noisy_latents(scheduler: FlowMatchEulerDiscreteScheduler,
                      latents: torch.Tensor, noise: torch.Tensor,
                      timesteps: torch.Tensor) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
    s = sigmas[timesteps].view(-1, 1, 1, 1)
    return (1.0 - s) * latents + s * noise


def flow_matching_loss(pred: torch.Tensor, noise: torch.Tensor,
                       latents: torch.Tensor, sigmas: torch.Tensor,
                       min_snr_gamma: float = 5.0) -> torch.Tensor:
    target = noise - latents
    diff = (pred.float() - target.float()) ** 2
    per_sample = diff.mean(dim=list(range(1, diff.ndim)))
    snr = (1.0 - sigmas) ** 2 / (sigmas ** 2 + 1e-8)
    weight = snr.clamp(max=min_snr_gamma)
    return (weight * per_sample).mean()


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int,
                  min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── LoRA injection ────────────────────────────────────────────────────────────

def inject_lora(transformer, rank: int, alpha: float,
                target_modules: list[str], dropout: float = 0.0):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none",
    )
    peft_model = get_peft_model(transformer, config)
    peft_model.print_trainable_parameters()
    return peft_model


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    scratch = os.environ.get("TRAINING_SCRATCH", "/scratch/training")
    default_out = Path(scratch) / "lora_nsfw_run01"

    parser = argparse.ArgumentParser(description="LoRA fine-tune on HF streaming dataset")
    # model
    parser.add_argument("--model_path",      default=None,
                        help="Pipeline path (default: Z-Image Turbo local snapshot)")
    parser.add_argument("--resume_adapter",  type=Path, default=None,
                        help="Existing PEFT adapter directory to resume/continue from")
    # data
    parser.add_argument("--dataset",         default="DownFlow/nsfw")
    parser.add_argument("--text_col",        default="text_en")
    parser.add_argument("--image_col",       default="image")
    parser.add_argument("--n_items",         type=int, default=20,
                        help="Number of dataset items to load into RAM and cycle through")
    # LoRA
    parser.add_argument("--rank",            type=int,   default=32)
    parser.add_argument("--lora_alpha",      type=float, default=None,
                        help="LoRA alpha (default: rank)")
    parser.add_argument("--target_modules",  nargs="+",
                        default=["to_q", "to_k", "to_v", "w1", "w2", "w3"])
    parser.add_argument("--lora_dropout",    type=float, default=0.05)
    # training
    parser.add_argument("--steps",           type=int,   default=2000)
    parser.add_argument("--batch_size",      type=int,   default=1)
    parser.add_argument("--grad_accum",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=5e-5)
    parser.add_argument("--warmup_steps",    type=int,   default=100)
    parser.add_argument("--weight_decay",    type=float, default=0.01)
    parser.add_argument("--res",             type=int,   default=512)
    parser.add_argument("--save_every",      type=int,   default=500)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--grad_ckpt",       action="store_true")
    parser.add_argument("--flip_aug",        action="store_true")
    parser.add_argument("--caption_dropout", type=float, default=0.05)
    parser.add_argument("--ema_decay",       type=float, default=0.9999)
    parser.add_argument("--timestep_bias",   type=float, default=1.2)
    parser.add_argument("--output_dir",      type=Path,  default=default_out)
    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = float(args.rank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    print(f"LoRA NSFW fine-tune  rank={args.rank}  alpha={args.lora_alpha}")
    print(f"Dataset: {args.dataset}  n_items={args.n_items}  steps={args.steps}")
    print(f"lr={args.lr:.1e}  grad_accum={args.grad_accum}  resume_adapter={args.resume_adapter}")

    # ── pipeline ──────────────────────────────────────────────────────────────
    model_path = args.model_path or MODEL_TURBO
    print(f"\nLoading pipeline from {model_path} ...")
    pipe = ZImagePipeline.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=False,
    ).to(device)

    pipe.vae.requires_grad_(False).eval()
    for name in ("text_encoder", "tokenizer", "cap_embedder"):
        mod = getattr(pipe, name, None)
        if mod is not None and hasattr(mod, "requires_grad_"):
            mod.requires_grad_(False)
            if hasattr(mod, "eval"): mod.eval()

    scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
    scheduler.set_timesteps(1000)

    # ── inject LoRA ───────────────────────────────────────────────────────────
    print("\nInjecting LoRA adapters ...")
    pipe.transformer.requires_grad_(False)
    peft_tr = inject_lora(
        pipe.transformer,
        rank=args.rank,
        alpha=args.lora_alpha,
        target_modules=args.target_modules,
        dropout=args.lora_dropout,
    )
    pipe.transformer = peft_tr

    if args.grad_ckpt:
        base = peft_tr.base_model.model
        if hasattr(base, "enable_gradient_checkpointing"):
            base.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled.")

    # ── resume from existing adapter ──────────────────────────────────────────
    if args.resume_adapter is not None:
        print(f"\nLoading adapter weights from {args.resume_adapter} ...")
        from safetensors.torch import load_file as st_load
        ckpt_file = args.resume_adapter / "adapter_model.safetensors"
        ckpt_state = st_load(str(ckpt_file), device="cpu")
        # PEFT saves  base_model.model.<path>.lora_A.weight
        # state_dict expects  base_model.model.<path>.lora_A.default.weight
        remapped = {}
        for k, v in ckpt_state.items():
            k2 = re.sub(
                r'\.(lora_A|lora_B|lora_embedding_A|lora_embedding_B)\.weight$',
                r'.\1.default.weight', k,
            )
            remapped[k2] = v
        missing, unexpected = peft_tr.load_state_dict(remapped, strict=False)
        non_lora_missing = [k for k in missing if "lora_" not in k]
        print(f"  Loaded {len(remapped)} tensors from adapter checkpoint.")
        if non_lora_missing:
            print(f"  Warning: {len(non_lora_missing)} non-LoRA keys missing: {non_lora_missing[:3]}")

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_st: dict[str, torch.Tensor] | None = None
    if args.ema_decay > 0:
        ema_st = {
            k: v.clone().detach().cpu()   # keep EMA on CPU to save VRAM
            for k, v in peft_tr.state_dict().items()
            if "lora_" in k
        }
        print(f"EMA enabled (decay={args.ema_decay}, {len(ema_st)} LoRA tensors)")

    # ── optimizer (LoRA params only) ──────────────────────────────────────────
    lora_params = [p for p in peft_tr.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params, lr=args.lr,
        betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    lr_sched = get_scheduler(optimizer, args.warmup_steps, args.steps)

    # ── dataset ───────────────────────────────────────────────────────────────
    records = load_n_items(
        args.dataset, args.image_col, args.text_col,
        args.n_items, args.res, args.flip_aug,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    data_iter = infinite_cycle(records, args.batch_size, flip_aug=args.flip_aug)

    # ── training loop ─────────────────────────────────────────────────────────
    peft_tr.train()
    eff_batch = args.batch_size * args.grad_accum
    print(f"\nStarting LoRA training: {args.steps} steps  eff_batch={eff_batch}\n")

    global_step = 0
    accum_loss  = 0.0
    accum_count = 0
    t0 = time.time()
    optimizer.zero_grad()

    for micro_step in range(args.steps * args.grad_accum):
        if global_step >= args.steps:
            break

        pixel_values, texts = next(data_iter)
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        # Caption dropout
        if torch.rand(1).item() < args.caption_dropout:
            texts = [""] * len(texts)

        latents  = encode_image(pipe.vae, pixel_values)
        cap_feats = encode_text(pipe, texts, device, dtype)

        B     = latents.shape[0]
        u     = torch.sigmoid(torch.randn(B, device=device) * args.timestep_bias)
        t_idx = (u * scheduler.config.num_train_timesteps).long().clamp(
                    0, scheduler.config.num_train_timesteps - 1)
        sigmas_t = scheduler.sigmas.to(device=device, dtype=dtype)[t_idx]
        noise    = torch.randn_like(latents)
        noisy_z  = get_noisy_latents(scheduler, latents, noise, t_idx)

        # ZImageTransformer2DModel forward: x=list[tensor], t=float, cap_feats=list[tensor]
        noisy_z_list = list(noisy_z.unsqueeze(2).unbind(0))   # list of (1, C, H, W)
        out = peft_tr(
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

        if (micro_step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()
            global_step += 1

            # EMA
            if ema_st is not None:
                with torch.no_grad():
                    for k, v in peft_tr.state_dict().items():
                        if k in ema_st:
                            ema_st[k].mul_(args.ema_decay).add_(
                                v.detach().cpu(), alpha=1.0 - args.ema_decay
                            )

            elapsed  = time.time() - t0
            sps      = global_step / max(elapsed, 1e-6)
            eta      = (args.steps - global_step) / max(sps, 1e-6)
            avg_loss = accum_loss / accum_count
            cur_lr   = optimizer.param_groups[0]["lr"]
            print(
                f"step {global_step:5d}/{args.steps}"
                f"  loss={avg_loss:.5f}"
                f"  lr={cur_lr:.2e}"
                f"  {sps:.2f}it/s"
                f"  ETA {eta/60:.1f}min",
                flush=True,
            )
            accum_loss  = 0.0
            accum_count = 0

            # Checkpoint
            if global_step % args.save_every == 0 or global_step == args.steps:
                ck = args.output_dir / f"adapter_checkpoint_{global_step:05d}"
                peft_tr.save_pretrained(str(ck))
                print(f"  -> checkpoint saved: {ck}")
                if ema_st is not None:
                    from safetensors.torch import save_file as st_save
                    ek = args.output_dir / f"adapter_checkpoint_{global_step:05d}_ema"
                    ek.mkdir(parents=True, exist_ok=True)
                    st_save(ema_st, str(ek / "adapter_model.safetensors"))
                    print(f"  -> EMA checkpoint: {ek}")

    # ── final save ────────────────────────────────────────────────────────────
    # Save EMA weights as final (smoother, more general than raw last step)
    final_dir = args.output_dir / "final_adapter"
    if ema_st is not None:
        from safetensors.torch import save_file as st_save
        final_dir.mkdir(parents=True, exist_ok=True)
        # Also save PEFT adapter_config.json alongside EMA weights
        peft_tr.save_pretrained(str(final_dir))  # writes config + non-EMA weights
        st_save(ema_st, str(final_dir / "adapter_model.safetensors"))  # overwrite with EMA
        print(f"\nFinal LoRA adapter (EMA weights) -> {final_dir}")
    else:
        peft_tr.save_pretrained(str(final_dir))
        print(f"\nFinal LoRA adapter -> {final_dir}")

    print(
        f"Load with:\n"
        f"  from diffusers import ZImagePipeline\n"
        f"  from peft import PeftModel\n"
        f"  pipe = ZImagePipeline.from_pretrained(MODEL_BASE, torch_dtype=torch.bfloat16)\n"
        f"  pipe.transformer = PeftModel.from_pretrained(pipe.transformer, '{final_dir}')\n"
        f"  pipe.to('cuda')"
    )


if __name__ == "__main__":
    main()
