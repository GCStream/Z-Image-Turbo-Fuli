#!/usr/bin/env python3
"""
Stage 3 — LoRA fine-tuning for artist face/style identity learning
===================================================================

Goal
----
Make Z-Image "know" each artist (fuliji) by name.  After training, a prompt
such as:

    portrait of 萌芽儿o0, outdoor sunshine, vivid colours

…should reliably produce that artist's likeness without any reference image.

Architecture choices
--------------------
We target two complementary layer groups in the Z-Image DiT transformer:

1. Attention projections  (to_q / to_k / to_v)
   These control *how* the model attends to text tokens and to spatial features.
   Teaching them the artist name → face mapping enables the "Builder" to focus
   on the right visual cues when the trigger word appears in the prompt.

2. Feed-forward projections  (w1 / w2 / w3)
   The MLP blocks encode per-token style and texture information.  Fine-tuning
   them lets the "Architect" translate the trigger token into the artist's
   characteristic lighting, colour palette, and expression style.

Regularisation dataset
----------------------
With only ~1 177 images across 405 artists over-specialisation is a real risk.
The model might "forget" how to do standard T2I if all training gradients push
in the artist-identity direction.

Solution: interleave samples from the existing `finetune_dataset/` (or any
`--reg_dataset` directory) at a configurable ratio (default 1 reg per 4 artist
steps).  Regularisation captions are *untouched* standard captions — they act
as an anchor so the model retains general T2I competence.

Caption format
--------------
Artist samples:
    portrait of {artist_name}, {fuliji_tags}, {image_tags}

Regularisation samples:
    (original caption from metadata.jsonl / text_en column unchanged)

Training loop
-------------
  - Flow matching loss (flow_matching_loss with min-SNR-5 weighting)
  - Logit-normal timestep sampling (sigma controlled by --timestep_bias)
  - Caption dropout on *artist* samples only (helps CFG amplify style)
  - EMA on LoRA adapter weights
  - LoRA adapter saved via PEFT to ``final_adapter/``

Usage
-----
  python3 stage3_finetune/train_lora_artist.py \\
      --parquet /home/test/fetch-telegram/fuliji_dataset.parquet \\
      --reg_dataset finetune_dataset \\
      --steps 2000 --lr 1e-4 --rank 32 \\
      --flip_aug --caption_dropout 0.05 --ema_decay 0.9999 \\
      --grad_ckpt \\
      --output_dir /scratch/training/lora_artist_run01

Inference
---------
  from diffusers import ZImagePipeline
  from peft import PeftModel

  pipe = ZImagePipeline.from_pretrained(MODEL_BASE, torch_dtype=torch.bfloat16)
  pipe.transformer = PeftModel.from_pretrained(
      pipe.transformer, "/scratch/training/lora_artist_run01/final_adapter"
  )
  pipe.to("cuda")
  image = pipe("portrait of 萌芽儿o0, professional lighting", ...).images[0]
"""

import argparse
import copy
import io
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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

# ── constants ─────────────────────────────────────────────────────────────────
VAE_SCALE    = 0.3611
VAE_SHIFT    = 0.1159
PATCH_MULTIPLE = 16


# ── VAE helpers ────────────────────────────────────────────────────────────────

def encode_image(vae: AutoencoderKL, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    return (latents - VAE_SHIFT) * VAE_SCALE


# ── text encoding ──────────────────────────────────────────────────────────────

def encode_text(pipe: ZImagePipeline, prompts: list[str],
                device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
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
    """Per-sample spatial MSE with min-SNR-γ weighting."""
    target = noise - latents
    diff = (pred.float() - target.float()) ** 2
    per_sample = diff.mean(dim=list(range(1, diff.ndim)))   # (B,)
    snr    = (1.0 - sigmas) ** 2 / (sigmas ** 2 + 1e-8)
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


# ── image transform ───────────────────────────────────────────────────────────

def _round_to_mult(x: int, mult: int) -> int:
    return (x // mult) * mult


def make_transform(min_side: int = 512, flip_aug: bool = False):
    """
    Resize shortest side to ``min_side``, floor-round to PATCH_MULTIPLE,
    centre-crop, normalise to [-1, 1].
    """
    to_tensor = transforms.ToTensor()
    normalise = transforms.Normalize([0.5], [0.5])

    def transform(img: Image.Image) -> torch.Tensor:
        if flip_aug and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.convert("RGB")
        w, h = img.size
        scale = min_side / min(w, h)
        new_w = max(min_side, int(w * scale))
        new_h = max(min_side, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        crop_w = _round_to_mult(new_w, PATCH_MULTIPLE)
        crop_h = _round_to_mult(new_h, PATCH_MULTIPLE)
        x0 = (new_w - crop_w) // 2
        y0 = (new_h - crop_h) // 2
        img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        return normalise(to_tensor(img))

    return transform


# ── artist dataset from parquet ───────────────────────────────────────────────

def build_caption(artist_name: str, fuliji_tags: list, image_tags: list) -> str:
    """
    Construct a trigger-tagged caption.

    Format: ``portrait of {artist_name}, {physical_traits}, {scene_tags}``

    The artist name acts as the trigger token.  Physical traits (fuliji_tags)
    appear on every image of that artist so the model learns to associate them
    with the trigger.  Scene / outfit tags (image_tags) ground the per-image
    context.
    """
    phys  = ", ".join(str(t) for t in fuliji_tags  if t)
    scene = ", ".join(str(t) for t in image_tags[:8]  if t)   # cap scene tags for brevity
    parts = [f"portrait of {artist_name}"]
    if phys:
        parts.append(phys)
    if scene:
        parts.append(scene)
    return ", ".join(parts)


class ParquetArtistDataset:
    """
    Loads the fuliji parquet once into RAM and yields shuffled (image, caption)
    pairs.  Each epoch reshuffles the order.
    """

    def __init__(self, parquet_path: str, res: int = 512, flip_aug: bool = False,
                 min_count: int = 1):
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        if min_count > 1:
            counts = df["fuliji"].value_counts()
            keep   = counts[counts >= min_count].index
            df     = df[df["fuliji"].isin(keep)]
            print(f"  Filtered to artists with >={min_count} images: "
                  f"{len(keep)} artists, {len(df)} images")
        self._transform = make_transform(res, flip_aug)
        self._records: list[tuple[bytes, str]] = []

        import numpy as np
        for _, row in df.iterrows():
            img_bytes = row["image"]["bytes"]
            artist    = str(row["fuliji"])
            raw_ft = row.get("fuliji_tags")
            raw_it = row.get("image_tags")
            f_tags = list(raw_ft) if (raw_ft is not None and len(raw_ft) > 0) else []
            i_tags = list(raw_it) if (raw_it is not None and len(raw_it) > 0) else []
            caption   = build_caption(artist, f_tags, i_tags)
            self._records.append((img_bytes, caption))

        print(f"  Artist dataset: {len(self._records)} images loaded into RAM.")

    def __len__(self):
        return len(self._records)

    def infinite_iter(self, batch_size: int = 1):
        """Yield (pixel_values_tensor, [caption]) batches forever."""
        indices = list(range(len(self._records)))
        while True:
            random.shuffle(indices)
            buf_imgs, buf_texts = [], []
            for idx in indices:
                raw_bytes, caption = self._records[idx]
                try:
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    buf_imgs.append(self._transform(img))
                    buf_texts.append(caption)
                except Exception:
                    continue
                if len(buf_imgs) == batch_size:
                    yield torch.stack(buf_imgs), buf_texts
                    buf_imgs, buf_texts = [], []


# ── regularisation dataset from local imagefolder ────────────────────────────

class RegDataset:
    """
    Loads a local ``finetune_dataset/`` or similar imagefolder with a
    ``metadata.jsonl`` file.  Returns (image, caption) pairs for anchoring.
    """

    def __init__(self, dataset_dir: str, text_col: str = "text_en",
                 res: int = 512, flip_aug: bool = False):
        from pathlib import Path as _Path
        ddir = _Path(dataset_dir)
        meta_path = ddir / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found in {ddir}")

        self._transform = make_transform(res, flip_aug=False)   # no flip for reg
        self._records: list[tuple[str, str]] = []               # (img_path, caption)

        with open(meta_path) as f:
            for line in f:
                row = json.loads(line)
                fname = row.get("file_name") or row.get("image")
                caption = str(row.get(text_col, "") or "")
                if fname and caption:
                    self._records.append((str(ddir / fname), caption))

        print(f"  Reg dataset: {len(self._records)} images from {dataset_dir}")

    def __len__(self):
        return len(self._records)

    def infinite_iter(self, batch_size: int = 1):
        indices = list(range(len(self._records)))
        while True:
            random.shuffle(indices)
            buf_imgs, buf_texts = [], []
            for idx in indices:
                fpath, caption = self._records[idx]
                try:
                    img = Image.open(fpath).convert("RGB")
                    buf_imgs.append(self._transform(img))
                    buf_texts.append(caption)
                except Exception:
                    continue
                if len(buf_imgs) == batch_size:
                    yield torch.stack(buf_imgs), buf_texts
                    buf_imgs, buf_texts = [], []


# ── LoRA injection ────────────────────────────────────────────────────────────

def inject_lora(transformer, rank: int, alpha: float,
                target_modules: list[str], dropout: float = 0.0):
    """
    Wrap ``transformer`` with PEFT LoRA adapters.

    Returns the peft-wrapped model.  Only the adapter weights are trainable;
    the base weights are frozen.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(transformer, config)
    peft_model.print_trainable_parameters()
    return peft_model


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    scratch = os.environ.get("TRAINING_SCRATCH", "/tmp")
    default_out = Path(scratch) / "lora_artist_run01"

    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for artist identity on Z-Image"
    )
    # ── data ──
    parser.add_argument("--parquet", default="/home/test/fetch-telegram/fuliji_dataset.parquet",
                        help="Path to fuliji_dataset.parquet")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Only train on artists with at least this many images. "
                             "E.g. --min_count 21 selects the 8 artists with >20 images.")
    parser.add_argument("--reg_dataset", default=str(_ROOT / "finetune_dataset"),
                        help="Local imagefolder for regularisation (metadata.jsonl required)")
    parser.add_argument("--reg_text_col", default="text_en",
                        help="Caption column name in reg dataset metadata.jsonl")
    parser.add_argument("--reg_ratio", type=float, default=0.25,
                        help="Fraction of optimizer steps that use a reg sample instead "
                             "of an artist sample. 0.25 = 1 reg per 4 artist steps.")
    # ── model ──
    parser.add_argument("--model_path", default=None,
                        help="Pipeline path override (default: Z-Image base)")
    # ── LoRA ──
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA rank (r). Higher = more capacity but more VRAM.")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha. Defaults to rank (scale=1.0).")
    parser.add_argument("--target_modules", nargs="+",
                        default=["to_q", "to_k", "to_v", "w1", "w2", "w3"],
                        help="Transformer linear modules to target with LoRA. "
                             "to_q/to_k/to_v = attention; w1/w2/w3 = MLP.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # ── training ──
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-step batch size (keep at 1 for variable resolutions)")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch×accum)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate for LoRA (higher than full-ft because "
                             "only adapter weights update)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Gradient checkpointing (saves ~40%% VRAM)")
    parser.add_argument("--flip_aug", action="store_true",
                        help="Random horizontal flip on artist images")
    parser.add_argument("--caption_dropout", type=float, default=0.05,
                        help="Fraction of artist steps where caption is zeroed "
                             "(trains unconditional mode for CFG amplification)")
    parser.add_argument("--timestep_bias", type=float, default=1.2,
                        help="Logit-normal sigma for timestep sampling. "
                             ">1 biases toward low-noise (style) steps.")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay for LoRA adapter weights. 0 = disabled.")
    parser.add_argument("--output_dir", type=Path, default=default_out)
    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = float(args.rank)   # scale=1 as default

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    print(f"LoRA artist training  rank={args.rank}  alpha={args.lora_alpha}")
    print(f"Target modules: {args.target_modules}")
    print(f"Steps={args.steps}  lr={args.lr:.1e}  grad_accum={args.grad_accum}")
    print(f"Reg ratio={args.reg_ratio}  caption_dropout={args.caption_dropout}")

    # ── load pipeline (base weights frozen) ──────────────────────────────────
    model_path = args.model_path or MODEL_BASE
    print(f"\nLoading pipeline from {model_path} …")
    pipe = ZImagePipeline.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=False,
    ).to(device)

    pipe.vae.requires_grad_(False).eval()
    for name in ("text_encoder", "tokenizer", "cap_embedder"):
        mod = getattr(pipe, name, None)
        if mod is not None and hasattr(mod, "requires_grad_"):
            mod.requires_grad_(False)
            if hasattr(mod, "eval"):
                mod.eval()

    scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
    scheduler.set_timesteps(1000)

    # ── inject LoRA into transformer ─────────────────────────────────────────
    print("\nInjecting LoRA adapters …")
    pipe.transformer.requires_grad_(False)   # freeze base weights first
    peft_transformer = inject_lora(
        pipe.transformer,
        rank=args.rank,
        alpha=args.lora_alpha,
        target_modules=args.target_modules,
        dropout=args.lora_dropout,
    )
    pipe.transformer = peft_transformer   # swap in peft model

    if args.grad_ckpt:
        # Try to enable grad checkpointing on the underlying base model
        base_model = peft_transformer.base_model.model
        if hasattr(base_model, "enable_gradient_checkpointing"):
            base_model.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled.")

    # ── EMA on LoRA parameters only ──────────────────────────────────────────
    # We deepcopy just the adapter state dict for lightweight EMA tracking.
    ema_state: dict[str, torch.Tensor] | None = None
    if args.ema_decay > 0:
        ema_state = {
            k: v.clone().detach()
            for k, v in peft_transformer.state_dict().items()
            if "lora_" in k
        }
        print(f"EMA enabled (decay={args.ema_decay}, {len(ema_state)} LoRA tensors)")

    # ── optimizer (only LoRA params) ─────────────────────────────────────────
    lora_params = [p for p in peft_transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    lr_sched = get_scheduler(optimizer, args.warmup_steps, args.steps)

    # ── datasets ─────────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    artist_ds = ParquetArtistDataset(args.parquet, res=args.res, flip_aug=args.flip_aug,
                                     min_count=args.min_count)
    artist_iter = artist_ds.infinite_iter(args.batch_size)

    reg_iter = None
    if args.reg_ratio > 0 and args.reg_dataset:
        try:
            reg_ds = RegDataset(args.reg_dataset, text_col=args.reg_text_col,
                                res=args.res)
            reg_iter = reg_ds.infinite_iter(args.batch_size)
        except FileNotFoundError as e:
            print(f"  Warning: reg dataset unavailable ({e}). Continuing without reg.")
            reg_iter = None

    # ── training loop ─────────────────────────────────────────────────────────
    peft_transformer.train()
    eff_batch = args.batch_size * args.grad_accum
    print(f"\nStarting LoRA training:  {args.steps} steps, eff_batch={eff_batch}\n")

    global_step = 0
    accum_loss  = 0.0
    accum_count = 0
    t0 = time.time()
    optimizer.zero_grad()

    # Track reg injection points: inject reg every 1/reg_ratio accum cycles
    reg_period = round(1.0 / args.reg_ratio) if args.reg_ratio > 0 else 0

    for micro_step in range(args.steps * args.grad_accum):
        if global_step >= args.steps:
            break

        # Decide this micro-step's data source
        use_reg = (
            reg_iter is not None
            and reg_period > 0
            and (micro_step % (reg_period * args.grad_accum)) == 0
        )

        if use_reg:
            pixel_values, texts = next(reg_iter)
            is_reg = True
        else:
            pixel_values, texts = next(artist_iter)
            is_reg = False

        pixel_values = pixel_values.to(device=device, dtype=dtype)

        # Encode image (frozen VAE)
        latents = encode_image(pipe.vae, pixel_values)

        # Caption dropout for artist samples only
        if not is_reg and torch.rand(1).item() < args.caption_dropout:
            texts = [""] * len(texts)

        # Encode text (frozen text encoder)
        cap_feats = encode_text(pipe, texts, device, dtype)

        # Logit-normal timestep sampling biased toward low-noise style steps
        B = latents.shape[0]
        u = torch.sigmoid(torch.randn(B, device=device) * args.timestep_bias)
        t_idx = (u * scheduler.config.num_train_timesteps).long().clamp(
            0, scheduler.config.num_train_timesteps - 1
        )
        sigmas_t = scheduler.sigmas.to(device=device, dtype=dtype)[t_idx]
        noise    = torch.randn_like(latents)
        noisy_z  = get_noisy_latents(scheduler, latents, noise, t_idx)

        # Forward through peft-wrapped transformer
        noisy_z_list = list(noisy_z.unsqueeze(2).unbind(0))
        out = peft_transformer(
            x=noisy_z_list,
            t=t_idx.float(),
            cap_feats=cap_feats,
            return_dict=False,
        )
        pred = torch.stack(out[0]).squeeze(2)

        loss = flow_matching_loss(pred, noise, latents, sigmas_t) / args.grad_accum
        loss.backward()
        accum_loss  += loss.item() * args.grad_accum
        accum_count += 1

        if accum_count == args.grad_accum:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()

            # EMA update on LoRA weights
            if ema_state is not None:
                with torch.no_grad():
                    current = peft_transformer.state_dict()
                    for k in ema_state:
                        ema_state[k].mul_(args.ema_decay).add_(
                            current[k].to(ema_state[k].device),
                            alpha=1.0 - args.ema_decay,
                        )

            global_step += 1
            elapsed = time.time() - t0
            sps     = global_step / elapsed
            eta     = (args.steps - global_step) / max(sps, 1e-6)
            avg_loss = accum_loss / accum_count
            src_tag  = "[reg]" if is_reg else "[art]"

            print(
                f"step {global_step:5d}/{args.steps}"
                f"  loss={avg_loss:.5f}"
                f"  lr={lr_sched.get_last_lr()[0]:.2e}"
                f"  {src_tag}"
                f"  {sps:.2f}it/s"
                f"  ETA {eta / 60:.1f}min",
                flush=True,
            )
            accum_loss  = 0.0
            accum_count = 0

            # Mid-run checkpoint (saves full adapter state + EMA)
            if global_step % args.save_every == 0:
                ckpt_dir = args.output_dir / f"adapter_checkpoint_{global_step:05d}"
                _save_adapter(peft_transformer, ema_state, ckpt_dir, is_ema=False)
                print(f"  → checkpoint saved: {ckpt_dir}")
                if ema_state is not None:
                    ckpt_ema_dir = args.output_dir / f"adapter_checkpoint_{global_step:05d}_ema"
                    _save_adapter(peft_transformer, ema_state, ckpt_ema_dir, is_ema=True)
                    print(f"  → EMA checkpoint: {ckpt_ema_dir}")

    # ── final save ─────────────────────────────────────────────────────────────
    final_dir = args.output_dir / "final_adapter"
    if ema_state is not None:
        _save_adapter(peft_transformer, ema_state, final_dir, is_ema=True)
        print(f"\nFinal LoRA adapter (EMA weights) → {final_dir}")
    else:
        _save_adapter(peft_transformer, ema_state, final_dir, is_ema=False)
        print(f"\nFinal LoRA adapter → {final_dir}")

    print(
        "\nLoad with:\n"
        "  from diffusers import ZImagePipeline\n"
        "  from peft import PeftModel\n"
        f"  pipe = ZImagePipeline.from_pretrained(MODEL_BASE, torch_dtype=torch.bfloat16)\n"
        f"  pipe.transformer = PeftModel.from_pretrained(pipe.transformer, '{final_dir}')\n"
        "  pipe.to('cuda')\n"
        "  image = pipe('portrait of 萌芽儿o0, outdoor sunshine', ...).images[0]"
    )


def _save_adapter(peft_transformer, ema_state: dict | None,
                  save_dir: Path, is_ema: bool):
    """
    Save the LoRA adapter.

    If ``is_ema=True`` and ``ema_state`` is provided, temporarily swap the
    EMA weights into the adapter before saving, then restore the live weights.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if is_ema and ema_state is not None:
        # Swap EMA weights in, save, swap back
        live_state = {k: peft_transformer.state_dict()[k].clone() for k in ema_state}
        sd = peft_transformer.state_dict()
        for k, v in ema_state.items():
            sd[k].copy_(v)
        peft_transformer.save_pretrained(str(save_dir))
        # Restore live weights
        sd2 = peft_transformer.state_dict()
        for k, v in live_state.items():
            sd2[k].copy_(v)
    else:
        peft_transformer.save_pretrained(str(save_dir))


if __name__ == "__main__":
    main()
