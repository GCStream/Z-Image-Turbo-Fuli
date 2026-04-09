#!/usr/bin/env python3
"""
Stage 3 — LoRA evaluation: base vs fine-tuned side-by-side
===========================================================

Loads a saved LoRA adapter, runs the same set of prompts through
the base Z-Image and the fine-tuned Z-Image, then assembles:
  - per-pair panels (original | LoRA)
  - summary grid
  - metadata.json

Also optionally tests Turbo transfer.

Usage:
  python3 stage3_finetune/eval_lora.py \
      --lora_path /scratch/training/lora_nsfw_r16/final_adapter \
      --n 20 --res 512
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

import torch
from diffusers import ZImagePipeline
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont

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

DATASET_NSFW = "DRDELATV/SHORT_NSFW"


def _load_env():
    env = _ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()


# ── prompts ────────────────────────────────────────────────────────────────────

def load_test_prompts(n_nsfw: int, n_sfw: int) -> list[dict]:
    """Mix of explicit NSFW prompts from the training dataset + diverse SFW prompts."""
    prompts = []

    # NSFW — same distribution as training data
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_NSFW, split="train", streaming=True,
                          token=os.environ.get("HF_TOKEN"))
        for row in ds:
            if row.get("text"):
                prompts.append({"prompt": row["text"].strip(), "category": "NSFW"})
            if len(prompts) >= n_nsfw:
                break
    except Exception as e:
        print(f"[WARN] Could not load NSFW dataset: {e}")

    # SFW — diverse controls
    sfw = [
        "a golden retriever puppy playing in autumn leaves",
        "a neon-lit cyberpunk alleyway at night, rainy, reflections",
        "a watercolor painting of a Japanese mountain village in spring",
        "an oil painting portrait of an elderly sea captain",
        "a hyper-realistic photo of a red apple on a marble surface",
        "concept art of a futuristic space station interior",
        "a smiling sloth wearing a bowtie and holding a book",
        "an impressionist painting of sunflowers in a field",
        "a detailed illustration of a dragon perched on a cliff",
        "a studio photo of a luxury wristwatch on black velvet",
    ]
    for p in sfw[:n_sfw]:
        prompts.append({"prompt": p, "category": "SFW"})

    return prompts


# ── generation ─────────────────────────────────────────────────────────────────

def gen(pipe, prompt: str, steps: int, guidance: float, res: int, seed: int,
        neg: str = "") -> Image.Image:
    gen_obj = torch.Generator("cuda").manual_seed(seed)
    kwargs = dict(prompt=prompt, height=res, width=res,
                  num_inference_steps=steps, guidance_scale=guidance, generator=gen_obj)
    if neg:
        kwargs["negative_prompt"] = neg
    with torch.no_grad():
        return pipe(**kwargs).images[0]


# ── panel ──────────────────────────────────────────────────────────────────────

def _label(w, text, bg, fg=(255, 255, 255), fs=18):
    s = Image.new("RGB", (w, 32), bg)
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 7), text, fill=fg, font=font)
    return s


def _prompt_strip(w, text, fs=13):
    wrapped = textwrap.wrap(text, width=max(1, w // (fs // 2 + 2)))[:3]
    lh = fs + 5
    h = lh * len(wrapped) + 10
    s = Image.new("RGB", (w, max(h, lh + 10)), (20, 20, 20))
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    for i, line in enumerate(wrapped):
        d.text((6, 5 + i * lh), line, fill=(200, 200, 200), font=font)
    return s


def build_panel(idx, img_base, img_lora, prompt, cat, res,
                col2_label="Z-Image + LoRA", img_turbo=None):
    ncols = 3 if img_turbo is not None else 2
    SEP = 6
    total_w = res * ncols + SEP * (ncols - 1)

    header = _label(total_w, f"#{idx + 1}  [{cat}]", (40, 20, 60))
    pstrip = _prompt_strip(total_w, prompt)

    LABEL_H = 32
    total_h = header.height + pstrip.height + LABEL_H + res

    canvas = Image.new("RGB", (total_w, total_h), (10, 10, 10))
    canvas.paste(header, (0, 0))
    canvas.paste(pstrip, (0, header.height))

    y_lbl = header.height + pstrip.height
    cols = [(img_base, "Z-Image base  (50 steps)", (0, 70, 140)),
            (img_lora, col2_label, (0, 120, 60))]
    if img_turbo is not None:
        cols.append((img_turbo, "Z-Image-Turbo + LoRA", (140, 70, 0)))

    for ci, (img, lbl, bg) in enumerate(cols):
        x = ci * (res + SEP)
        canvas.paste(_label(res, lbl, bg), (x, y_lbl))
        canvas.paste(img, (x, y_lbl + LABEL_H))

    return canvas


def build_grid(panel_paths, thumb_w=960):
    thumbs = []
    for p in panel_paths:
        img = Image.open(p)
        scale = thumb_w / img.width
        thumbs.append(img.resize((thumb_w, int(img.height * scale)), Image.LANCZOS))
    if not thumbs:
        return Image.new("RGB", (thumb_w, 100))
    grid = Image.new("RGB", (thumb_w, sum(t.height for t in thumbs)))
    y = 0
    for t in thumbs:
        grid.paste(t, (0, y))
        y += t.height
    return grid


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path",  type=Path,
                        default=Path("/scratch/training/lora_nsfw_r16/final_adapter"))
    parser.add_argument("--n_nsfw",     type=int, default=15)
    parser.add_argument("--n_sfw",      type=int, default=10)
    parser.add_argument("--res",        type=int, default=512)
    parser.add_argument("--steps",      type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        type=Path,
                        default=_ROOT / "outputs" / "eval_lora")
    parser.add_argument("--with_turbo", action="store_true",
                        help="Also generate with Turbo + LoRA transfer (3-column panels)")
    args = parser.parse_args()

    panels_dir = args.out / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # ── load base pipeline ─────────────────────────────────────────────────────
    print("Loading Z-Image base …")
    pipe_base = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")

    # ── load LoRA pipeline ─────────────────────────────────────────────────────
    print(f"Loading Z-Image + LoRA ({args.lora_path}) …")
    pipe_lora = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe_lora.transformer = PeftModel.from_pretrained(
        pipe_lora.transformer, str(args.lora_path),
    )
    pipe_lora.transformer.eval()

    # ── optional Turbo + LoRA transfer ────────────────────────────────────────
    pipe_turbo_lora = None
    if args.with_turbo:
        print("Loading Z-Image-Turbo + LoRA transfer …")
        pipe_turbo_lora = ZImagePipeline.from_pretrained(
            MODEL_TURBO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
        ).to("cuda")
        lora_state = {k: v for k, v in pipe_lora.transformer.state_dict().items()
                      if "lora_" in k}
        missing, _ = pipe_turbo_lora.transformer.load_state_dict(lora_state, strict=False)
        print(f"  Turbo transfer: {len(lora_state)} tensors, {len(missing)} missing")

    # ── prompts ────────────────────────────────────────────────────────────────
    print(f"\nBuilding test prompts ({args.n_nsfw} NSFW + {args.n_sfw} SFW) …")
    prompts = load_test_prompts(args.n_nsfw, args.n_sfw)
    print(f"  Total: {len(prompts)}")

    # ── generate ───────────────────────────────────────────────────────────────
    print(f"\nGenerating {len(prompts)} pairs at {args.res}px …\n")
    metadata = []
    panel_paths = []
    neg = "blurry, low quality, deformed, text, watermark"

    for i, t in enumerate(prompts):
        prompt = t["prompt"]
        cat    = t["category"]
        seed   = args.seed + i

        print(f"[{i+1:3d}/{len(prompts)}] [{cat}] {prompt[:80]}")

        img_base = gen(pipe_base, prompt, args.steps, 4.0, args.res, seed, neg)
        img_lora = gen(pipe_lora, prompt, args.steps, 4.0, args.res, seed, neg)

        img_turbo = None
        if pipe_turbo_lora is not None:
            img_turbo = gen(pipe_turbo_lora, prompt, 9, 0.0, args.res, seed)

        col2_lbl = f"Z-Image + LoRA r{16}  (50 steps)"
        panel = build_panel(i, img_base, img_lora, prompt, cat, args.res,
                             col2_label=col2_lbl, img_turbo=img_turbo)
        pp = panels_dir / f"{i:03d}_panel.jpg"
        panel.save(pp, quality=88)
        panel_paths.append(pp)

        img_base.save(panels_dir / f"{i:03d}_base.jpg", quality=90)
        img_lora.save(panels_dir / f"{i:03d}_lora.jpg", quality=90)

        metadata.append({"idx": i, "prompt": prompt, "category": cat, "seed": seed})
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ── summary ────────────────────────────────────────────────────────────────
    print("\nBuilding summary grid …")
    grid = build_grid(panel_paths)
    grid_path = args.out / "summary_grid.jpg"
    grid.save(grid_path, quality=85)
    print(f"  {grid_path}  {grid.size}")
    print(f"  {args.out}/metadata.json  ({len(metadata)} entries)")


if __name__ == "__main__":
    main()
