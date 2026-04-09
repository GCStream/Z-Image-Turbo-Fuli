#!/usr/bin/env python3
"""
Stage 3B — Full fine-tune evaluation: Base | Base-FT | Turbo | Turbo-FT
=========================================================================
Generates 4-column side-by-side panels and a summary grid.

Columns per panel:
  1. Z-Image base      (50 steps, CFG=4)
  2. Z-Image FullFT    (50 steps, CFG=4)
  3. Z-Image Turbo     (8 steps,  CFG=0)
  4. Z-Image Turbo-FT  (8 steps,  CFG=0)

Prompts: sampled from finetune_dataset/metadata.jsonl (text_en).

Usage:
  python3 stage3_finetune/eval_fullft.py
  python3 stage3_finetune/eval_fullft.py \
      --ft_path /scratch/training/fullft_nsfw_run03/final_transformer \
      --turbo_ft_path /scratch/training/fullft_turbo_run01/final_transformer \
      --n_train 15 --n_sfw 5 --res 512 --seed 42
"""

import argparse
import json
import os
import random
import textwrap
from pathlib import Path

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel
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

DATASET_DIR = _ROOT / "finetune_dataset"

SFW_PROMPTS = [
    "a golden retriever puppy playing in autumn leaves",
    "a neon-lit cyberpunk alleyway at night, rainy, reflections on the wet ground",
    "a watercolor painting of a Japanese mountain village in spring",
    "an oil painting portrait of an elderly sea captain with a weathered face",
    "a hyper-realistic photo of a red apple on a marble surface, studio lighting",
    "concept art of a futuristic space station interior, dramatic lighting",
    "a smiling sloth wearing a bowtie holding a book, cute illustration",
]


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

def load_prompts(n_train: int, n_sfw: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    prompts = []

    meta = DATASET_DIR / "metadata.jsonl"
    if meta.exists():
        rows = [json.loads(l) for l in open(meta) if l.strip()]
        rows = [r for r in rows if r.get("text_en") and r.get("status", "ok") in ("ok", "ok_mono")]
        rng.shuffle(rows)
        for r in rows[:n_train]:
            prompts.append({"prompt": r["text_en"], "category": "train-dist",
                            "source": r.get("source", "")})
    else:
        print("[WARN] finetune_dataset/metadata.jsonl not found; falling back to SFW only")

    for p in SFW_PROMPTS[:n_sfw]:
        prompts.append({"prompt": p, "category": "SFW", "source": ""})

    return prompts


# ── generation ─────────────────────────────────────────────────────────────────

def gen(pipe, prompt: str, steps: int, guidance: float,
        res_h: int, res_w: int, seed: int, neg: str = "") -> Image.Image:
    g = torch.Generator("cuda").manual_seed(seed)
    kwargs = dict(prompt=prompt, height=res_h, width=res_w,
                  num_inference_steps=steps, guidance_scale=guidance, generator=g)
    if neg:
        kwargs["negative_prompt"] = neg
    with torch.no_grad():
        return pipe(**kwargs).images[0]


# ── panel helpers ──────────────────────────────────────────────────────────────

def _label_bar(w: int, text: str, bg, fg=(255, 255, 255), fs: int = 18) -> Image.Image:
    s = Image.new("RGB", (w, 34), bg)
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 8), text, fill=fg, font=font)
    return s


def _prompt_strip(w: int, text: str, fs: int = 13) -> Image.Image:
    wrapped = textwrap.wrap(text, width=max(1, w // (fs // 2 + 2)))[:4]
    lh = fs + 5
    h = lh * len(wrapped) + 12
    s = Image.new("RGB", (w, max(h, lh + 12)), (22, 22, 22))
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    for i, line in enumerate(wrapped):
        d.text((6, 6 + i * lh), line, fill=(200, 200, 200), font=font)
    return s


def build_panel(idx: int, cols_data: list[tuple],
                prompt: str, category: str, source: str) -> Image.Image:
    """cols_data: list of (image, label_text, bg_color)."""
    SEP = 6
    LABEL_H = 34
    ncols = len(cols_data)
    res_w, res_h = cols_data[0][0].size
    total_w = res_w * ncols + SEP * (ncols - 1)

    hdr_text = f"#{idx + 1}  [{category}]"
    if source:
        hdr_text += f"  {source}"
    header  = _label_bar(total_w, hdr_text, (40, 20, 60))
    pstrip  = _prompt_strip(total_w, prompt)
    total_h = header.height + pstrip.height + LABEL_H + res_h
    canvas  = Image.new("RGB", (total_w, total_h), (10, 10, 10))

    canvas.paste(header, (0, 0))
    canvas.paste(pstrip, (0, header.height))

    y = header.height + pstrip.height
    for ci, (img, lbl, bg) in enumerate(cols_data):
        x = ci * (res_w + SEP)
        canvas.paste(_label_bar(res_w, lbl, bg), (x, y))
        canvas.paste(img, (x, y + LABEL_H))

    return canvas


def build_grid(panel_paths: list[Path], thumb_w: int = 960) -> Image.Image:
    thumbs = []
    for p in panel_paths:
        img   = Image.open(p)
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
    parser.add_argument("--ft_path",       type=Path,
                        default=Path("/scratch/training/fullft_nsfw_run03/final_transformer"))
    parser.add_argument("--turbo_ft_path", type=Path,
                        default=Path("/scratch/training/fullft_turbo_run01/final_transformer"))
    parser.add_argument("--n_train", type=int, default=15,
                        help="# prompts from training distribution")
    parser.add_argument("--n_sfw",   type=int, default=5,
                        help="# SFW control prompts")
    parser.add_argument("--res",     type=int, default=512)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--neg",     type=str,
                        default="blurry, low quality, deformed, watermark, text, logo")
    parser.add_argument("--out",     type=Path,
                        default=_ROOT / "outputs" / "eval_fullft")
    args = parser.parse_args()

    panels_dir = args.out / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # ── load base (50 steps, CFG=4) ────────────────────────────────────────────
    print("Loading Z-Image base …")
    pipe_base = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe_base.set_progress_bar_config(disable=True)

    # ── load base fine-tuned ───────────────────────────────────────────────────
    print(f"Loading Z-Image FullFT from {args.ft_path} …")
    pipe_ft = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe_ft.transformer = ZImageTransformer2DModel.from_pretrained(
        args.ft_path, torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe_ft.set_progress_bar_config(disable=True)

    # ── load Turbo (8 steps, CFG=0) ────────────────────────────────────────────
    print("Loading Z-Image Turbo …")
    pipe_turbo = ZImagePipeline.from_pretrained(
        MODEL_TURBO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe_turbo.set_progress_bar_config(disable=True)

    # ── load Turbo fine-tuned ──────────────────────────────────────────────────
    print(f"Loading Z-Image Turbo-FT from {args.turbo_ft_path} …")
    pipe_turbo_ft = ZImagePipeline.from_pretrained(
        MODEL_TURBO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe_turbo_ft.transformer = ZImageTransformer2DModel.from_pretrained(
        args.turbo_ft_path, torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe_turbo_ft.set_progress_bar_config(disable=True)

    # ── prompts ────────────────────────────────────────────────────────────────
    prompts = load_prompts(args.n_train, args.n_sfw, args.seed)
    total   = len(prompts)
    print(f"\n{total} prompts ({args.n_train} train-dist + {args.n_sfw} SFW)\n")

    panel_paths = []
    metadata    = []

    for i, item in enumerate(prompts):
        prompt   = item["prompt"]
        category = item["category"]
        source   = item.get("source", "")
        short    = prompt[:80] + ("…" if len(prompt) > 80 else "")
        print(f"[{i+1:2d}/{total}] [{category}] {short}", flush=True)

        img_base     = gen(pipe_base,     prompt, steps=50, guidance=4.0,
                           res_h=args.res, res_w=args.res, seed=args.seed, neg=args.neg)
        img_ft       = gen(pipe_ft,       prompt, steps=50, guidance=4.0,
                           res_h=args.res, res_w=args.res, seed=args.seed, neg=args.neg)
        img_turbo    = gen(pipe_turbo,    prompt, steps=8,  guidance=0.0,
                           res_h=args.res, res_w=args.res, seed=args.seed)
        img_turbo_ft = gen(pipe_turbo_ft, prompt, steps=8,  guidance=0.0,
                           res_h=args.res, res_w=args.res, seed=args.seed)

        cols_data = [
            (img_base,     "Base  (50 steps, CFG=4)",       (0,   70, 140)),
            (img_ft,       "Base-FT  (50 steps, CFG=4)",    (0,  130,  50)),
            (img_turbo,    "Turbo  (8 steps, CFG=0)",       (140, 70,   0)),
            (img_turbo_ft, "Turbo-FT  (8 steps, CFG=0)",    (120,  0, 120)),
        ]
        panel = build_panel(i, cols_data, prompt, category, source)
        panel_path = panels_dir / f"panel_{i:03d}.jpg"
        panel.save(panel_path, quality=92)
        panel_paths.append(panel_path)

        metadata.append({
            "idx": i, "prompt": prompt, "category": category, "source": source,
            "panel": str(panel_path.relative_to(_ROOT)),
        })
        print(f"  → saved {panel_path.name}", flush=True)

    # ── summary grid ──────────────────────────────────────────────────────────
    grid = build_grid(panel_paths)
    grid_path = args.out / "summary_grid.jpg"
    grid.save(grid_path, quality=88)
    print(f"\nSummary grid → {grid_path}")

    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(f"Metadata     → {args.out / 'metadata.json'}")
    print(f"\nDone. {total} panels in {panels_dir}")


if __name__ == "__main__":
    main()
