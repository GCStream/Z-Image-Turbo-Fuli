#!/usr/bin/env python3
"""
Stage 3 — LoRA artist eval: Original | Base | LoRA
===================================================

Generates 3-column panels for each sampled artist image:
  1. Original photo  (from fuliji_dataset.parquet — the ground truth)
  2. Base model output  (Z-Image base, 50 steps, CFG=4)
  3. LoRA-FT output     (same prompt + adapter, 50 steps, CFG=4)

The prompt fed to cols 2 & 3 is the same trigger-tagged caption used during
training:
    portrait of {artist_name}, {fuliji_tags}, {image_tags[:8]}

This directly measures how well the LoRA learned to reproduce the artist's
likeness from the trigger token alone.

Usage
-----
  # After lora_artist_run01 finishes:
  python3 stage3_finetune/eval_lora_artist.py

  python3 stage3_finetune/eval_lora_artist.py \\
      --adapter /scratch/training/lora_artist_run01/final_adapter \\
      --n_artist 20 --n_sfw 5 --seed 42 \\
      --out outputs/eval_lora_artist
"""

import argparse
import io
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
from diffusers import ZImagePipeline
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel

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

PARQUET = Path("/home/test/fetch-telegram/fuliji_dataset.parquet")
PATCH_MULTIPLE = 16

# CJK-capable font — required for Chinese artist names in panel labels.
# Noto Sans CJK covers simplified/traditional Chinese, Japanese, Korean.
CJK_FONT = _ROOT / "assets" / "fonts" / "NotoSansCJK-Regular.ttc"
LATIN_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LATIN_FONT      = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _get_font(size: int, bold: bool = False) -> "ImageFont.FreeTypeFont":
    """Return a font that can render CJK characters."""
    if CJK_FONT.exists():
        try:
            return ImageFont.truetype(str(CJK_FONT), size)
        except Exception:
            pass
    fallback = LATIN_FONT_BOLD if bold else LATIN_FONT
    try:
        return ImageFont.truetype(fallback, size)
    except Exception:
        return ImageFont.load_default()

SFW_PROMPTS = [
    "a golden retriever puppy playing in autumn leaves",
    "a neon-lit cyberpunk alleyway at night, rainy, reflections on the wet ground",
    "a watercolor painting of a Japanese mountain village in spring",
    "an oil painting portrait of an elderly sea captain with a weathered face",
    "a hyper-realistic photo of a red apple on a marble surface, studio lighting",
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


# ── caption builder (mirrors train_lora_artist.py) ───────────────────────────

def build_caption(artist_name: str, fuliji_tags, image_tags) -> str:
    phys  = ", ".join(str(t) for t in list(fuliji_tags)  if t)
    scene = ", ".join(str(t) for t in list(image_tags)[:8] if t)
    parts = [f"portrait of {artist_name}"]
    if phys:
        parts.append(phys)
    if scene:
        parts.append(scene)
    return ", ".join(parts)


# ── image resizing helpers ────────────────────────────────────────────────────

def _round_down(x: int, mult: int) -> int:
    return (x // mult) * mult


def fit_to_res(img: Image.Image, res: int) -> Image.Image:
    """
    Resize so shortest side = res, then centre-crop both dims to
    the nearest PATCH_MULTIPLE below the resized dimensions.
    Matches the preprocessing used in train_lora_artist.py.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = res / min(w, h)
    nw = max(res, int(w * scale))
    nh = max(res, int(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    cw = _round_down(nw, PATCH_MULTIPLE)
    ch = _round_down(nh, PATCH_MULTIPLE)
    x0 = (nw - cw) // 2
    y0 = (nh - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def square_crop(img: Image.Image, size: int) -> Image.Image:
    """Centre-square-crop then resize to (size, size) for uniform panel columns."""
    w, h = img.size
    s = min(w, h)
    x0 = (w - s) // 2
    y0 = (h - s) // 2
    return img.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.LANCZOS)


# ── dataset sampling ──────────────────────────────────────────────────────────

def sample_artist_rows(parquet: Path, n: int, seed: int) -> list[dict]:
    """
    Sample ``n`` rows, trying to cover diverse artists.
    Returns list of dicts with keys: image_bytes, artist, caption, fuliji_tags, image_tags.
    """
    df = pd.read_parquet(parquet)
    rng = random.Random(seed)

    # Group by artist, pick one image per artist, then sample n artists
    artists = df["fuliji"].unique().tolist()
    rng.shuffle(artists)
    selected = artists[:n]

    rows = []
    for artist in selected:
        group = df[df["fuliji"] == artist]
        row = group.sample(1, random_state=seed).iloc[0]
        ft = list(row["fuliji_tags"]) if hasattr(row["fuliji_tags"], "__iter__") else []
        it = list(row["image_tags"])  if hasattr(row["image_tags"],  "__iter__") else []
        rows.append({
            "image_bytes": row["image"]["bytes"],
            "artist":      str(row["fuliji"]),
            "gallery":     str(row.get("gallery", "")),
            "caption":     build_caption(row["fuliji"], ft, it),
            "fuliji_tags": ft,
            "image_tags":  it,
        })
    return rows


# ── generation ────────────────────────────────────────────────────────────────

def gen(pipe, prompt: str, steps: int, guidance: float,
        res: int, seed: int, neg: str = "") -> Image.Image:
    g = torch.Generator("cuda").manual_seed(seed)
    kwargs = dict(
        prompt=prompt, height=res, width=res,
        num_inference_steps=steps, guidance_scale=guidance, generator=g,
    )
    if neg:
        kwargs["negative_prompt"] = neg
    with torch.no_grad():
        return pipe(**kwargs).images[0]


# ── panel rendering ───────────────────────────────────────────────────────────

def _label_bar(w: int, text: str, bg, fg=(255, 255, 255), fs: int = 18) -> Image.Image:
    bar = Image.new("RGB", (w, 34), bg)
    d   = ImageDraw.Draw(bar)
    font = _get_font(fs, bold=True)
    d.text((8, 8), text, fill=fg, font=font)
    return bar


def _prompt_strip(w: int, text: str, fs: int = 13) -> Image.Image:
    # Use a character-width estimate safe for CJK (each CJK char ≈ 2 latin chars).
    # textwrap doesn't know about CJK widths so we wrap manually.
    max_chars = max(1, w // (fs + 2))
    words = text
    lines, line = [], ""
    for ch in words:
        line += ch
        # rough display-width estimate: CJK counts as 2, ASCII as 1
        dw = sum(2 if ord(c) > 0x2E7F else 1 for c in line)
        if dw >= max_chars:
            lines.append(line)
            line = ""
        if len(lines) == 5:
            break
    if line and len(lines) < 5:
        lines.append(line)
    if not lines:
        lines = [text[:max_chars]]
    lh = fs + 5
    h  = lh * len(lines) + 12
    s  = Image.new("RGB", (w, max(h, lh + 12)), (22, 22, 22))
    d  = ImageDraw.Draw(s)
    font = _get_font(fs)
    for i, ln in enumerate(lines):
        d.text((6, 6 + i * lh), ln, fill=(200, 200, 200), font=font)
    return s


def build_panel(idx: int, orig: Image.Image,
                img_base: Image.Image, img_lora: Image.Image,
                artist: str, gallery: str, caption: str,
                category: str = "artist") -> Image.Image:
    """
    3-column panel:
      Col 0  Original photo   — dark green header
      Col 1  Base model       — dark blue header
      Col 2  LoRA model       — dark red header
    """
    SEP     = 6
    LABEL_H = 34
    imgs    = [orig, img_base, img_lora]
    labels  = [
        ("Original", (20, 80, 20)),
        ("Base (50-step CFG=4)", (20, 40, 100)),
        ("LoRA (50-step CFG=4)", (120, 30, 30)),
    ]
    res = orig.width
    total_w = res * 3 + SEP * 2

    hdr_text = f"#{idx + 1}  {artist}"
    if gallery:
        hdr_text += f"  ·  {gallery}"
    hdr_text += f"  [{category}]"

    header = _label_bar(total_w, hdr_text, (40, 20, 60))
    pstrip = _prompt_strip(total_w, caption)

    total_h = header.height + pstrip.height + LABEL_H + res
    canvas  = Image.new("RGB", (total_w, total_h), (10, 10, 10))
    canvas.paste(header, (0, 0))
    canvas.paste(pstrip, (0, header.height))

    y = header.height + pstrip.height
    for ci, (img, (lbl, bg)) in enumerate(zip(imgs, labels)):
        x = ci * (res + SEP)
        canvas.paste(_label_bar(res, lbl, bg), (x, y))
        canvas.paste(img, (x, y + LABEL_H))

    return canvas


def build_grid(panel_paths: list[Path], thumb_w: int = 960) -> Image.Image:
    thumbs = []
    for p in panel_paths:
        img   = Image.open(p)
        scale = thumb_w / img.width
        thumbs.append(img.resize((thumb_w, int(img.height * scale)), Image.LANCZOS))
    grid = Image.new("RGB", (thumb_w, sum(t.height for t in thumbs)))
    y = 0
    for t in thumbs:
        grid.paste(t, (0, y))
        y += t.height
    return grid


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-column eval: Original | Base | LoRA")
    parser.add_argument("--adapter", type=Path,
                        default=Path("/scratch/training/lora_artist_run01/final_adapter"),
                        help="Path to PEFT adapter directory (final_adapter/)")
    parser.add_argument("--model_path", default=None,
                        help="Base pipeline path override (default: Z-Image base)")
    parser.add_argument("--turbo", action="store_true",
                        help="Use Z-Image Turbo as base (8-step, CFG=0)")
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--n_artist", type=int, default=20,
                        help="# artist images to evaluate (one per distinct artist)")
    parser.add_argument("--artist_filter", nargs="+", default=None,
                        help="Restrict eval to these specific artists (space-separated)")
    parser.add_argument("--n_sfw",    type=int, default=5,
                        help="# SFW anchor prompts (no original image, just base vs lora)")
    parser.add_argument("--res",      type=int, default=512,
                        help="Generation resolution (square)")
    parser.add_argument("--steps",    type=int, default=None,
                        help="Inference steps (default: 8 for Turbo, 50 for base)")
    parser.add_argument("--cfg",      type=float, default=None,
                        help="CFG scale (default: 0 for Turbo, 4 for base)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA weight multiplier (default 1.0 = alpha/rank = 1)")
    parser.add_argument("--neg",      type=str,
                        default="blurry, low quality, deformed, watermark, text, logo")
    parser.add_argument("--out",      type=Path,

                        default=_ROOT / "outputs" / "eval_lora_artist")
    args = parser.parse_args()

    # Resolve defaults for turbo vs base
    is_turbo    = args.turbo
    model_path  = args.model_path or (MODEL_TURBO if is_turbo else MODEL_BASE)
    steps       = args.steps  if args.steps  is not None else (8  if is_turbo else 50)
    cfg         = args.cfg    if args.cfg    is not None else (0.0 if is_turbo else 4.0)
    neg_prompt  = "" if is_turbo else args.neg

    panels_dir = args.out / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # ── load base pipeline ────────────────────────────────────────────────────
    print(f"Loading base pipeline from {model_path} …")
    pipe_base = ZImagePipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe_base.set_progress_bar_config(disable=True)

    # ── load LoRA pipeline ────────────────────────────────────────────────────
    print(f"Loading LoRA adapter from {args.adapter} …")
    pipe_lora = ZImagePipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe_lora.transformer = PeftModel.from_pretrained(
        pipe_lora.transformer,
        str(args.adapter),
        is_trainable=False,
    )
    if args.lora_scale != 1.0:
        for module in pipe_lora.transformer.modules():
            if hasattr(module, "scaling"):
                module.scaling = {k: v * args.lora_scale for k, v in module.scaling.items()}
        print(f"LoRA scale set to {args.lora_scale}x")
    pipe_lora.set_progress_bar_config(disable=True)

    # ── sample data ───────────────────────────────────────────────────────────
    if args.artist_filter:
        import pandas as _pd
        _df = _pd.read_parquet(args.parquet)
        artist_rows = []
        for artist in args.artist_filter:
            group = _df[_df["fuliji"] == artist]
            if len(group) == 0:
                print(f"[WARN] artist '{artist}' not found in parquet, skipping")
                continue
            row = group.sample(1, random_state=args.seed).iloc[0]
            ft = list(row["fuliji_tags"]) if hasattr(row["fuliji_tags"], "__iter__") else []
            it = list(row["image_tags"])  if hasattr(row["image_tags"],  "__iter__") else []
            artist_rows.append({
                "image_bytes": row["image"]["bytes"],
                "artist":      str(row["fuliji"]),
                "gallery":     str(row.get("gallery", "")),
                "caption":     build_caption(row["fuliji"], ft, it),
                "fuliji_tags": ft,
                "image_tags":  it,
            })
        print(f"\nUsing {len(artist_rows)} artist filter rows from {args.parquet} …")
    else:
        print(f"\nSampling {args.n_artist} artist rows from {args.parquet} …")
        artist_rows = sample_artist_rows(args.parquet, args.n_artist, args.seed)

    panel_paths = []
    metadata    = []
    total       = len(artist_rows) + args.n_sfw
    print(f"{total} panels ({args.n_artist} artist + {args.n_sfw} SFW)\n")

    # ── artist panels ─────────────────────────────────────────────────────────
    for i, row in enumerate(artist_rows):
        caption = row["caption"]
        artist  = row["artist"]
        short   = caption[:80] + ("…" if len(caption) > 80 else "")
        print(f"[{i+1:2d}/{total}] {artist}  {short}", flush=True)

        # Original image — square-cropped to res
        try:
            orig_pil = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
            orig_sq  = square_crop(orig_pil, args.res)
        except Exception as e:
            print(f"  [WARN] corrupt image for {artist}, using blank: {e}")
            orig_sq = Image.new("RGB", (args.res, args.res), (30, 30, 30))

        img_base = gen(pipe_base, caption, steps=steps, guidance=cfg,
                       res=args.res, seed=args.seed, neg=neg_prompt)
        img_lora = gen(pipe_lora, caption, steps=steps, guidance=cfg,
                       res=args.res, seed=args.seed, neg=neg_prompt)

        panel = build_panel(i, orig_sq, img_base, img_lora,
                            artist=artist, gallery=row["gallery"],
                            caption=caption, category="artist")
        p = panels_dir / f"panel_{i:03d}.jpg"
        panel.save(p, quality=90)
        panel_paths.append(p)
        print(f"  → {p.name}")

        metadata.append({
            "idx": i, "category": "artist",
            "artist": artist, "gallery": row["gallery"],
            "caption": caption,
        })

    # ── SFW anchor panels ─────────────────────────────────────────────────────
    sfw_start = len(artist_rows)
    blank     = Image.new("RGB", (args.res, args.res), (30, 30, 30))
    blank_d   = ImageDraw.Draw(blank)
    blank_font = _get_font(18)
    blank_d.text((10, args.res // 2 - 10), "N/A (SFW prompt)",
                 fill=(100, 100, 100), font=blank_font)

    for j, sfw in enumerate(SFW_PROMPTS[:args.n_sfw]):
        idx = sfw_start + j
        print(f"[{idx+1:2d}/{total}] [SFW] {sfw[:80]}", flush=True)

        img_base = gen(pipe_base, sfw, steps=steps, guidance=cfg,
                       res=args.res, seed=args.seed, neg=neg_prompt)
        img_lora = gen(pipe_lora, sfw, steps=steps, guidance=cfg,
                       res=args.res, seed=args.seed, neg=neg_prompt)

        panel = build_panel(idx, blank, img_base, img_lora,
                            artist="SFW control", gallery="",
                            caption=sfw, category="SFW")
        p = panels_dir / f"panel_{idx:03d}.jpg"
        panel.save(p, quality=90)
        panel_paths.append(p)
        print(f"  → {p.name}")

        metadata.append({"idx": idx, "category": "SFW", "caption": sfw})

    # ── summary grid ──────────────────────────────────────────────────────────
    grid_path = args.out / "summary_grid.jpg"
    build_grid(panel_paths).save(grid_path, quality=85)
    meta_path = args.out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(f"\nSummary grid → {grid_path}")
    print(f"Metadata     → {meta_path}")
    print(f"\nDone. {len(panel_paths)} panels in {panels_dir}")


if __name__ == "__main__":
    main()
