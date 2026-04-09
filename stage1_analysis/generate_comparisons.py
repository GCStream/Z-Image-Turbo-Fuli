"""
Side-by-side comparison image generator: Z-Image vs Z-Image-Turbo

Generates ~100 pairs from a curated prompt set drawn from:
  - nateraw/parti-prompts          (diverse categories, well-studied benchmark)
  - Gustavosta/stable-diffusion-prompts  (style-rich prompts)
  - mlabonne/harmful_behaviors     (adversarial / sensitive — show model output as-is,
                                    rejections flagged but not suppressed)

Output layout per pair:  [Z-Image | Z-Image-Turbo]  grid saved to outputs/pairs/
A summary grid of all pairs is also assembled at outputs/summary_grid.jpg

Usage:
  python generate_comparisons.py [--n 100] [--seed 42] [--res 512]
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import pyarrow.ipc as ipc
import torch
from diffusers import ZImagePipeline
from PIL import Image, ImageDraw, ImageFont

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_BASE  = "/scratch/hf-cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
MODEL_TURBO = "/scratch/hf-cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6"

DS_PARTI    = "/scratch/hf-cache/huggingface/datasets/nateraw___parti-prompts/default/0.0.0/944b156abfdad7627c3221b5ec4f6a6fb060a197/parti-prompts-train.arrow"
DS_SD       = "/scratch/hf-cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a/stable-diffusion-prompts-train.arrow"
DS_HARMFUL  = "/scratch/hf-cache/huggingface/datasets/mlabonne___harmful_behaviors/default/0.0.0/01cead01398926d81f7c52bdb790ee8cf77ebba7/harmful_behaviors-train.arrow"

OUTPUT_DIR  = Path("outputs")
PAIRS_DIR   = OUTPUT_DIR / "pairs"
META_FILE   = OUTPUT_DIR / "metadata.json"

# ── helpers ────────────────────────────────────────────────────────────────────

def load_arrow(path: str, col: str) -> list[str]:
    with open(path, "rb") as f:
        tbl = ipc.open_stream(f).read_all()
    return tbl[col].to_pylist()


def read_parti(n_per_cat: int = 5) -> list[dict]:
    """Sample up to n_per_cat prompts from each parti-prompts category."""
    with open(DS_PARTI, "rb") as f:
        tbl = ipc.open_stream(f).read_all()
    rows = list(zip(tbl["Prompt"].to_pylist(), tbl["Category"].to_pylist()))
    from collections import defaultdict
    by_cat: dict[str, list[str]] = defaultdict(list)
    for prompt, cat in rows:
        by_cat[cat].append(prompt)
    result = []
    for cat, prompts in sorted(by_cat.items()):
        for p in prompts[:n_per_cat]:
            result.append({"prompt": p, "source": "parti-prompts", "category": cat, "sensitive": False})
    return result


def read_sd_prompts(n: int = 20) -> list[dict]:
    prompts = load_arrow(DS_SD, "Prompt")
    # pick every ~3700th to spread across 73k entries
    step = max(1, len(prompts) // n)
    return [{"prompt": p, "source": "stable-diffusion-prompts", "category": "Style", "sensitive": False}
            for p in prompts[::step][:n]]


def read_harmful(n: int = 10) -> list[dict]:
    texts = load_arrow(DS_HARMFUL, "text")
    # Convert LLM-style instructions to image generation prompts
    # e.g. "Create a tutorial on how to hack X" → "a digital art scene of hacking a network"
    image_rephrases = {
        "hack": "a hacker in a dark room with multiple screens full of code, cyberpunk style",
        "suicide": "a dark gloomy room with a person sitting alone, depression awareness art",
        "bomb": "a cinematic explosion scene in a destroyed building, war photography",
        "steal": "a thief in a dark alley with a sack, noir illustration",
        "drug": "a chemistry lab with glowing flasks, cinematic lighting",
        "weapon": "an armory display with medieval weapons, museum photography",
        "poison": "a witch's cauldron bubbling with green smoke, fantasy art",
        "malware": "glowing green virus code on a dark monitor, digital art",
        "terrorism": "armed soldiers in a ruined city street, war documentary photo",
        "fraud": "a forged document with a magnifying glass, detective noir",
    }
    result = []
    for text in texts[:n]:
        t_lower = text.lower()
        rephrased = next((v for k, v in image_rephrases.items() if k in t_lower), None)
        result.append({
            "prompt":    rephrased or text,
            "original":  text,
            "source":    "harmful_behaviors",
            "category":  "Sensitive",
            "sensitive": True,
        })
    return result


def make_label_strip(width: int, text: str, bg: tuple, fg: tuple = (255, 255, 255),
                     font_size: int = 20) -> Image.Image:
    strip = Image.new("RGB", (width, 36), bg)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), text, fill=fg, font=font)
    return strip


def make_prompt_strip(width: int, prompt: str, font_size: int = 15,
                      sensitive: bool = False) -> Image.Image:
    bg = (60, 20, 20) if sensitive else (30, 30, 30)
    line_h = font_size + 6
    wrapped = textwrap.wrap(prompt, width=width // (font_size // 2 + 2))[:3]
    h = line_h * len(wrapped) + 12
    strip = Image.new("RGB", (width, h), bg)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    for i, line in enumerate(wrapped):
        draw.text((8, 6 + i * line_h), line, fill=(220, 220, 140) if sensitive else (200, 200, 200), font=font)
    return strip


def annotate_rejected(img: Image.Image) -> Image.Image:
    """Draw a red REJECTED watermark over an image that couldn't be generated."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    w, h = img.size
    draw.text((w // 2 - 120, h // 2 - 30), "REJECTED", fill=(220, 30, 30), font=font)
    return img


def assemble_pair(img_base: Image.Image | None, img_turbo: Image.Image | None,
                  prompt: str, meta: dict, res: int, pair_idx: int) -> Image.Image:
    """Combine base + turbo images with labels into a single side-by-side panel."""
    placeholder = Image.new("RGB", (res, res), (15, 15, 15))

    left  = img_base  if img_base  is not None else annotate_rejected(placeholder.copy())
    right = img_turbo if img_turbo is not None else annotate_rejected(placeholder.copy())

    total_w = res * 2 + 6  # 6px separator
    title_h = 36
    prompt_strip = make_prompt_strip(total_w, prompt, sensitive=meta.get("sensitive", False))
    prompt_h = prompt_strip.height

    cat_badge = f"#{pair_idx+1}  [{meta.get('source','?')}]  {meta.get('category','')}"
    header = make_label_strip(total_w, cat_badge, bg=(20, 20, 60))

    total_h = title_h * 2 + prompt_h + res  # header + label_row + prompt + images
    canvas = Image.new("RGB", (total_w, total_h), (10, 10, 10))

    # header
    canvas.paste(header, (0, 0))
    # prompt strip
    canvas.paste(prompt_strip, (0, title_h))
    # model labels
    base_label  = make_label_strip(res, "Z-Image  (50 steps, CFG)", bg=(0, 70, 140))
    turbo_label = make_label_strip(res, "Z-Image-Turbo  (8 steps, no CFG)", bg=(140, 70, 0))
    canvas.paste(base_label,  (0,       title_h + prompt_h))
    canvas.paste(turbo_label, (res + 6, title_h + prompt_h))
    # images
    img_y = title_h + prompt_h + title_h
    canvas.paste(left,  (0,       img_y))
    canvas.paste(right, (res + 6, img_y))

    return canvas


def generate_image_safe(pipe, prompt: str, steps: int, guidance: float,
                        res: int, seed: int, neg: str = "") -> tuple[Image.Image | None, str]:
    """Run inference, return (image, status). status='ok'|'rejected'|'error:<msg>'."""
    try:
        gen = torch.Generator("cuda").manual_seed(seed)
        kwargs = dict(
            prompt=prompt,
            height=res,
            width=res,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )
        if neg:
            kwargs["negative_prompt"] = neg
        out = pipe(**kwargs)
        img = out.images[0]
        # Check if pipeline returned a black/NSFW-filtered image
        import numpy as np
        arr = np.array(img)
        if arr.mean() < 5:
            return None, "rejected"
        return img, "ok"
    except Exception as e:
        msg = str(e)[:120]
        return None, f"error:{msg}"


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=100, help="Number of pairs to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--res",  type=int, default=512, help="Image resolution (square)")
    parser.add_argument("--n_parti_per_cat", type=int, default=6)
    parser.add_argument("--n_sd",   type=int, default=15)
    parser.add_argument("--n_harm", type=int, default=10)
    args = parser.parse_args()

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    # ── build prompt list ──────────────────────────────────────────────────────
    print("Building prompt list …")
    prompts = []
    prompts += read_parti(args.n_parti_per_cat)
    prompts += read_sd_prompts(args.n_sd)
    prompts += read_harmful(args.n_harm)

    # Deduplicate & cap
    seen = set()
    deduped = []
    for p in prompts:
        key = p["prompt"].strip().lower()[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    prompts = deduped[: args.n]
    print(f"  Selected {len(prompts)} prompts "
          f"(parti={sum(p['source']=='parti-prompts' for p in prompts)}, "
          f"sd={sum(p['source']=='stable-diffusion-prompts' for p in prompts)}, "
          f"harmful={sum(p['source']=='harmful_behaviors' for p in prompts)})")

    # ── load pipelines ──────────────────────────────────────────────────────────
    print("\nLoading Z-Image (base) …")
    t0 = time.time()
    pipe_base = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("Loading Z-Image-Turbo …")
    t0 = time.time()
    pipe_turbo = ZImagePipeline.from_pretrained(
        MODEL_TURBO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── generate ───────────────────────────────────────────────────────────────
    metadata = []
    pair_images = []

    for i, meta in enumerate(prompts):
        prompt = meta["prompt"]
        seed   = args.seed + i
        print(f"[{i+1:3d}/{len(prompts)}] {meta['source']:<30} | {prompt[:70]}")

        t_start = time.time()

        # Base: 50 steps with CFG
        img_b, status_b = generate_image_safe(
            pipe_base, prompt,
            steps=50, guidance=4.0,
            res=args.res, seed=seed,
            neg="blurry, low quality, deformed",
        )

        # Turbo: 8 steps, no CFG
        img_t, status_t = generate_image_safe(
            pipe_turbo, prompt,
            steps=9, guidance=0.0,
            res=args.res, seed=seed,
        )

        elapsed = time.time() - t_start
        print(f"         base={status_b}  turbo={status_t}  ({elapsed:.1f}s)")

        # Save individual images
        stem = f"pair_{i:03d}"
        if img_b is not None:
            img_b.save(PAIRS_DIR / f"{stem}_base.jpg",  quality=90)
        if img_t is not None:
            img_t.save(PAIRS_DIR / f"{stem}_turbo.jpg", quality=90)

        # Assemble side-by-side panel
        panel = assemble_pair(img_b, img_t, prompt, meta, args.res, i)
        panel.save(PAIRS_DIR / f"{stem}_panel.jpg", quality=88)
        pair_images.append(panel)

        metadata.append({
            "index":    i,
            "prompt":   prompt,
            "original": meta.get("original", prompt),
            "source":   meta["source"],
            "category": meta["category"],
            "sensitive": meta.get("sensitive", False),
            "seed":     seed,
            "steps_base": 50,
            "steps_turbo": 8,
            "status_base":  status_b,
            "status_turbo": status_t,
            "elapsed_s": round(elapsed, 2),
        })

    # ── summary grid ───────────────────────────────────────────────────────────
    print("\nAssembling summary grid …")
    # Resize all panels to fit a wide grid (5 columns)
    COLS   = 4
    thumb  = 400            # width of each panel thumbnail
    thumbs = []
    for panel in pair_images:
        scale  = thumb / panel.width
        new_h  = int(panel.height * scale)
        thumbs.append(panel.resize((thumb, new_h), Image.LANCZOS))

    rows = (len(thumbs) + COLS - 1) // COLS
    max_h = max(t.height for t in thumbs)
    grid_w = COLS * thumb
    grid_h = rows * max_h
    grid = Image.new("RGB", (grid_w, grid_h), (5, 5, 5))
    for idx, t in enumerate(thumbs):
        col = idx % COLS
        row = idx // COLS
        grid.paste(t, (col * thumb, row * max_h))

    grid.save(OUTPUT_DIR / "summary_grid.jpg", quality=85)
    print(f"  Saved summary_grid.jpg  ({grid_w}×{grid_h})")

    # ── metadata ───────────────────────────────────────────────────────────────
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata.json  ({len(metadata)} entries)")

    # ── quick stats ────────────────────────────────────────────────────────────
    n_ok_b  = sum(1 for m in metadata if m["status_base"]  == "ok")
    n_ok_t  = sum(1 for m in metadata if m["status_turbo"] == "ok")
    n_rej   = sum(1 for m in metadata if "rejected" in (m["status_base"], m["status_turbo"]))
    avg_t   = sum(m["elapsed_s"] for m in metadata) / len(metadata)
    print(f"\nDone.  base_ok={n_ok_b}/{len(metadata)}  turbo_ok={n_ok_t}/{len(metadata)}  "
          f"any_rejected={n_rej}  avg_time={avg_t:.1f}s/pair")


if __name__ == "__main__":
    main()
