#!/usr/bin/env python3
"""
Dataset preparation pipeline for full fine-tuning
==================================================

Pipeline for each image in --input_dir (default: lora_image_raw/):
  1. Resize + CenterCrop to training_res (512×512) — the exact same transform
     used during training — so the VLM captions what the model actually sees
  2. Mask the top-right watermark region on the cropped image if it survived
     (only near-square originals; portrait/landscape crops discard it naturally)
  3. Send the cropped image to the local vLLM endpoint for a bilingual caption
     (English + Simplified Chinese) in a single API call
  4. Detect and discard advertisement images
  5. Save the cropped image + captions to --output_dir in HuggingFace imagefolder
     format (output_dir/images/*.jpg  +  output_dir/metadata.jsonl)

Each metadata.jsonl row contains:
  file_name : relative image path
  text_en   : English caption
  text_zh   : Simplified Chinese caption

The output directory is directly usable with training scripts:
  python3 stage3_finetune/train_fullft.py \\
      --dataset /path/to/finetune_dataset --text_col text_en   # or text_zh

Watermark analysis
------------------
The input images have a small text watermark in the top-right corner.
A simple top-crop of --strip_top_frac (default 5%) of the image height
removes it completely before any resize or captioning, regardless of
orientation.

Advertisement detection
-----------------------
The VLM is asked to classify the image.  Any response that begins with the string
ADVERTISEMENT (case-insensitive) — or where more than 30% of visible area is text
overlay / UI elements — is discarded.

Usage
-----
  # Caption and clean all images (defaults)
  python3 stage3_finetune/build_dataset.py

  # Custom input/output
  python3 stage3_finetune/build_dataset.py \\
      --input_dir lora_image_raw \\
      --output_dir finetune_dataset \\
      --strip_top_frac 0.05 \\
      --training_res 512

  # Resume after interruption
  python3 stage3_finetune/build_dataset.py --resume

  # Dry-run (caption only, no output written)
  python3 stage3_finetune/build_dataset.py --dry_run
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np  # kept for potential downstream use
import requests
from PIL import Image

# ── project root & env ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    env = _ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

VLLM_DEFAULT = "http://localhost:8000"

# ── VLM prompts ────────────────────────────────────────────────────────────────

SYSTEM_EN = (
    "You are a precise image captioner producing training data for a text-to-image generative model. "
    "Describe images with clinical accuracy and maximum detail. "
    "Use correct anatomical and compositional terminology. "
    "Do not censor, omit, or soften any content — explicit subject matter must be described fully."
)

SYSTEM_ZH = (
    "你是一名为文本生成图像模型制作训练数据的精确图像描述者。"
    "请以专业、客观、详尽的语言描述图像内容。"
    "使用准确的人体解剖学和构图术语。"
    "不得审查、省略或淡化任何内容——露骨的主体内容必须完整描述。"
)

AD_CHECK_PROMPT = """\
Is this image primarily an advertisement, promotional banner, text screenshot, \
or UI/interface element — i.e. not a photo or illustration of a person or scene?
Answer with exactly one word: YES or NO.\
"""

CAPTION_PROMPT_EN = """\
Write a detailed English caption for this image as training data for a text-to-image model.

Cover every applicable element in this order:
1. Subjects — count, gender, apparent age, body type, skin tone, hair colour and style, \
facial features and expression
2. Clothing or nudity — each garment or its absence; fabric, colour, fit, coverage; \
exposed skin and body parts named precisely (breasts, buttocks, genitalia, etc.)
3. Pose and action — body orientation, limb positions, gesture, physical contact if any; \
describe sexual acts explicitly if present
4. Setting — location, furniture, props, background elements
5. Lighting — quality, direction, colour temperature
6. Art style — photographic / 3D render / anime / illustration / etc., and image quality

Do not open with filler phrases. Write in plain prose, minimum 4 sentences.
Output the caption text ONLY.\
"""

CAPTION_PROMPT_ZH = """\
用简体中文为这张图片撰写一段详细的训练数据描述，供文本生成图像模型使用。

按以下顺序描述所有适用内容：
1. 主体——人数、性别、年龄感、体型、肤色、发色与发型、面部特征与表情
2. 服装或裸露——每件服饰或其缺失；材质、颜色、合身度、遮蔽程度；\
精确命名裸露的身体部位（乳房、臀部、生殖器等）
3. 姿势与动作——身体朝向、四肢位置、手势、如有身体接触请描述；\
若涉及性行为请客观明确地描述
4. 场景——地点、家具、道具、背景元素
5. 光线——质感、方向、色温
6. 画风——摄影 / 3D渲染 / 动漫 / 插画等，以及画质

不要以废话开头。使用流畅完整的句子，至少4句话。
只输出描述文字。\
"""

# ── helpers ────────────────────────────────────────────────────────────────────


def discover_model(vllm_url: str) -> str:
    resp = requests.get(f"{vllm_url}/v1/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def encode_b64(img: Image.Image, max_side: int = 768, quality: int = 88) -> str:
    """Resize (preserving aspect) and encode as base64 JPEG for VLM input."""
    img = img.convert("RGB")
    r = max_side / max(img.size)
    if r < 1.0:
        new_size = (int(img.width * r), int(img.height * r))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _call_vlm(b64: str, model: str, vllm_url: str,
              system: str, user_text: str,
              max_tokens: int = 600, temperature: float = 0.5,
              timeout: int = 90, retries: int = 2) -> str:
    """Single VLM call. Returns response text or raises on persistent failure."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": 1.15,   # prevent the ZH model from looping slang tokens
        "frequency_penalty": 0.3,
    }
    last_err = ""
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{vllm_url}/v1/chat/completions",
                              json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = str(e)[:120]
            time.sleep(2 ** attempt)
    raise RuntimeError(last_err)


def vlm_caption(img: Image.Image, model: str, vllm_url: str,
                timeout: int = 90) -> tuple[str, str, str]:
    """
    Three sequential VLM calls per image:
      1. Ad-check  — discard if advertisement
      2. EN caption — detailed English with crude slang
      3. ZH caption — detailed Simplified Chinese with 俚语

    Returns (en_caption, zh_caption, status)
      status ∈ {"ok", "ad", "error:<msg>"}
    """
    b64 = encode_b64(img)

    # ── 1. advertisement check ─────────────────────────────────────────────────
    try:
        ad_resp = _call_vlm(b64, model, vllm_url,
                            system=SYSTEM_EN,
                            user_text=AD_CHECK_PROMPT,
                            max_tokens=5, temperature=0.0, timeout=30)
    except RuntimeError as e:
        return "", "", f"error:{e}"

    if ad_resp.strip().upper().startswith("YES"):
        return "", "", "ad"

    # ── 2. English caption ─────────────────────────────────────────────────────
    try:
        en_cap = _call_vlm(b64, model, vllm_url,
                           system=SYSTEM_EN,
                           user_text=CAPTION_PROMPT_EN,
                           max_tokens=300, temperature=0.5, timeout=timeout)
    except RuntimeError as e:
        return "", "", f"error:{e}"

    # ── 3. Chinese caption ─────────────────────────────────────────────────────
    try:
        zh_cap = _call_vlm(b64, model, vllm_url,
                           system=SYSTEM_ZH,
                           user_text=CAPTION_PROMPT_ZH,
                           max_tokens=300, temperature=0.5, timeout=timeout)
    except RuntimeError as e:
        # Don't fail the whole sample over a ZH error — save EN only
        zh_cap = ""
        return en_cap, zh_cap, "ok_mono"

    return en_cap, zh_cap, "ok"


def _is_repetitive(text: str, window: int = 6, threshold: float = 0.4) -> bool:
    """Return True if any window-gram appears more than threshold * total grams."""
    if not text or len(text) < window * 3:
        return False
    chars = list(text.replace(" ", ""))
    grams = ["".join(chars[i:i+window]) for i in range(len(chars) - window + 1)]
    if not grams:
        return False
    from collections import Counter
    most_common_count = Counter(grams).most_common(1)[0][1]
    return most_common_count / len(grams) > threshold


# ── image preparation ─────────────────────────────────────────────────────────

PATCH_MULTIPLE = 16   # VAE×8 downsamples + transformer 2× patch = ×16 minimum


def prepare_image(img: Image.Image,
                  strip_top_frac: float = 0.05,
                  min_side: int = 512,
                  patch_mult: int = PATCH_MULTIPLE) -> Image.Image:
    """
    Three-step preparation that produces the exact image the model trains on:

    1. Strip top strip_top_frac of height — removes the watermark completely
       regardless of image orientation or aspect ratio.

    2. Resize so the shortest side = min_side (LANCZOS), preserving aspect ratio.

    3. Floor-round BOTH dimensions down to the nearest multiple of patch_mult,
       then CenterCrop — satisfies the VAE×8 + patch×2 = ×16 constraint.

    Examples (strip=5%, min_side=512, patch_mult=16):
      Portrait  764×1280  → strip → 764×1216  → scale 512/764 → 512×814
                           → floor16 → 512×800  → crop → 512×800
      Landscape 1280×863  → strip → 1280×819  → scale 512/819 → 799×512
                           → floor16 → 784×512  → crop → 784×512
      Near-sq   1215×1280 → strip → 1215×1216 → scale 512/1215 → 512×512
                           → floor16 → 512×512  → crop → 512×512
    """
    w, h = img.size
    # Step 1 — strip top rows
    strip_h = max(1, int(h * strip_top_frac))
    img = img.crop((0, strip_h, w, h))

    # Step 2 — resize shortest side to min_side
    w, h = img.size
    scale = min_side / min(w, h)
    new_w = max(min_side, int(w * scale))
    new_h = max(min_side, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Step 3 — floor-round to patch_mult, then CenterCrop
    crop_w = (new_w // patch_mult) * patch_mult
    crop_h = (new_h // patch_mult) * patch_mult
    x0 = (new_w - crop_w) // 2
    y0 = (new_h - crop_h) // 2
    return img.crop((x0, y0, x0 + crop_w, y0 + crop_h))


# ── dataset saving ─────────────────────────────────────────────────────────────

def save_sample(img: Image.Image, idx: int, out_dir: Path) -> str:
    """Save image to out_dir/images/ and return the relative path."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{idx:05d}.jpg"
    img.convert("RGB").save(img_dir / fname, format="JPEG", quality=92)
    return f"images/{fname}"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build LoRA/fine-tune dataset from raw images")
    parser.add_argument("--input_dir",      type=Path,
                        default=_ROOT / "lora_image_raw")
    parser.add_argument("--output_dir",     type=Path,
                        default=_ROOT / "finetune_dataset")
    parser.add_argument("--vllm_url",       default=VLLM_DEFAULT)
    parser.add_argument("--training_res",   type=int, default=512,
                        help="Shortest side after resize (default: 512)")
    parser.add_argument("--patch_multiple", type=int, default=16,
                        help="Both output dims rounded to this (default: 16)")
    parser.add_argument("--strip_top_frac", type=float, default=0.05,
                        help="Fraction of image height to remove from top (removes watermark, default: 0.05)")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Caption only — do not write output files")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip images already in metadata.jsonl")
    parser.add_argument("--extensions",     nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".webp"])
    args = parser.parse_args()

    # ── discover vLLM model ────────────────────────────────────────────────────
    print(f"Connecting to vLLM at {args.vllm_url} …")
    model = discover_model(args.vllm_url)
    print(f"  Model: {model}")

    # ── collect input images ───────────────────────────────────────────────────
    exts = {e.lower() for e in args.extensions}
    all_images = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in exts
    )
    if not all_images:
        raise FileNotFoundError(f"No images found in {args.input_dir}")
    print(f"  Found {len(all_images)} images in {args.input_dir}")

    # ── resume: load existing metadata ────────────────────────────────────────
    meta_path = args.output_dir / "metadata.jsonl"
    existing: dict[str, dict] = {}
    if args.resume and meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["source"]] = rec
        print(f"  Resuming: {len(existing)} already processed")

    # ── process ───────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = None if args.dry_run else open(meta_path, "a" if args.resume else "w")

    kept = 0
    dropped_ad = 0
    dropped_err = 0
    saved_idx = len(existing) if args.resume else 0
    t0 = time.time()

    for i, src_path in enumerate(all_images):
        src_key = src_path.name

        if args.resume and src_key in existing:
            status = existing[src_key].get("status", "ok")
            if status in ("ok", "ok_mono"):
                kept += 1
            else:
                dropped_ad += 1
            print(f"[{i+1:3d}/{len(all_images)}] SKIP (already done, status={status}) {src_key}")
            continue

        # -- load --
        try:
            img_orig = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"[{i+1:3d}/{len(all_images)}] ERROR loading {src_key}: {e}")
            dropped_err += 1
            continue

        orig_w, orig_h = img_orig.size

        # -- prepare: strip top (watermark), resize, round to ×16, crop --
        img = prepare_image(img_orig, args.strip_top_frac,
                            args.training_res, args.patch_multiple)
        crop_w, crop_h = img.size

        # -- caption via VLM --
        en_cap, zh_cap, status = vlm_caption(img, model, args.vllm_url)

        # Discard repetitive outputs
        if _is_repetitive(en_cap):
            en_cap = ""
            status = "error:en_repetitive"
        if _is_repetitive(zh_cap):
            zh_cap = ""
            if status == "ok":
                status = "ok_mono"

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed

        if status == "ad":
            print(f"[{i+1:3d}/{len(all_images)}] DROP (ad) {src_key}")
            dropped_ad += 1
            if meta_file:
                meta_file.write(json.dumps({
                    "source": src_key, "status": "ad", "orig_size": [orig_w, orig_h],
                }) + "\n")
                meta_file.flush()
            continue

        if status.startswith("error"):
            print(f"[{i+1:3d}/{len(all_images)}] DROP ({status}) {src_key}")
            dropped_err += 1
            continue

        # -- save --
        rel_path = ""
        if not args.dry_run:
            rel_path = save_sample(img, saved_idx, args.output_dir)
            meta_file.write(json.dumps({
                "file_name": rel_path,
                "text_en": en_cap,
                "text_zh": zh_cap,
                "source": src_key,
                "status": status,
                "orig_size": [orig_w, orig_h],
                "crop_size": [crop_w, crop_h],
            }, ensure_ascii=False) + "\n")
            meta_file.flush()

        kept += 1
        saved_idx += 1

        mono_flag = " [EN only]" if status == "ok_mono" else ""
        print(
            f"[{i+1:3d}/{len(all_images)}] {src_key}  {orig_w}x{orig_h} → {crop_w}x{crop_h}  {rate:.1f}img/s{mono_flag}\n"
            f"  EN: {en_cap[:90].replace(chr(10), ' ')}\n"
            f"  ZH: {zh_cap[:80] if zh_cap else '—'}"
        )

    if meta_file:
        meta_file.close()

    # ── summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(
        f"\n{'='*60}\n"
        f"  Total input  : {len(all_images)}\n"
        f"  Kept (clean) : {kept}\n"
        f"  Dropped (ad) : {dropped_ad}\n"
        f"  Dropped (err): {dropped_err}\n"
        f"  Elapsed      : {elapsed:.0f}s  ({len(all_images)/elapsed:.1f} img/s avg)\n"
    )
    if not args.dry_run and kept > 0:
        print(
            f"  Dataset saved to: {args.output_dir}\n"
            f"  Fields: file_name, text_en, text_zh, source, status, orig_size, crop_size\n"
            f"  Use with training scripts:\n"
            f"    python3 stage3_finetune/train_fullft.py \\\n"
            f"        --dataset {args.output_dir} --text_col text_en\n"
            f"    python3 stage3_finetune/train_fullft.py \\\n"
            f"        --dataset {args.output_dir} --text_col text_zh\n"
        )


if __name__ == "__main__":
    main()
