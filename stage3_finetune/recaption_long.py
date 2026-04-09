"""One-shot script: re-caption entries where EN word count > 400 or ZH char count > 400,
using 300-token limit. Updates metadata.jsonl in-place."""

import argparse, base64, io, json, sys, time
from pathlib import Path

import requests
from PIL import Image

# ── same prompts as build_dataset.py ──────────────────────────────────────────
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

Do not open with filler phrases. Write in plain prose, minimum 3 sentences.
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

不要以废话开头。使用流畅完整的句子，至少3句话。
只输出描述文字。\
"""


def _img_to_b64(img: Image.Image, max_side: int = 1024) -> str:
    img = img.convert("RGB")
    r = max_side / max(img.size)
    if r < 1.0:
        img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _call_vlm(b64: str, model: str, vllm_url: str,
              system: str, user_text: str, max_tokens: int = 300,
              temperature: float = 0.5, timeout: int = 90, retries: int = 2) -> str:
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
        "repetition_penalty": 1.15,
        "frequency_penalty": 0.3,
    }
    last_err = ""
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = str(e)[:120]
            time.sleep(2 ** attempt)
    raise RuntimeError(last_err)


def _get_model(vllm_url: str) -> str:
    r = requests.get(f"{vllm_url}/v1/models", timeout=10)
    return r.json()["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="finetune_dataset")
    ap.add_argument("--vllm_url", default="http://localhost:8000")
    ap.add_argument("--en_word_limit", type=int, default=400)
    ap.add_argument("--zh_char_limit", type=int, default=400)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta_path = dataset_dir / "metadata.jsonl"

    rows = []
    with open(meta_path) as f:
        for line in f:
            rows.append(json.loads(line))

    model = _get_model(args.vllm_url)
    print(f"Model: {model}")

    to_fix = [
        (i, r) for i, r in enumerate(rows)
        if len(r.get("text_en", "").split()) > args.en_word_limit
        or len(r.get("text_zh", "")) > args.zh_char_limit
    ]
    print(f"Entries to re-caption: {len(to_fix)} / {len(rows)}")

    for n, (i, row) in enumerate(to_fix):
        src = row["source"]
        img_path = dataset_dir / "images" / row["file_name"].replace("images/", "")
        en_words = len(row.get("text_en", "").split())
        zh_chars = len(row.get("text_zh", ""))
        print(f"[{n+1:3d}/{len(to_fix)}] {src}  EN={en_words}w  ZH={zh_chars}c", end="  ", flush=True)

        if args.dry_run:
            print("DRY RUN")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"ERROR loading image: {e}")
            continue

        b64 = _img_to_b64(img)
        updated = False

        if en_words > args.en_word_limit:
            try:
                new_en = _call_vlm(b64, model, args.vllm_url,
                                   system=SYSTEM_EN, user_text=CAPTION_PROMPT_EN,
                                   max_tokens=args.max_tokens)
                rows[i]["text_en"] = new_en
                updated = True
                print(f"EN→{len(new_en.split())}w", end="  ", flush=True)
            except Exception as e:
                print(f"EN ERR:{e}", end="  ", flush=True)

        if zh_chars > args.zh_char_limit:
            try:
                new_zh = _call_vlm(b64, model, args.vllm_url,
                                   system=SYSTEM_ZH, user_text=CAPTION_PROMPT_ZH,
                                   max_tokens=args.max_tokens)
                rows[i]["text_zh"] = new_zh
                updated = True
                print(f"ZH→{len(new_zh)}c", end="  ", flush=True)
            except Exception as e:
                print(f"ZH ERR:{e}", end="  ", flush=True)

        if updated:
            # Write back after every entry to survive interruption
            with open(meta_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("✓")
        else:
            print()

    print(f"\nDone. {len(to_fix)} entries processed.")


if __name__ == "__main__":
    main()
