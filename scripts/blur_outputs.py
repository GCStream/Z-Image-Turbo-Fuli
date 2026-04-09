"""
blur_outputs.py  –  Blur all generated output images before public repo push.

Applies a strong Gaussian blur (default radius=30) to every .jpg/.png under
outputs/.  Overwrites files in-place.  Run once before `git add` / `git push`.

Usage:
    python3 scripts/blur_outputs.py [--radius 30] [--dry-run]
"""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageFilter

_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = _ROOT / "outputs"


def blur_image(path: Path, radius: int) -> None:
    img = Image.open(path).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    # Preserve format; JPEG quality kept reasonable so file isn't huge
    if path.suffix.lower() in (".jpg", ".jpeg"):
        blurred.save(path, format="JPEG", quality=80, optimize=True)
    else:
        blurred.save(path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blur all output images.")
    parser.add_argument("--radius", type=int, default=30,
                        help="Gaussian blur radius (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files that would be blurred without modifying them")
    parser.add_argument("--dir", type=Path, default=OUTPUTS,
                        help="Directory to search (default: outputs/)")
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in args.dir.rglob("*") if p.suffix.lower() in exts)

    if not images:
        print("No images found.")
        sys.exit(0)

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Blurring {len(images)} images "
          f"(radius={args.radius}) in {args.dir} …")

    for i, path in enumerate(images, 1):
        rel = path.relative_to(_ROOT)
        if args.dry_run:
            print(f"  would blur: {rel}")
        else:
            blur_image(path, args.radius)
            if i % 50 == 0 or i == len(images):
                print(f"  {i}/{len(images)}  {rel}")

    if not args.dry_run:
        print(f"\nDone. {len(images)} images blurred.")


if __name__ == "__main__":
    main()
