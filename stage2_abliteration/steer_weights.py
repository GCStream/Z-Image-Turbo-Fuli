#!/usr/bin/env python3
"""
Stage 2B — Apply rank-1 ablation to Z-Image transformer weights
================================================================

Given the per-layer concept directions from find_directions.py, this script
surgically removes those directions from every specified weight matrix that
WRITES to the residual stream, then saves the modified model to disk.

Theory
------
For a weight matrix W that produces residual-stream output (W @ x),
the rank-1 ablation projects out direction d from W's row space:

    W' = W - alpha * outer(d, d @ W)
       = W - alpha * d ⊗ (d^T W)

This ensures that for any input x, the component of (W' @ x) along d is
reduced by alpha × its original magnitude. With alpha=1.0 the direction is
fully removed; values 0 < alpha < 1 provide partial suppression.

Target weight modules (those that write to the residual stream)
---------------------------------------------------------------
  layers[i].attention.to_out.0    (3840 × 3840)   — attention output projection
  layers[i].feed_forward.w2       (3840 × 10240)  — MLP down/output projection

These are selected here because:
  1. They are the only modules in each block that write back to the residual stream
  2. Editing W_q / W_k / W_v / W_up would affect QK geometry and scale, risking
     catastrophic quality degradation
  3. noise_refiner and context_refiner are left untouched by default

Usage
-----
  python3 stage2_abliteration/steer_weights.py [options]

  --directions PATH    directions .pt file from find_directions.py
                       (default: stage2_abliteration/directions/nsfw_vs_sfw.pt)
  --layers INT [INT …] which layers to ablate (default: 16 17 … 29)
  --components STR     "attn", "mlp", or "both" (default: "both")
  --alpha FLOAT        ablation strength; 1.0 = full removal (default: 1.0)
  --out PATH           output directory for modified model
                       (default: $TRAINING_SCRATCH/z-image-abliterated or
                                 /tmp/z-image-abliterated)
  --dry-run            print what would change, don't save
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from diffusers import ZImagePipeline

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

MODEL_BASE = (
    "/scratch/hf-cache/huggingface/hub/"
    "models--Tongyi-MAI--Z-Image/snapshots/"
    "04cc4abb7c5069926f75c9bfde9ef43d49423021"
)

DEFAULT_DIRECTIONS = Path(__file__).parent / "directions" / "nsfw_vs_sfw.pt"


def _load_env():
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

# ── ablation ───────────────────────────────────────────────────────────────────

def ablate_weight(W: torch.Tensor, d: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Apply rank-1 ablation to weight matrix W given direction d.

    W must have shape (out_features, in_features) where out_features == len(d).
    (i.e. W writes to the residual stream of dimension len(d).)

      W' = W - alpha * outer(d, d @ W)

    Parameters
    ----------
    W     : (out, in) float tensor — the weight to modify
    d     : (out,)   float unit vector — the direction to remove
    alpha : ablation strength in [0, 1] (1.0 = full removal)
    """
    if W.shape[0] != d.shape[0]:
        raise ValueError(
            f"Shape mismatch: W is {W.shape} but direction d is {d.shape}. "
            "W's first dim must equal len(d)."
        )
    d = d.to(dtype=W.dtype, device=W.device)
    # d @ W : (in,)    — projection of each input basis onto d
    # outer : (out, in) — rank-1 update
    update = torch.outer(d, d @ W)
    return W - alpha * update


def ablate_layer(
    transformer,
    layer_idx: int,
    direction: torch.Tensor,
    components: str,
    alpha: float,
    dry_run: bool,
):
    """Modify the specified weight matrices of layers[layer_idx] in-place."""
    block = transformer.layers[layer_idx]
    dim   = direction.shape[0]   # 3840

    targets: list[tuple[str, torch.nn.Linear]] = []

    if components in ("attn", "both"):
        targets.append(("attention.to_out.0", block.attention.to_out[0]))

    if components in ("mlp", "both"):
        targets.append(("feed_forward.w2", block.feed_forward.w2))

    for name, linear in targets:
        W_orig = linear.weight.data.float()
        W_new  = ablate_weight(W_orig, direction.float(), alpha)

        diff_norm  = (W_new - W_orig).norm().item()
        frac_change = diff_norm / W_orig.norm().item() * 100

        print(
            f"  layer {layer_idx:2d} · {name:<28}  "
            f"ΔW norm={diff_norm:.4f}  ({frac_change:.2f}% change)"
        )

        if not dry_run:
            linear.weight.data = W_new.to(dtype=linear.weight.data.dtype)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    scratch = os.environ.get("TRAINING_SCRATCH", "/tmp")
    default_out = Path(scratch) / "z-image-abliterated"

    parser = argparse.ArgumentParser(description="Apply rank-1 ablation to Z-Image transformer weights")
    parser.add_argument(
        "--directions", type=Path, default=DEFAULT_DIRECTIONS,
        help=f"Directions .pt file (default: {DEFAULT_DIRECTIONS})",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layer indices to ablate (default: use recommended_layer from directions file)",
    )
    parser.add_argument(
        "--components", choices=["attn", "mlp", "both"], default="both",
        help="Which output projection(s) to modify (default: both)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Ablation strength in [0,1]; 1.0 = full direction removal (default: 1.0)",
    )
    parser.add_argument(
        "--out", type=Path, default=default_out,
        help=f"Output directory for modified model (default: {default_out})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving")
    args = parser.parse_args()

    # ── load directions ────────────────────────────────────────────────────────
    if not args.directions.exists():
        sys.exit(
            f"[ERROR] Directions file not found: {args.directions}\n"
            "Run find_directions.py first."
        )
    payload = torch.load(args.directions, map_location="cpu", weights_only=False)
    directions: dict[int, torch.Tensor] = payload["directions"]
    recommended = payload.get("recommended_layer", None)
    redundant   = set(payload.get("redundant_layers", []))

    # Default to recommended layer if --layers not specified
    if args.layers is None:
        if recommended is not None:
            args.layers = [recommended]
            print(f"[INFO] Using recommended_layer={recommended} from directions file.")
        else:
            args.layers = sorted(directions.keys())
            print(f"[INFO] No recommended_layer in file; using all {len(args.layers)} layers.")

    print(f"Loaded directions: {len(directions)} layers, "
          f"concept_a={payload['concept_a']!r}, concept_b={payload.get('concept_b','?')!r}")
    print(f"  Applying to layers: {args.layers}")
    print(f"  Components: {args.components}   alpha={args.alpha}   dry_run={args.dry_run}")

    # Warn if many redundant layers would compound the ablation
    n_redundant_in_target = sum(1 for i in args.layers if i in redundant)
    if len(args.layers) > 3 and n_redundant_in_target > 0:
        effective_alpha = args.alpha * len(args.layers)
        print(
            f"  ⚠ WARNING: {n_redundant_in_target}/{len(args.layers)} target layers share near-identical "
            f"directions (cos>0.99).\n"
            f"    Effective alpha ≈ {effective_alpha:.1f}x — this WILL destroy generation quality.\n"
            f"    Recommended: --layers {recommended} --alpha {args.alpha}\n"
            f"    Or:          --layers <all> --alpha {args.alpha / len(args.layers):.4f}"
        )

    # ── load model ─────────────────────────────────────────────────────────────
    print("Loading Z-Image (base) …")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    print("  Done.\n")

    # ── apply ablation ─────────────────────────────────────────────────────────
    print("Applying rank-1 ablation …")
    for layer_idx in sorted(args.layers):
        d = directions[layer_idx]
        ablate_layer(pipe.transformer, layer_idx, d, args.components, args.alpha, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] No weights modified.")
        return

    # ── save modified model ────────────────────────────────────────────────────
    print(f"\nSaving modified pipeline → {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    # Save just the transformer (the large part that changed)
    transformer_out = args.out / "transformer"
    pipe.transformer.save_pretrained(transformer_out)

    # Copy the rest of the pipeline components from the original (symlink-safe copy)
    original = Path(MODEL_BASE)
    for item in original.iterdir():
        if item.name == "transformer":
            continue   # already saved
        dest = args.out / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest, symlinks=False)
        else:
            shutil.copy2(item, dest)

    # Write ablation provenance metadata
    meta = {
        "base_model":   str(MODEL_BASE),
        "directions":   str(args.directions),
        "layers":       args.layers,
        "components":   args.components,
        "alpha":        args.alpha,
        "concept_a":    payload["concept_a"],
        "concept_b":    payload["concept_b"],
    }
    import json
    (args.out / "ablation_config.json").write_text(json.dumps(meta, indent=2))

    print(f"  transformer → {transformer_out}")
    print(f"  ablation_config.json written")
    print(f"\nDone. Load steered model with:")
    print(f"  ZImagePipeline.from_pretrained('{args.out}')")


if __name__ == "__main__":
    main()
