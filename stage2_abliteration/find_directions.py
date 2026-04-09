#!/usr/bin/env python3
"""
Stage 2A — Find concept directions in Z-Image residual stream
==============================================================

Strategy
--------
For a pair of concept sets (A = "positive", B = "negative"), we:
  1. Run N denoising steps through Z-Image for each prompt
  2. Capture the residual stream at a MIDDLE-NOISE timestep via a forward hook
     (middle steps encode semantic layout; pure-noise steps mostly encode structure)
  3. Hook each `layers[i]` output to capture (batch, seq, dim=3840) hidden states
  4. Mean-pool over the sequence dimension → (n_prompts, 3840) per layer
  5. Direction: d = normalize(mean(A_acts) - mean(B_acts))
  6. Coherence check: compute inter-layer cosine; if two layers share cos > 0.99,
     report them as redundant — ablating both gives 2× the intended suppression
  7. Report a recommended minimal layer set and save full payload

Critical design requirements for a valid concept pair
------------------------------------------------------
  BAD:  NSFW anime vs diverse parti-prompts (animals/food/scenes)
         → direction captures "anime portrait" vs "everything else",
           compounding with ablation to destroy general generation ability.
  GOOD: NSFW anime vs SFW anime (same style distribution, different content)
         → direction isolates explicit content, preserving general capabilities.

  By default this script constructs the SFW set from stable-diffusion-prompts
  filtered to anime/character entries that do NOT contain NSFW keywords.

Architecture context (Z-Image, S3-DiT single-stream)
------------------------------------------------------
- 30 transformer blocks: `transformer.layers[0..29]`
- Residual stream dim: 3840
- Each block output shape: (batch, seq_len, 3840)
- Output projections that write to residual stream:
    layers[i].attention.to_out.0     (3840 × 3840)
    layers[i].feed_forward.w2        (3840 × 10240)

Usage
-----
  python3 stage2_abliteration/find_directions.py

  Optional overrides:
    --n_each N          prompts per concept (default 50)
    --layer_start INT   first layer to hook (default 0)
    --layer_end   INT   last layer to hook  (default 29)
    --collect_step INT  which denoising step to capture (default: middle of --collect_steps)
    --collect_steps INT total denoising steps for collection pass (default 10)
    --out PATH          output file (default stage2_abliteration/directions/nsfw_vs_sfw.pt)
    --res INT           latent resolution for activation collection (default 256 for speed)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.ipc as ipc
import torch
from diffusers import ZImagePipeline

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

MODEL_BASE = (
    "/scratch/hf-cache/huggingface/hub/"
    "models--Tongyi-MAI--Z-Image/snapshots/"
    "04cc4abb7c5069926f75c9bfde9ef43d49423021"
)

DATASET_NSFW = "DRDELATV/SHORT_NSFW"

DS_SD = (
    "/scratch/hf-cache/huggingface/datasets/"
    "Gustavosta___stable-diffusion-prompts/default/0.0.0/"
    "d816d4a05cb89bde39dd99284c459801e1e7e69a/stable-diffusion-prompts-train.arrow"
)

# Keywords that must appear in an SD prompt to count as anime/character style
_ANIME_INCLUDE = (
    "anime", "manga", "character", "girl", "boy", "woman", "man",
    "portrait", "illustration", "digital art", "fantasy",
)
# Keywords that disqualify a prompt as SFW reference
_NSFW_EXCLUDE = (
    "nude", "naked", "nsfw", "explicit", "breast", "nipple", "pussy",
    "penis", "genitalia", "erotic", "hentai", "porn", "sex", "lewd",
    "lingerie", "underwear", "topless", "exposed", "uncensored",
)


def _load_env():
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()


# ── prompt loading ─────────────────────────────────────────────────────────────

def load_nsfw_prompts(n: int) -> list[str]:
    """
    Load explicit captions directly from DRDELATV/SHORT_NSFW.
    This dataset has a 'text' column with real explicit image descriptions —
    no VLM sanitization involved.
    """
    from datasets import load_dataset
    ds = load_dataset(
        DATASET_NSFW,
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    prompts = []
    for row in ds:
        if row.get("text"):
            prompts.append(row["text"].strip())
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        print(f"[WARN] Only {len(prompts)} NSFW prompts available (requested {n})")
    return prompts[:n]


def load_sfw_prompts(n: int) -> list[str]:
    """
    Load SFW prompts that are STYLE-MATCHED to the NSFW concept set.

    We draw from stable-diffusion-prompts filtered to:
      - contain at least one of _ANIME_INCLUDE keywords (same style distribution)
      - contain none of _NSFW_EXCLUDE keywords (content is clean)

    This avoids the critical failure mode of using *diverse* SFW prompts
    (animals, food, landscapes) which produce a direction that captures
    "anime portrait" vs "everything else" rather than explicit content.
    """
    with open(DS_SD, "rb") as f:
        tbl = ipc.open_stream(f).read_all()
    all_prompts = tbl["Prompt"].to_pylist()

    filtered: list[str] = []
    for p in all_prompts:
        pl = p.lower()
        if any(kw in pl for kw in _ANIME_INCLUDE) and not any(kw in pl for kw in _NSFW_EXCLUDE):
            filtered.append(p)
        if len(filtered) >= n * 4:   # gather a pool then subsample
            break

    if len(filtered) < n:
        print(f"  [WARN] Only {len(filtered)} style-matched SFW prompts found; using all.")
        return filtered

    # Subsample evenly from pool to maximise diversity
    step = max(1, len(filtered) // n)
    return filtered[::step][:n]


# ── activation collection ──────────────────────────────────────────────────────

def collect_activations(
    pipe: ZImagePipeline,
    prompts: list[str],
    layer_start: int,
    layer_end: int,
    res: int,
    collect_steps: int = 10,
    collect_step: int = -1,
    seed: int = 0,
) -> dict[int, torch.Tensor]:
    """
    Collect residual-stream activations at each hooked layer for a list of prompts.

    Activations are captured at a MIDDLE-NOISE timestep, not the initial pure-noise
    step. At the first denoising step the residual stream is dominated by noise
    structure; semantic content is encoded around t≈0.5 (collect_step ≈ steps//2).

    For each prompt:
      - Run `collect_steps` denoising steps
      - Capture the output of layers[i] only at step `collect_step`
      - Mean-pool over sequence tokens → one vector per prompt per layer

    Returns
    -------
    dict mapping layer_idx → float32 tensor of shape (n_prompts, 3840)
    """
    if collect_step < 0:
        collect_step = collect_steps // 2   # default: middle step

    accumulated: dict[int, list[torch.Tensor]] = {i: [] for i in range(layer_start, layer_end + 1)}
    step_counter: dict[str, int] = {"n": 0}

    hooks = []
    for i in range(layer_start, layer_end + 1):
        def make_hook(idx: int):
            def hook(module, inp, out):
                # Each full pipeline forward call fires all layer hooks once per step.
                # We use a shared counter and only accumulate at the target step.
                # The counter is reset per-prompt; we track via closure over a list.
                pass   # replaced below per-prompt
            return hook
        hooks.append(pipe.transformer.layers[i].register_forward_hook(make_hook(i)))
    # Remove placeholder hooks
    for h in hooks:
        h.remove()
    hooks.clear()

    try:
        for j, prompt in enumerate(prompts):
            print(f"  [{j + 1:3d}/{len(prompts)}] {prompt[:80]}", flush=True)
            gen = torch.Generator("cuda").manual_seed(seed + j)

            # Per-prompt step counter for the hook
            capture_buf: dict[int, torch.Tensor] = {}
            _step = {"n": 0}

            def make_hook(idx: int):
                def hook(module, inp, out):
                    if _step["n"] == collect_step:
                        hs = out[0] if isinstance(out, tuple) else out
                        # With CFG (guidance_scale > 0) the batch dimension is 2
                        # (uncond, cond). Take only the conditional half (index -1).
                        # With guidance_scale=0 batch is 1; both cases handled safely.
                        hs_cond = hs[-1:].float()            # (1, seq, 3840)
                        capture_buf[idx] = hs_cond.mean(dim=1).squeeze(0).detach().cpu()  # (3840,)
                return hook
            active_hooks = [
                pipe.transformer.layers[i].register_forward_hook(make_hook(i))
                for i in range(layer_start, layer_end + 1)
            ]

            # Step counter hook on layer 0 to count denoising steps
            def step_tick(module, inp, out):
                _step["n"] += 1
            tick_hook = pipe.transformer.layers[layer_start].register_forward_pre_hook(
                lambda m, i: step_ticker()
            )
            def step_ticker():
                _step["n"] += 1
            tick_hook.remove()
            # Simpler: count via hook on the first layer and use pre-hook
            _step["n"] = 0
            def pre_tick(module, inp):
                _step["n"] += 1
            tick = pipe.transformer.layers[layer_start].register_forward_pre_hook(pre_tick)

            try:
                with torch.no_grad():
                    pipe(
                        prompt,
                        height=res,
                        width=res,
                        num_inference_steps=collect_steps,
                        guidance_scale=4.0,
                        negative_prompt="blurry, low quality",
                        generator=gen,
                    )
            finally:
                for h in active_hooks:
                    h.remove()
                tick.remove()

            # Store captured activations
            for i in range(layer_start, layer_end + 1):
                if i in capture_buf:
                    accumulated[i].append(capture_buf[i])
                else:
                    # Fallback: step index may not have fired — use last step
                    print(f"  [WARN] Layer {i}: no activation captured at step {collect_step}")

    except Exception:
        raise

    return {
        i: torch.stack(acts)   # (n_prompts, 3840)
        for i, acts in accumulated.items()
        if acts
    }


# ── direction computation ──────────────────────────────────────────────────────

def compute_directions(
    acts_a: dict[int, torch.Tensor],
    acts_b: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """
    For each layer, compute the unit-norm direction pointing from concept B to concept A.

      d = normalize(mean(A) - mean(B))

    Returns
    -------
    dict mapping layer_idx → float32 unit vector of shape (3840,)
    """
    directions: dict[int, torch.Tensor] = {}
    for i in acts_a:
        diff = acts_a[i].mean(dim=0) - acts_b[i].mean(dim=0)   # (3840,)
        norm = diff.norm()
        if norm < 1e-8:
            print(f"  [WARN] layer {i}: near-zero direction norm ({norm:.2e}), skipping")
            continue
        directions[i] = (diff / norm).float()
    return directions


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect concept-pair activations and compute abliteration directions")
    parser.add_argument("--n_each",         type=int,  default=50,  help="Prompts per concept (default 50)")
    parser.add_argument("--layer_start",    type=int,  default=0,   help="First layer to probe (default 0)")
    parser.add_argument("--layer_end",      type=int,  default=29,  help="Last layer to probe  (default 29)")
    parser.add_argument("--collect_steps",  type=int,  default=10,  help="Denoising steps for activation collection (default 10)")
    parser.add_argument("--collect_step",   type=int,  default=-1,  help="Which step index to capture (-1 = middle, default)")
    parser.add_argument("--res",            type=int,  default=256, help="Resolution (default 256)")
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "directions" / "nsfw_vs_sfw.pt",
        help="Output .pt file path",
    )
    args = parser.parse_args()
    if args.collect_step < 0:
        args.collect_step = args.collect_steps // 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── load prompts ───────────────────────────────────────────────────────────
    print(f"Loading concept prompts ({args.n_each} each) …")
    nsfw_prompts = load_nsfw_prompts(args.n_each)
    sfw_prompts  = load_sfw_prompts(args.n_each)
    print(f"  NSFW (A): {len(nsfw_prompts)} prompts")
    print(f"  SFW  (B): {len(sfw_prompts)}  prompts")

    # ── load model ─────────────────────────────────────────────────────────────
    print("\nLoading Z-Image (base) …")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    print(f"  Done. Probing layers {args.layer_start}–{args.layer_end} at {args.res}×{args.res}\n")

    # ── collect activations ────────────────────────────────────────────────────
    print(f"\n=== Concept A (NSFW) activations  [capturing at step {args.collect_step} of {args.collect_steps}] ===")
    acts_a = collect_activations(
        pipe, nsfw_prompts, args.layer_start, args.layer_end, args.res,
        collect_steps=args.collect_steps, collect_step=args.collect_step,
        seed=args.seed,
    )

    print("\n=== Concept B (SFW, style-matched) activations ===")
    acts_b = collect_activations(
        pipe, sfw_prompts, args.layer_start, args.layer_end, args.res,
        collect_steps=args.collect_steps, collect_step=args.collect_step,
        seed=args.seed + 1000,
    )

    # ── compute directions ─────────────────────────────────────────────────────
    print("\nComputing per-layer directions …")
    directions = compute_directions(acts_a, acts_b)

    # ── layer coherence analysis ────────────────────────────────────────────────
    sorted_layers = sorted(directions.keys())
    print("\nLayer  |  raw diff norm  |  cos(prev_layer)  |  direction norm")
    print("-" * 68)
    raw_norms = {}
    for j, i in enumerate(sorted_layers):
        raw_diff = acts_a[i].mean(dim=0) - acts_b[i].mean(dim=0)
        raw_norms[i] = raw_diff.norm().item()
        if j == 0:
            cos_prev = "   —  "
        else:
            prev = sorted_layers[j - 1]
            cos_prev = f"{torch.dot(directions[prev], directions[i]).item():+.4f}"
        print(f"  {i:2d}   |  {raw_norms[i]:10.2f}   |  {cos_prev:>12}     |  {directions[i].norm():.6f}")

    # Identify redundant layers (cos > 0.99 with predecessor)
    print("\n── Redundancy analysis ──")
    redundant = set()
    for j in range(1, len(sorted_layers)):
        i_prev = sorted_layers[j - 1]
        i_curr = sorted_layers[j]
        cos = torch.dot(directions[i_prev], directions[i_curr]).item()
        if cos > 0.99:
            redundant.add(i_curr)

    # Recommend: keep only the single highest-norm non-redundant layer
    non_redundant = [i for i in sorted_layers if i not in redundant]
    if non_redundant:
        best_layer = max(non_redundant, key=lambda i: raw_norms[i])
        print(f"  Non-redundant layers (cos < 0.99 with predecessor): {non_redundant}")
        print(f"  Recommended single-layer ablation: layer {best_layer}  (highest raw diff norm)")
        print(f"  ⚠ Ablating ALL {len(sorted_layers)} layers with α=1.0 is equivalent to α≈{len(sorted_layers):.0f}x.")
        print(f"    Use --layers {best_layer} --alpha 1.0 OR --layers {' '.join(str(x) for x in sorted_layers)} --alpha {1/len(sorted_layers):.3f}")
    else:
        best_layer = sorted_layers[-1]
        print(f"  All layers are highly coherent (best by norm: {best_layer})")

    # ── save ───────────────────────────────────────────────────────────────────
    payload = {
        "directions":       directions,        # {layer_idx: (3840,) float32 unit vector}
        "concept_a":        "nsfw",
        "concept_b":        "sfw_anime_matched",
        "n_a":              len(nsfw_prompts),
        "n_b":              len(sfw_prompts),
        "layer_start":      args.layer_start,
        "layer_end":        args.layer_end,
        "collect_step":     args.collect_step,
        "collect_steps":    args.collect_steps,
        "model":            "Tongyi-MAI/Z-Image",
        "res":              args.res,
        "recommended_layer": best_layer,
        "redundant_layers": sorted(redundant),
        "raw_diff_norms":   raw_norms,
    }
    torch.save(payload, args.out)
    print(f"\nSaved {len(directions)} directions → {args.out}")


if __name__ == "__main__":
    main()
