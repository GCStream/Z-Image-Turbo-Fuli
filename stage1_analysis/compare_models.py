"""
Block-by-block weight similarity analysis: Z-Image vs Z-Image-Turbo

Metrics per tensor / per block:
  - cosine_sim   : cosine similarity of flattened weight vectors  (1.0 = identical)
  - l2_norm_diff : ||w_base - w_turbo||_2  / ||w_base||_2  (relative L2 change)
  - mean_diff    : |mean(w_base) - mean(w_turbo)|
  - std_diff     : |std(w_base)  - std(w_turbo)|

Grouping:
  - individual tensor results
  - aggregated per logical block (e.g. layers.{i}, context_refiner.{i}, …)
  - summary by block category
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open

# ── paths ──────────────────────────────────────────────────────────────────────
BASE  = Path("/scratch/hf-cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer")
TURBO = Path("/scratch/hf-cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6/transformer")

# ── helpers ────────────────────────────────────────────────────────────────────

def load_index(path: Path) -> dict[str, str]:
    """Return {tensor_name: shard_filename} from safetensors index."""
    with open(path / "diffusion_pytorch_model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]

def open_shards(base: Path, index: dict[str, str]) -> dict[str, "safe_open"]:
    """Open every unique shard file once, return {filename: handle}."""
    handles = {}
    for fname in set(index.values()):
        handles[fname] = safe_open(base / fname, framework="pt", device="cpu")
    return handles

def get_tensor(handles, index, key) -> torch.Tensor:
    fname = index[key]
    return handles[fname].get_tensor(key).float()

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    norm_a = a.norm()
    if norm_a < 1e-12:
        return float("nan")
    return ((a - b).norm() / norm_a).item()

def block_name(key: str) -> str:
    """Extract logical block name, e.g. 'layers.7', 'context_refiner.0', …"""
    m = re.match(r"^((?:layers|context_refiner|noise_refiner)\.\d+|[^.]+)", key)
    return m.group(1) if m else key.split(".")[0]

def block_category(block: str) -> str:
    if block.startswith("layers."):
        return "layers (DiT blocks)"
    if block.startswith("context_refiner."):
        return "context_refiner"
    if block.startswith("noise_refiner."):
        return "noise_refiner"
    if "embedder" in block or "embedder" in block:
        return "embedders"
    if "final_layer" in block:
        return "final_layer"
    return "other"

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading shard index files …")
idx_base  = load_index(BASE)
idx_turbo = load_index(TURBO)

all_keys_base  = set(idx_base.keys())
all_keys_turbo = set(idx_turbo.keys())
common_keys    = sorted(all_keys_base & all_keys_turbo)
only_base      = sorted(all_keys_base  - all_keys_turbo)
only_turbo     = sorted(all_keys_turbo - all_keys_base)

print(f"  Base  keys : {len(all_keys_base)}")
print(f"  Turbo keys : {len(all_keys_turbo)}")
print(f"  Common     : {len(common_keys)}")
if only_base:
    print(f"  Only in base  : {only_base}")
if only_turbo:
    print(f"  Only in turbo : {only_turbo}")
print()

print("Opening shard files …")
handles_base  = open_shards(BASE,  idx_base)
handles_turbo = open_shards(TURBO, idx_turbo)
print(f"  Base  shards: {len(handles_base)}")
print(f"  Turbo shards: {len(handles_turbo)}")
print()

# ── tensor-level comparison ────────────────────────────────────────────────────
print("Computing per-tensor metrics …")

tensor_results = {}   # key -> dict of metrics
block_buckets  = defaultdict(list)   # block_name -> list of metric dicts

for key in common_keys:
    tb = get_tensor(handles_base,  idx_base,  key)
    tt = get_tensor(handles_turbo, idx_turbo, key)

    if tb.shape != tt.shape:
        print(f"  [SHAPE MISMATCH] {key}: base={tuple(tb.shape)} turbo={tuple(tt.shape)}")
        continue

    cs  = cosine_sim(tb, tt)
    rl2 = rel_l2(tb, tt)
    md  = abs(tb.mean().item() - tt.mean().item())
    sd  = abs(tb.std().item()  - tt.std().item())
    identical = torch.equal(tb, tt)

    rec = dict(cosine_sim=cs, rel_l2=rl2, mean_diff=md, std_diff=sd,
               identical=identical, shape=tuple(tb.shape), numel=tb.numel())
    tensor_results[key] = rec

    blk = block_name(key)
    block_buckets[blk].append(rec)

print(f"  Processed {len(tensor_results)} tensors\n")

# ── block-level aggregation ────────────────────────────────────────────────────
def agg(records: list[dict]) -> dict:
    cs  = [r["cosine_sim"] for r in records if r["cosine_sim"] == r["cosine_sim"]]
    rl2 = [r["rel_l2"]     for r in records if r["rel_l2"]     == r["rel_l2"]]
    return dict(
        n_tensors   = len(records),
        numel_total = sum(r["numel"] for r in records),
        mean_cosine = sum(cs) / len(cs) if cs else float("nan"),
        min_cosine  = min(cs) if cs else float("nan"),
        mean_rel_l2 = sum(rl2) / len(rl2) if rl2 else float("nan"),
        max_rel_l2  = max(rl2) if rl2 else float("nan"),
        n_identical = sum(r["identical"] for r in records),
    )

block_agg = {blk: agg(recs) for blk, recs in sorted(block_buckets.items(),
             key=lambda x: (block_category(x[0]), x[0]))}

# ── print block table ──────────────────────────────────────────────────────────
COL = 28
print("=" * 110)
print(f"{'Block':<{COL}}  {'#tensors':>8}  {'params':>12}  {'mean_cosine':>12}  {'min_cosine':>11}  {'mean_relL2':>11}  {'max_relL2':>10}  {'identical':>10}")
print("=" * 110)

prev_cat = None
for blk, m in block_agg.items():
    cat = block_category(blk)
    if cat != prev_cat:
        print(f"  ── {cat} ──")
        prev_cat = cat
    ident_frac = f"{m['n_identical']}/{m['n_tensors']}"
    print(f"  {blk:<{COL-2}}  {m['n_tensors']:>8}  {m['numel_total']:>12,}  "
          f"{m['mean_cosine']:>12.6f}  {m['min_cosine']:>11.6f}  "
          f"{m['mean_rel_l2']:>11.6f}  {m['max_rel_l2']:>10.6f}  {ident_frac:>10}")

print("=" * 110)
print()

# ── category-level summary ─────────────────────────────────────────────────────
from collections import defaultdict as dd2

cat_buckets = dd2(list)
for blk, recs in block_buckets.items():
    cat_buckets[block_category(blk)].extend(recs)

cat_agg = {cat: agg(recs) for cat, recs in cat_buckets.items()}

print("=" * 90)
print("Category summary")
print("=" * 90)
print(f"{'Category':<30}  {'#tensors':>8}  {'params':>12}  {'mean_cosine':>12}  {'mean_relL2':>11}")
print("-" * 90)
for cat, m in sorted(cat_agg.items()):
    print(f"  {cat:<28}  {m['n_tensors']:>8}  {m['numel_total']:>12,}  "
          f"{m['mean_cosine']:>12.6f}  {m['mean_rel_l2']:>11.6f}")
print("=" * 90)
print()

# ── most-changed tensors ───────────────────────────────────────────────────────
print("Top 20 most-changed tensors (by cosine divergence from 1.0):")
print("-" * 80)
ranked = sorted(tensor_results.items(), key=lambda x: x[1]["cosine_sim"])
for key, m in ranked[:20]:
    print(f"  cosine={m['cosine_sim']:.6f}  relL2={m['rel_l2']:.6f}  {key}")

print()
print("Top 20 least-changed tensors:")
print("-" * 80)
for key, m in reversed(ranked[-20:]):
    print(f"  cosine={m['cosine_sim']:.6f}  relL2={m['rel_l2']:.6f}  {key}")

print()

# ── identical tensors ──────────────────────────────────────────────────────────
n_identical = sum(1 for m in tensor_results.values() if m["identical"])
print(f"Bit-identical tensors: {n_identical} / {len(tensor_results)}")

# ── per-layer cosine heatmap (text) ───────────────────────────────────────────
layer_cos = {}
for blk, m in block_agg.items():
    if blk.startswith("layers."):
        idx = int(blk.split(".")[1])
        layer_cos[idx] = m["mean_cosine"]

if layer_cos:
    print()
    print("Per DiT layer mean cosine similarity (layers 0–29):")
    print("  [1.000 = identical, lower = more diverged after distillation]")
    print()
    for i in range(30):
        v = layer_cos.get(i, float("nan"))
        bar_len = int((1.0 - v) * 400) if v == v else 0
        bar = "█" * min(bar_len, 60)
        print(f"  layer {i:2d}: {v:.6f}  {bar}")
    print()
