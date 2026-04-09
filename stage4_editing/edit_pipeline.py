"""
Stage 4 — Editing Support: Z-Image-Edit and Z-Image-Omni-Base inference

Models (pending release)
------------------------
  Tongyi-MAI/Z-Image-Edit       -- edited on Z-Image for instruction-based editing
  Tongyi-MAI/Z-Image-Omni-Base  -- generation + editing base, highest diversity

Both use the same ZImagePipeline from diffusers but with an image conditioning input.
Reference pipeline call (once models are released):

  from diffusers import ZImagePipeline
  pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Edit", ...)

Planned sub-scripts
-------------------
  edit_pipeline.py   -- wrapper: load image + edit instruction → edited image
  eval_editing.py    -- benchmark on edit_bench, emu_edit_test_set, magic_brush

Cached evaluation datasets (available immediately, no download needed)
-----------------------------------------------------------------------
  LonelVino/edit_bench             HF_DATASETS_CACHE/LonelVino___edit_bench/
  facebook/emu_edit_test_set       HF_DATASETS_CACHE/facebook___emu_edit_test_set/
  osunlp/magic_brush               HF_DATASETS_CACHE/osunlp___magic_brush/
  timbrooks/instructpix2pix-...    HF_DATASETS_CACHE/timbrooks___instructpix2pix.../

Usage (once implemented)
------------------------
  python3 stage4_editing/edit_pipeline.py \\
      --image input.jpg \\
      --instruction "change the background to a snowy mountain" \\
      --out edited.jpg

  python3 stage4_editing/eval_editing.py \\
      --dataset edit_bench \\
      --out stage4_editing/outputs/edit_bench_results/
"""

# Placeholder — implementation coming in Stage 4 (after Z-Image-Edit release)
raise NotImplementedError("Stage 4 not yet implemented. See module docstring.")
