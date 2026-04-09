"""
Merge LoRA adapter weights into the base model and push to HuggingFace Hub.

Usage:
    python3 scripts/merge_and_push.py \
        --base_path /scratch/hf-cache/... \
        --adapter_path /scratch/training/.../final_adapter \
        --output_dir /tmp/merged_model \
        --repo_id DownFlow/Z-Image-Turbo-Fuli \
        [--private]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model and push to HF Hub")
    parser.add_argument("--base_path", required=True, help="Path to base diffusers pipeline")
    parser.add_argument("--adapter_path", required=True, help="Path to PEFT LoRA adapter dir")
    parser.add_argument("--output_dir", required=True, help="Local dir to save merged model")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo ID to push to")
    parser.add_argument("--private", action="store_true", help="Create/push as private repo")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    return parser.parse_args()


def main():
    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: No HF token found. Set HF_TOKEN env var or pass --token.")

    base_path = args.base_path
    adapter_path = args.adapter_path
    output_dir = args.output_dir

    print(f"Base model : {base_path}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Output dir : {output_dir}")
    print(f"Repo ID    : {args.repo_id}")
    print()

    # -------------------------------------------------------------------------
    # 1. Load pipeline (CPU first to avoid OOM flash; move to GPU for save ops)
    # -------------------------------------------------------------------------
    print("Loading pipeline...")
    import torch

    # Import the pipeline class dynamically via diffusers auto-loading
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # -------------------------------------------------------------------------
    # 2. Apply LoRA to transformer and merge
    # -------------------------------------------------------------------------
    print("Loading and merging LoRA adapter into transformer...")
    from peft import PeftModel

    transformer = pipe.transformer
    transformer = PeftModel.from_pretrained(transformer, adapter_path)
    transformer = transformer.merge_and_unload()
    transformer = transformer.to(torch.bfloat16)
    pipe.transformer = transformer

    print("LoRA merged successfully.")

    # -------------------------------------------------------------------------
    # 3. Save merged pipeline locally
    # -------------------------------------------------------------------------
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"Output dir already exists, removing: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    print(f"Saving merged pipeline to {output_dir} ...")
    pipe.save_pretrained(output_dir)
    print("Saved.")

    # -------------------------------------------------------------------------
    # 4. Push to Hub using huggingface_hub
    # -------------------------------------------------------------------------
    print(f"\nPushing to HuggingFace Hub: {args.repo_id} ...")
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Ensure repo exists
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"Repo ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Warning: repo creation step: {e}")

    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload merged Z-Image-Turbo + Fuliji LoRA (merge_and_unload)",
    )

    print(f"\nDone! Model live at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
