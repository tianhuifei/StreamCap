"""Download a faster-whisper model into the project's local models directory."""

from __future__ import annotations

import argparse
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
WHISPER_MODELS_DIR = os.path.join(project_root, "models", "whisper")

SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")


def download_model(model_name: str, mirror: str | None = None) -> str:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: {', '.join(SUPPORTED_MODELS)}")

    if mirror:
        os.environ["HF_ENDPOINT"] = mirror

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("huggingface_hub is required. Install it with: pip install huggingface_hub") from e

    target_dir = os.path.join(WHISPER_MODELS_DIR, model_name)
    os.makedirs(target_dir, exist_ok=True)

    repo_id = f"Systran/faster-whisper-{model_name}"
    print(f"Downloading {repo_id} to {target_dir}")
    snapshot_download(repo_id=repo_id, local_dir=target_dir)
    print(f"Done. Model saved to: {target_dir}")
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a faster-whisper model into models/whisper/")
    parser.add_argument(
        "model",
        nargs="?",
        default="medium",
        choices=SUPPORTED_MODELS,
        help="Model size to download (default: medium)",
    )
    parser.add_argument(
        "--mirror",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Hugging Face mirror endpoint (default: https://hf-mirror.com)",
    )
    args = parser.parse_args()
    download_model(args.model, mirror=args.mirror)


if __name__ == "__main__":
    main()
