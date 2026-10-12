from __future__ import annotations

import argparse
from pathlib import Path

INFERENCE_FILES = [
    "config.yaml", "head.pt", "metadata.json", "README.md", "LICENSE*",
    "backbone/*.json", "backbone/*.safetensors", "backbone/*.model",
    "backbone/merges.txt", "backbone/vocab.*", "backbone/tokenizer*",
    "tokenizer/*", "backbone_adapter/*.json", "backbone_adapter/*.safetensors",
    "backbone_full/*.json", "backbone_full/*.safetensors", "backbone_full/*.model",
    "backbone_full/merges.txt", "backbone_full/vocab.*", "backbone_full/tokenizer*",
]


def download_model(repo_id="LIQIIIII/Jeva-4B", *, revision=None, local_dir=None):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=repo_id, revision=revision, local_dir=local_dir,
        allow_patterns=INFERENCE_FILES,
    ))


def main():
    parser = argparse.ArgumentParser(description="Download Jeva model files for local inference")
    parser.add_argument("repo_id", nargs="?", default="LIQIIIII/Jeva-4B")
    parser.add_argument("--revision", help="Commit hash or revision to pin")
    parser.add_argument("--local-dir")
    args = parser.parse_args()
    print(download_model(args.repo_id, revision=args.revision, local_dir=args.local_dir))


if __name__ == "__main__":
    main()
