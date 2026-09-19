#!/usr/bin/env python3
"""Publish a validated WhiteSpacer PEFT adapter to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/WhiteSpacer"))
    parser.add_argument("--evaluation", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--repo-id", default="KaztoRay/WhiteSpacer")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Missing trained model: {args.model_dir}")
    report = json.loads(args.evaluation.read_text())
    if report["passed"] != report["total"]:
        raise RuntimeError("Refusing to publish: evaluation gate did not pass")
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.model_dir,
        commit_message="Release WhiteSpacer adapter",
    )


if __name__ == "__main__":
    main()

