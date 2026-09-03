"""Download the exact Hugging Face revisions required by the final pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main() -> None:
    models = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    for name in ("minilm", "clip"):
        model = models["embedders"][name]
        destination = snapshot_download(
            repo_id=model["model_id"], revision=model["revision"], local_files_only=False
        )
        print(f"{name}: {destination}")


if __name__ == "__main__":
    main()
