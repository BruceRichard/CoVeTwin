"""Download CoVeTwin-compatible model checkpoints from Hugging Face."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = PROJECT_ROOT / "pretrain"

snapshot_download(repo_id="Caoza/PhysX-Anything", local_dir=PRETRAIN_ROOT)
snapshot_download(
    repo_id="microsoft/TRELLIS-image-large",
    local_dir=PRETRAIN_ROOT / "trellis",
)
