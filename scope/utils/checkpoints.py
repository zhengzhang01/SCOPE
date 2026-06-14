import os
from pathlib import Path
from typing import Optional, Union


SCOPE_CHECKPOINT_NAME = "checkpoint.pt"
SCOPE_CHECKPOINT_REPO_ID = os.environ.get("SCOPE_CHECKPOINT_REPO_ID", "zhengzhang01/SCOPE")
SCOPE_CHECKPOINT_FILENAME = os.environ.get("SCOPE_CHECKPOINT_FILENAME", SCOPE_CHECKPOINT_NAME)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_checkpoint_path() -> Path:
    """Return the canonical SCOPE checkpoint path, with local bundle fallbacks."""
    root = repo_root()
    candidates = [
        root.parent / "checkpoint" / SCOPE_CHECKPOINT_NAME,
        root / "checkpoint" / SCOPE_CHECKPOINT_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def local_checkpoint_candidates() -> list[Path]:
    root = repo_root()
    return [
        root.parent / "checkpoint" / SCOPE_CHECKPOINT_NAME,
        root / "checkpoint" / SCOPE_CHECKPOINT_NAME,
    ]


def resolve_checkpoint_path(
    checkpoint: Optional[Union[str, Path]] = "auto",
    *,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """Resolve a local checkpoint path, downloading the default checkpoint when needed."""
    if checkpoint is None:
        checkpoint = "auto"

    checkpoint_str = str(checkpoint).strip()
    if checkpoint_str == "" or checkpoint_str.lower() == "auto":
        for candidate in local_checkpoint_candidates():
            if candidate.exists():
                return candidate
        return download_checkpoint(repo_id=repo_id, filename=filename)

    checkpoint_path = Path(checkpoint_str)
    if checkpoint_path.exists():
        return checkpoint_path

    if checkpoint_str.endswith(".pt"):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return download_checkpoint(repo_id=checkpoint_str, filename=filename)


def download_checkpoint(repo_id: Optional[str] = None, filename: Optional[str] = None) -> Path:
    """Download the SCOPE checkpoint from Hugging Face Hub and return the cached path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to auto-download the SCOPE checkpoint. "
            "Install dependencies with `pip install -r requirements.txt`, or pass "
            "`--checkpoint /path/to/checkpoint.pt`."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cached_path = hf_hub_download(
        repo_id=repo_id or SCOPE_CHECKPOINT_REPO_ID,
        repo_type="model",
        filename=filename or SCOPE_CHECKPOINT_FILENAME,
        token=token,
    )
    return Path(cached_path)
