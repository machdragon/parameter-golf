"""
Shared preflight for LAWA/KURE Modal trainers: dataset + tokenizer must exist on the volume.

Modal never reads your laptop's ./data/ — only the named volume mounted at /vol.
"""

from __future__ import annotations

import os

VOL_DATASET_DIR = "/vol/datasets/fineweb10B_sp1024"
VOL_TOKENIZER_FILE = "/vol/tokenizers/fineweb_1024_bpe.model"


def ensure_modal_training_data() -> None:
    """Raise with actionable instructions if required paths are missing on the volume."""
    problems: list[str] = []
    if not os.path.isdir(VOL_DATASET_DIR):
        problems.append(f"Missing directory: {VOL_DATASET_DIR}")
    if not os.path.isfile(VOL_TOKENIZER_FILE):
        problems.append(f"Missing file: {VOL_TOKENIZER_FILE}")
    if not problems:
        return
    msg = (
        "[modal] Dataset/tokenizer are not on Modal volume `parameter-golf-data` "
        "(mounted at /vol).\n"
        + "\n".join(f"  • {p}" for p in problems)
        + "\n\n"
        "Modal does not copy ./data from your machine automatically.\n"
        "From the repo root, on the same Modal account you use for `modal run`, run once:\n\n"
        "  ./scripts/modal_sync_data.sh\n\n"
        "That requires locally:\n"
        "  ./data/datasets/fineweb10B_sp1024/\n"
        "  ./data/tokenizers/fineweb_1024_bpe.model\n\n"
        "Verify on the volume:\n"
        "  .venv-modal/bin/modal volume ls parameter-golf-data\n"
        "  .venv-modal/bin/modal volume ls parameter-golf-data tokenizers\n"
        "  .venv-modal/bin/modal volume ls parameter-golf-data datasets\n\n"
        "Each Modal workspace/account has its own volume — run sync again when you switch accounts."
    )
    raise RuntimeError(msg)
