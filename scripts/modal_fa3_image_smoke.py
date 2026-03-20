"""
Quick Modal check: same FA3 image stack + volume as modal_train_lawa / modal_train_kure_r2_ttt.

Verifies on 1× H100 (briefly):
  - torch + CUDA
  - flash_attn_3 / flash_attn_interface import
  - parameter-golf-data volume: dataset dir + tokenizer file exist and tokenizer loads

Run (from repo root, after `modal setup`):
  .venv-modal/bin/modal run scripts/modal_fa3_image_smoke.py

Prerequisite: ./scripts/modal_sync_data.sh (same Modal account).

If a long training run is in progress, wait for it to finish (or use another workspace)
so you do not compete for H100 quota.

Image definition: `modal_image_fa3_pytorch.py` (`uv_pip_install` on the shared registry image).
"""

from __future__ import annotations

import modal

from modal_image_fa3_pytorch import pytorch_fa3_hopper_image
from modal_train_volume_check import VOL_DATASET_DIR, VOL_TOKENIZER_FILE, ensure_modal_training_data

app = modal.App("parameter-golf-fa3-image-smoke")
DATA_VOLUME = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

image = pytorch_fa3_hopper_image().add_local_python_source("modal_train_volume_check")


@app.function(
    image=image,
    gpu="H100:1",
    timeout=600,
    volumes={"/vol": DATA_VOLUME},
)
def verify_stack_and_volume() -> str:
    """Import FA3 on GPU path + confirm synced FineWeb + tokenizer on the volume."""
    import os

    import sentencepiece as spm
    import torch
    import flash_attn_interface

    ensure_modal_training_data()

    entries = sorted(os.listdir(VOL_DATASET_DIR))
    preview = entries[:5]
    more = f" (+{len(entries) - 5} more)" if len(entries) > 5 else ""

    sp = spm.SentencePieceProcessor(model_file=VOL_TOKENIZER_FILE)
    vocab_size = sp.vocab_size()

    lines = [
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}",
        f"flash_attn_interface={flash_attn_interface.__file__}",
        f"dataset_dir={VOL_DATASET_DIR!r} shard_count={len(entries)} sample={preview!r}{more}",
        f"tokenizer={VOL_TOKENIZER_FILE!r} vocab_size={vocab_size}",
        "[modal] FA3 image + volume checks OK.",
    ]
    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    print(verify_stack_and_volume.remote())
