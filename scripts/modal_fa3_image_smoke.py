"""
Quick Modal check: same FA3 image stack as modal_train_lawa / modal_train_kure_r2_ttt.

Run (from repo root, after `modal setup`):
  .venv-modal/bin/modal run scripts/modal_fa3_image_smoke.py

Keep _FA3_* in sync with those scripts if you change the base or find-links URL.
"""

from __future__ import annotations

import subprocess

import modal

_FA3_PYTORCH_BASE = "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel"
_FA3_FIND_LINKS = "https://windreamer.github.io/flash-attention3-wheels/cu126_torch2100"

app = modal.App("parameter-golf-fa3-image-smoke")

image = (
    modal.Image.from_registry(_FA3_PYTORCH_BASE)
    .pip_install("numpy", "sentencepiece", "zstandard")
    .pip_install("flash_attn_3", find_links=_FA3_FIND_LINKS)
)


@app.function(image=image, gpu="H100:1", timeout=600)
def verify_imports() -> str:
    return subprocess.check_output(
        [
            "python",
            "-c",
            "import torch; import flash_attn_interface; "
            "print('torch', torch.__version__); "
            "print('flash_attn_interface', flash_attn_interface.__file__)",
        ],
        text=True,
    ).strip()


@app.local_entrypoint()
def main() -> None:
    print(verify_imports.remote())
