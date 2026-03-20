"""
PyTorch Hub CUDA image + prebuilt FlashAttention 3 wheels for Modal.

`pytorch/pytorch` images ship Debian's PEP 668–managed `python3` on PATH. Modal's
`Image.pip_install` runs `python -m pip install`, which can hit that interpreter
and fail with *externally-managed-environment*. Use Conda's Python explicitly via
`run_commands` (Modal: https://modal.com/docs/guide/images#run-shell-commands-with-run_commands).

Wheel index must match CUDA × PyTorch on the base image:
https://windreamer.github.io/flash-attention3-wheels/
"""

from __future__ import annotations

import modal

PYTORCH_FA3_BASE = "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel"
FA3_WHEEL_FIND_LINKS = "https://windreamer.github.io/flash-attention3-wheels/cu126_torch2100"
# Official PyTorch Docker images install the stack under Conda.
_CONDA_PYTHON = "/opt/conda/bin/python"


def pytorch_fa3_hopper_image() -> modal.Image:
    """PyTorch 2.10 + cu12.6 + flash_attn_3 wheel (no Hopper source compile)."""
    return modal.Image.from_registry(PYTORCH_FA3_BASE).run_commands(
        f"{_CONDA_PYTHON} -m pip install --no-cache-dir numpy sentencepiece zstandard",
        (
            f"{_CONDA_PYTHON} -m pip install --no-cache-dir flash_attn_3 "
            f"--find-links {FA3_WHEEL_FIND_LINKS}"
        ),
    )
