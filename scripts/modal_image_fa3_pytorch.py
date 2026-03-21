"""
PyTorch Hub CUDA image + prebuilt FlashAttention 3 wheels for Modal.

`pytorch/pytorch` images can expose a distro-managed `python3` on PATH. Modal's
`Image.pip_install` runs `python -m pip install`, which can hit that interpreter
and fail with *externally-managed-environment*. Modal's documented fix for
registry-based images is `uv_pip_install`, which installs against the active
Python without emitting the raw `python -m pip install` step that trips PEP 668.

Wheel index must match CUDA × PyTorch on the base image:
https://windreamer.github.io/flash-attention3-wheels/
"""

from __future__ import annotations

import modal

PYTORCH_FA3_BASE = "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel"
FA3_WHEEL_FIND_LINKS = "https://windreamer.github.io/flash-attention3-wheels/cu126_torch2100"


def pytorch_fa3_hopper_image() -> modal.Image:
    """PyTorch 2.10 + cu12.6 + flash_attn_3 wheel (no Hopper source compile)."""
    return (
        modal.Image.from_registry(PYTORCH_FA3_BASE)
        .uv_pip_install("numpy", "sentencepiece", "zstandard")
        .uv_pip_install("flash_attn_3", find_links=FA3_WHEEL_FIND_LINKS)
    )
