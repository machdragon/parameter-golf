import subprocess

import modal

app = modal.App("parameter-golf-fa3-image-smoke")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel")
    .apt_install("git")
    .pip_install(
        "numpy",
        "sentencepiece",
        "zstandard",
        "flash-attn>=2.7",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention",
        "cd /tmp/flash-attention/hopper && python setup.py install",
        "rm -rf /tmp/flash-attention",
        gpu="H100",
    )
)


@app.function(image=image, gpu="H100", timeout=20 * 60)
def smoke() -> str:
    gpu_info = subprocess.check_output(
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        ],
        text=True,
    ).strip()

    import torch
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    q = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_3_func(q, k, v, causal=True)
    return f"GPU={gpu_info}; FA3_out_shape={tuple(out.shape)}; dtype={out.dtype}"


@app.local_entrypoint()
def main() -> None:
    print("Building FA3/H100 image and running smoke test...")
    print(smoke.remote())
