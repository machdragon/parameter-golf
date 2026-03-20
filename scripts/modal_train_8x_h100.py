"""
Modal: 8× H100 training with root train_gpt.py (no FlashAttention source build).

For record-specific scripts (e.g. LAWA + FA3 Hopper), use modal_train_lawa.py instead.
"""

import os
import subprocess
from typing import Dict

import modal

app = modal.App("parameter-golf-train-8xh100")
DATA_VOLUME = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("git")
    .pip_install(
        "numpy",
        "tqdm",
        "huggingface-hub",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
        "datasets",
        "tiktoken",
        "sentencepiece",
    )
    .add_local_file("train_gpt.py", remote_path="/root/train_gpt.py")
)


@app.function(image=image, gpu="H100:8", timeout=2 * 60 * 60, volumes={"/vol": DATA_VOLUME})
def train(
    data_root: str,
    tokenizer_relpath: str,
    run_id: str,
    extra_env: Dict[str, str],
) -> int:
    run_dir = f"/vol/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)
    env = {
        **os.environ,
        "DATA_PATH": f"/vol/{data_root}",
        "TOKENIZER_PATH": f"/vol/{tokenizer_relpath}",
        "RUN_ID": run_id,
        **extra_env,
    }
    try:
        return subprocess.run(
            ["torchrun", "--nproc_per_node=8", "/root/train_gpt.py"],
            env=env,
            text=True,
            cwd=run_dir,
            check=False,
        ).returncode
    finally:
        try:
            print(f"[modal] volume path {run_dir!r} after train: {sorted(os.listdir(run_dir))[:30]}")
        except OSError as e:
            print(f"[modal] could not list {run_dir!r}: {e}")
        DATA_VOLUME.commit()


@app.local_entrypoint()
def main(
    run_id: str = "modal-8x-run",
    data_root: str = "datasets/fineweb10B_sp1024",
    tokenizer_relpath: str = "tokenizers/fineweb_1024_bpe.model",
    iterations: int = 20000,
    max_wallclock_seconds: float = 600.0,
) -> None:
    rc = train.remote(
        data_root=data_root,
        tokenizer_relpath=tokenizer_relpath,
        run_id=run_id,
        extra_env={
            "ITERATIONS": str(iterations),
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        },
    )
    if rc != 0:
        raise RuntimeError(f"Remote training failed with exit code {rc}")
    print(
        f"Artifacts on Modal volume `parameter-golf-data`: runs/{run_id}/ "
        f"(download with: modal volume get parameter-golf-data runs/{run_id} ./modal_runs/{run_id})"
    )
