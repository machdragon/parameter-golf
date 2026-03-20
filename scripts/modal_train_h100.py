import os
import subprocess
from typing import Dict

import modal

app = modal.App("parameter-golf-train-h100")
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


@app.function(image=image, gpu="H100:1", timeout=60 * 60)
def gpu_info() -> str:
    return subprocess.check_output(
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        ],
        text=True,
    ).strip()


@app.function(
    image=image,
    gpu="H100:1",
    timeout=6 * 60 * 60,
    volumes={"/vol": DATA_VOLUME},
)
def train(
    data_root: str,
    tokenizer_relpath: str,
    run_id: str,
    extra_env: Dict[str, str],
) -> int:
    env = {
        "DATA_PATH": f"/vol/{data_root}",
        "TOKENIZER_PATH": f"/vol/{tokenizer_relpath}",
        "RUN_ID": run_id,
        **extra_env,
    }
    command = "python /root/train_gpt.py"
    result = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    return result.returncode


@app.local_entrypoint()
def main(
    run_id: str = "modal-h100-run",
    data_root: str = "datasets/fineweb10B_sp1024",
    tokenizer_relpath: str = "tokenizers/fineweb_1024_bpe.model",
    iterations: int = 20000,
    max_wallclock_seconds: float = 600.0,
) -> None:
    print("Remote GPU:", gpu_info.remote())
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
