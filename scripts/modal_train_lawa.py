import os
import subprocess
from typing import Dict

import modal

# Training script baked into the image (path relative to repo root).
# Default: LAWA frontier record on this fork. Override for other records.
LOCAL_TRAIN_GPT = "records/track_10min_16mb/lawa_frontier/train_gpt.py"

app = modal.App("parameter-golf-lawa-frontier")
DATA_VOLUME = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel")
    # Base image has no git; required before cloning flash-attention for FA3 Hopper build.
    .apt_install("git")
    .pip_install(
        "numpy",
        "sentencepiece",
        "zstandard",
        "flash-attn>=2.7",
    )
    .run_commands(
        # Build FA3 Hopper kernels (provides flash_attn_interface)
        "git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention",
        "cd /tmp/flash-attention/hopper && python setup.py install",
        "rm -rf /tmp/flash-attention",
        gpu="H100",
    )
    .add_local_file(
        LOCAL_TRAIN_GPT,
        remote_path="/root/train_gpt.py",
    )
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=2 * 60 * 60,
    volumes={"/vol": DATA_VOLUME},
)
def train(run_id: str, extra_env: Dict[str, str]) -> int:
    run_dir = f"/vol/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    env = {
        **os.environ,
        # Paths
        "DATA_PATH": "/vol/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/vol/tokenizers/fineweb_1024_bpe.model",
        "RUN_ID": run_id,
        # PR #198 defaults
        "NUM_LAYERS": "11",
        "MUON_WD": "0.04",
        "ADAM_WD": "0.04",
        "BIGRAM_VOCAB_SIZE": "2048",
        "LAWA_ENABLED": "1",
        "LAWA_EMA_DECAY": "0.995",
        "ITERATIONS": "20000",
        "MAX_WALLCLOCK_SECONDS": "600",
        # Overrides
        **extra_env,
    }
    try:
        result = subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=8",
                "/root/train_gpt.py",
            ],
            env=env,
            text=True,
            cwd=run_dir,
            check=False,
        )
        return result.returncode
    finally:
        DATA_VOLUME.commit()


@app.local_entrypoint()
def main(
    run_id: str = "lawa-frontier-run",
    seed: int = 1337,
    lawa_decay: float = 0.995,
    lawa_enabled: int = 1,
    max_wallclock: float = 600.0,
) -> None:
    extra_env = {
        "SEED": str(seed),
        "LAWA_EMA_DECAY": str(lawa_decay),
        "LAWA_ENABLED": str(lawa_enabled),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
    }
    print(f"Launching LAWA frontier training: run_id={run_id}")
    print(f"  seed={seed}, lawa_decay={lawa_decay}, lawa_enabled={lawa_enabled}")
    rc = train.remote(run_id=run_id, extra_env=extra_env)
    if rc != 0:
        raise RuntimeError(f"Remote training failed with exit code {rc}")
    print("Training completed successfully.")
    print(
        f"Artifacts on Modal volume `parameter-golf-data`: runs/{run_id}/ "
        f"(download with: modal volume get parameter-golf-data runs/{run_id} ./modal_runs/{run_id})"
    )
