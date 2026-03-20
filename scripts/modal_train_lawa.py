import os
import subprocess
from typing import Dict

import modal

from modal_train_volume_check import ensure_modal_training_data

# Training script baked into the image (path relative to repo root).
# Default: LAWA frontier record on this fork. Override for other records.
LOCAL_TRAIN_GPT = "records/track_10min_16mb/lawa_frontier/train_gpt.py"

# Prebuilt FlashAttention 3 (Hopper) — no git clone / GPU compile during image build.
# Index must match CUDA × PyTorch on the base image:
# https://windreamer.github.io/flash-attention3-wheels/
_FA3_PYTORCH_BASE = "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel"
_FA3_FIND_LINKS = "https://windreamer.github.io/flash-attention3-wheels/cu126_torch2100"

app = modal.App("parameter-golf-lawa-frontier")
DATA_VOLUME = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

image = (
    modal.Image.from_registry(_FA3_PYTORCH_BASE)
    .pip_install("numpy", "sentencepiece", "zstandard")
    .pip_install("flash_attn_3", find_links=_FA3_FIND_LINKS)
).add_local_file(
    LOCAL_TRAIN_GPT,
    remote_path="/root/train_gpt.py",
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=2 * 60 * 60,
    volumes={"/vol": DATA_VOLUME},
)
def train(run_id: str, extra_env: Dict[str, str]) -> int:
    ensure_modal_training_data()
    run_dir = f"/vol/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    # Belt-and-suspenders: some stacks only respect process CWD (not subprocess cwd).
    os.chdir(run_dir)
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
        try:
            names = sorted(os.listdir(run_dir))
            tail = " …" if len(names) > 30 else ""
            print(f"[modal] volume path {run_dir!r} after train: {names[:30]}{tail}")
        except OSError as e:
            print(f"[modal] could not list {run_dir!r}: {e}")
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
