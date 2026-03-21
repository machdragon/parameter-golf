import os
import subprocess
from typing import Dict

import modal

from modal_image_fa3_pytorch import pytorch_fa3_hopper_image
from modal_train_volume_check import ensure_modal_training_data

# Training script: PR #287 base + Overtone init
LOCAL_TRAIN_GPT = "records/track_10min_16mb/2026-03-21_XSA_Overtone_EMA997/train_gpt.py"

app = modal.App("parameter-golf-xsa-overtone")
DATA_VOLUME = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

# FA3 image: see modal_image_fa3_pytorch.py (`uv_pip_install` on the shared PyTorch base).
image = (
    pytorch_fa3_hopper_image()
    .add_local_python_source("modal_image_fa3_pytorch", "modal_train_volume_check")
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
    ensure_modal_training_data()
    run_dir = f"/vol/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)
    env = {
        **os.environ,
        # Paths
        "DATA_PATH": "/vol/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/vol/tokenizers/fineweb_1024_bpe.model",
        "RUN_ID": run_id,
        # Model shape (PR #287 proven config)
        "NUM_LAYERS": "11",
        "BIGRAM_VOCAB_SIZE": "2048",
        "XSA_LAST_N": "4",
        # EMA (PR #287 proven config)
        "EMA_ENABLED": "1",
        "EMA_DECAY": "0.997",
        "SWA_ENABLED": "0",
        # Optimizer (PR #287 proven config)
        "MATRIX_LR": "0.025",
        "SCALAR_LR": "0.025",
        "TIED_EMBED_LR": "0.035",
        "MUON_MOMENTUM": "0.99",
        "MUON_MOMENTUM_WARMUP_START": "0.92",
        "MUON_MOMENTUM_WARMUP_STEPS": "1500",
        "WARMDOWN_ITERS": "3000",
        "MUON_WD": "0.04",
        "ADAM_WD": "0.04",
        # Training
        "ITERATIONS": "9000",
        "MAX_WALLCLOCK_SECONDS": "600",
        "EVAL_STRIDE": "64",
        # Overrides from CLI
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
    run_id: str = "xsa-overtone-001",
    seed: int = 1337,
    max_wallclock: float = 600.0,
    ema_decay: float = 0.997,
    num_layers: int = 11,
    xsa_last_n: int = 4,
) -> None:
    extra_env = {
        "SEED": str(seed),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "EMA_DECAY": str(ema_decay),
        "NUM_LAYERS": str(num_layers),
        "XSA_LAST_N": str(xsa_last_n),
    }
    print(f"Launching XSA + Overtone + EMA training: run_id={run_id}")
    print(f"  seed={seed}, ema_decay={ema_decay}, num_layers={num_layers}, xsa_last_n={xsa_last_n}")
    rc = train.remote(run_id=run_id, extra_env=extra_env)
    if rc != 0:
        raise RuntimeError(f"Remote training failed with exit code {rc}")
    print("Training completed successfully.")
    print(
        f"Artifacts on Modal volume `parameter-golf-data`: runs/{run_id}/\n"
        f"Download with:\n"
        f"  modal volume get parameter-golf-data runs/{run_id} ./modal_runs/{run_id}"
    )
