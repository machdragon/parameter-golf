import os
import subprocess
from typing import Dict

import modal

# Compatibility wrapper script name; defaults are periodic KURE(0.001, every=8), no tanh.
LOCAL_TRAIN_GPT = "records/track_10min_16mb/2026-03-21_KURE001_Tanh_XSA_EMA997/train_gpt.py"

app = modal.App("parameter-golf-kure001-xsa-ema997")
DATA_VOLUME = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


def ensure_modal_training_data() -> None:
    data_path = "/vol/datasets/fineweb10B_sp1024"
    tok_path = "/vol/tokenizers/fineweb_1024_bpe.model"
    missing = [p for p in (data_path, tok_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing Modal training data/tokenizer on volume: "
            + ", ".join(missing)
            + ". Sync first with scripts/modal_sync_data.sh."
        )


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
        # KURE periodic dial
        "KURE_LAMBDA": "0.001",
        "R2_LAMBDA": "0.0",
        "KURE_EVERY": "8",
        "TANH_REPARAM": "0",
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
            tail = " ..." if len(names) > 30 else ""
            print(f"[modal] volume path {run_dir!r} after train: {names[:30]}{tail}")
        except OSError as exc:
            print(f"[modal] could not list {run_dir!r}: {exc}")
        DATA_VOLUME.commit()


@app.local_entrypoint()
def main(
    run_id: str = "kure001-xsa-ema997-001",
    seed: int = 1337,
    max_wallclock: float = 600.0,
    ema_decay: float = 0.997,
    num_layers: int = 11,
    xsa_last_n: int = 4,
    kure_lambda: float = 0.001,
    r2_lambda: float = 0.0,
    kure_every: int = 8,
    tanh_reparam: bool = False,
) -> None:
    extra_env = {
        "SEED": str(seed),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "EMA_DECAY": str(ema_decay),
        "NUM_LAYERS": str(num_layers),
        "XSA_LAST_N": str(xsa_last_n),
        "KURE_LAMBDA": str(kure_lambda),
        "R2_LAMBDA": str(r2_lambda),
        "KURE_EVERY": str(kure_every),
        "TANH_REPARAM": "1" if tanh_reparam else "0",
    }
    print(f"Launching periodic-KURE + XSA + EMA training: run_id={run_id}")
    print(
        f"  seed={seed}, ema_decay={ema_decay}, num_layers={num_layers}, xsa_last_n={xsa_last_n}, "
        f"kure_lambda={kure_lambda}, r2_lambda={r2_lambda}, kure_every={kure_every}, "
        f"tanh_reparam={int(tanh_reparam)}"
    )
    rc = train.remote(run_id=run_id, extra_env=extra_env)
    if rc != 0:
        raise RuntimeError(f"Remote training failed with exit code {rc}")
    print("Training completed successfully.")
    print(
        f"Artifacts on Modal volume `parameter-golf-data`: runs/{run_id}/\\n"
        f"Download with:\\n"
        f"  modal volume get parameter-golf-data runs/{run_id} ./modal_runs/{run_id}"
    )
