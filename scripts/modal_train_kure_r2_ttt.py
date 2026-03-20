import os
import subprocess
from typing import Dict

import modal

# Training script: KURE/R2 + tanh reparam + parallel EMA + LoRA TTT
LOCAL_TRAIN_GPT = "records/track_10min_16mb/2026-03-20_LAWA_KURE_R2_LoRATTT/train_gpt.py"

app = modal.App("parameter-golf-kure-r2-ttt")
DATA_VOLUME = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

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
    os.chdir(run_dir)
    env = {
        **os.environ,
        # Paths
        "DATA_PATH": "/vol/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/vol/tokenizers/fineweb_1024_bpe.model",
        "RUN_ID": run_id,
        # Model shape (PR #198/201 defaults)
        "NUM_LAYERS": "11",
        "BIGRAM_VOCAB_SIZE": "2048",
        # Optimizer
        "MUON_WD": "0.04",
        "ADAM_WD": "0.04",
        # LAWA parallel EMA (new defaults)
        "LAWA_ENABLED": "1",
        "LAWA_EMA_DECAY": "0.999",
        # KURE/R2 quantization-aware regularization
        "KURE_LAMBDA": "0.01",
        "R2_LAMBDA": "0.01",
        # Tanh reparameterization
        "TANH_REPARAM": "1",
        # LoRA TTT at eval
        "TTT_LORA_ENABLED": "1",
        "TTT_LORA_RANK": "8",
        "TTT_LORA_LR": "0.01",
        "TTT_CHUNK_SIZE": "256",
        "TTT_EVAL_SEQ_LEN": "1024",
        "TTT_BATCH_SIZE": "64",
        # Training
        "ITERATIONS": "20000",
        "MAX_WALLCLOCK_SECONDS": "600",
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
    run_id: str = "kure-r2-ttt-001",
    seed: int = 1337,
    max_wallclock: float = 600.0,
    kure_lambda: float = 0.01,
    r2_lambda: float = 0.01,
    tanh_reparam: int = 1,
    ttt_lora_enabled: int = 1,
) -> None:
    extra_env = {
        "SEED": str(seed),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "KURE_LAMBDA": str(kure_lambda),
        "R2_LAMBDA": str(r2_lambda),
        "TANH_REPARAM": str(tanh_reparam),
        "TTT_LORA_ENABLED": str(ttt_lora_enabled),
    }
    print(f"Launching KURE/R2 + Tanh + Parallel EMA + LoRA TTT training: run_id={run_id}")
    print(f"  seed={seed}, kure_lambda={kure_lambda}, r2_lambda={r2_lambda}")
    print(f"  tanh_reparam={tanh_reparam}, ttt_lora_enabled={ttt_lora_enabled}")
    rc = train.remote(run_id=run_id, extra_env=extra_env)
    if rc != 0:
        raise RuntimeError(f"Remote training failed with exit code {rc}")
    print("Training completed successfully.")
    print(
        f"Artifacts on Modal volume `parameter-golf-data`: runs/{run_id}/\n"
        f"Download with:\n"
        f"  modal volume get parameter-golf-data runs/{run_id} ./modal_runs/{run_id}"
    )
