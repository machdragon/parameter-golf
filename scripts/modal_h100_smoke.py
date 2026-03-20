import subprocess

import modal

app = modal.App("h100-smoke-test")


@app.function(gpu="H100:1", timeout=600)
def check_h100() -> str:
    """Runs on a remote H100 worker and reports GPU info."""
    return subprocess.check_output(
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        ],
        text=True,
    ).strip()


@app.local_entrypoint()
def main() -> None:
    print("Requesting 1x H100 worker...")
    gpu_info = check_h100.remote()
    print("Remote GPU:", gpu_info)
