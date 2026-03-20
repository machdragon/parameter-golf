# Cloud runbook — Parameter Golf

Copy-paste flows for **RunPod** (submission-style) and **Modal** (remote GPU without managing pods).  
GCP / Azure: same ideas (VM + data disk + bundle results); sections reserved below.

**Branch / worktree:** lives on `docs/runbook-cloud-runs` (from `main`). To open a second checkout without switching branches:

```bash
git worktree add ../parameter-golf-runbook docs/runbook-cloud-runs
```

---

## Conventions

| Item | Location |
|------|----------|
| Training log | `logs/<RUN_ID>.txt` |
| Quantized artifact | `final_model.int8.ptz` (and optional `final_model.pt`) |
| Repro metadata | `logs/<RUN_ID>.meta.txt` (optional but recommended) |
| Parsed metrics JSON | `logs/<RUN_ID>.metrics.json` (optional; from script) |

Use a **unique `RUN_ID`** per prod run (e.g. `prod_8xh100_20260320`).

---

## RunPod (recommended for submission / eval match)

Official template: **[Parameter Golf on RunPod Hub](https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j)** — image `runpod/parameter-golf:latest` (Python, PyTorch, deps pre-installed).

### Faster iteration

- Attach a **[network volume](https://docs.runpod.io/pods/storage/create-network-volumes)** so `/workspace` (or your clone path) survives pod stops — avoid re-clone + re-download.
- Small data while debugging:  
  `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards N`

### 1× H100 smoke (from repo root)

```bash
RUN_ID=smoke_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### 8× H100 (submission wallclock)

```bash
RUN_ID=prod_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### After the run — save results off the pod

Training writes under the **current working directory** (`final_model.*`, `logs/<RUN_ID>.txt`).

1. **Optional:** record environment metadata (still on the pod, repo root):

   ```bash
   export RUN_ID=prod_8xh100
   ./scripts/make_run_meta.sh
   ```

2. **Optional:** JSON summary from the log:

   ```bash
   python3 scripts/extract_run_metrics.py "logs/${RUN_ID}.txt" -o "logs/${RUN_ID}.metrics.json"
   ```

3. **Bundle** for one file to transfer:

   ```bash
   ./scripts/bundle_run_results.sh "$RUN_ID" "logs/${RUN_ID}_bundle.tar.gz"
   ```

4. **Transfer** (many pods use SSH that does **not** expose SCP/SFTP — use RunPod’s tool):

   **On the pod**

   ```bash
   cd /workspace/parameter-golf   # or your path
   runpodctl send "logs/${RUN_ID}_bundle.tar.gz"
   ```

   **On your laptop**

   ```bash
   runpodctl receive <ONE_TIME_CODE>
   ```

   Docs: [Transfer files](https://docs.runpod.io/pods/storage/transfer-files), [SSH setup](https://docs.runpod.io/pods/configuration/use-ssh).

5. **Alternative:** enable **full SSH** (public IP + port) and use `scp`/`rsync` if you prefer.

---

## Modal (quick remote GPU)

### One-time

```bash
python3 -m venv .venv-modal
.venv-modal/bin/pip install -U pip modal
.venv-modal/bin/modal setup
```

Sync dataset + tokenizer to volume **`parameter-golf-data`** (paths mirror local `data/` layout):

```bash
./scripts/modal_sync_data.sh
```

### 1× H100 — baseline `train_gpt.py`

```bash
.venv-modal/bin/modal run scripts/modal_train_h100.py --run-id my-modal-1x
```

### 8× H100 — baseline `train_gpt.py` (no FlashAttention source build)

Faster image build than the LAWA/FA3 script:

```bash
.venv-modal/bin/modal run scripts/modal_train_8x_h100.py --run-id my-modal-8x
```

### 8× H100 — record-specific `train_gpt.py` (LAWA / FA3 Hopper)

`scripts/modal_train_lawa.py` uses a **heavy** image (FlashAttention Hopper compile on first build).  
Edit **`LOCAL_TRAIN_GPT`** at the top of that file to your record path, e.g.  
`records/track_10min_16mb/<your_record>/train_gpt.py`.

```bash
.venv-modal/bin/modal run scripts/modal_train_lawa.py --run-id lawa-test-001
```

### After the run — artifacts on the Modal volume

Training uses **`cwd=/vol/runs/<run_id>`**, so logs and `final_model.*` persist under volume path **`runs/<run_id>/`**.

Commit is performed automatically at the end of each training function.

**Download to your machine:**

```bash
mkdir -p modal_runs
.venv-modal/bin/modal volume get parameter-golf-data runs/<run_id> ./modal_runs/<run_id>
```

Then bundle from that directory:

```bash
./scripts/bundle_run_results.sh "<run_id>" "./modal_runs/<run_id>_bundle.tar.gz" "./modal_runs/<run_id>"
```

---

## GCP / Azure (placeholder)

When you have credits:

1. **VM**: NVIDIA GPU + CUDA driver; use the same **Python / PyTorch / deps** stack as RunPod eval (see `requirements.txt` and upstream challenge docs).
2. **Disk**: attach a persistent data disk for the repo + `data/` (like a RunPod network volume).
3. **Results**: same **`bundle_run_results.sh` + optional `make_run_meta.sh` / `extract_run_metrics.py`**, then `gsutil cp` / `az storage blob upload` (or SCP).

Add concrete machine images and commands here once you standardize on a SKU.

---

## Troubleshooting

| Symptom | Likely cause |
|--------|----------------|
| `git: not found` during Modal image build | Add `.apt_install("git")` before `git clone` in the image definition. |
| Modal build very slow once | Compiling CUDA extensions (e.g. FA Hopper). Cached until you change an earlier image layer — see [Modal images](https://modal.com/docs/guide/images). |
| SCP fails on RunPod | Use **`runpodctl send/receive`** or full SSH — see links above. |
| Empty folder after Modal run | Ensure training finished; check `modal volume ls parameter-golf-data runs/` — volume writes need **`commit()`** (handled in our scripts). |
