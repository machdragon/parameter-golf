# Cloud runbook — Parameter Golf

Copy-paste flows for **RunPod** (submission-style) and **Modal** (remote GPU without managing pods).  
GCP / Azure: same ideas (VM + data disk + bundle results); sections reserved below.

To open a **second checkout** without switching branches:

```bash
git worktree add ../parameter-golf-second main
```

---

## RunPod — optimized next run (3 commands)

**First time with a [network volume](#runpod-network-volume-setup)?** Seed `/workspace` once (clone + FineWeb), then use the steps below on every pod.

From repo root on the pod (e.g. `cd /workspace/parameter-golf` after `git pull`):

1. **Preflight** (Python/torch/CUDA, GPU count, dataset + tokenizer):

   ```bash
   ./scripts/runpod_preflight.sh 8
   ```

   Use `1` instead of `8` on a 1×H100 pod.

2. **Train** (sets `RUN_ID`, `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE`, runs `torchrun`):

   ```bash
   ./scripts/runpod_train.sh 8x --run-id prod_8xh100_$(date -u +%Y%m%d_%H%MZ)
   ```

   Smoke on 1×H100: `./scripts/runpod_train.sh 1x --run-id smoke_001`

   Record script: add `--train-gpt records/track_10min_16mb/<your_record>/train_gpt.py`

3. **Finish** (meta + `metrics.json` + tarball for `runpodctl`):

   ```bash
   ./scripts/runpod_finish.sh "<same RUN_ID as step 2>"
   ```

Nothing needs to be “built” locally for RunPod: the **[Parameter Golf template](https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j)** image already includes PyTorch and deps. These scripts only **validate paths** and **standardize** env + post-run packaging.

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

### RunPod network volume setup

You create and attach volumes in the **[RunPod console](https://console.runpod.io)** (we cannot do that from this repo).

A **network volume** is separate billable storage (~[\$0.07/GB/mo for the first 1 TB](https://docs.runpod.io/storage/network-volumes)) that **persists when pods are deleted**. When attached to a Pod, it **replaces the normal volume disk** and is mounted at **`/workspace`**. That way you **pay once** (time + egress) to download FineWeb, then **later pods** mostly run **training only**.

Constraints from RunPod ([network volumes](https://docs.runpod.io/storage/network-volumes), [storage options](https://docs.runpod.io/pods/storage/sync-volumes)):

- **Secure Cloud** pods only (community cloud may not offer network volumes).
- Attach the volume **only when deploying** the Pod — **not** after the fact (no hot-attach).
- Pick a **datacenter** for the volume; GPU SKUs you can choose may **depend on that region**.

#### Step 1 — Create the volume

1. Open RunPod **[Storage](https://console.runpod.io/user/storage)** (or **New Network Volume** from the product UI).
2. Click **New Network Volume** / **Create Network Volume**.
3. Choose a **datacenter** (note it — use the **same** region when picking GPUs).
4. Set a **name** (e.g. `parameter-golf-data`) and **size** (e.g. **100 GB+** if you want full `sp1024` shards + repo + checkpoints; size can **increase** later, not shrink).
5. Create the volume.

Optional **API** (replace `RUNPOD_API_KEY` and pick a valid `dataCenterId` from their API/docs):

```bash
curl --request POST \
  --url https://rest.runpod.io/v1/networkvolumes \
  --header 'Authorization: Bearer RUNPOD_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{"name":"parameter-golf-data","size":100,"dataCenterId":"US-KS-2"}'
```

See [POST /networkvolumes](https://docs.runpod.io/api-reference/network-volumes/POST/networkvolumes).

#### Step 2 — Deploy a Pod with that volume

1. Go to **[Deploy a Pod](https://console.runpod.io/deploy)**.
2. Under storage, choose **Network volume** and select the volume you created (this mounts it as **`/workspace`**).
3. Select **GPU** (must be available in a datacenter compatible with the volume).
4. Use template **[Parameter Golf](https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j)** / image `runpod/parameter-golf:latest` as usual.
5. Deploy.

#### Step 3 — One-time seed: clone + FineWeb onto the volume

On the pod (SSH or web terminal), **`/workspace` is your persistent disk**. Populate it once:

**Option A — curl installer (no prior clone):**

```bash
curl -fsSL "https://raw.githubusercontent.com/machdragon/parameter-golf/main/scripts/runpod_seed_workspace.sh" | bash
```

Use a **fork** if needed:

```bash
export GIT_REPO_URL=https://github.com/YOUR_USER/parameter-golf.git
curl -fsSL "https://raw.githubusercontent.com/machdragon/parameter-golf/main/scripts/runpod_seed_workspace.sh" | bash
```

Small subset while testing (saves download time):

```bash
export TRAIN_SHARDS=1
curl -fsSL "https://raw.githubusercontent.com/machdragon/parameter-golf/main/scripts/runpod_seed_workspace.sh" | bash
```

**Option B — manual:**

```bash
cd /workspace
git clone https://github.com/machdragon/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024   # add --train-shards N if desired
```

#### Step 4 — Later pods (same volume)

Deploy a **new** Pod, attach the **same** network volume again. Then:

```bash
cd /workspace/parameter-golf
git pull
./scripts/runpod_preflight.sh 8
./scripts/runpod_train.sh 8x --run-id prod_8xh100_$(date -u +%Y%m%d_%H%MZ)
```

No re-download if `data/` is already on the volume.

**Optional:** [S3-compatible API](https://docs.runpod.io/storage/s3-api) can upload data **without** a running GPU pod to reduce wasted GPU time (advanced).

### Faster iteration (summary)

- **Network volume** = persistent `/workspace`; see above.
- **Smaller data** while debugging: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards N` or `TRAIN_SHARDS=N` with `runpod_seed_workspace.sh`.

### Manual env (equivalent to `runpod_train.sh`)

If you prefer explicit exports:

**1× H100 smoke**

```bash
RUN_ID=smoke_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**8× H100 (submission wallclock)**

```bash
RUN_ID=prod_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### After the run — save results off the pod

Training writes under the **current working directory** (`final_model.*`, `logs/<RUN_ID>.txt`).

**One shot:** `./scripts/runpod_finish.sh "$RUN_ID"` (runs the steps below).

**Piecemeal:**

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

**Verify the volume** (path defaults to `/` on the volume — you should see `datasets/`, `tokenizers/`, and after a good run `runs/`):

```bash
.venv-modal/bin/modal volume ls parameter-golf-data
.venv-modal/bin/modal volume ls parameter-golf-data runs
```

**Download to your machine:**

```bash
mkdir -p modal_runs
.venv-modal/bin/modal volume get parameter-golf-data runs/<run_id> ./modal_runs/<run_id>
```

**If `volume get` says `No such file or directory`:** the run likely used an **older** Modal definition that wrote logs and checkpoints under **`/root`** (container disk). That data is **not** on `parameter-golf-data` and is **gone** after the container exits. Recover by copying **function logs / stdout** from the Modal app run page (e.g. **Go to function logs**). Then `git pull` and re-run with the current `scripts/modal_train_lawa.py` (uses `/vol/runs/<run_id>` + `chdir` + `commit`); the log line `[modal] volume path '/vol/runs/…' after train: [...]` confirms files landed on the volume.

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
| `modal volume get` → **No such file or directory** for `runs/<id>` | **`runs/`** was never created: old deploy wrote to **`/root`** only (ephemeral). List root: `modal volume ls parameter-golf-data`. Recover from **Modal UI logs**; redeploy latest scripts. |
| **`Timed out waiting for final app logs`** (local CLI) | Often harmless; remote job may still finish. Confirm in the Modal **app run** page. |
