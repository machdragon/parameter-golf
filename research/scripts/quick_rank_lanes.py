#!/usr/bin/env python3
"""
Run stock train_gpt.py with quick_protocol + named presets from research/lane_env_presets.json.
Parses val_bpb from process output (last match). Writes JSON + CSV under research/sweeps/output/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRESETS_PATH = _REPO_ROOT / "research" / "lane_env_presets.json"
_OUT_DIR = _REPO_ROOT / "research" / "sweeps" / "output"
_LOG_DIR = _REPO_ROOT / "research" / "sweeps" / "logs"

_VAL_BPB_RE = re.compile(r"val_bpb[:\s=]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _find_torchrun(repo: Path) -> list[str]:
    venv = repo / ".venv" / "bin" / "torchrun"
    if venv.is_file():
        return [str(venv)]
    return ["torchrun"]


def _parse_val_bpb(text: str) -> float | None:
    matches = _VAL_BPB_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _load_presets() -> dict:
    with open(_PRESETS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _merge_env(quick: dict, preset_env: dict, extra: dict[str, str]) -> dict[str, str]:
    out = {**quick, **preset_env, **extra}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick rank train_gpt presets (screening only).")
    ap.add_argument(
        "preset_ids",
        nargs="*",
        default=["baseline", "lane_2_kv_heads_2", "lower_lr_like_sota"],
        help="Preset ids from lane_env_presets.json (default: baseline lane_2_kv_heads_2 lower_lr_like_sota)",
    )
    ap.add_argument("--repo-root", type=Path, default=_REPO_ROOT, help="Parameter golf repo root")
    ap.add_argument("--nproc", type=int, default=1, help="torchrun --nproc_per_node")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    train_script = repo / "train_gpt.py"
    if not train_script.is_file():
        print(f"train_gpt.py not found under {repo}", file=sys.stderr)
        return 1

    data_path = os.environ.get("DATA_PATH")
    tok_path = os.environ.get("TOKENIZER_PATH")
    if not data_path or not tok_path:
        print(
            "Set DATA_PATH and TOKENIZER_PATH to your dataset and tokenizer .model paths.",
            file=sys.stderr,
        )
        return 1

    raw = _load_presets()
    quick = raw["quick_protocol"]["env"]
    preset_list = {p["id"]: p for p in raw["presets"]}

    missing = [pid for pid in args.preset_ids if pid not in preset_list]
    if missing:
        print(f"Unknown preset id(s): {missing}. Known: {sorted(preset_list.keys())}", file=sys.stderr)
        return 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    launcher = _find_torchrun(repo)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[dict] = []

    for preset_id in args.preset_ids:
        meta = preset_list[preset_id]
        env = _merge_env(quick, meta.get("env") or {}, {})
        env.setdefault("VOCAB_SIZE", os.environ.get("VOCAB_SIZE", "1024"))
        env["DATA_PATH"] = data_path
        env["TOKENIZER_PATH"] = tok_path
        env["RUN_ID"] = f"quick_rank_{preset_id}_{ts}"

        cmd = [
            *launcher,
            "--standalone",
            f"--nproc_per_node={args.nproc}",
            str(train_script),
        ]
        log_path = _LOG_DIR / f"{env['RUN_ID']}.log"

        row: dict = {
            "preset_id": preset_id,
            "lane_ref": meta.get("lane_ref"),
            "notes": meta.get("notes"),
            "command": cmd,
            "env": {k: env[k] for k in sorted(env.keys())},
        }

        if args.dry_run:
            print("Would run:", " ".join(cmd))
            print("Env:", json.dumps(env, indent=2))
            results.append({**row, "dry_run": True})
            continue

        print(f"=== {preset_id} ===", flush=True)
        proc_env = os.environ.copy()
        proc_env.update(env)

        t0 = datetime.now(timezone.utc)
        p = subprocess.run(
            cmd,
            cwd=repo,
            env=proc_env,
            capture_output=True,
            text=True,
        )
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        log_path.write_text(out, encoding="utf-8")
        print(out[-4000:] if len(out) > 4000 else out, flush=True)
        t1 = datetime.now(timezone.utc)
        elapsed_s = (t1 - t0).total_seconds()
        val_bpb = _parse_val_bpb(out)

        row.update(
            {
                "returncode": p.returncode,
                "elapsed_sec": elapsed_s,
                "val_bpb": val_bpb,
                "log_file": str(log_path),
            }
        )
        results.append(row)
        print(f"val_bpb={val_bpb} elapsed_s={elapsed_s:.1f} rc={p.returncode}", flush=True)

    # Delta vs first preset (usually baseline)
    base_vpb = None
    for r in results:
        if r.get("val_bpb") is not None:
            base_vpb = r["val_bpb"]
            break
    if base_vpb is not None:
        for r in results:
            v = r.get("val_bpb")
            r["delta_val_bpb_vs_first"] = (v - base_vpb) if v is not None else None

    out_json = _OUT_DIR / f"quick_rank_{ts}.json"
    payload = {
        "created_utc": ts,
        "repo": str(repo),
        "results": results,
        "lanes_not_env_only": raw.get("lanes_not_env_only"),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_csv = _OUT_DIR / f"quick_rank_{ts}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "preset_id",
                "lane_ref",
                "val_bpb",
                "delta_val_bpb_vs_first",
                "elapsed_sec",
                "returncode",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "preset_id": r.get("preset_id"),
                    "lane_ref": r.get("lane_ref"),
                    "val_bpb": r.get("val_bpb"),
                    "delta_val_bpb_vs_first": r.get("delta_val_bpb_vs_first"),
                    "elapsed_sec": r.get("elapsed_sec"),
                    "returncode": r.get("returncode"),
                }
            )

    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_csv}", flush=True)
    return 0 if all(r.get("returncode", 0) == 0 for r in results if not r.get("dry_run")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
