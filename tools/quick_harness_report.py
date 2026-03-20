#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

QUICK_METRIC_RE = re.compile(
    r"quick_metric step:(?P<step>\d+) val_bpb:(?P<val_bpb>[-+0-9.eE]+) train_time_ms:(?P<train_time_ms>[-+0-9.eE]+)"
)
ROUNDTRIP_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
SUBMISSION_INT8_ZLIB_RE = re.compile(
    r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes"
)


def parse_quick_metric(log_path: Path) -> dict[str, float | int | str]:
    if not log_path.exists():
        raise ValueError(f"log file does not exist: {log_path}")
    last_match: re.Match[str] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = QUICK_METRIC_RE.search(line)
            if match is not None:
                last_match = match
    if last_match is None:
        raise ValueError(
            f"No quick_metric line found in {log_path}. "
            "Expected: quick_metric step:<int> val_bpb:<float> train_time_ms:<float>"
        )
    return {
        "step": int(last_match.group("step")),
        "val_bpb": float(last_match.group("val_bpb")),
        "train_time_ms": float(last_match.group("train_time_ms")),
    }


def parse_total_submission_int8_zlib_bytes(log_path: Path) -> int | None:
    """Last `Total submission size int8+zlib:` line if present."""
    last: int | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = SUBMISSION_INT8_ZLIB_RE.search(line)
            if match is not None:
                last = int(match.group("bytes"))
    return last


def parse_final_int8_zlib_roundtrip_exact(log_path: Path) -> dict[str, float] | None:
    """Last `final_int8_zlib_roundtrip_exact` line if present (skipped when SKIP_POST_TRAIN_EVAL=1)."""
    last: re.Match[str] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = ROUNDTRIP_EXACT_RE.search(line)
            if match is not None:
                last = match
    if last is None:
        return None
    return {
        "val_loss": float(last.group("val_loss")),
        "val_bpb": float(last.group("val_bpb")),
    }


def command_snapshot(args: argparse.Namespace) -> int:
    profile = args.profile
    log_path = Path(args.log).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = parse_quick_metric(log_path)
    roundtrip = parse_final_int8_zlib_roundtrip_exact(log_path)
    payload = {
        "profile": profile,
        "log_path": str(log_path),
        "step": metric["step"],
        "val_bpb": metric["val_bpb"],
        "train_time_ms": metric["train_time_ms"],
    }
    if roundtrip is not None:
        payload["final_int8_zlib_roundtrip_exact"] = roundtrip
    out_path = out_dir / f"{profile}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"snapshot profile={profile} step={payload['step']} "
        f"val_bpb={payload['val_bpb']:.8f} train_time_ms={payload['train_time_ms']:.0f}"
    )
    print(f"wrote {out_path}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    runtime_factor = float(args.runtime_factor)

    if not baseline_path.exists():
        raise ValueError(f"baseline snapshot does not exist: {baseline_path}")
    if not candidate_path.exists():
        raise ValueError(f"candidate snapshot does not exist: {candidate_path}")
    if runtime_factor <= 0:
        raise ValueError(f"runtime_factor must be > 0, got {runtime_factor}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    bpb_improved = float(candidate["val_bpb"]) < float(baseline["val_bpb"])
    runtime_ceiling = float(baseline["train_time_ms"]) * runtime_factor
    runtime_ok = float(candidate["train_time_ms"]) <= runtime_ceiling
    passed = bpb_improved and runtime_ok

    print(
        f"baseline val_bpb={float(baseline['val_bpb']):.8f} train_time_ms={float(baseline['train_time_ms']):.0f}"
    )
    print(
        f"candidate val_bpb={float(candidate['val_bpb']):.8f} train_time_ms={float(candidate['train_time_ms']):.0f}"
    )
    print(
        f"delta val_bpb={float(candidate['val_bpb']) - float(baseline['val_bpb']):+.8f} "
        f"delta_train_time_ms={float(candidate['train_time_ms']) - float(baseline['train_time_ms']):+.0f}"
    )
    print(
        f"gate bpb_improved={int(bpb_improved)} runtime_ok={int(runtime_ok)} "
        f"(candidate_train_time_ms <= {runtime_factor:.2f}x baseline = {runtime_ceiling:.0f})"
    )
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


def command_promotion(args: argparse.Namespace) -> int:
    """Extract quick_metric, roundtrip exact val_bpb, and int8+zlib payload bytes vs limit."""
    limit = int(args.limit_bytes)
    rows: list[dict[str, object]] = []
    for raw in args.entry:
        name, sep, path_str = raw.partition("=")
        if not sep:
            raise ValueError(f"expected name=log_path, got: {raw!r}")
        name = name.strip()
        log_path = Path(path_str.strip()).resolve()
        metric = parse_quick_metric(log_path)
        roundtrip = parse_final_int8_zlib_roundtrip_exact(log_path)
        nbytes = parse_total_submission_int8_zlib_bytes(log_path)
        under = nbytes is not None and nbytes <= limit
        rows.append(
            {
                "name": name,
                "log_path": str(log_path),
                "quick_metric_val_bpb": metric["val_bpb"],
                "train_time_ms": metric["train_time_ms"],
                "final_int8_zlib_roundtrip_exact": roundtrip,
                "total_submission_int8_zlib_bytes": nbytes,
                "limit_bytes": limit,
                "under_16mb_limit": under,
            }
        )

    out_path = Path(args.out).resolve() if args.out else None
    payload = {"entries": rows, "limit_bytes": limit}
    text = json.dumps(payload, indent=2) + "\n"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(text, end="")

    missing = [r["name"] for r in rows if r["final_int8_zlib_roundtrip_exact"] is None]
    if missing:
        raise ValueError(f"missing final_int8_zlib_roundtrip_exact in: {', '.join(missing)}")
    missing_b = [r["name"] for r in rows if r["total_submission_int8_zlib_bytes"] is None]
    if missing_b:
        raise ValueError(f"missing Total submission size int8+zlib in: {', '.join(missing_b)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse and compare quick harness logs.")
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot", help="Parse one quick run log and write a profile snapshot JSON.")
    snapshot.add_argument("--profile", choices=("baseline", "candidate"), required=True)
    snapshot.add_argument("--log", required=True)
    snapshot.add_argument("--out-dir", required=True)
    snapshot.set_defaults(func=command_snapshot)

    compare = sub.add_parser("compare", help="Compare candidate snapshot against baseline snapshot.")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--runtime-factor", default=1.10)
    compare.set_defaults(func=command_compare)

    promotion = sub.add_parser(
        "promotion",
        help="Extract quick_metric, final_int8_zlib_roundtrip_exact, and int8+zlib bytes from logs.",
    )
    promotion.add_argument(
        "--entry",
        action="append",
        required=True,
        metavar="NAME=LOG_PATH",
        help="Repeatable: label and harness log path (e.g. baseline=logs/quick_harness/baseline.latest.log).",
    )
    promotion.add_argument(
        "--limit-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="Max allowed int8+zlib payload bytes (default: 16 MiB).",
    )
    promotion.add_argument(
        "--out",
        help="Write JSON summary to this path (default: print to stdout).",
    )
    promotion.set_defaults(func=command_promotion)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
