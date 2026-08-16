#!/usr/bin/env python
"""Safely wait for an idle GPU and run the FedJam four-branch experiment once.

The monitor never terminates another process.  It requires a GPU to have no
reported compute process, at least the configured free memory, and low GPU
utilization for two consecutive polls.  A final check is performed immediately
before launching the evaluator.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "autoresearch" / "fedjam-four-branches-260816"
DEFAULT_LOCK = Path("/tmp/spectramemad_fedjam_four_branches.lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=24.0)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dino-model", default="vit_base_patch14_dinov2")
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    return parser.parse_args()


def run_query(query: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def gpu_snapshot() -> list[dict]:
    rows = []
    for line in run_query("gpu=index,uuid,memory.free,utilization.gpu").splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            continue
        index, uuid, free_mib, utilization = fields
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "free_mib": int(float(free_mib)),
                "utilization": int(float(utilization)),
            }
        )

    busy_uuids: set[str] = set()
    try:
        for line in run_query("compute-apps=gpu_uuid,pid").splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) >= 2:
                busy_uuids.add(fields[0])
    except subprocess.CalledProcessError:
        # A transient nvidia-smi query failure is treated as unsafe by the
        # caller because the returned snapshot will contain no candidate.
        busy_uuids = {row["uuid"] for row in rows}

    for row in rows:
        row["has_compute_process"] = row["uuid"] in busy_uuids
    return rows


def safe_candidates(snapshot: list[dict], args: argparse.Namespace) -> list[dict]:
    min_free_mib = int(args.min_free_gib * 1024)
    return sorted(
        [
            row
            for row in snapshot
            if not row["has_compute_process"]
            and row["free_mib"] >= min_free_mib
            and row["utilization"] <= args.max_utilization
        ],
        key=lambda row: row["free_mib"],
        reverse=True,
    )


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_once(args: argparse.Namespace, gpu_index: int, log_handle) -> int:
    output_root = Path(args.output_root).resolve()
    command = [
        sys.executable,
        str(ROOT / "tools" / "eval_fedjam_four_branches.py"),
        "--output-root",
        str(output_root),
        "--device",
        "cuda",
        "--batch-size",
        str(args.batch_size),
        "--dino-model",
        args.dino_model,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[monitor] launching on physical GPU {gpu_index}: {' '.join(command)}", flush=True)
    log_handle.write(f"[monitor] launching on physical GPU {gpu_index}\n")
    log_handle.flush()
    process = subprocess.run(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    return int(process.returncode)


def main() -> int:
    args = parse_args()
    args.poll_seconds = max(1, min(args.poll_seconds, 60))
    args.stable_polls = max(2, args.stable_polls)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "gpu_monitor.log"
    status_path = output_root / "monitor_status.json"

    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[monitor] another monitor is already running", flush=True)
            return 2

        summary_path = output_root / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
            if summary.get("status") == "complete":
                print(f"[monitor] result already complete: {summary_path}", flush=True)
                return 0

        with log_path.open("a", encoding="utf-8") as log_handle:
            stable_index: int | None = None
            stable_count = 0
            while True:
                try:
                    snapshot = gpu_snapshot()
                except (OSError, subprocess.CalledProcessError) as exc:
                    print(f"[monitor] nvidia-smi unavailable: {exc}", flush=True)
                    stable_index = None
                    stable_count = 0
                    time.sleep(args.poll_seconds)
                    continue

                candidates = safe_candidates(snapshot, args)
                compact = [
                    {
                        "gpu": row["index"],
                        "free_gib": round(row["free_mib"] / 1024.0, 1),
                        "util": row["utilization"],
                        "compute": row["has_compute_process"],
                    }
                    for row in snapshot
                ]
                print(f"[monitor] snapshot={compact} candidates={[row['index'] for row in candidates]}", flush=True)
                log_handle.write(f"[monitor] snapshot={compact}\n")
                log_handle.flush()

                if candidates:
                    candidate = candidates[0]
                    if candidate["index"] == stable_index:
                        stable_count += 1
                    else:
                        stable_index = candidate["index"]
                        stable_count = 1
                    write_status(
                        status_path,
                        {
                            "status": "waiting_for_stable_gpu",
                            "candidate_gpu": stable_index,
                            "stable_count": stable_count,
                            "required_stable_polls": args.stable_polls,
                            "last_snapshot": snapshot,
                        },
                    )
                    if stable_count >= args.stable_polls:
                        final_candidates = safe_candidates(gpu_snapshot(), args)
                        final = next(
                            (row for row in final_candidates if row["index"] == stable_index),
                            None,
                        )
                        if final is not None:
                            write_status(
                                status_path,
                                {
                                    "status": "running",
                                    "gpu": stable_index,
                                    "last_snapshot": final,
                                },
                            )
                            code = run_once(args, stable_index, log_handle)
                            write_status(
                                status_path,
                                {
                                    "status": "complete" if code == 0 else "failed",
                                    "returncode": code,
                                    "gpu": stable_index,
                                },
                            )
                            return code
                        stable_index = None
                        stable_count = 0
                else:
                    stable_index = None
                    stable_count = 0
                    write_status(
                        status_path,
                        {"status": "waiting", "last_snapshot": snapshot},
                    )

                time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
