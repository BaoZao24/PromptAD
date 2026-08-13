#!/usr/bin/env python
"""Launch resumable OFDMA generation workers outside a terminal session."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ofdma_target_scene_coldstart_v1.json"
DEFAULT_PYTHON = Path("/mnt/data/wangbei/envs/ofdma-sim-py310/bin/python")
GENERATOR = REPO_ROOT / "tools" / "generate_ofdma_target_scene_dataset.py"

WORKERS = (
    {"worker_id": "validation_0_2", "gpu": "0", "split": "validation", "indexes": "0-2"},
    {"worker_id": "validation_3_4", "gpu": "0", "split": "validation", "indexes": "3-4"},
    {"worker_id": "test_0_4", "gpu": "1", "split": "test", "indexes": "0-4"},
    {"worker_id": "test_5_9", "gpu": "1", "split": "test", "indexes": "5-9"},
    {"worker_id": "test_10_14", "gpu": "2", "split": "test", "indexes": "10-14"},
    {"worker_id": "test_15_19", "gpu": "2", "split": "test", "indexes": "15-19"},
    {"worker_id": "test_20_24", "gpu": "3", "split": "test", "indexes": "20-24"},
    {"worker_id": "test_25_29", "gpu": "3", "split": "test", "indexes": "25-29"},
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_indexes(value: str) -> list[int]:
    indexes: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            indexes.extend(range(start, end + 1))
        else:
            indexes.append(int(part))
    return indexes


def scene_id(split: str, index: int) -> str:
    return f"{'val' if split == 'validation' else 'test'}_{index:03d}"


def worker_complete(output_root: Path, worker: dict[str, str]) -> bool:
    return all(
        (
            output_root
            / "scenes"
            / worker["split"]
            / scene_id(worker["split"], index)
            / "_COMPLETE.json"
        ).is_file()
        for index in parse_indexes(worker["indexes"])
    )


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return {"workers": []}
    return read_json(state_path)


def launch(args: argparse.Namespace) -> None:
    protocol = read_json(args.config.resolve())
    output_root = Path(protocol["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "generation_logs"
    logs_dir.mkdir(exist_ok=True)
    state_path = output_root / "generation_workers.json"
    previous = {
        item["worker_id"]: item for item in load_state(state_path).get("workers", [])
    }
    records: list[dict[str, Any]] = []

    for worker in WORKERS:
        if worker_complete(output_root, worker):
            records.append({**worker, "status": "complete", "pid": None})
            continue
        old = previous.get(worker["worker_id"])
        if old and old.get("pid") and process_alive(int(old["pid"])):
            records.append({**old, "status": "running"})
            continue

        log_path = logs_dir / f"{worker['worker_id']}.log"
        command = [
            str(args.python),
            str(GENERATOR),
            "--config",
            str(args.config.resolve()),
            "--mode",
            "formal",
            "--split",
            worker["split"],
            "--scene-indexes",
            worker["indexes"],
            "--resume",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = worker["gpu"]
        environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
        log_handle = log_path.open("a")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        records.append(
            {
                **worker,
                "status": "running",
                "pid": process.pid,
                "log_path": str(log_path),
                "command": command,
            }
        )
    write_json_atomic(state_path, {"workers": records})
    print(
        json.dumps(
            {
                "started_or_running": sum(
                    item["status"] == "running" for item in records
                ),
                "complete": sum(item["status"] == "complete" for item in records),
                "state": str(state_path),
            }
        )
    )


def status(args: argparse.Namespace) -> None:
    protocol = read_json(args.config.resolve())
    output_root = Path(protocol["output_root"]).resolve()
    state_path = output_root / "generation_workers.json"
    state = load_state(state_path)
    workers: list[dict[str, Any]] = []
    for worker in WORKERS:
        stored = next(
            (
                item
                for item in state.get("workers", [])
                if item["worker_id"] == worker["worker_id"]
            ),
            {},
        )
        complete = worker_complete(output_root, worker)
        alive = bool(stored.get("pid")) and process_alive(int(stored["pid"]))
        workers.append(
            {
                "worker_id": worker["worker_id"],
                "complete": complete,
                "alive": alive,
                "pid": stored.get("pid"),
            }
        )
    completed_scenes = len(list(output_root.glob("scenes/*/*/_COMPLETE.json")))
    completed_observations = len(list(output_root.glob("scenes/*/*/images/*/observation.json")))
    print(
        json.dumps(
            {
                "completed_scenes": completed_scenes,
                "expected_scenes": 35,
                "completed_observations": completed_observations,
                "expected_observations": 7140,
                "workers": workers,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "start":
        launch(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
