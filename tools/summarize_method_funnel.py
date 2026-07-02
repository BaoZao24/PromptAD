#!/usr/bin/env python
"""Summarize pooled RF method-funnel result CSVs.

The script is intentionally read-only for training artifacts. It collects
available result tables, compares every valid method against pooled_rf_rgb,
and writes small summary files under the same experiment directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METHOD_STATUS = {
    "pooled_rf_rgb": "valid",
    "pooled_rf_rgb_vcpa": "pending",
    "pooled_rf_rgb_visual_adapter": "valid",
    "pooled_rf_rgb_grouped_meanmax": "pending_after_fix",
    "pooled_rf_rgb_object_agnostic": "invalid_vacuous_in_pooled",
}
EXPECTED_METHODS = tuple(DEFAULT_METHOD_STATUS)
EXPECTED_TASKS = ("cls", "seg")


def metric_for_task(task: str) -> str:
    if task == "cls":
        return "i_roc"
    if task == "seg":
        return "p_roc"
    raise ValueError(f"Unsupported task: {task}")


def discover_result_files(root: Path) -> list[Path]:
    return sorted(root.glob("results_*_pooled_rf_rgb*.csv"))


def parse_task(path: Path) -> str:
    name = path.name
    if name.startswith("results_cls_"):
        return "cls"
    if name.startswith("results_seg_"):
        return "seg"
    raise ValueError(f"Cannot parse task from {path}")


def parse_method(path: Path, task: str) -> str:
    prefix = f"results_{task}_"
    return path.stem.removeprefix(prefix)


def summarize_file(path: Path, root: Path) -> dict:
    task = parse_task(path)
    method = parse_method(path, task)
    metric = metric_for_task(task)
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise ValueError(f"{path} missing metric column {metric}")

    row = {
        "method": method,
        "task": task,
        "status": status_for_method(method),
        "result_file": str(path.relative_to(root)),
        "rows": len(df),
        "overall": df[metric].mean(),
    }
    for dataset, value in df.groupby("dataset")[metric].mean().items():
        signal = dataset.replace("_signal", "")
        row[f"{signal}_avg"] = value
    pulse_m40 = df[(df["dataset"] == "pulse_signal") & (df["jsr"] == "m40db")]
    if not pulse_m40.empty:
        row["pulse_m40_avg"] = pulse_m40[metric].mean()
    return row


def status_for_method(method: str) -> str:
    if method.endswith("_PREFIX_BUG"):
        return "invalid_prefix_bug"
    return DEFAULT_METHOD_STATUS.get(method, "unknown")


def add_missing_expected_rows(summary: pd.DataFrame) -> pd.DataFrame:
    existing = {(row.method, row.task) for row in summary.itertuples()}
    rows = []
    for task in EXPECTED_TASKS:
        for method in EXPECTED_METHODS:
            if (method, task) in existing:
                continue
            status = DEFAULT_METHOD_STATUS[method]
            if status == "valid":
                status = "pending"
            rows.append({
                "method": method,
                "task": task,
                "status": status,
                "result_file": "",
                "rows": 0,
            })
    if not rows:
        return summary
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)


def build_delta(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for task, group in summary.groupby("task"):
        base = group[group["method"] == "pooled_rf_rgb"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        numeric_cols = [
            col for col in group.columns
            if col not in {"method", "task", "status", "result_file", "rows"}
            and pd.api.types.is_numeric_dtype(group[col])
        ]
        for _, row in group.iterrows():
            record = {
                "method": row["method"],
                "task": task,
                "status": row["status"],
            }
            for col in numeric_cols:
                record[f"delta_{col}"] = row[col] - base_row[col]
            records.append(record)
    return pd.DataFrame(records)


def write_markdown(summary: pd.DataFrame, delta: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Method Funnel Summary",
        "",
        "自动汇总 `results_*.csv`。旧 `object_agnostic` 和修复前 `grouped_meanmax` 不应作为有效结论。",
        "",
        "## Available Results",
        "",
        summary.sort_values(["task", "method"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Delta vs pooled_rf_rgb",
        "",
        delta.sort_values(["task", "method"]).to_markdown(index=False, floatfmt=".4f") if not delta.empty else "No baseline delta available.",
        "",
        "## Reading Rules",
        "",
        "- `valid`: 可作为当前有效结果。",
        "- `pending`: 计划中但结果尚未落盘。",
        "- `pending_after_fix`: 旧结果失效，必须用修复后的代码重跑。",
        "- `invalid_prefix_bug`: 旧 grouped 结果，因 pooled RF 判定 bug 已作废。",
        "- `invalid_vacuous_in_pooled`: 在 pooled universal 设定下没有有效对照意义。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="analysis_outputs/20260627_method_funnel")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_result_files(root)
    rows = [summarize_file(path, root) for path in files]
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit(f"No result CSV files found under {root}")
    summary = add_missing_expected_rows(summary)

    delta = build_delta(summary)
    summary_path = out_dir / "method_funnel_summary.csv"
    delta_path = out_dir / "method_funnel_delta_vs_baseline.csv"
    markdown_path = out_dir / "method_funnel_summary.md"
    summary.sort_values(["task", "method"]).to_csv(summary_path, index=False)
    delta.sort_values(["task", "method"]).to_csv(delta_path, index=False)
    write_markdown(summary, delta, markdown_path)

    print(f"wrote {summary_path}")
    print(f"wrote {delta_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
