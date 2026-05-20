import argparse
import csv
from pathlib import Path


DATASETS = ["burst_signal", "chirp_signal", "dsss_signal"]
SCENES = [
    "WeaponMuseum_spectrum",
    "Playground_spectrum",
    "TimeSquare_spectrum",
    "Gymnasium_spectrum",
]
NOISES = ["m10db", "m20db", "m30db"]


def read_i_roc(root, dataset, scene, noise):
    path = (
        Path(root)
        / dataset
        / scene
        / noise
        / "normal_75_25"
        / "k_1"
        / "csv"
        / "Seed_111-results.csv"
    )
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    key = f"{dataset}-{scene}"
    for row in rows:
        if row[""] == key:
            return float(row["i_roc"])
    raise ValueError(f"Missing row {key} in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic-root", default="result_split_prompt/generic_rgb")
    parser.add_argument("--rf-domain-root", default="result_split_prompt/rf_domain_rgb")
    parser.add_argument(
        "--rf-structured-root",
        default="result_split_prompt/rf_signal_structured_rgb",
    )
    parser.add_argument(
        "--output",
        default="analysis_outputs/prompt_three_way_compare/prompt_three_way_compare.csv",
    )
    args = parser.parse_args()

    methods = [
        ("generic + rgb", args.generic_root),
        ("rf_domain + rgb", args.rf_domain_root),
        ("rf_signal_structured + rgb", args.rf_structured_root),
    ]

    rows = []
    for dataset in DATASETS:
        for scene in SCENES:
            for noise in NOISES:
                row = {"dataset": dataset, "scene": scene, "noise": noise}
                for method, root in methods:
                    row[method] = read_i_roc(root, dataset, scene, noise)
                rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("method,burst,chirp,dsss,overall")
    for method, _ in methods:
        dataset_means = []
        for dataset in DATASETS:
            vals = [row[method] for row in rows if row["dataset"] == dataset]
            dataset_means.append(sum(vals) / len(vals))
        overall = sum(dataset_means) / len(dataset_means)
        print(
            f"{method},{dataset_means[0]:.4f},"
            f"{dataset_means[1]:.4f},{dataset_means[2]:.4f},{overall:.4f}"
        )

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
