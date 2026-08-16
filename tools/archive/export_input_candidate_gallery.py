import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path("/mnt/data/wangbei/data/datasets")
OUT_DIR = Path("analysis_outputs/01_figures/input_candidates_20260626")

ROWS = [
    ("burst", "Playground_spectrum", "m30db", "normal"),
    ("burst", "Playground_spectrum", "m30db", "abnormal"),
    ("chirp", "Playground_spectrum", "m30db", "normal"),
    ("chirp", "Playground_spectrum", "m30db", "abnormal"),
    ("pulse", "Playground_spectrum", "m30db", "normal"),
    ("pulse", "Playground_spectrum", "m30db", "abnormal"),
]


def minmax(arr):
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    return (arr - lo) / max(hi - lo, 1e-6)


def read_gray(path, size=240):
    image = Image.open(path).convert("L").resize((size, size), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def clahe(gray):
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return op.apply(u8).astype(np.float32) / 255.0


def weak_row_residual(gray, alpha=0.1):
    background = np.median(gray, axis=1, keepdims=True)
    residual = np.abs(gray - background)
    return minmax(gray + alpha * residual)


def local2d_residual(gray, win=15):
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    background = cv2.medianBlur(u8, win).astype(np.float32) / 255.0
    return minmax(np.abs(gray - background))


def tophat(gray, kernel_size=13):
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    white_hat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    black_hat = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    return minmax(np.maximum(white_hat, black_hat))


def multiscale_residual(gray, kernels=(7, 15, 31)):
    residuals = []
    for k in kernels:
        background = cv2.GaussianBlur(gray, (k, k), 0)
        residuals.append(np.abs(gray - background))
    return minmax(np.maximum.reduce(residuals))


def local_variance(gray, win=15):
    mean = cv2.blur(gray, (win, win))
    mean_sq = cv2.blur(gray * gray, (win, win))
    return minmax(np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)))


def edge_gradient(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return minmax(np.sqrt(gx * gx + gy * gy))


def stack3(c1, c2, c3):
    return np.clip(np.stack([c1, c2, c3], axis=-1), 0.0, 1.0)


def variants(gray):
    contrast = clahe(gray)
    weak = weak_row_residual(gray)
    local = local2d_residual(gray)
    hat = tophat(gray)
    multi = multiscale_residual(gray)
    lvar = local_variance(gray)
    grad = edge_gradient(gray)
    return {
        "rgb/original": stack3(gray, gray, gray),
        "gray3": stack3(gray, gray, gray),
        "gray_local2d_edge": stack3(gray, local, grad),
        "clahe_gray": stack3(contrast, gray, gray),
        "old_gray_residual": stack3(contrast, weak, gray),
        "local2d": stack3(contrast, local, gray),
        "tophat": stack3(contrast, hat, gray),
        "multiscale": stack3(contrast, multi, gray),
        "local_var": stack3(contrast, lvar, gray),
        "edge_grad": stack3(contrast, grad, gray),
    }


def middle_channels(gray):
    return {
        "gray": gray,
        "CLAHE": clahe(gray),
        "old row residual": weak_row_residual(gray),
        "local2d residual": local2d_residual(gray),
        "tophat": tophat(gray),
        "multiscale residual": multiscale_residual(gray),
        "local variance": local_variance(gray),
        "edge gradient": edge_gradient(gray),
    }


def pick_sample(dataset, scene, jsr, split):
    folder = DATA_ROOT / dataset / scene / split / jsr
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files under {folder}")
    return files[len(files) // 2]


def draw_grid(samples, view_fn, titles, out_path, cmap=None):
    fig, axes = plt.subplots(
        nrows=len(samples),
        ncols=len(titles),
        figsize=(2.25 * len(titles), 2.05 * len(samples)),
        constrained_layout=True,
    )
    for row_idx, (row_label, gray, _) in enumerate(samples):
        views = view_fn(gray)
        for col_idx, title in enumerate(titles):
            ax = axes[row_idx, col_idx]
            ax.imshow(views[title], cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_triplets(samples, out_path):
    triplet_names = [
        "gray_local2d_edge",
        "old_gray_residual",
        "local2d",
        "tophat",
        "multiscale",
        "local_var",
        "edge_grad",
    ]
    fig, axes = plt.subplots(
        nrows=len(samples),
        ncols=len(triplet_names) * 3,
        figsize=(2.0 * len(triplet_names) * 3, 1.9 * len(samples)),
        constrained_layout=True,
    )
    for row_idx, (row_label, gray, _) in enumerate(samples):
        contrast = clahe(gray)
        channel_sets = {
            "gray_local2d_edge": (gray, local2d_residual(gray), edge_gradient(gray)),
            "old_gray_residual": (contrast, weak_row_residual(gray), gray),
            "local2d": (contrast, local2d_residual(gray), gray),
            "tophat": (contrast, tophat(gray), gray),
            "multiscale": (contrast, multiscale_residual(gray), gray),
            "local_var": (contrast, local_variance(gray), gray),
            "edge_grad": (contrast, edge_gradient(gray), gray),
        }
        for group_idx, name in enumerate(triplet_names):
            for c_idx, channel in enumerate(channel_sets[name]):
                ax = axes[row_idx, group_idx * 3 + c_idx]
                ax.imshow(channel, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(f"{name}\nC{c_idx + 1}", fontsize=8)
                if group_idx == 0 and c_idx == 0:
                    ax.set_ylabel(row_label, fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(samples, out_dir):
    lines = [
        "# Input Candidate Gallery",
        "",
        "Generated from the updated local RF datasets. This is only a visual screening step; it is not a metric result.",
        "",
        "Rows use `Playground_spectrum` and `m30db` for burst/chirp/pulse, with one normal and one abnormal sample per anomaly type.",
        "",
        "Files:",
        "",
        "- `candidate_composites.png`: simulated 3-channel input sent to CLIP.",
        "- `candidate_middle_channels.png`: the main enhancement channel for each candidate.",
        "- `candidate_triplets.png`: individual C1/C2/C3 channels for the active enhancement candidates.",
        "",
        "Sample files:",
        "",
    ]
    for label, _, path in samples:
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            "",
            "Reading guide:",
            "",
            "- `gray3` is the pure black/white baseline: the same grayscale image repeated across 3 channels.",
            "- `gray_local2d_edge` is gray + local 2D residual + edge gradient.",
            "- `clahe_gray` keeps the shape conservative: contrast-enhanced gray plus original gray.",
            "- `old_gray_residual` is the current morph input: CLAHE + row-wise weak residual + gray.",
            "- `local2d`, `tophat`, `multiscale`, `local_var`, and `edge_grad` are candidate third-channel replacements.",
            "",
            "A good universal candidate should make abnormal structure clearer without making normal samples look equally abnormal.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--size", type=int, default=240)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for dataset, scene, jsr, split in ROWS:
        path = pick_sample(dataset, scene, jsr, split)
        gray = read_gray(path, args.size)
        samples.append((f"{dataset} {split}", gray, path))

    composite_titles = [
        "rgb/original",
        "gray3",
        "gray_local2d_edge",
        "clahe_gray",
        "old_gray_residual",
        "local2d",
        "tophat",
        "multiscale",
        "local_var",
        "edge_grad",
    ]
    middle_titles = [
        "gray",
        "CLAHE",
        "old row residual",
        "local2d residual",
        "tophat",
        "multiscale residual",
        "local variance",
        "edge gradient",
    ]

    draw_grid(
        samples,
        variants,
        composite_titles,
        args.out_dir / "candidate_composites.png",
    )
    draw_grid(
        samples,
        middle_channels,
        middle_titles,
        args.out_dir / "candidate_middle_channels.png",
        cmap="gray",
    )
    draw_triplets(samples, args.out_dir / "candidate_triplets.png")
    write_readme(samples, args.out_dir)

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
