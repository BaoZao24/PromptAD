#!/usr/bin/env python
"""Produce RAW TTA-view PNGs (no rendering / no annotations) for paper figures.

Output: 8 folders under analysis_outputs/01_figures/paper_arch_tta/tta_view_samples/

  SUPPORT side (normal):
    normal_originals/    - 5 normal baseline PNGs
    normal_blur/         - 5 normal PNGs after cv2.GaussianBlur(3x3)
    normal_shift_up/     - 5 normal PNGs after warpAffine dy=-4 px
    normal_shift_down/   - 5 normal PNGs after warpAffine dy=+4 px

  TEST side (abnormal):
    abnormal_originals/  - 5 abnormal input PNGs
    tta_blur/            - 5 abnormal PNGs after cv2.GaussianBlur(3x3)
    tta_shift_up/        - 5 abnormal PNGs after warpAffine dy=-4 px
    tta_shift_down/      - 5 abnormal PNGs after warpAffine dy=+4 px

  groundtruth/          - 5 anomaly-mask PNGs
  README.md             - folder map

All PNGs are exact raw transform outputs (cv2.imwrite), no matplotlib.
Transforms match tools/eval_cls_vit_patchcore_gallery.py (_augment_array):
  blur       = cv2.GaussianBlur(ksize=3x3, sigma=0)
  shift_up   = cv2.warpAffine(dy=-4 px, BORDER_REFLECT_101)
  shift_down = cv2.warpAffine(dy=+4 px, BORDER_REFLECT_101)
"""
import os, cv2, numpy as np

PAIRS = [
    ("burst_signal",
     "/mnt/data/wangbei/data/datasets/burst/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_burst_m30db_t00000-04000_f92.00-93.50MHz_patch006_abnormal.png",
     "/mnt/data/wangbei/data/datasets/burst/TimeSquare_spectrum/normal/m30db/TimeSquare_spectrum_burst_m30db_t04000-08000_f92.00-93.50MHz_patch054_normal.png"),
    ("chirp_signal",
     "/mnt/data/wangbei/data/datasets/chirp/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_chirp_m30db_t14000-18000_f105.60-107.10MHz_patch191_abnormal.png",
     "/mnt/data/wangbei/data/datasets/chirp/TimeSquare_spectrum/normal/m30db/TimeSquare_spectrum_chirp_m30db_t08000-12000_f105.60-107.10MHz_patch119_normal.png"),
    ("dsss_signal",
     "/mnt/data/wangbei/data/datasets/dsss/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_dsss_m30db_t10000-14000_f105.60-107.10MHz_patch143_abnormal.png",
     "/mnt/data/wangbei/data/datasets/dsss/TimeSquare_spectrum/normal/m30db/TimeSquare_spectrum_dsss_m30db_t14000-18000_f105.60-107.10MHz_patch191_normal.png"),
    ("pulse_signal",
     "/mnt/data/wangbei/data/datasets/pulse/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_pulse_m30db_t12000-16000_f100.00-101.50MHz_patch160_abnormal.png",
     "/mnt/data/wangbei/data/datasets/pulse/TimeSquare_spectrum/normal/m30db/TimeSquare_spectrum_pulse_m30db_t08000-12000_f100.00-101.50MHz_patch112_normal.png"),
    ("wideband_pulse",
     "/mnt/data/wangbei/data/datasets/wideband_pulse/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_wideband_pulse_m30db_t04000-08000_f92.00-93.50MHz_patch054_abnormal.png",
     "/mnt/data/wangbei/data/datasets/wideband_pulse/TimeSquare_spectrum/normal/m30db/TimeSquare_spectrum_wideband_pulse_m30db_t02000-06000_f92.00-93.50MHz_patch030_normal.png"),
]

SHIFT_PX = 4
BLUR_KSIZE = (3, 3)


def _shift(arr, dy):
    h, w = arr.shape[:2]
    mat = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(arr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def gt_path_of(abn):
    return os.path.join(os.path.dirname(abn).replace("/abnormal/", "/groundtruth/"),
                        os.path.basename(abn).replace("_abnormal.png", "_groundtruth.png"))


def gt_area_frac(gt):
    return float((gt > 0).sum()) / float(gt.size)


def make_views(img):
    return [cv2.GaussianBlur(img, BLUR_KSIZE, 0),
            _shift(img, dy=-SHIFT_PX),
            _shift(img, dy=+SHIFT_PX)]


def main():
    base = "analysis_outputs/01_figures/paper_arch_tta/tta_view_samples"
    dirs = {
        # support side (normal)
        "normal_originals":  os.path.join(base, "normal_originals"),
        "normal_blur":       os.path.join(base, "normal_blur"),
        "normal_shift_up":   os.path.join(base, "normal_shift_up"),
        "normal_shift_down": os.path.join(base, "normal_shift_down"),
        # test side (abnormal)
        "abnormal_originals": os.path.join(base, "abnormal_originals"),
        "tta_blur":           os.path.join(base, "tta_blur"),
        "tta_shift_up":       os.path.join(base, "tta_shift_up"),
        "tta_shift_down":     os.path.join(base, "tta_shift_down"),
        "groundtruth":        os.path.join(base, "groundtruth"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    summary = []
    for sig, abn, norm in PAIRS:
        img      = cv2.imread(abn,  cv2.IMREAD_GRAYSCALE)
        norm_img = cv2.imread(norm, cv2.IMREAD_GRAYSCALE)
        gt       = cv2.imread(gt_path_of(abn), cv2.IMREAD_GRAYSCALE)
        assert img is not None and norm_img is not None and gt is not None
        if gt.shape != img.shape:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        blur, sup, sdn = make_views(img)
        nblur, nsup, nsdn = make_views(norm_img)

        # support (normal) side
        cv2.imwrite(os.path.join(dirs["normal_originals"],  f"{sig}.png"), norm_img)
        cv2.imwrite(os.path.join(dirs["normal_blur"],       f"{sig}.png"), nblur)
        cv2.imwrite(os.path.join(dirs["normal_shift_up"],   f"{sig}.png"), nsup)
        cv2.imwrite(os.path.join(dirs["normal_shift_down"], f"{sig}.png"), nsdn)
        # test (abnormal) side
        cv2.imwrite(os.path.join(dirs["abnormal_originals"], f"{sig}.png"), img)
        cv2.imwrite(os.path.join(dirs["tta_blur"],           f"{sig}.png"), blur)
        cv2.imwrite(os.path.join(dirs["tta_shift_up"],       f"{sig}.png"), sup)
        cv2.imwrite(os.path.join(dirs["tta_shift_down"],     f"{sig}.png"), sdn)
        cv2.imwrite(os.path.join(dirs["groundtruth"],        f"{sig}.png"), gt)
        summary.append((sig, gt_area_frac(gt)))
        print(f"  {sig:18s}  anomaly={gt_area_frac(gt)*100:5.2f}%  wrote 9 PNGs")

    readme = (
        "# TTA view samples — folder map (m30db, TimeSquare scene)\n\n"
        "All 5 signal classes x 9 PNGs each. **Files are raw transform outputs — no annotation / no overlay.**\n\n"
        "## Folder layout (2x4 grid, support vs test x 4 views)\n\n"
        "| | identity (= original) | blur | shift up | shift down |\n"
        "|---|---|---|---|---|\n"
        "| **support (normal)**  | `normal_originals/`  | `normal_blur/`  | `normal_shift_up/`  | `normal_shift_down/`  |\n"
        "| **test (abnormal)**   | `abnormal_originals/` | `tta_blur/`     | `tta_shift_up/`     | `tta_shift_down/`     |\n\n"
        "Plus `groundtruth/` (5 anomaly-mask PNGs) and this README.\n\n"
        "## Transform parameters (match model `_augment_array`)\n\n"
        "- `blur`        : `cv2.GaussianBlur(ksize=3x3, sigma=0)`\n"
        "- `shift up`    : `cv2.warpAffine(dy = -4 px, BORDER_REFLECT_101)`\n"
        "- `shift down`  : `cv2.warpAffine(dy = +4 px, BORDER_REFLECT_101)`\n\n"
        "## Filename convention\n\n"
        "`{signal_class}.png` in every folder, e.g. `burst_signal.png`.\n\n"
        "## Anomaly area (of 256x256 patch)\n\n"
        "| signal | anomaly % |\n|---|---:|\n"
        + "\n".join(f"| {sig} | {f*100:.2f} % |" for sig, f in summary)
        + "\n\n## Notes\n\n"
        "- All samples are 256x256 grayscale PNG (uint8, 0-255).\n"
        "- `paired_tta=stft_shift_blur` has 4 views: identity/blur/shift-up/shift-down. Identity is a no-op so it is omitted (the \"identity column\" is just `*_originals/`).\n"
        "- `abnormal` patches were picked as the **largest-anomaly** sample per class; `normal` baselines are the **closest patch number** in the same f-band (training and test sets do not share patch indices).\n"
        "- `burst/chirp/pulse` anomalies are inherently tiny (0.2-0.4% of patch) at m30db - that is the real signal-level anomaly character, not a picking mistake.\n"
    )
    with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"\n  wrote {os.path.join(base, 'README.md')}")


if __name__ == "__main__":
    main()
