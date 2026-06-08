#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LEVELS = ["m20db", "m30db", "m40db"]
COLS = [
    "gray",
    "clahe_c20_t4",
    "clahe_c20_t16",
    "clahe_c5_t2",
    "clahe_c8_t2",
    "clahe_c8_t4",
    "clahe_c8_t16",
    "groundtruth",
]


def load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def minmax(x):
    x = x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / max(hi - lo, 1e-6)


def to_u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def gray_contrast(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(to_u8(gray)).astype(np.float32) / 255.0


def variants(img_path, gt_path=None):
    gray = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
    data = {
        "gray": gray,
        "clahe_c20_t4": gray_contrast(gray, clip_limit=20.0, tile_grid_size=(4, 4)),
        "clahe_c20_t16": gray_contrast(gray, clip_limit=20.0, tile_grid_size=(16, 16)),
        "clahe_c5_t2": gray_contrast(gray, clip_limit=5.0, tile_grid_size=(2, 2)),
        "clahe_c8_t2": gray_contrast(gray, clip_limit=8.0, tile_grid_size=(2, 2)),
        "clahe_c8_t4": gray_contrast(gray, clip_limit=8.0, tile_grid_size=(4, 4)),
        "clahe_c8_t16": gray_contrast(gray, clip_limit=8.0, tile_grid_size=(16, 16)),
    }
    if gt_path is not None and gt_path.exists():
        data["groundtruth"] = np.array(Image.open(gt_path).convert("L"))
    else:
        data["groundtruth"] = np.zeros_like(to_u8(gray))
    return data


def choose_public_normal(root: Path, count=3):
    meas = sorted([p for p in root.iterdir() if p.is_dir()])
    picks = []
    for folder in meas[:count]:
        images = sorted(folder.glob("*.png"))
        if images:
            picks.append((f"Normal\n{folder.name}", images[len(images)//2], None))
    return picks


def choose_abnormal(root: Path, level: str):
    imgs = sorted((root / "abnormal" / level).glob("*.png"))
    if not imgs:
        return None
    img = imgs[len(imgs)//2]
    gt = root / "groundtruth" / level / img.name.replace("_abnormal.png", "_groundtruth.png")
    return (f"DSSS abnormal\n{level}", img, gt)


def array_to_tile(arr, tile_w, tile_h):
    if arr.ndim == 2:
        pil = Image.fromarray(to_u8(arr) if arr.dtype != np.uint8 else arr).convert("RGB")
    else:
        pil = Image.fromarray(arr).convert("RGB")
    return pil.resize((tile_w, tile_h), Image.Resampling.BILINEAR)


def pretty_col(col):
    mapping = {
        "gray": "gray",
        "clahe_c20_t4": "clahe c20 t4",
        "clahe_c20_t16": "clahe c20 t16",
        "clahe_c5_t2": "clahe c5 t2",
        "clahe_c8_t2": "clahe c8 t2",
        "clahe_c8_t4": "clahe c8 t4",
        "clahe_c8_t16": "clahe c8 t16",
        "groundtruth": "groundtruth",
    }
    return mapping.get(col, col.replace("_", " "))


def make_sheet(rows, out_path: Path, tile_w=220, tile_h=220, left_w=260, top_h=78, gap=10):
    title_font = load_font(28)
    header_font = load_font(21)
    row_font = load_font(22)
    small_font = load_font(16)

    width = left_w + gap + len(COLS) * (tile_w + gap)
    height = top_h + gap + len(rows) * (tile_h + gap)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 18), "Public RF_SPE_PNG DSSS CLAHE Parameter Comparison", fill=(0, 0, 0), font=title_font)
    draw.text(
        (22, 50),
        "Compare aggressive CLAHE settings: clipLimit {20, 8, 5} x tileGridSize {2, 4, 16}",
        fill=(80, 80, 80),
        font=small_font,
    )

    x0 = left_w + gap
    for idx, col in enumerate(COLS):
        x = x0 + idx * (tile_w + gap)
        draw.rounded_rectangle((x, 18, x + tile_w, 62), radius=8, fill=(240, 240, 240), outline=(210, 210, 210))
        label = pretty_col(col)
        bbox = draw.textbbox((0, 0), label, font=header_font)
        tx = x + (tile_w - (bbox[2] - bbox[0])) // 2
        ty = 30
        draw.text((tx, ty), label, fill=(0, 0, 0), font=header_font)

    y0 = top_h + gap
    for r_idx, (row_label, img_path, gt_path) in enumerate(rows):
        y = y0 + r_idx * (tile_h + gap)
        draw.rounded_rectangle((15, y, left_w - 5, y + tile_h), radius=10, fill=(245, 245, 245), outline=(220, 220, 220))
        lines = row_label.split("\n")
        line_y = y + 62
        for line in lines:
            draw.text((25, line_y), line, fill=(0, 0, 0), font=row_font)
            line_y += 34
        draw.text((25, y + tile_h - 30), Path(img_path).name, fill=(90, 90, 90), font=small_font)

        feats = variants(img_path, gt_path)
        for c_idx, col in enumerate(COLS):
            x = x0 + c_idx * (tile_w + gap)
            tile = array_to_tile(feats[col], tile_w, tile_h)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(210, 210, 210), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-normal-root", default="/mnt/data/wangbei/data/RF_SPE_PNG/RF_Spectrum_Public_Dataset")
    parser.add_argument("--dsss-root", default="/mnt/data/wangbei/data/RF_SPE_PNG/dsss")
    parser.add_argument("--out", default="analysis_outputs/public_dsss_contrast_compare/contact_sheet.png")
    args = parser.parse_args()

    rows = []
    rows.extend(choose_public_normal(Path(args.public_normal_root), count=3))
    dsss_root = Path(args.dsss_root)
    for level in LEVELS:
        picked = choose_abnormal(dsss_root, level)
        if picked is not None:
            rows.append(picked)
    make_sheet(rows, Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
