#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def minmax(x):
    x = x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / max(hi - lo, 1e-6)


def robust(x, percentile=99.5):
    x = np.maximum(x.astype(np.float32), 0.0)
    scale = float(np.percentile(x, percentile))
    return np.clip(x / max(scale, 1e-6), 0.0, 1.0)


def to_u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def label_tile(tile, label):
    tile = Image.fromarray(to_u8(tile)).convert("RGB")
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bbox[2] + 8, bbox[3] + 6), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=font)
    return tile


def variants(path):
    gray = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    row_bg = np.median(gray, axis=1, keepdims=True)
    row_bg_smooth = cv2.GaussianBlur(row_bg, (1, 31), 0)
    col_bg = np.median(gray, axis=0, keepdims=True)
    col_bg_smooth = cv2.GaussianBlur(col_bg, (31, 1), 0)
    abs_row_res = np.abs(gray - row_bg)
    pos_row_res = np.maximum(gray - row_bg_smooth, 0.0)
    abs_col_res = np.abs(gray - col_bg)
    pos_col_res = np.maximum(gray - col_bg_smooth, 0.0)

    blur_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=9.0, sigmaY=3.0)
    local_abs_res = np.abs(gray - blur_bg)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(to_u8(gray)).astype(np.float32) / 255.0

    # Stronger but still conservative DSSS residual candidates.
    weak_a02 = minmax(gray + 0.2 * abs_row_res)
    weak_a05 = minmax(gray + 0.5 * abs_row_res)
    pos_boost = minmax(gray + 0.6 * robust(pos_row_res, 99.0))
    local_boost = minmax(gray + 0.35 * robust(local_abs_res, 99.0))
    residual_mix = minmax(gray + 0.35 * robust(abs_row_res, 99.0) + 0.25 * robust(local_abs_res, 99.0))
    col_boost = minmax(gray + 0.6 * robust(pos_col_res, 99.0))
    row_col_mix = minmax(gray + 0.25 * robust(abs_row_res, 99.0) + 0.45 * robust(pos_col_res, 99.0))

    return [
        ("gray", gray),
        ("gray_contrast", gray_contrast),
        ("weak_residual_a0.2", weak_a02),
        ("weak_residual_a0.5", weak_a05),
        ("abs_row_residual", robust(abs_row_res, 99.0)),
        ("positive_row_boost", pos_boost),
        ("positive_col_boost", col_boost),
        ("row_col_mix", row_col_mix),
        ("local_abs_boost", local_boost),
        ("row_local_mix", residual_mix),
    ]


def make_sheet(samples, out_path, tile_w=220):
    rows = []
    for title, path in samples:
        vars_ = variants(path)
        tiles = []
        for label, img in vars_:
            pil = label_tile(img, label)
            h = int(tile_w * pil.height / pil.width)
            tiles.append(pil.resize((tile_w, h), Image.Resampling.BILINEAR))
        title_h = 24
        row_w = tile_w * len(tiles)
        row_h = title_h + tiles[0].height
        row = Image.new("RGB", (row_w, row_h), (245, 245, 245))
        draw = ImageDraw.Draw(row)
        draw.text((4, 5), title, fill=(0, 0, 0), font=ImageFont.load_default())
        for idx, tile in enumerate(tiles):
            row.paste(tile, (idx * tile_w, title_h))
        rows.append(row)

    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def pick_samples(root):
    root = Path(root)
    selected = []
    for scene in ["Playground_spectrum", "WeaponMuseum_spectrum", "TimeSquare_spectrum", "Gymnasium_spectrum"]:
        for level in ["m20db", "m30db"]:
            normal = sorted((root / scene / "normal" / level).glob("*.png"))
            abnormal = sorted((root / scene / "abnormal" / level).glob("*.png"))
            if normal:
                selected.append((f"{scene} {level} normal", normal[len(normal) // 2]))
            if abnormal:
                selected.append((f"{scene} {level} abnormal", abnormal[len(abnormal) // 2]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/data/wangbei/data/datasets/dsss")
    parser.add_argument("--out", default="analysis_outputs/dsss_residual_variant_compare/contact_sheet.png")
    args = parser.parse_args()

    samples = pick_samples(args.root)
    make_sheet(samples, Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
