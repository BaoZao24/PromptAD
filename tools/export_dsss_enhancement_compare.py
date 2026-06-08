#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCENES = ["Playground_spectrum", "WeaponMuseum_spectrum", "TimeSquare_spectrum", "Gymnasium_spectrum"]
LEVELS = ["m20db", "m30db"]


def minmax(x):
    x = x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / max(hi - lo, 1e-6)


def to_u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def gamma_enhance(gray, gamma=0.7):
    x = np.clip(gray, 0.0, 1.0)
    return np.power(x, gamma)


def log_enhance(gray, gain=12.0):
    x = np.clip(gray, 0.0, 1.0).astype(np.float32)
    return np.log1p(gain * x) / np.log1p(np.array(gain, dtype=np.float32))


def weak_residual(gray, alpha=0.2):
    row_bg = np.median(gray, axis=1, keepdims=True)
    residual = np.abs(gray - row_bg)
    return minmax(gray + alpha * residual)


def clahe_contrast(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(to_u8(gray)).astype(np.float32) / 255.0


def panel_label(image, label):
    tile = Image.fromarray(to_u8(image)).convert('RGB')
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bbox[2] + 8, bbox[3] + 6), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=font)
    return tile


def make_variants(path):
    gray = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return [
        ('gray', gray),
        ('contrast_clahe', clahe_contrast(gray, clip_limit=2.0, tile_grid_size=(8, 8))),
        ('gamma_0.7', gamma_enhance(gray, gamma=0.7)),
        ('gamma_0.5', gamma_enhance(gray, gamma=0.5)),
        ('log_gain12', log_enhance(gray, gain=12.0)),
        ('weak_residual_a0.2', weak_residual(gray, alpha=0.2)),
    ]


def pick_samples(root: Path):
    samples = []
    for scene in SCENES:
        for level in LEVELS:
            normal = sorted((root / scene / 'normal' / level).glob('*.png'))
            abnormal = sorted((root / scene / 'abnormal' / level).glob('*.png'))
            if normal:
                samples.append((f'{scene} {level} normal', normal[len(normal) // 2]))
            if abnormal:
                samples.append((f'{scene} {level} abnormal', abnormal[len(abnormal) // 2]))
    return samples


def make_sheet(samples, out_path: Path, tile_w=220):
    rows = []
    title_font = ImageFont.load_default()
    for title, path in samples:
        variants = make_variants(path)
        tiles = []
        for label, img in variants:
            pil = panel_label(img, label)
            h = int(tile_w * pil.height / pil.width)
            tiles.append(pil.resize((tile_w, h), Image.Resampling.BILINEAR))
        title_h = 26
        row_w = tile_w * len(tiles)
        row_h = title_h + tiles[0].height
        row = Image.new('RGB', (row_w, row_h), (245, 245, 245))
        draw = ImageDraw.Draw(row)
        draw.text((4, 5), title, fill=(0, 0, 0), font=title_font)
        for idx, tile in enumerate(tiles):
            row.paste(tile, (idx * tile_w, title_h))
        rows.append(row)

    out = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/data/wangbei/data/datasets/dsss')
    parser.add_argument('--out', default='analysis_outputs/dsss_enhancement_compare/contact_sheet.png')
    args = parser.parse_args()

    samples = pick_samples(Path(args.root))
    make_sheet(samples, Path(args.out))
    print(args.out)


if __name__ == '__main__':
    main()
