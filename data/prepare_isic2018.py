#!/usr/bin/env python3
"""
Prepare ISIC2018 Task 1 (Lesion Segmentation) for EdgeAlign-Mamba.
This script expects a raw folder containing two subfolders:
  raw/images  -> original lesion images (jpg/png)
  raw/masks   -> corresponding segmentation masks (png), often named like ISIC_XXXXXX_segmentation.png
It will:
  - Match images and masks by ID
  - Convert images to RGB PNG and masks to single-channel PNG (0/255)
  - Split into train/val by ratio
  - Write into the EdgeAlign-Mamba dataset layout:
      out_dir/train/images/*.png
      out_dir/train/masks/*.png
      out_dir/val/images/*.png
      out_dir/val/masks/*.png
Usage:
  python data/prepare_isic2018.py --raw-dir PATH_TO_RAW_ISIC2018 --out-dir data/isic2018 --val-ratio 0.3 --seed 42
"""
import argparse
import os
import re
import sys
import random
import csv
from typing import Dict, List, Tuple
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_EXTS = {'.png', '.bmp', '.tif', '.tiff'}

ID_RE = re.compile(r"^(ISIC_\d+)")
SEG_SUFFIX_RE = re.compile(r"^(ISIC_\d+)_segmentation$")


def list_files(folder: str, exts: set) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def id_from_image_name(name: str) -> str:
    base = os.path.splitext(name)[0]
    m = ID_RE.match(base)
    return m.group(1) if m else base


def id_from_mask_name(name: str) -> str:
    base = os.path.splitext(name)[0]
    m = SEG_SUFFIX_RE.match(base)
    if m:
        return m.group(1)
    # fallback: try general ID
    m2 = ID_RE.match(base)
    return m2.group(1) if m2 else base


def pair_images_masks(img_dir: str, mask_dir: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    img_files = list_files(img_dir, IMG_EXTS)
    mask_files = list_files(mask_dir, MASK_EXTS)
    img_map = {id_from_image_name(f): f for f in img_files}
    mask_map = {id_from_mask_name(f): f for f in mask_files}
    common_ids = sorted(set(img_map.keys()) & set(mask_map.keys()))
    return common_ids, img_map, mask_map


def ensure_dirs(out_dir: str):
    for p in [
        os.path.join(out_dir, 'train', 'images'),
        os.path.join(out_dir, 'train', 'masks'),
        os.path.join(out_dir, 'val', 'images'),
        os.path.join(out_dir, 'val', 'masks'),
    ]:
        os.makedirs(p, exist_ok=True)


def save_image_to_png(src_path: str, dst_path: str):
    img = Image.open(src_path).convert('RGB')
    img.save(dst_path, format='PNG')


def save_mask_to_png(src_path: str, dst_path: str):
    m = Image.open(src_path).convert('L')
    # binarize: any non-zero -> 255
    m = m.point(lambda p: 255 if p > 0 else 0)
    m.save(dst_path, format='PNG')


def process_and_split(common_ids: List[str], img_map: Dict[str, str], mask_map: Dict[str, str],
                       img_dir: str, mask_dir: str, out_dir: str, val_ratio: float, seed: int):
    rnd = random.Random(seed)
    ids = common_ids.copy()
    rnd.shuffle(ids)
    n_total = len(ids)
    n_val = int(round(n_total * val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    with open(os.path.join(out_dir, 'split_manifest.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'split'])
        writer.writerows((id_, 'val' if id_ in val_ids else 'train') for id_ in ids)

    for i, id_ in enumerate(ids, 1):
        img_src = os.path.join(img_dir, img_map[id_])
        mask_src = os.path.join(mask_dir, mask_map[id_])
        split = 'val' if id_ in val_ids else 'train'
        img_dst = os.path.join(out_dir, split, 'images', f'{id_}.png')
        mask_dst = os.path.join(out_dir, split, 'masks', f'{id_}.png')
        save_image_to_png(img_src, img_dst)
        save_mask_to_png(mask_src, mask_dst)
        if i % 200 == 0 or i == n_total:
            print(f'Processed {i}/{n_total} pairs...')


def main():
    parser = argparse.ArgumentParser(description='Prepare ISIC2018 dataset for EdgeAlign-Mamba')
    parser.add_argument('--raw-dir', type=str, required=True, help='Path to raw dir containing images/ and masks/')
    parser.add_argument('--out-dir', type=str, default=os.path.join('data', 'isic2018'), help='Output dataset dir')
    parser.add_argument('--val-ratio', type=float, default=0.3, help='Validation split ratio (default 0.3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    parser.add_argument('--check-only', action='store_true', help='Only check pairing status, do not convert/split')
    args = parser.parse_args()

    img_dir = os.path.join(args.raw_dir, 'images')
    mask_dir = os.path.join(args.raw_dir, 'masks')

    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        print(f'ERROR: raw dir must contain "images" and "masks" subfolders. Got: {img_dir}, {mask_dir}')
        sys.exit(1)

    common_ids, img_map, mask_map = pair_images_masks(img_dir, mask_dir)

    extra_imgs = sorted(set(img_map.keys()) - set(mask_map.keys()))
    extra_masks = sorted(set(mask_map.keys()) - set(img_map.keys()))

    print(f'Total images: {len(img_map)} | masks: {len(mask_map)} | pairs: {len(common_ids)}')
    if extra_imgs:
        print(f'WARNING: {len(extra_imgs)} images without masks (examples): {extra_imgs[:5]}')
    if extra_masks:
        print(f'WARNING: {len(extra_masks)} masks without images (examples): {extra_masks[:5]}')

    if args.check_only:
        return

    ensure_dirs(args.out_dir)
    process_and_split(common_ids, img_map, mask_map, img_dir, mask_dir, args.out_dir, args.val_ratio, args.seed)
    print('Done. Output at:', args.out_dir)


if __name__ == '__main__':
    main()
