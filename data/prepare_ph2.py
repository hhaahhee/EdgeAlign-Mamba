#!/usr/bin/env python3
"""Prepare all PH2 lesion images and masks for frozen-checkpoint evaluation."""

import argparse
import csv
import re
from pathlib import Path

from PIL import Image


SAMPLE_RE = re.compile(r"(IMD\d+)", re.IGNORECASE)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def discover_pairs(raw_dir):
    raw_root = Path(raw_dir)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"PH2 raw directory not found: {raw_root}")
    images = {}
    masks = {}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = SAMPLE_RE.search(path.name)
        if not match:
            continue
        sample_id = match.group(1).upper()
        descriptor = str(path).lower()
        target = masks if "lesion" in descriptor else images
        if sample_id in target:
            raise ValueError(f"Multiple candidate files for {sample_id}: {target[sample_id]} and {path}")
        target[sample_id] = path

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"PH2 pairing failed. Images without masks: {missing_masks[:5]}; "
            f"masks without images: {missing_images[:5]}"
        )
    pairs = [(sample_id, images[sample_id], masks[sample_id]) for sample_id in sorted(images)]
    if not pairs:
        raise ValueError(f"No PH2 image-mask pairs found under {raw_root}")
    return pairs


def prepare(raw_dir, out_dir, expected_count=200, overwrite=False):
    pairs = discover_pairs(raw_dir)
    if expected_count > 0 and len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} PH2 pairs, found {len(pairs)}")

    output_root = Path(out_dir) / "external_test"
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    manifest_path = Path(out_dir) / "ph2_manifest.csv"
    existing = [path for folder in (image_dir, mask_dir) if folder.is_dir() for path in folder.iterdir()]
    if (existing or manifest_path.is_file()) and not overwrite:
        raise FileExistsError(
            f"Prepared output already exists under {out_dir}. Use --overwrite to replace it."
        )
    if overwrite:
        for path in existing:
            if path.is_file():
                path.unlink()
        if manifest_path.is_file():
            manifest_path.unlink()
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    raw_root = Path(raw_dir).resolve()
    manifest_rows = []
    for sample_id, image_path, mask_path in pairs:
        output_image = image_dir / f"{sample_id}.png"
        output_mask = mask_dir / f"{sample_id}.png"
        Image.open(image_path).convert("RGB").save(output_image, format="PNG")
        mask = Image.open(mask_path).convert("L").point(lambda value: 255 if value > 0 else 0)
        mask.save(output_mask, format="PNG")
        manifest_rows.append(
            (sample_id, str(image_path.resolve().relative_to(raw_root)), str(mask_path.resolve().relative_to(raw_root)))
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "source_image", "source_mask"])
        writer.writerows(manifest_rows)
    print(f"Prepared {len(pairs)} PH2 pairs at {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PH2 for EdgeAlign-Mamba external evaluation")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", default="data/ph2")
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prepared dataset")
    args = parser.parse_args()
    prepare(args.raw_dir, args.out_dir, args.expected_count, args.overwrite)


if __name__ == "__main__":
    main()
