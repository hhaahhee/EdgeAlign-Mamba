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
    images = {}
    masks = {}
    for path in sorted(Path(raw_dir).rglob("*")):
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
    return [(sample_id, images[sample_id], masks[sample_id]) for sample_id in sorted(images)]


def prepare(raw_dir, out_dir, expected_count=200):
    pairs = discover_pairs(raw_dir)
    if expected_count > 0 and len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} PH2 pairs, found {len(pairs)}")

    output_root = Path(out_dir) / "external_test"
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for sample_id, image_path, mask_path in pairs:
        output_image = image_dir / f"{sample_id}.png"
        output_mask = mask_dir / f"{sample_id}.png"
        Image.open(image_path).convert("RGB").save(output_image, format="PNG")
        mask = Image.open(mask_path).convert("L").point(lambda value: 255 if value > 0 else 0)
        mask.save(output_mask, format="PNG")
        manifest_rows.append((sample_id, str(image_path), str(mask_path)))

    with (Path(out_dir) / "ph2_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "source_image", "source_mask"])
        writer.writerows(manifest_rows)
    print(f"Prepared {len(pairs)} PH2 pairs at {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PH2 for EdgeAlign-Mamba external evaluation")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", default="data/ph2")
    parser.add_argument("--expected-count", type=int, default=200)
    args = parser.parse_args()
    prepare(args.raw_dir, args.out_dir, args.expected_count)


if __name__ == "__main__":
    main()
