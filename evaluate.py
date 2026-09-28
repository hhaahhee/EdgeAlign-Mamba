#!/usr/bin/env python3
"""Evaluate a frozen EdgeAlign-Mamba checkpoint on ISIC2018 or PH2."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_setting import setting_config
from datasets.dataset import NPY_datasets
from engine import calculate_binary_metrics
from models.edgealign_mamba import EdgeAlignMamba


NEAREST_RESAMPLE = getattr(Image, "Resampling", Image).NEAREST


def calculate_boundary_metrics(prediction, target, threshold=0.5, tolerance=2.0):
    """Return per-image HD95 (pixels) and boundary F1."""
    prediction = np.asarray(prediction).squeeze() >= threshold
    target = np.asarray(target).squeeze() >= 0.5
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            f"Boundary metrics require equal 2-D masks, got {prediction.shape} and {target.shape}"
        )

    prediction_empty = not np.any(prediction)
    target_empty = not np.any(target)
    if prediction_empty and target_empty:
        return 0.0, 1.0, True, True
    if prediction_empty or target_empty:
        diagonal = float(np.hypot(*prediction.shape))
        return diagonal, 0.0, prediction_empty, target_empty

    structure = np.ones((3, 3), dtype=bool)
    prediction_boundary = np.logical_xor(
        prediction, binary_erosion(prediction, structure=structure, border_value=0)
    )
    target_boundary = np.logical_xor(
        target, binary_erosion(target, structure=structure, border_value=0)
    )

    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)
    prediction_distances = distance_to_target[prediction_boundary]
    target_distances = distance_to_prediction[target_boundary]

    hd95 = max(
        float(np.percentile(prediction_distances, 95)),
        float(np.percentile(target_distances, 95)),
    )
    precision = float(np.mean(prediction_distances <= tolerance))
    recall = float(np.mean(target_distances <= tolerance))
    bf1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return hd95, bf1, False, False


def prepare_evaluation_masks(prediction, mask_path, evaluation_size, threshold=0.5):
    """Prepare one binary prediction/GT pair shared by all evaluation metrics."""
    prediction = np.asarray(prediction).squeeze()
    if prediction.ndim != 2:
        raise ValueError(f"Expected a 2-D prediction, got shape {prediction.shape}")

    prediction_image = Image.fromarray(
        (prediction >= threshold).astype(np.uint8) * 255,
        mode="L",
    )
    if prediction_image.size != evaluation_size:
        prediction_image = prediction_image.resize(evaluation_size, NEAREST_RESAMPLE)

    with Image.open(mask_path) as target_image:
        target_image = target_image.convert("L").resize(
            evaluation_size,
            NEAREST_RESAMPLE,
        )
        target = np.asarray(target_image) >= 127

    return np.asarray(prediction_image) >= 127, target


def build_model(config):
    cfg = config.model_config
    return EdgeAlignMamba(
        num_classes=cfg['num_classes'],
        input_channels=cfg['input_channels'],
        depths=cfg['depths'],
        depths_decoder=cfg['depths_decoder'],
        drop_path_rate=cfg['drop_path_rate'],
        load_ckpt_path=None,
        use_mbam=cfg['use_mbam'],
        mbam_stages=cfg['mbam_stages'],
        mbam_on_skip=cfg['mbam_on_skip'],
        mbam_branch_ratio=cfg['mbam_branch_ratio'],
        mbam_dilation=cfg['mbam_dilation'],
        mbam_ca_ratio=cfg['mbam_ca_ratio'],
        mbam_branch_mask=cfg['mbam_branch_mask'],
        mbam_use_boundary_branch=cfg['mbam_use_boundary_branch'],
        mbam_use_scale_selection=cfg['mbam_use_scale_selection'],
        mbam_use_channel_attention=cfg['mbam_use_channel_attention'],
        mbam_use_edge_gate=cfg['mbam_use_edge_gate'],
        mbam_use_res_scale=cfg['mbam_use_res_scale'],
        use_dual_skip=cfg['use_dual_skip'],
        dual_skip_stages=cfg['dual_skip_stages'],
        mamba_skip_gate_ratio=cfg['mamba_skip_gate_ratio'],
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    state_dict = {
        key: value for key, value in state_dict.items()
        if 'total_ops' not in key and 'total_params' not in key
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing: {incompatible.missing_keys}; "
            f"unexpected: {incompatible.unexpected_keys}"
        )


def evaluate(model, loader, device, threshold, evaluation_size):
    model.eval()
    all_predictions = []
    all_targets = []
    per_sample = []
    with torch.no_grad():
        for images, _masks, sample_ids, mask_paths in tqdm(loader, desc="Evaluating"):
            probabilities = model(images.to(device, non_blocking=True).float())
            predictions = probabilities.squeeze(1).cpu().numpy()
            evaluation_prediction, evaluation_target = prepare_evaluation_masks(
                predictions[0], mask_paths[0], evaluation_size, threshold=threshold
            )
            all_predictions.append(evaluation_prediction[None, ...])
            all_targets.append(evaluation_target[None, ...])
            metrics = calculate_binary_metrics(
                evaluation_prediction,
                evaluation_target,
                threshold=0.5,
            )
            hd95, bf1, prediction_empty, target_empty = calculate_boundary_metrics(
                evaluation_prediction, evaluation_target, threshold=0.5, tolerance=2.0
            )
            metrics.update({
                'hd95': hd95,
                'bf1': bf1,
                'prediction_empty': prediction_empty,
                'target_empty': target_empty,
            })
            metrics['sample_id'] = sample_ids[0]
            per_sample.append(metrics)
    pooled = calculate_binary_metrics(
        np.concatenate(all_predictions),
        np.concatenate(all_targets),
        threshold=0.5,
    )
    pooled.update({
        'hd95': float(np.mean([row['hd95'] for row in per_sample])),
        'bf1': float(np.mean([row['bf1'] for row in per_sample])),
        'empty_prediction_count': int(sum(row['prediction_empty'] for row in per_sample)),
        'empty_target_count': int(sum(row['target_empty'] for row in per_sample)),
        'boundary_tolerance_pixels': 2.0,
    })
    return pooled, per_sample


def save_results(output_dir, dataset_name, pooled, per_sample):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{dataset_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(pooled, handle, indent=2)

    percentage_metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'accuracy', 'bf1']
    with (output_dir / f"{dataset_name}_metrics_percent.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *percentage_metrics, "hd95_pixels", "empty_prediction_count", "empty_target_count"])
        writer.writerow([
            dataset_name,
            *[f"{100 * pooled[name]:.2f}" for name in percentage_metrics],
            f"{pooled['hd95']:.2f}",
            pooled['empty_prediction_count'],
            pooled['empty_target_count'],
        ])

    with (output_dir / f"{dataset_name}_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'sample_id', *percentage_metrics, 'hd95', 'prediction_empty', 'target_empty',
                'tp', 'tn', 'fp', 'fn'
            ],
        )
        writer.writeheader()
        for row in per_sample:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Frozen-checkpoint evaluation for EdgeAlign-Mamba")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["isic18", "ph2"], default="ph2")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default="results/external_evaluation")
    args = parser.parse_args()

    config = setting_config
    data_path = args.data_path or (config.external_data_path if args.dataset == 'ph2' else config.data_path)
    split = args.split or ('external_test' if args.dataset == 'ph2' else 'val')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    dataset = NPY_datasets(
        data_path,
        config,
        train=False,
        split=split,
        return_id=True,
        return_mask_path=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    evaluation_size = (config.input_size_w, config.input_size_h)
    pooled, per_sample = evaluate(
        model,
        loader,
        device,
        config.threshold,
        evaluation_size,
    )
    save_results(args.output_dir, args.dataset, pooled, per_sample)
    displayed = {
        name: f"{100 * pooled[name]:.2f}"
        for name in ['dice', 'iou', 'sensitivity', 'specificity', 'accuracy', 'bf1']
    }
    displayed['hd95_pixels'] = f"{pooled['hd95']:.2f}"
    displayed['empty_prediction_count'] = pooled['empty_prediction_count']
    print(displayed)


if __name__ == "__main__":
    main()
