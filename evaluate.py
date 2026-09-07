#!/usr/bin/env python3
"""Evaluate a frozen EdgeAlign-Mamba checkpoint on ISIC2018 or PH2."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_setting import setting_config
from datasets.dataset import NPY_datasets
from engine import calculate_binary_metrics
from models.edgealign_mamba import EdgeAlignMamba


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


def evaluate(model, loader, device, threshold):
    model.eval()
    all_predictions = []
    all_targets = []
    per_sample = []
    with torch.no_grad():
        for images, masks, sample_ids in tqdm(loader, desc="Evaluating"):
            probabilities = model(images.to(device, non_blocking=True).float())
            predictions = probabilities.squeeze(1).cpu().numpy()
            targets = masks.squeeze(1).numpy()
            all_predictions.append(predictions)
            all_targets.append(targets)
            metrics = calculate_binary_metrics(predictions, targets, threshold)
            metrics['sample_id'] = sample_ids[0]
            per_sample.append(metrics)
    pooled = calculate_binary_metrics(np.concatenate(all_predictions), np.concatenate(all_targets), threshold)
    return pooled, per_sample


def save_results(output_dir, dataset_name, pooled, per_sample):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{dataset_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(pooled, handle, indent=2)

    metric_names = ['dice', 'iou', 'sensitivity', 'specificity', 'accuracy']
    with (output_dir / f"{dataset_name}_metrics_percent.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *metric_names])
        writer.writerow([dataset_name, *[f"{100 * pooled[name]:.2f}" for name in metric_names]])

    with (output_dir / f"{dataset_name}_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', *metric_names, 'tp', 'tn', 'fp', 'fn'])
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
    dataset = NPY_datasets(data_path, config, train=False, split=split, return_id=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    pooled, per_sample = evaluate(model, loader, device, config.threshold)
    save_results(args.output_dir, args.dataset, pooled, per_sample)
    print({name: f"{100 * pooled[name]:.2f}" for name in ['dice', 'iou', 'sensitivity', 'specificity', 'accuracy']})


if __name__ == "__main__":
    main()
