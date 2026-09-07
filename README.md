# EdgeAlign-Mamba

EdgeAlign-Mamba is a binary skin-lesion segmentation model derived from VM-UNet. It retains the VM-UNet visual state-space encoder-decoder and adds Multi-scale Boundary-aware Attention Modules (MBAMs) plus Lightweight Mamba Skip Gates (LMSGs). The active experiment protocol trains on ISIC2018 and evaluates the frozen best checkpoint on PH2.

## Environment

- Python 3.10
- PyTorch 2.1.2
- CUDA 11.8
- NVIDIA GeForce RTX 4080 SUPER
- Input size: 256 x 256 RGB

Install the pinned Python dependencies with:

```bash
pip install -r requirements.txt
```

CUDA-specific Mamba packages may require wheels matching PyTorch 2.1.2 and CUDA 11.8.

If the model import fails, verify that `mamba-ssm`, `causal-conv1d`, PyTorch,
and CUDA are mutually compatible. EdgeAlign-Mamba reports a direct import error
when the required CUDA selective-scan implementation is unavailable.

## Pre-trained VMamba initialization

The default configuration initializes the VM-UNet backbone from
`vmamba_small_e238_ema.pth`. The official VM-UNet project provides the
pre-trained-weight download links in the
[pre-trained weights section of its README](https://github.com/JCruan519/VM-UNet/blob/main/README.md#2-prepare-the-pre_trained-weights).
Please use those official links to obtain the file. At the time of writing, one
of the links listed by VM-UNet points to this
[Google Drive folder](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing).

Download the checkpoint and place it as follows:

```text
pre_trained_weights/
  vmamba_small_e238_ema.pth
```

The local directory name intentionally differs from VM-UNet's documented
`pretrained_weights/` directory and matches `configs/config_setting.py`. Model
weights are excluded from Git by `.gitignore`. The checkpoint is an upstream
artifact and is neither mirrored nor redistributed by this repository; consult
VM-UNet's repository and its linked download services for the applicable terms.

## Model configuration

The experiment entry point is `models/edgealign_mamba.py`, which exposes `EdgeAlignMamba`. The internal `vmunet` attribute is intentionally retained so checkpoints produced before the public rename remain compatible.

The configured encoder and decoder depths are `[2, 2, 2, 2]` and `[2, 2, 2, 1]`. MBAM is enabled before encoder VSS stages 0, 1, and 2. At decoder stages 1, 2, and 3, a deeper skip is added before the decoder layer and the adjacent shallower skip is fused after upsampling through LMSG.

## Data layout

ISIC2018 is used for model development:

```text
data/isic2018/
  train/images/
  train/masks/
  val/images/
  val/masks/
  split_manifest.csv
```

Prepare the official ISIC2018 Task 1 images and masks with the reproducible 70:30 split:

```bash
python data/prepare_isic2018.py --raw-dir PATH_TO_RAW_ISIC2018 --out-dir data/isic2018 --val-ratio 0.3 --seed 42
```

PH2 is reserved for external evaluation and is never used for training, checkpoint selection, or fine-tuning:

```text
data/ph2/
  external_test/images/
  external_test/masks/
  ph2_manifest.csv
```

Prepare all 200 PH2 image-mask pairs with:

```bash
python data/prepare_ph2.py --raw-dir PATH_TO_RAW_PH2 --out-dir data/ph2 --expected-count 200
```

## Preprocessing and augmentation

This repository preserves the preprocessing used for the reported experiments. Images are converted to RGB and masks to one-channel values in `[0, 1]`. For ISIC2018, scalar training statistics are 157.561 and 26.706, while validation/external preprocessing uses 149.034 and 32.022. After this standardization, every image is rescaled by its own global minimum and maximum to `[0, 255]`.

The training pipeline applies paired horizontal and vertical flips independently with probability 0.5. A rotation angle is sampled once when the training transform is initialized, uniformly from 0 to 360 degrees, and that fixed run-level angle is applied to a sample with probability 0.5. Torchvision's current default interpolation is retained for both rotation and resize. Finally, images and masks are resized to 256 x 256; tensors are cast to float32 when transferred to the GPU. Validation and PH2 evaluation contain no stochastic augmentation.

## Training

The default experiment uses batch size 32 for 100 epochs, AdamW with learning rate `1e-3`, betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `1e-2`. The scheduler is cosine annealing with `T_max=10` and `eta_min=1e-5`. The loss is the equally weighted sum of binary cross-entropy and soft Dice loss. Seed 42 is applied to Python, NumPy, and PyTorch; cuDNN deterministic mode is enabled and benchmark mode is disabled.

```bash
python train.py
```

The checkpoint with the minimum ISIC2018 validation loss is saved as the best model. The post-training evaluation in `train.py` is a best-checkpoint validation evaluation, not an independent test.

## Frozen-checkpoint evaluation

Evaluate the same selected checkpoint on all PH2 cases without fine-tuning:

```bash
python evaluate.py --checkpoint PATH_TO_BEST_CHECKPOINT --dataset ph2 --data-path data/ph2 --output-dir results/ph2_external
```

The evaluator reports pooled pixel-level Dice, foreground IoU, sensitivity, specificity, and accuracy at a threshold of 0.5. It writes a JSON summary, a two-decimal percentage CSV, and per-sample results.

## Complexity reporting

`train.py` retains the existing THOP profiling call and logs FLOPs and parameter counts for internal use. Complexity values are not required for the current manuscript, but the profiling code is preserved as requested.

## Upstream code

EdgeAlign-Mamba is derived from the official
[VM-UNet](https://github.com/JCruan519/VM-UNet) implementation and retains its
VMamba-based visual state-space backbone. VM-UNet is distributed under the
Apache License 2.0. The underlying visual state-space implementation originates
from [VMamba](https://github.com/MzeroMiko/VMamba), which is distributed under
the MIT License. EdgeAlign-Mamba adds the MBAM and LMSG modules and adapts the
training and evaluation pipeline for the experiments described above.

The repository-level `LICENSE` contains the Apache License 2.0 for this project
and retains the VMamba MIT copyright and permission notice for the applicable
upstream portions. Please cite both upstream works when using this code:

```bibtex
@article{ruan2024vmunet,
  title   = {VM-UNet: Vision Mamba UNet for Medical Image Segmentation},
  author  = {Ruan, Jiacheng and Li, Jincheng and Xiang, Suncheng},
  journal = {arXiv preprint arXiv:2402.02491},
  year    = {2024}
}

@article{liu2024vmamba,
  title   = {VMamba: Visual State Space Model},
  author  = {Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and
             Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal = {arXiv preprint arXiv:2401.10166},
  year    = {2024}
}
```

Only the ISIC2018-PH2 training and evaluation pipeline is included in this repository.
