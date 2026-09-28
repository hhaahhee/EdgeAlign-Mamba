# EdgeAlign-Mamba

Official implementation of EdgeAlign-Mamba for binary skin-lesion segmentation. The model is derived from VM-UNet and adds the Multi-scale Boundary-aware Attention Module (MBAM) and Lightweight Mamba Skip Gate (LMSG).

## Requirements

- Python 3.10
- PyTorch 2.1.2
- CUDA 11.8
- NVIDIA GPU with CUDA support

```bash
pip install -r requirements.txt
```

The reported experiments used an NVIDIA GeForce RTX 4080 SUPER and 256×256 RGB inputs.

## Pre-trained weights

Download `vmamba_small_e238_ema.pth` using the links provided in the [official VM-UNet README](https://github.com/JCruan519/VM-UNet/blob/main/README.md#2-prepare-the-pre_trained-weights), then place it at:

```text
pre_trained_weights/vmamba_small_e238_ema.pth
```

## Data

Prepare ISIC2018 with the reported 70:30 split (seed 42):

```bash
python data/prepare_isic2018.py --raw-dir PATH_TO_RAW_ISIC2018 --out-dir data/isic2018 --val-ratio 0.3 --seed 42
```

Prepare all 200 PH2 cases for external evaluation:

```bash
python data/prepare_ph2.py --raw-dir PATH_TO_RAW_PH2 --out-dir data/ph2 --expected-count 200
```

See [data/README.md](data/README.md) for the required raw-data layout.

## Training

Default settings are defined in `configs/config_setting.py` (batch size 32, 100 epochs, AdamW, and learning rate 1e-3). The random seeds used for the reported repeated experiments are specified in the paper.

```bash
python train.py
```

The checkpoint with the lowest ISIC2018 validation loss is selected.

## Evaluation

Evaluate the selected checkpoint on PH2 without fine-tuning:

```bash
python evaluate.py --checkpoint PATH_TO_BEST_CHECKPOINT --dataset ph2 --data-path data/ph2 --output-dir results/ph2_external
```

The evaluator reports IoU, Dice, accuracy, specificity, sensitivity, HD95, and BF1, and writes summary and per-sample files.
Each RGB image is independently min-max rescaled to [0, 255]. Images use bilinear interpolation, masks use nearest-neighbor interpolation followed by binary thresholding, and all region and boundary metrics share the same 256×256 binary mask pair.

## License and upstream code

This repository is based on [VM-UNet](https://github.com/JCruan519/VM-UNet) and its [VMamba](https://github.com/MzeroMiko/VMamba) backbone. The applicable Apache-2.0 and VMamba MIT notices are retained in [LICENSE](LICENSE). Please cite the original projects when using this code.
