# EdgeAlign-Mamba data preparation

Use `prepare_isic2018.py` to create the seeded ISIC2018 training and validation folders, and use `prepare_ph2.py` to create the PH2 `external_test` folder. Both scripts pair images and masks by sample ID, convert images to RGB PNG, convert masks to binary single-channel PNG, and write a manifest for auditability.

```bash
python data/prepare_isic2018.py --raw-dir PATH_TO_RAW_ISIC2018 --out-dir data/isic2018 --val-ratio 0.3 --seed 42
python data/prepare_ph2.py --raw-dir PATH_TO_RAW_PH2 --out-dir data/ph2 --expected-count 200
```

The dataset loader validates exact stem-based image-mask correspondence before training or evaluation. PH2 must remain external-only and must not be included in checkpoint selection.
