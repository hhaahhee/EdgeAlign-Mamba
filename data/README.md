# Data preparation

Required ISIC2018 raw layout:

```text
PATH_TO_RAW_ISIC2018/
  images/
  masks/
```

PH2 may retain its original directory layout; the preparation script discovers `IMD*` images and lesion masks recursively.

```bash
python data/prepare_isic2018.py --raw-dir PATH_TO_RAW_ISIC2018 --out-dir data/isic2018 --val-ratio 0.3 --seed 42
python data/prepare_ph2.py --raw-dir PATH_TO_RAW_PH2 --out-dir data/ph2 --expected-count 200
```

Both scripts validate one-to-one image-mask pairing and refuse to overwrite an existing prepared dataset unless `--overwrite` is supplied.
