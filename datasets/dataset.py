from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _files_by_stem(folder: Path):
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            if path.stem in files:
                raise ValueError(f"Duplicate sample ID '{path.stem}' in {folder}")
            files[path.stem] = path
    return files


class NPY_datasets(Dataset):
    """Paired RGB image and binary-mask dataset used by ISIC2018 and PH2."""

    def __init__(
        self,
        path_Data,
        config,
        train=True,
        split=None,
        return_id=False,
        return_mask_path=False,
    ):
        super().__init__()
        self.split = split or ("train" if train else "val")
        self.return_id = return_id
        self.return_mask_path = return_mask_path
        root = Path(path_Data) / self.split
        image_dir = root / "images"
        mask_dir = root / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Expected paired folders at {image_dir} and {mask_dir}")

        images = _files_by_stem(image_dir)
        masks = _files_by_stem(mask_dir)
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise ValueError(
                "Image/mask ID mismatch. "
                f"Images without masks: {missing_masks[:5]}; "
                f"masks without images: {missing_images[:5]}"
            )

        self.sample_ids = sorted(images)
        if not self.sample_ids:
            raise ValueError(f"No paired samples found under {root}")
        self.data = [(images[sample_id], masks[sample_id]) for sample_id in self.sample_ids]
        self.transformer = config.train_transformer if self.split == "train" else config.test_transformer

    def __getitem__(self, index):
        image_path, mask_path = self.data[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.expand_dims(np.asarray(Image.open(mask_path).convert("L")), axis=2) / 255.0
        image, mask = self.transformer((image, mask))
        if self.return_id and self.return_mask_path:
            return image, mask, self.sample_ids[index], str(mask_path)
        if self.return_id:
            return image, mask, self.sample_ids[index]
        if self.return_mask_path:
            return image, mask, str(mask_path)
        return image, mask

    def __len__(self):
        return len(self.data)
