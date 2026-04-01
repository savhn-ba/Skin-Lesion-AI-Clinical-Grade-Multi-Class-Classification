"""HAM10000 Dataset and DataLoader helpers.

The HAM10000 dataset contains 10,015 dermoscopy images labelled with one of
seven diagnostic categories.  The dataset is heavily imbalanced: the majority
class (NV — melanocytic nevi) represents ~67 % of all samples while the
rarest classes (DF, VASC) account for only ~1 % each.

This module provides:

* ``HAM10000Dataset`` — a ``torch.utils.data.Dataset`` that reads images from
  one or more image directories and maps label strings to integer indices.
* ``make_weighted_sampler`` — constructs a ``WeightedRandomSampler`` that
  draws each class with equal probability per epoch, counteracting the natural
  class imbalance.
* ``build_dataloaders`` — a convenience function that returns train/val
  ``DataLoader`` objects from a config dictionary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.transforms import get_train_transforms, get_val_transforms

# Canonical class ordering — index 0 is MEL (Melanoma) so that the cost matrix
# defined in configs/default.yaml is aligned with class index 0.
CLASS_NAMES: List[str] = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Mapping from the lowercase dx labels used in the HAM10000 CSV to our class names.
_DX_TO_CLASS: Dict[str, str] = {
    "mel": "MEL",
    "nv": "NV",
    "bcc": "BCC",
    "akiec": "AKIEC",
    "bkl": "BKL",
    "df": "DF",
    "vasc": "VASC",
}

# Map class name → integer index.
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


class HAM10000Dataset(Dataset):
    """PyTorch Dataset for the HAM10000 dermoscopy benchmark.

    Parameters
    ----------
    metadata_csv:
        Path to the HAM10000 metadata CSV file.  Must contain at least the
        columns ``image_id`` and ``dx``.
    image_dirs:
        One or more directories that contain the ``.jpg`` image files.
    transform:
        Optional callable applied to each PIL image before returning it.
    indices:
        Optional list of integer row indices to use (for train/val splitting).
        If ``None`` the entire CSV is used.
    """

    def __init__(
        self,
        metadata_csv: str | os.PathLike,
        image_dirs: List[str | os.PathLike],
        transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.transform = transform

        df = pd.read_csv(metadata_csv)
        # Normalise dx to uppercase class names.
        df["label"] = df["dx"].str.strip().str.lower().map(_DX_TO_CLASS)
        if df["label"].isna().any():
            unknown = df.loc[df["label"].isna(), "dx"].unique().tolist()
            raise ValueError(f"Unknown dx values in metadata: {unknown}")

        df["class_idx"] = df["label"].map(CLASS_TO_IDX)

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.df = df.reset_index(drop=True)

        # Build a lookup from image_id → full path.
        self._image_paths: Dict[str, Path] = {}
        for image_dir in image_dirs:
            for p in Path(image_dir).glob("*.jpg"):
                self._image_paths[p.stem] = p

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id: str = row["image_id"]
        label: int = int(row["class_idx"])

        path = self._image_paths.get(image_id)
        if path is None:
            raise FileNotFoundError(
                f"Image '{image_id}.jpg' not found in any of the provided image "
                "directories.  Please make sure you have downloaded both parts of "
                "the HAM10000 dataset."
            )

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def targets(self) -> List[int]:
        """Integer class index for every sample (used by WeightedRandomSampler)."""
        return self.df["class_idx"].tolist()


def make_weighted_sampler(targets: List[int], num_classes: int) -> WeightedRandomSampler:
    """Return a sampler that gives each class equal sampling probability.

    Parameters
    ----------
    targets:
        List of integer class labels for every sample in the dataset.
    num_classes:
        Total number of classes.

    Returns
    -------
    WeightedRandomSampler
        Draws ``len(targets)`` samples per epoch with replacement so that
        each class is represented roughly equally.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float)
    for t in targets:
        class_counts[t] += 1

    # Avoid division by zero for classes with no samples.
    class_counts = class_counts.clamp(min=1)
    class_weights = 1.0 / class_counts

    sample_weights = torch.tensor([class_weights[t] for t in targets], dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and validation DataLoaders from a config dictionary.

    The config dictionary should mirror the structure of ``configs/default.yaml``
    (i.e. ``cfg["data"]``, ``cfg["training"]``, etc.).

    Parameters
    ----------
    cfg:
        Parsed YAML configuration dictionary.

    Returns
    -------
    train_loader, val_loader
        Both loaders yield ``(image_tensor, label_int)`` tuples.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    metadata_csv = data_cfg["metadata_csv"]
    image_dirs = data_cfg["image_dirs"]
    val_split = data_cfg.get("val_split", 0.2)
    split_seed = data_cfg.get("split_seed", 42)
    num_workers = data_cfg.get("num_workers", 4)
    batch_size = train_cfg.get("batch_size", 32)
    use_weighted = train_cfg.get("weighted_sampling", True)
    num_classes = cfg["model"].get("num_classes", 7)

    # Load full metadata to compute split indices.
    full_df = pd.read_csv(metadata_csv)
    all_indices = list(range(len(full_df)))
    labels = (
        full_df["dx"].str.strip().str.lower().map(_DX_TO_CLASS).map(CLASS_TO_IDX).tolist()
    )

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_split,
        stratify=labels,
        random_state=split_seed,
    )

    train_ds = HAM10000Dataset(
        metadata_csv=metadata_csv,
        image_dirs=image_dirs,
        transform=get_train_transforms(),
        indices=train_idx,
    )
    val_ds = HAM10000Dataset(
        metadata_csv=metadata_csv,
        image_dirs=image_dirs,
        transform=get_val_transforms(),
        indices=val_idx,
    )

    sampler = None
    shuffle = True
    if use_weighted:
        sampler = make_weighted_sampler(train_ds.targets, num_classes)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
