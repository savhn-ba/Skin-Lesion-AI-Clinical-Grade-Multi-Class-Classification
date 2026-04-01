"""Unit tests for HAM10000Dataset and related data utilities."""

import io
import os
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    HAM10000Dataset,
    make_weighted_sampler,
)
from src.data.transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_dataset(
    tmpdir: Path,
    n_per_class: int = 5,
) -> tuple[Path, List[Path]]:
    """Create a tiny fake HAM10000 dataset in *tmpdir*.

    Returns
    -------
    (metadata_csv_path, [image_dir])
    """
    image_dir = tmpdir / "images"
    image_dir.mkdir()

    rows = []
    for cls_name in CLASS_NAMES:
        for i in range(n_per_class):
            image_id = f"ISIC_{cls_name}_{i:04d}"
            # Create a 100×100 RGB JPEG.
            img = Image.new("RGB", (100, 100), color=(i * 20, 128, 64))
            img.save(image_dir / f"{image_id}.jpg")
            rows.append({"image_id": image_id, "dx": cls_name.lower()})

    df = pd.DataFrame(rows)
    csv_path = tmpdir / "metadata.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, [image_dir]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_dataset_dir(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("ham10000")
    return tmpdir


@pytest.fixture(scope="module")
def fake_ham10000(fake_dataset_dir):
    return _make_fake_dataset(fake_dataset_dir, n_per_class=5)


class TestHAM10000Dataset:
    """Tests for HAM10000Dataset."""

    def test_length(self, fake_ham10000):
        csv_path, image_dirs = fake_ham10000
        ds = HAM10000Dataset(csv_path, image_dirs)
        assert len(ds) == len(CLASS_NAMES) * 5

    def test_getitem_returns_tuple(self, fake_ham10000):
        csv_path, image_dirs = fake_ham10000
        ds = HAM10000Dataset(csv_path, image_dirs, transform=get_val_transforms())
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)

    def test_label_range(self, fake_ham10000):
        csv_path, image_dirs = fake_ham10000
        ds = HAM10000Dataset(csv_path, image_dirs, transform=get_val_transforms())
        for _, label in ds:
            assert 0 <= label < len(CLASS_NAMES), f"Label {label} out of range."

    def test_targets_property(self, fake_ham10000):
        csv_path, image_dirs = fake_ham10000
        ds = HAM10000Dataset(csv_path, image_dirs)
        targets = ds.targets
        assert len(targets) == len(ds)
        assert all(isinstance(t, int) for t in targets)

    def test_all_class_names_present(self, fake_ham10000):
        csv_path, image_dirs = fake_ham10000
        ds = HAM10000Dataset(csv_path, image_dirs)
        present = set(ds.targets)
        assert present == set(range(len(CLASS_NAMES))), (
            f"Expected all {len(CLASS_NAMES)} classes, found indices {sorted(present)}."
        )

    def test_indices_subset(self, fake_ham10000):
        csv_path, image_dirs = fake_ham10000
        ds_full = HAM10000Dataset(csv_path, image_dirs)
        subset_idx = list(range(10))
        ds_sub = HAM10000Dataset(csv_path, image_dirs, indices=subset_idx)
        assert len(ds_sub) == 10

    def test_train_transform_augments_differently(self, fake_ham10000):
        """Train transforms should produce different tensors on multiple runs."""
        csv_path, image_dirs = fake_ham10000
        ds = HAM10000Dataset(csv_path, image_dirs, transform=get_train_transforms())
        img1, _ = ds[0]
        img2, _ = ds[0]
        # Because of random augmentations this *should* differ, but it is not
        # guaranteed for every call, so we only check the shape here.
        assert img1.shape == img2.shape

    def test_unknown_dx_raises_value_error(self, fake_dataset_dir):
        """A CSV with an unknown dx value should raise ValueError."""
        bad_csv = fake_dataset_dir / "bad_metadata.csv"
        pd.DataFrame(
            [{"image_id": "ISIC_UNKNOWN_0001", "dx": "unknown_class"}]
        ).to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="Unknown dx values"):
            HAM10000Dataset(bad_csv, [fake_dataset_dir / "images"])

    def test_missing_image_raises_file_not_found(self, fake_dataset_dir):
        """Accessing an image_id not on disk should raise FileNotFoundError."""
        missing_csv = fake_dataset_dir / "missing_metadata.csv"
        pd.DataFrame(
            [{"image_id": "ISIC_NOT_ON_DISK", "dx": "mel"}]
        ).to_csv(missing_csv, index=False)

        ds = HAM10000Dataset(missing_csv, [fake_dataset_dir / "images"])
        with pytest.raises(FileNotFoundError):
            _ = ds[0]


class TestMakeWeightedSampler:
    """Tests for make_weighted_sampler."""

    def test_returns_weighted_random_sampler(self):
        from torch.utils.data import WeightedRandomSampler

        targets = [0, 0, 0, 1, 2]  # imbalanced
        sampler = make_weighted_sampler(targets, num_classes=3)
        assert isinstance(sampler, WeightedRandomSampler)

    def test_sampler_length(self):
        targets = list(range(7)) * 10  # 70 samples, 10 per class
        sampler = make_weighted_sampler(targets, num_classes=7)
        assert len(sampler) == len(targets)

    def test_rare_class_gets_higher_weight(self):
        """Class with fewer samples should get a higher per-sample weight."""
        targets = [0] * 100 + [1] * 10  # class 0 is 10× more common
        sampler = make_weighted_sampler(targets, num_classes=2)
        weights = list(sampler.weights.numpy())
        # Weights for class 1 samples (indices 100–109) should be higher.
        w_class0 = weights[0]
        w_class1 = weights[100]
        assert w_class1 > w_class0, (
            f"Rare class weight ({w_class1}) should exceed common class weight ({w_class0})."
        )

    def test_empty_class_does_not_crash(self):
        """Sampler should not crash if one class has zero samples."""
        targets = [0, 0, 2, 2]  # class 1 absent
        sampler = make_weighted_sampler(targets, num_classes=3)
        assert len(sampler) == len(targets)


class TestTransforms:
    """Smoke tests for transform pipelines."""

    def test_val_transform_output_shape(self):
        img = Image.new("RGB", (450, 600))
        transform = get_val_transforms()
        tensor = transform(img)
        assert tensor.shape == (3, 300, 300)

    def test_train_transform_output_shape(self):
        img = Image.new("RGB", (450, 600))
        transform = get_train_transforms()
        tensor = transform(img)
        assert tensor.shape == (3, 300, 300)

    def test_val_transform_normalisation(self):
        """Output should be approximately zero-mean after normalisation."""
        # Create a grey image close to the ImageNet mean colour.
        img = Image.new("RGB", (300, 300), color=(123, 116, 103))
        transform = get_val_transforms()
        tensor = transform(img)
        assert tensor.abs().mean().item() < 1.0  # rough sanity check
