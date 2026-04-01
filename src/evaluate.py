"""Evaluation entry-point for the HAM10000 skin-lesion classifier.

Usage
-----
    python -m src.evaluate \\
        --config configs/default.yaml \\
        --checkpoint outputs/best_model.pth

The script loads the validation split (using the same seed as training so
there is no leakage), runs inference, and reports:

* Overall accuracy
* Per-class sensitivity (recall) and specificity
* Per-class F1-score
* Macro-averaged ROC-AUC
* Full confusion matrix (saved as a PNG heatmap)
* Per-class classification report (printed to stdout and saved as a text file)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    HAM10000Dataset,
    _DX_TO_CLASS,
    make_weighted_sampler,
)
from src.data.transforms import get_val_transforms
from src.models.classifier import build_model
from sklearn.model_selection import train_test_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the HAM10000 skin-lesion classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint.  Overrides config.",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _build_val_loader(cfg: dict) -> DataLoader:
    """Reconstruct the validation DataLoader using the same split as training."""
    data_cfg = cfg["data"]
    metadata_csv = data_cfg["metadata_csv"]
    image_dirs = data_cfg["image_dirs"]
    val_split = data_cfg.get("val_split", 0.2)
    split_seed = data_cfg.get("split_seed", 42)
    num_workers = data_cfg.get("num_workers", 4)
    batch_size = cfg["training"].get("batch_size", 32)

    full_df = pd.read_csv(metadata_csv)
    all_indices = list(range(len(full_df)))
    labels = (
        full_df["dx"].str.strip().str.lower().map(_DX_TO_CLASS).map(CLASS_TO_IDX).tolist()
    )

    _, val_idx = train_test_split(
        all_indices,
        test_size=val_split,
        stratify=labels,
        random_state=split_seed,
    )

    val_ds = HAM10000Dataset(
        metadata_csv=metadata_csv,
        image_dirs=image_dirs,
        transform=get_val_transforms(),
        indices=val_idx,
    )

    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (true_labels, predicted_labels, probability_matrix)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Inference"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

    return (
        np.array(all_labels, dtype=int),
        np.array(all_preds, dtype=int),
        np.vstack(all_probs),
    )


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> pd.DataFrame:
    """Return a DataFrame with sensitivity and specificity per class."""
    rows = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rows.append(
            {
                "class": CLASS_NAMES[c],
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    """Save a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to '{save_path}'.")


def evaluate(cfg: dict, checkpoint_path: str) -> Dict[str, float]:
    """Run full evaluation and return a metrics dictionary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model
    model = build_model(cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # ------------------------------------------------------------------ data
    val_loader = _build_val_loader(cfg)

    # ---------------------------------------------------------------- inference
    y_true, y_pred, y_prob = run_inference(model, val_loader, device)

    # --------------------------------------------------------------- metrics
    acc = float((y_true == y_pred).mean())
    print(f"\nOverall accuracy: {acc:.4f}")

    per_class_df = compute_per_class_metrics(y_true, y_pred, len(CLASS_NAMES))
    print("\nPer-class sensitivity and specificity:")
    print(per_class_df.to_string(index=False))

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    print("\nClassification report:")
    print(report)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        print(f"Macro-averaged ROC-AUC: {auc:.4f}")
    except ValueError as exc:
        auc = float("nan")
        print(f"ROC-AUC could not be computed: {exc}")

    # ------------------------------------------------------------ save artefacts
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w") as fh:
        fh.write(report)

    metrics_path = results_dir / "per_class_metrics.csv"
    per_class_df.to_csv(metrics_path, index=False)

    plot_confusion_matrix(y_true, y_pred, results_dir / "confusion_matrix.png")

    return {
        "accuracy": acc,
        "macro_auc": auc,
        "mel_sensitivity": float(
            per_class_df.loc[per_class_df["class"] == "MEL", "sensitivity"].values[0]
        ),
    }


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    ckpt = args.checkpoint or cfg["evaluation"]["checkpoint"]
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: '{ckpt}'.  "
            "Run 'python -m src.train' first to generate a checkpoint."
        )

    evaluate(cfg, ckpt)


if __name__ == "__main__":
    main()
