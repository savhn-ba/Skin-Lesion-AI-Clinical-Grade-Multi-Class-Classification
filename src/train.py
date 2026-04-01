"""Training entry-point for the HAM10000 skin-lesion classifier.

Usage
-----
    python -m src.train --config configs/default.yaml

The training loop implements a two-phase fine-tuning strategy:

Phase 1
    Only the new classifier head and the last ``unfreeze_blocks`` EfficientNet
    blocks are trained at a relatively high learning rate.  The rest of the
    backbone remains frozen.

Phase 2
    All layers are unfrozen and trained at a lower learning rate.

In both phases a ``CosineAnnealingLR`` scheduler is used to smoothly decay the
learning rate to zero by the end of the phase.

After every epoch the model is evaluated on the validation set.  The checkpoint
with the lowest validation loss is saved to ``{output_dir}/best_model.pth``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

from src.data.dataset import build_dataloaders
from src.losses.cost_sensitive import AsymmetricCostSensitiveLoss
from src.models.classifier import build_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the HAM10000 skin-lesion classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _train_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _val_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one validation epoch and return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(cfg: dict) -> None:
    """Full training loop driven by the provided config dictionary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("Building DataLoaders …")
    train_loader, val_loader = build_dataloaders(cfg)

    # ------------------------------------------------------------------ model
    print("Building model …")
    model = build_model(cfg)
    model.to(device)

    # ------------------------------------------------------------------ loss
    cost_matrix_list = cfg["loss"]["cost_matrix"]
    criterion = AsymmetricCostSensitiveLoss.from_list(cost_matrix_list)
    criterion.to(device)

    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "best_model.pth"

    # ---------------------------------------------------------- training phases
    for phase_idx, phase_cfg in enumerate(cfg["training"]["phases"], start=1):
        n_epochs = phase_cfg["epochs"]
        lr = phase_cfg["lr"]
        unfreeze_blocks = phase_cfg.get("unfreeze_blocks", -1)

        print(f"\n=== Phase {phase_idx} — {n_epochs} epochs, lr={lr} ===")

        # Freeze / unfreeze backbone layers.
        model.freeze_backbone()
        model.unfreeze_last_n_blocks(unfreeze_blocks)

        # Only pass trainable parameters to the optimiser.
        optimizer = optim.Adam(
            model.get_trainable_params(),
            lr=lr,
            weight_decay=cfg["training"].get("weight_decay", 1e-4),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 100)

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = _val_epoch(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"    ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

    print(f"\nTraining complete.  Best model saved to '{best_ckpt_path}'.")


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
