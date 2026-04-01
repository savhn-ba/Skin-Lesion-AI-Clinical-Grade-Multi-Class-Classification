"""EfficientNet-B3 classifier for HAM10000 skin-lesion classification.

The backbone is loaded from the ``timm`` library with ImageNet pre-trained
weights.  The classifier head is replaced with a dropout + linear layer for
the target number of classes.

Two-phase fine-tuning is supported:

* Phase 1 — only the new head and the last ``n`` EfficientNet blocks are
  trainable; all earlier layers are frozen.
* Phase 2 — all layers are unfrozen for full fine-tuning at a lower learning
  rate.
"""

from __future__ import annotations

from typing import List, Optional

import timm
import torch
import torch.nn as nn


class SkinLesionClassifier(nn.Module):
    """EfficientNet-B3 fine-tuned for seven-class skin-lesion classification.

    Parameters
    ----------
    num_classes:
        Number of output classes (default: 7 for HAM10000).
    pretrained:
        If ``True`` (default), initialise the backbone with ImageNet weights.
    dropout:
        Dropout probability applied before the final linear layer.
    backbone:
        Name of the timm model to use as backbone.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone: str = "efficientnet_b3",
    ) -> None:
        super().__init__()
        # Load backbone without the default classifier head.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove the original head.
        )
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Image batch of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(N, num_classes)``.
        """
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (only the classifier head trains)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last ``n`` EfficientNet blocks in the backbone.

        Parameters
        ----------
        n:
            Number of blocks to unfreeze from the end.  Pass ``-1`` to unfreeze
            all backbone parameters.
        """
        if n == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        # EfficientNet blocks are stored in backbone.blocks (a Sequential of
        # MBConv block groups).
        blocks: Optional[nn.Sequential] = getattr(self.backbone, "blocks", None)
        if blocks is None:
            # Fallback: unfreeze the whole backbone.
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        block_list: List[nn.Module] = list(blocks)
        unfreeze_from = max(0, len(block_list) - n)
        for i, block in enumerate(block_list):
            for param in block.parameters():
                param.requires_grad = i >= unfreeze_from

        # Also unfreeze the final conv / batch-norm after the blocks.
        for attr in ("conv_head", "bn2", "act2"):
            layer = getattr(self.backbone, attr, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return all parameters that have ``requires_grad=True``."""
        return [p for p in self.parameters() if p.requires_grad]


def build_model(cfg: dict) -> SkinLesionClassifier:
    """Construct a :class:`SkinLesionClassifier` from a config dictionary.

    Parameters
    ----------
    cfg:
        Parsed YAML configuration dictionary (expects a ``"model"`` key).

    Returns
    -------
    SkinLesionClassifier
    """
    model_cfg = cfg["model"]
    return SkinLesionClassifier(
        num_classes=model_cfg.get("num_classes", 7),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
        backbone=model_cfg.get("backbone", "efficientnet_b3"),
    )
