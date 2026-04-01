"""Unit tests for the SkinLesionClassifier model."""

import pytest
import torch

from src.models.classifier import SkinLesionClassifier


@pytest.fixture(scope="module")
def small_model():
    """A small EfficientNet-B0 model (faster to instantiate in tests)."""
    return SkinLesionClassifier(
        num_classes=7,
        pretrained=False,  # avoid downloading weights in CI
        dropout=0.0,
        backbone="efficientnet_b0",
    )


class TestSkinLesionClassifier:
    """Tests for SkinLesionClassifier."""

    def test_output_shape(self, small_model):
        """Forward pass should return (N, num_classes) logits."""
        x = torch.randn(2, 3, 224, 224)
        logits = small_model(x)
        assert logits.shape == (2, 7), f"Expected (2, 7), got {logits.shape}"

    def test_output_is_logits_not_probs(self, small_model):
        """Outputs should NOT sum to 1 (they are raw logits, not probabilities)."""
        x = torch.randn(4, 3, 224, 224)
        logits = small_model(x)
        row_sums = logits.sum(dim=1)
        # Softmax probabilities sum to 1; logits generally do not.
        assert not torch.allclose(row_sums, torch.ones(4), atol=0.1), (
            "Output appears to be probabilities rather than logits."
        )

    def test_freeze_backbone(self, small_model):
        """After freezing, backbone parameters should have requires_grad=False."""
        small_model.freeze_backbone()
        backbone_params = list(small_model.backbone.parameters())
        assert all(not p.requires_grad for p in backbone_params), (
            "Some backbone parameters are still trainable after freeze_backbone()."
        )
        # Classifier head must still be trainable.
        head_params = list(small_model.classifier.parameters())
        assert all(p.requires_grad for p in head_params)

    def test_unfreeze_all(self, small_model):
        """Unfreezing with n=-1 should make all backbone params trainable."""
        small_model.freeze_backbone()
        small_model.unfreeze_last_n_blocks(-1)
        backbone_params = list(small_model.backbone.parameters())
        assert all(p.requires_grad for p in backbone_params)

    def test_unfreeze_last_n_blocks(self, small_model):
        """Unfreezing last 2 blocks should make at least some backbone params trainable."""
        small_model.freeze_backbone()
        small_model.unfreeze_last_n_blocks(2)
        trainable = [p for p in small_model.backbone.parameters() if p.requires_grad]
        assert len(trainable) > 0, (
            "No backbone parameters became trainable after unfreeze_last_n_blocks(2)."
        )

    def test_get_trainable_params_changes_with_freeze(self, small_model):
        """get_trainable_params should return fewer params when backbone is frozen."""
        small_model.unfreeze_last_n_blocks(-1)
        all_trainable = small_model.get_trainable_params()

        small_model.freeze_backbone()
        frozen_trainable = small_model.get_trainable_params()

        assert len(frozen_trainable) < len(all_trainable), (
            "Freezing backbone should reduce the number of trainable parameters."
        )

    def test_gradients_flow_through_head(self, small_model):
        """Gradients should flow back to the classifier head during training."""
        small_model.freeze_backbone()
        # Ensure classifier head is trainable.
        for p in small_model.classifier.parameters():
            p.requires_grad = True

        x = torch.randn(2, 3, 224, 224)
        logits = small_model(x)
        loss = logits.sum()
        loss.backward()

        for p in small_model.classifier.parameters():
            assert p.grad is not None, "Classifier head parameter has no gradient."

    def test_different_batch_sizes(self, small_model):
        """Model should handle different batch sizes without error."""
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 3, 224, 224)
            out = small_model(x)
            assert out.shape[0] == bs

    def test_build_model_from_config(self):
        """build_model should correctly parse a config dict."""
        from src.models.classifier import build_model

        cfg = {
            "model": {
                "num_classes": 7,
                "pretrained": False,
                "dropout": 0.2,
                "backbone": "efficientnet_b0",
            }
        }
        model = build_model(cfg)
        assert isinstance(model, SkinLesionClassifier)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 7)
