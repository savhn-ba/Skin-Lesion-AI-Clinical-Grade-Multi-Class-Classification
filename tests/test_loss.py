"""Unit tests for the AsymmetricCostSensitiveLoss."""

import pytest
import torch

from src.losses.cost_sensitive import AsymmetricCostSensitiveLoss


def _identity_cost(n: int) -> torch.Tensor:
    """Cost matrix where all off-diagonal entries are 1 and diagonal is 0."""
    return (1.0 - torch.eye(n)).float()


def _mel_priority_cost(n: int, mel_cost: float = 5.0) -> torch.Tensor:
    """Off-diagonal costs where row 0 (MEL) has elevated penalty."""
    mat = _identity_cost(n)
    mat[0, :] = mel_cost
    mat[0, 0] = 0.0  # correct prediction is still free
    return mat


class TestAsymmetricCostSensitiveLoss:
    """Tests for AsymmetricCostSensitiveLoss."""

    def test_zero_loss_for_perfect_predictions(self):
        """Loss should be ≈ 0 when predictions are perfect (one-hot logits)."""
        n = 7
        cost = _identity_cost(n)
        criterion = AsymmetricCostSensitiveLoss(cost)

        # Very large logit for the correct class → softmax ≈ one-hot.
        logits = torch.full((4, n), -1e9)
        targets = torch.tensor([0, 1, 3, 6])
        for i, t in enumerate(targets):
            logits[i, t] = 1e9

        loss = criterion(logits, targets)
        assert loss.item() < 1e-3, f"Expected near-zero loss, got {loss.item()}"

    def test_loss_positive_for_wrong_predictions(self):
        """Loss should be positive when predictions are confidently wrong."""
        n = 7
        cost = _identity_cost(n)
        criterion = AsymmetricCostSensitiveLoss(cost)

        # True label is 0 (MEL) but model predicts class 1 with high confidence.
        logits = torch.full((1, n), -1e9)
        logits[0, 1] = 1e9
        targets = torch.tensor([0])

        loss = criterion(logits, targets)
        assert loss.item() > 0.5

    def test_mel_miss_more_costly_than_nv_miss(self):
        """A missed Melanoma should incur higher loss than a missed NV."""
        n = 7
        cost = _mel_priority_cost(n, mel_cost=5.0)

        # Model always predicts class 2 (BCC) with full confidence.
        logits = torch.full((2, n), -1e9)
        logits[:, 2] = 1e9

        # Sample 0: true = MEL (index 0), sample 1: true = NV (index 1)
        targets = torch.tensor([0, 1])

        # Use reduction="none" to get per-sample losses.
        criterion_none = AsymmetricCostSensitiveLoss(cost, reduction="none")
        losses = criterion_none(logits, targets)

        assert losses[0].item() > losses[1].item(), (
            f"MEL loss ({losses[0].item():.4f}) should exceed "
            f"NV loss ({losses[1].item():.4f})"
        )

    def test_reduction_mean_vs_sum(self):
        """'mean' and 'sum' reductions should be consistent."""
        n = 7
        cost = _identity_cost(n)
        logits = torch.randn(8, n)
        targets = torch.randint(0, n, (8,))

        loss_mean = AsymmetricCostSensitiveLoss(cost, reduction="mean")(logits, targets)
        loss_sum = AsymmetricCostSensitiveLoss(cost, reduction="sum")(logits, targets)
        loss_none = AsymmetricCostSensitiveLoss(cost, reduction="none")(logits, targets)

        assert torch.allclose(loss_mean, loss_none.mean(), atol=1e-5)
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-5)

    def test_from_list_constructor(self):
        """from_list should produce the same result as from_tensor."""
        n = 3
        matrix_list = [[0.0, 2.0, 2.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        matrix_tensor = torch.tensor(matrix_list, dtype=torch.float)

        crit_list = AsymmetricCostSensitiveLoss.from_list(matrix_list)
        crit_tensor = AsymmetricCostSensitiveLoss(matrix_tensor)

        logits = torch.randn(5, n)
        targets = torch.randint(0, n, (5,))

        assert torch.allclose(
            crit_list(logits, targets), crit_tensor(logits, targets), atol=1e-6
        )

    def test_invalid_reduction_raises(self):
        """Passing an unsupported reduction should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid reduction"):
            AsymmetricCostSensitiveLoss(_identity_cost(3), reduction="invalid")

    def test_device_movement(self):
        """Cost matrix buffer should follow the module to a new device."""
        cost = _identity_cost(7)
        criterion = AsymmetricCostSensitiveLoss(cost)
        # Moving to CPU should work without error.
        criterion = criterion.to("cpu")
        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        loss = criterion(logits, targets)
        assert loss.device.type == "cpu"

    def test_batch_size_one(self):
        """Loss should work with a single-sample batch."""
        n = 7
        cost = _identity_cost(n)
        criterion = AsymmetricCostSensitiveLoss(cost)
        logits = torch.randn(1, n)
        targets = torch.tensor([3])
        loss = criterion(logits, targets)
        assert loss.ndim == 0  # scalar

    def test_gradients_flow(self):
        """Backward pass should produce non-zero gradients on the logits."""
        n = 7
        cost = _identity_cost(n)
        criterion = AsymmetricCostSensitiveLoss(cost)
        logits = torch.randn(4, n, requires_grad=True)
        targets = torch.randint(0, n, (4,))
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0
