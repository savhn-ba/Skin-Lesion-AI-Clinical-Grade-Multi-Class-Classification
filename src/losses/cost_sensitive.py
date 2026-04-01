"""Asymmetric cost-sensitive loss for clinical-grade skin-lesion classification.

Standard cross-entropy treats all misclassifications as equally costly.  In a
clinical setting, missing a **Melanoma** (false negative) can delay treatment
and worsen patient outcomes, while a false positive merely requires a follow-up
appointment.  The ``AsymmetricCostSensitiveLoss`` encodes this asymmetry via a
cost matrix ``C`` where ``C[i, j]`` represents the cost incurred when a sample
with true class ``i`` is classified as class ``j``.

The loss for a single sample is computed as:

    L = sum_j ( C[y, j] * p_j )

where ``p`` is the predicted probability vector (after softmax) and ``y`` is
the true class.  This is equivalent to a weighted sum of the predicted
probabilities over all *incorrect* predictions, penalised by the corresponding
entry in the cost matrix.  A zero cost on the diagonal means the model is not
penalised for correct predictions.

When averaged over a mini-batch, the expected loss becomes:

    L_batch = (1 / N) * sum_i sum_j ( C[y_i, j] * p_{i,j} )

which the model minimises by assigning low probability to high-cost incorrect
classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricCostSensitiveLoss(nn.Module):
    """Differentiable asymmetric cost-sensitive classification loss.

    Parameters
    ----------
    cost_matrix:
        A ``(num_classes, num_classes)`` tensor where entry ``[i, j]`` is the
        cost of predicting class ``j`` when the true class is ``i``.  The
        diagonal should be zero (no cost for a correct prediction).  Larger
        values in row ``i`` increase the penalty for misclassifying samples of
        class ``i``.
    reduction:
        ``"mean"`` (default) averages the loss over the batch;
        ``"sum"`` returns the total loss;
        ``"none"`` returns per-sample losses.
    """

    def __init__(
        self,
        cost_matrix: torch.Tensor,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction '{reduction}'.  Choose 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction
        # Register as a buffer so the matrix is moved to the correct device
        # automatically when .to(device) is called.
        self.register_buffer("cost_matrix", cost_matrix.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the asymmetric cost-sensitive loss.

        Parameters
        ----------
        logits:
            Raw (un-normalised) model outputs of shape ``(N, C)``.
        targets:
            Integer class indices of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value (or per-sample tensor if ``reduction="none"``).
        """
        probs = F.softmax(logits, dim=1)  # (N, C)

        # Gather the cost row for each sample's true class: shape (N, C).
        costs = self.cost_matrix[targets]  # (N, C)

        # Per-sample loss: expected cost under the predicted distribution.
        sample_losses = (costs * probs).sum(dim=1)  # (N,)

        if self.reduction == "mean":
            return sample_losses.mean()
        if self.reduction == "sum":
            return sample_losses.sum()
        return sample_losses

    @classmethod
    def from_list(
        cls,
        cost_matrix_list: list,
        reduction: str = "mean",
    ) -> "AsymmetricCostSensitiveLoss":
        """Construct from a nested Python list (e.g. loaded from YAML).

        Parameters
        ----------
        cost_matrix_list:
            A list of lists representing the cost matrix rows.
        reduction:
            See :class:`AsymmetricCostSensitiveLoss`.
        """
        matrix = torch.tensor(cost_matrix_list, dtype=torch.float)
        return cls(cost_matrix=matrix, reduction=reduction)
