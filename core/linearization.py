"""Evaluate the ACPR anisotropic objective and its local pose derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anisotropic import anisotropic_terms, covariance_rotation_gradient
from .residual_calibration import huber_loss_weights, tukey_loss_weights
from .rigid_motion import residual_jacobians


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """One evaluation on a fixed correspondence set at a supplied pose.

    ``metric`` is the weighted residual-Jacobian approximation J.T Sigma^-1 J.
    It is positive semidefinite and may be singular; it is not the exact
    Hessian of the objective. No damping or support acceptance is applied.
    """

    objective: float
    gradient: np.ndarray
    metric: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    covariance_gradient: np.ndarray
    positive_weights: int
    weight_sum: float
    effective_samples: float


def evaluate_objective(
    source: np.ndarray,
    target: np.ndarray,
    source_covariances: np.ndarray,
    target_covariances: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    pivot: np.ndarray,
    *,
    loss: str,
    scale: float,
) -> ObjectiveEvaluation:
    """Return the radial loss, exact local gradient, and approximate metric.

    ``loss`` is 'huber' or 'tukey'. ``scale`` is respectively the Huber
    transition or the calibrated Tukey cutoff. There is no log-determinant
    term. Point coordinates and covariances use the normalized convention
    described in the README. The pivot and correspondences stay fixed
    within a stage; a Tukey cutoff must also stay fixed.

    Both residual motion and source-covariance rotation contribute to the
    gradient. Loss and derivatives are divided by the number of supplied
    correspondences, including any with zero radial weight.

    This function performs no correspondence selection, scale recalibration,
    optimization loop, damping, line search, or update acceptance.
    """
    if loss not in ("huber", "tukey"):
        raise ValueError("loss must be 'huber' or 'tukey'")
    transformed, rotated_covariances, precision, z, distances = anisotropic_terms(
        source, target, source_covariances, target_covariances, rotation, translation
    )
    if loss == "huber":
        losses, weights = huber_loss_weights(distances, scale)
    else:
        losses, weights = tukey_loss_weights(distances, scale)
    jacobians = residual_jacobians(transformed, pivot)

    residual_gradient = np.einsum("nki,nk->ni", jacobians, z)
    covariance_gradient = np.zeros_like(residual_gradient)
    covariance_gradient[:, :3] = covariance_rotation_gradient(rotated_covariances, z)
    gradient = np.mean(
        weights[:, None] * (residual_gradient + covariance_gradient), axis=0
    )
    metric = np.einsum(
        "n,nki,nkl,nlj->ij", weights, jacobians, precision, jacobians
    ) / len(distances)

    weight_sum = float(weights.sum())
    squared_weight_sum = float(np.dot(weights, weights))
    effective_samples = (
        weight_sum * weight_sum / squared_weight_sum if squared_weight_sum > 0.0 else 0.0
    )
    return ObjectiveEvaluation(
        objective=float(losses.mean()),
        gradient=gradient,
        metric=metric,
        distances=distances,
        weights=weights,
        covariance_gradient=np.mean(weights[:, None] * covariance_gradient, axis=0),
        positive_weights=int(np.count_nonzero(weights > 0.0)),
        weight_sum=weight_sum,
        effective_samples=effective_samples,
    )
