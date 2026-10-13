"""Residual-scale calibration and radial robust losses used by ACPR."""

from __future__ import annotations

import numpy as np


def _residual_vector(distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances must be a nonempty one-dimensional array")
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise ValueError("distances must be finite and nonnegative")
    return distances


def calibrate_cutoff(
    initial_distances: np.ndarray,
    chi_median: float,
    tukey_multiplier: float,
) -> tuple[float, float]:
    """Return sigma_0 and cutoff b from initial Mahalanobis distances.

    chi_median is the median of the chi distribution for the residual
    dimension. No experiment-specific multiplier is supplied here.
    Calibration is performed once at the initial anisotropic estimate;
    the resulting cutoff is held fixed during the refinement stage.

    The complete method retains the incoming pose when the residual scale
    is invalid. This standalone routine reports that condition as an error.
    """
    initial_distances = _residual_vector(initial_distances)
    chi_median = float(chi_median)
    tukey_multiplier = float(tukey_multiplier)
    if not np.isfinite(chi_median) or chi_median <= 0.0:
        raise ValueError("chi_median must be finite and positive")
    if not np.isfinite(tukey_multiplier) or tukey_multiplier <= 0.0:
        raise ValueError("tukey_multiplier must be finite and positive")

    sigma = float(np.median(initial_distances) / chi_median)
    if not np.isfinite(sigma) or sigma <= np.sqrt(np.finfo(float).eps):
        raise ValueError("the initial residual scale is invalid or effectively zero")
    cutoff = tukey_multiplier * sigma
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("the calibrated cutoff must be finite and positive")
    return sigma, cutoff


def tukey_loss_weights(
    distances: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-correspondence Tukey losses and radial weights.

    The weight is rho'(u)/u, with its continuous limit at zero. It is zero
    at and beyond the fixed cutoff. Losses are averaged over the full fixed
    correspondence set by evaluate_objective, including zero-weight pairs.
    """
    distances = _residual_vector(distances)
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")

    ratio2 = np.square(distances / cutoff)
    inside = ratio2 < 1.0
    weights = np.zeros_like(distances)
    weights[inside] = np.square(1.0 - ratio2[inside])
    losses = np.full_like(distances, cutoff * cutoff / 6.0)
    losses[inside] = (cutoff * cutoff / 6.0) * (
        1.0 - np.power(1.0 - ratio2[inside], 3)
    )
    return losses, weights


def huber_loss_weights(
    distances: np.ndarray,
    transition: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial Huber losses and weights for the initial GLS objective.

    The loss is u^2 / 2 below the transition delta and delta * (u - delta/2)
    above it. The corresponding weight is one or delta/u. The continuous
    weight at zero is one; no observations are rejected by this loss.
    """
    distances = _residual_vector(distances)
    transition = float(transition)
    if not np.isfinite(transition) or transition <= 0.0:
        raise ValueError("transition must be finite and positive")
    weights = np.ones_like(distances)
    outside = distances > transition
    weights[outside] = transition / distances[outside]
    losses = np.empty_like(distances)
    losses[~outside] = 0.5 * np.square(distances[~outside])
    losses[outside] = transition * (distances[outside] - 0.5 * transition)
    return losses, weights
