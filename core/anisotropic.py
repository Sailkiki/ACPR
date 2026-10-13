"""Selected anisotropic computations from ACPR; no registration optimizer."""

from __future__ import annotations

import numpy as np

from .rigid_motion import skew


_ROTATION_GENERATORS = np.stack([skew(axis) for axis in np.eye(3)])


def anisotropic_terms(
    source: np.ndarray,
    target: np.ndarray,
    source_covariances: np.ndarray,
    target_covariances: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a supplied pose on fixed, already selected correspondences.

    Points have shape (N, 3); covariance arrays have shape (N, 3, 3).
    The rotation has shape (3, 3), and the translation has shape (3,).
    Covariances must be symmetric positive definite. Coordinates, translation,
    and covariances follow the support-normalized convention in the README.

    Returns transformed points x, rotated source covariances A, combined
    precisions Sigma^-1, precision-weighted residuals z, and distances u.
    This function neither selects correspondences nor estimates a pose.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_covariances = np.asarray(source_covariances, dtype=float)
    target_covariances = np.asarray(target_covariances, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)

    if source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
        raise ValueError("source must be a nonempty (N, 3) array")
    if target.shape != source.shape:
        raise ValueError("target must contain one point per source point")
    covariance_shape = (len(source), 3, 3)
    if (source_covariances.shape != covariance_shape
            or target_covariances.shape != covariance_shape):
        raise ValueError("covariance arrays must have shape (N, 3, 3)")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rotation and translation must have shapes (3, 3) and (3,)")
    arrays = (source, target, source_covariances, target_covariances,
              rotation, translation)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("all inputs must be finite")

    transformed = source @ rotation.T + translation
    residual = transformed - target
    rotated_source = rotation[None] @ source_covariances @ rotation.T[None]
    precision = np.linalg.inv(rotated_source + target_covariances)
    precision_residual = np.einsum("nij,nj->ni", precision, residual)
    squared_distance = np.einsum("ni,ni->n", residual, precision_residual)
    distances = np.sqrt(np.maximum(squared_distance, 0.0))
    return transformed, rotated_source, precision, precision_residual, distances


def covariance_rotation_gradient(
    rotated_source: np.ndarray,
    precision_residual: np.ndarray,
) -> np.ndarray:
    """Return only the covariance part of d(0.5*u_i**2)/d(omega).

    Inputs are A and z from anisotropic_terms. The (N, 3) output contains
    per-correspondence derivatives for left rotation increments. Residual
    Jacobian terms and robust weights are deliberately outside this routine.
    """
    rotated_source = np.asarray(rotated_source, dtype=float)
    precision_residual = np.asarray(precision_residual, dtype=float)
    if (precision_residual.ndim != 2 or precision_residual.shape[1] != 3
            or len(precision_residual) == 0
            or rotated_source.shape != (len(precision_residual), 3, 3)):
        raise ValueError("expected A with shape (N, 3, 3) and z with shape (N, 3)")
    if not (np.isfinite(rotated_source).all()
            and np.isfinite(precision_residual).all()):
        raise ValueError("A and z must be finite")

    contribution = np.empty_like(precision_residual)
    for axis, generator in enumerate(_ROTATION_GENERATORS):
        derivative = generator[None] @ rotated_source - rotated_source @ generator[None]
        contribution[:, axis] = -0.5 * np.einsum(
            "ni,nij,nj->n", precision_residual, derivative, precision_residual
        )
    return contribution
