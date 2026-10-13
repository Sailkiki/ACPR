"""Geometric covariance models for neighborhoods supplied by the caller."""

from __future__ import annotations

import numpy as np


def covariances_from_neighborhoods(
    neighborhoods: np.ndarray,
    *,
    surface_variation_max: float,
    covariance_epsilon: float,
    covariance_magnitude: float,
    rank_relative: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct anisotropic or isotropic covariance models using local PCA.

    ``neighborhoods`` has shape (N, K, 3), with K >= 3. Each neighborhood
    contains the neighbors of one correspondence endpoint, excluding that
    endpoint itself. Neighbor lookup and neighborhood-size selection belong
    to the unreleased data preparation code.

    A reliable plane has two resolved tangential directions and a smallest
    eigenvalue fraction no greater than ``surface_variation_max``. Its
    covariance is kappa * (I - (1 - epsilon) * n n.T). Other neighborhoods
    retain kappa * I. Covariances are dimensionless; their physical values
    are obtained by multiplying by the squared support scale.

    Returns covariance matrices (N, 3, 3), a planar mask (N,), and surface
    variations (N,). Rank-deficient neighborhoods have NaN variation.
    All model and reliability parameters are explicit caller inputs.
    """
    neighborhoods = np.asarray(neighborhoods, dtype=float)
    if (neighborhoods.ndim != 3 or neighborhoods.shape[2] != 3
            or neighborhoods.shape[0] == 0 or neighborhoods.shape[1] < 3):
        raise ValueError("neighborhoods must have shape (N, K, 3), N > 0, K >= 3")
    if not np.isfinite(neighborhoods).all():
        raise ValueError("neighborhood coordinates must be finite")
    surface_variation_max = float(surface_variation_max)
    covariance_epsilon = float(covariance_epsilon)
    covariance_magnitude = float(covariance_magnitude)
    rank_relative = float(rank_relative)
    parameters = (surface_variation_max, covariance_epsilon,
                  covariance_magnitude, rank_relative)
    if not np.isfinite(parameters).all():
        raise ValueError("covariance parameters must be finite")
    if not 0.0 <= surface_variation_max < 1.0 / 3.0:
        raise ValueError("surface_variation_max must be in [0, 1/3)")
    if not 0.0 < covariance_epsilon < 1.0:
        raise ValueError("covariance_epsilon must be in (0, 1)")
    if covariance_magnitude <= 0.0 or not 0.0 < rank_relative < 1.0:
        raise ValueError("positive magnitude and rank_relative in (0, 1) required")

    count, neighbor_count, _ = neighborhoods.shape
    identity = np.eye(3)
    covariances = np.tile(covariance_magnitude * identity, (count, 1, 1))
    planar = np.zeros(count, dtype=bool)
    variations = np.full(count, np.nan)

    for index, neighbors in enumerate(neighborhoods):
        centered = neighbors - neighbors.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(
            centered.T @ centered / neighbor_count
        )
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if (eigenvalues[-1] <= 0.0
                or eigenvalues[1] <= rank_relative * eigenvalues[-1]):
            continue
        variations[index] = eigenvalues[0] / eigenvalues.sum()
        if variations[index] <= surface_variation_max:
            normal = eigenvectors[:, 0]
            covariances[index] = covariance_magnitude * (
                identity - (1.0 - covariance_epsilon) * np.outer(normal, normal)
            )
            planar[index] = True

    return covariances, planar, variations
