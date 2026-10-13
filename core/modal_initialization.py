"""Point-displacement metric and mode selection for supplied pose hypotheses.

This NumPy implementation follows the pose-distance and kernel-mode formulas
in the manuscript. It does not include the graph-based hypothesis generator.
"""

from __future__ import annotations

import numpy as np


def source_moments(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the centroid and population second moment of source endpoints.

    Source coordinates are in physical units. The second moment uses the
    divisor N, not N - 1, to reproduce the mean squared point displacement.
    """
    source = np.asarray(source, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
        raise ValueError("source must be a nonempty (N, 3) array")
    if not np.isfinite(source).all():
        raise ValueError("source coordinates must be finite")
    mean = source.mean(axis=0)
    centered = source - mean
    return mean, centered.T @ centered / len(source)


def _hypotheses(
    rotations: np.ndarray, translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or len(rotations) == 0 or translations.shape != (len(rotations), 3)):
        raise ValueError("expected rotations (H, 3, 3) and translations (H, 3)")
    if not (np.isfinite(rotations).all() and np.isfinite(translations).all()):
        raise ValueError("hypotheses must be finite")
    if (not np.allclose(rotations.swapaxes(1, 2) @ rotations,
                        np.eye(3), atol=1e-7, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-7, rtol=0)):
        raise ValueError("hypotheses must contain proper rotations")
    return rotations, translations


def _distance_squared(
    rotation: np.ndarray,
    translation: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    mean: np.ndarray,
    moment: np.ndarray,
    support: float,
) -> np.ndarray:
    rotation_difference = rotation[None] - rotations
    centroid_displacement = rotation_difference @ mean + translation - translations
    center_term = np.einsum("ni,ni->n", centroid_displacement, centroid_displacement)
    extent_term = np.einsum(
        "nij,jk,nik->n", rotation_difference, moment, rotation_difference
    )
    return np.maximum(center_term + extent_term, 0.0) / (support * support)


def hypothesis_distances(
    reference_index: int,
    rotations: np.ndarray,
    translations: np.ndarray,
    source: np.ndarray,
    support: float,
) -> np.ndarray:
    """Return normalized RMS displacements from one hypothesis to all others.

    Source points and hypothesis translations use the same physical units.
    ``support`` is the branch support s_b in those units. Each output equals
    sqrt(mean(||T_reference(p_i) - T_j(p_i)||^2)) / support without explicitly
    transforming all source points for every pair of hypotheses.
    """
    rotations, translations = _hypotheses(rotations, translations)
    if (not isinstance(reference_index, (int, np.integer))
            or not 0 <= reference_index < len(rotations)):
        raise ValueError("reference_index must index an existing hypothesis")
    support = float(support)
    if not np.isfinite(support) or support <= 0.0:
        raise ValueError("support must be finite and positive")
    mean, moment = source_moments(source)
    return np.sqrt(_distance_squared(
        rotations[reference_index], translations[reference_index],
        rotations, translations, mean, moment, support,
    ))


def select_modal_hypothesis(
    rotations: np.ndarray,
    translations: np.ndarray,
    source: np.ndarray,
    support: float,
    bandwidth: float,
) -> tuple[int, np.ndarray]:
    """Select an existing hypothesis using a compact displacement kernel.

    Each score sums (1 - (d / bandwidth)^2)^2 over distances d < bandwidth,
    including the hypothesis itself. ``bandwidth`` is dimensionless. Ties
    return the first maximum in the supplied order; rotations are not averaged.

    Returns the selected index and all H support scores. Pairwise evaluation
    takes O(H^2) work while holding only O(H) temporary distances. Hypothesis
    generation, consensus refitting, and branch selection are not included.
    """
    rotations, translations = _hypotheses(rotations, translations)
    support, bandwidth = float(support), float(bandwidth)
    if (not np.isfinite([support, bandwidth]).all()
            or support <= 0.0 or bandwidth <= 0.0):
        raise ValueError("support and bandwidth must be finite and positive")
    mean, moment = source_moments(source)
    scores = np.empty(len(rotations))
    for index, (rotation, translation) in enumerate(zip(rotations, translations)):
        distance2 = _distance_squared(
            rotation, translation, rotations, translations, mean, moment, support
        )
        ratio2 = distance2 / (bandwidth * bandwidth)
        inside = ratio2 < 1.0
        scores[index] = np.square(1.0 - ratio2[inside]).sum()
    return int(np.argmax(scores)), scores
