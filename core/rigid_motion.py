"""Rigid-motion differentials used by the ACPR anisotropic objective."""

from __future__ import annotations

import math

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    """Return [v]_x such that [v]_x @ u == cross(v, u)."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("vector must contain three finite entries")
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def residual_jacobians(transformed: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Return [-[x_i - pivot]_x | I] for centered left pose increments.

    The output has shape (N, 3, 6). Rotation occupies the first three
    columns, followed by translation. The pivot is fixed within a stage.
    """
    transformed = np.asarray(transformed, dtype=float)
    pivot = np.asarray(pivot, dtype=float)
    if (transformed.ndim != 2 or transformed.shape[1] != 3
            or len(transformed) == 0 or pivot.shape != (3,)):
        raise ValueError("expected transformed points (N, 3) and a pivot (3,)")
    if not (np.isfinite(transformed).all() and np.isfinite(pivot).all()):
        raise ValueError("transformed points and pivot must be finite")

    jacobians = np.zeros((len(transformed), 3, 6))
    jacobians[:, :, :3] = np.stack([-skew(v) for v in transformed - pivot])
    jacobians[:, :, 3:] = np.eye(3)
    return jacobians


def left_increment(pose: np.ndarray, pivot: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply one supplied SE(3) increment around a fixed pivot.

    ``delta = [omega, velocity]`` uses radians for omega and the same units
    as the pose translation for velocity. Translation is mapped through
    the SO(3) left Jacobian, including its continuous limit at zero.

    This is a motion primitive. It neither computes an optimization step
    nor decides whether a proposed update should be accepted.
    """
    pose = np.asarray(pose, dtype=float)
    pivot = np.asarray(pivot, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if pose.shape != (4, 4) or pivot.shape != (3,) or delta.shape != (6,):
        raise ValueError("expected pose (4, 4), pivot (3,), and delta (6,)")
    if not all(np.isfinite(a).all() for a in (pose, pivot, delta)):
        raise ValueError("pose, pivot, and delta must be finite")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0):
        raise ValueError("pose must be a homogeneous rigid transformation")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(rotation) - 1.0) > 1e-7):
        raise ValueError("pose must contain a proper rotation")

    omega, velocity = delta[:3], delta[3:]
    theta = float(np.linalg.norm(omega))
    generator = skew(omega)
    if theta < 1e-6:
        theta2 = theta * theta
        a = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        c = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / (theta * theta)
        c = (theta - math.sin(theta)) / (theta ** 3)
    generator2 = generator @ generator
    increment_rotation = np.eye(3) + a * generator + b * generator2
    left_jacobian = np.eye(3) + b * generator + c * generator2

    result = np.eye(4)
    result[:3, :3] = increment_rotation @ rotation
    result[:3, 3] = (
        increment_rotation @ (pose[:3, 3] - pivot)
        + pivot + left_jacobian @ velocity
    )
    return result
