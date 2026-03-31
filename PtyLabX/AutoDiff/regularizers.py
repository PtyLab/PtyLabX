"""Regularizers for AD-based ptychographic reconstruction.

Each regularizer takes ``(state, static)`` and returns a scalar penalty.
They are composed with the data loss via ``build_loss()`` in ``__init__.py``.
"""

from __future__ import annotations

import jax.numpy as jnp

from PtyLabX.AutoDiff.state import PtychographyState, StaticConfig


def object_tv(state: PtychographyState, static: StaticConfig, weight: float = 1e-4, aleph: float = 1e-3) -> jnp.ndarray:
    """Differentiable total variation penalty on the object.

    Uses forward finite differences (shift-and-subtract) which are fully
    differentiable, unlike the existing ``Regularizers.grad_TV`` which uses
    central differences designed for direct application.

    Parameters
    ----------
    state : PtychographyState
        Current reconstruction state.
    static : StaticConfig
        Non-differentiable metadata (unused here, kept for API consistency).
    weight : float
        Regularization strength.
    aleph : float
        Small constant to avoid ``sqrt(0)`` (smooth approximation of L1).

    Returns
    -------
    jnp.ndarray
        Scalar TV penalty.
    """
    obj = state.object
    grad_x = obj[..., :, 1:] - obj[..., :, :-1]
    grad_y = obj[..., 1:, :] - obj[..., :-1, :]
    # Pad to same shape (forward differences lose one pixel)
    grad_x = jnp.pad(grad_x, [(0, 0)] * (obj.ndim - 1) + [(0, 1)])
    grad_y = jnp.pad(grad_y, [(0, 0)] * (obj.ndim - 2) + [(0, 1), (0, 0)])
    tv = jnp.sum(jnp.sqrt(jnp.abs(grad_x) ** 2 + jnp.abs(grad_y) ** 2 + aleph))
    return weight * tv
