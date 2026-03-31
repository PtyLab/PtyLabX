"""Loss functions for AD-based ptychographic reconstruction.

Each loss takes ``(I_measured, I_predicted)`` and returns a scalar.
The forward model and loss are composed via ``build_loss()`` in ``__init__.py``.

"""

from __future__ import annotations

import jax.numpy as jnp


def amplitude_loss(I_meas: jnp.ndarray, I_pred: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error on amplitudes (Gaussian noise model).

    Equivalent to ``||sqrt(I_meas) - sqrt(I_pred)||^2`` averaged over pixels.
    Acts as a variance-stabilising transform at moderate-to-high photon counts
    (approximates the Anscombe transform without the +3/8 shift).

    Best for: high-flux optical ptychography.
    """
    return jnp.mean((jnp.sqrt(I_meas + 1e-10) - jnp.sqrt(I_pred + 1e-10)) ** 2)


def poisson_loss(I_meas: jnp.ndarray, I_pred: jnp.ndarray) -> jnp.ndarray:
    """Poisson negative log-likelihood loss.

    ``mean(I_pred - I_meas * log(I_pred))``, with a small epsilon to avoid ``log(0)``.

    Best for: low-flux / X-ray / electron ptychography.
    """
    return jnp.mean(I_pred - I_meas * jnp.log(I_pred + 1e-10))


def mad_amplitude_loss(I_meas: jnp.ndarray, I_pred: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute deviation on amplitudes.

    ``mean(|sqrt(I_meas) - sqrt(I_pred)|)`` — more robust to outliers than MSE.
    Experimental: needs further investigation on convergence behaviour.
    """
    return jnp.mean(jnp.abs(jnp.sqrt(I_meas + 1e-10) - jnp.sqrt(I_pred + 1e-10)))
