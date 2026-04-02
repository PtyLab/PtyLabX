"""Single-slice CPM forward model for AD-based ptychographic reconstruction.

This is the simplest forward model: extract object patch at each scan position,
multiply by probe, propagate to detector (Fraunhofer), compute intensity.

Supports both object-only and blind (object + probe) reconstruction via the
``PtychographyState`` — if ``state.probe`` is ``None``, a frozen probe must be
supplied via ``known_probe``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from PtyLabX.AutoDiff.propagators import propagate_fraunhofer
from PtyLabX.AutoDiff.state import PtychographyState, StaticConfig


def single_slice_forward(
    state: PtychographyState,
    position_indices: jax.Array,
    static: StaticConfig,
    known_probe: jax.Array | None = None,
) -> jax.Array:
    """Predict diffraction intensities for a batch of scan positions.

    For each position index, extracts the object patch via
    ``jax.lax.dynamic_slice``, multiplies by the probe, propagates to the
    detector plane (Fraunhofer FFT), and computes the squared modulus.

    Parameters
    ----------
    state : PtychographyState
        Current reconstruction state (object, and optionally probe).
    position_indices : jax.Array
        1-D integer array of scan position indices for this batch.
    static : StaticConfig
        Non-differentiable metadata (positions, pixel sizes, propagator type, etc.).
    known_probe : jax.Array | None
        If ``state.probe`` is ``None`` (object-only reconstruction), this must
        supply the fixed probe array.

    Returns
    -------
    jax.Array
        Predicted intensities, shape ``(batch_size, Np, Np)``.
    """
    probe = state.probe if state.probe is not None else known_probe
    Np = static.Np

    def _single_position(pos_idx: jax.Array) -> jax.Array:
        """Forward model for one scan position."""
        assert probe is not None
        row, col = static.positions[pos_idx, 0], static.positions[pos_idx, 1]
        # Extract object patch — dynamic_slice is vmap/grad compatible
        object_patch = jax.lax.dynamic_slice(state.object, (0, 0, 0, 0, row, col), (*state.object.shape[:4], Np, Np))
        # Exit surface wave
        esw = object_patch * probe
        # Propagate to detector (Fraunhofer far-field)
        ESW = propagate_fraunhofer(esw, static.fftshift_switch)
        # Intensity: sum over all mode dimensions (nlambda, nosm, npsm, nslice)
        I_est = jnp.sum(jnp.abs(ESW) ** 2, axis=(0, 1, 2, 3))
        return I_est

    # Vectorise over the batch of position indices
    return jax.vmap(_single_position)(position_indices)
