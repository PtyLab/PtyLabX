"""Pure-function propagators for AD-based reconstruction.

These wrap the existing ``fft2c`` / ``ifft2c`` utilities as thin, JIT-friendly
functions that take only array + scalar arguments (no mutable Reconstruction or
Params objects), making them safe for ``jax.grad`` and ``jax.vmap``.
"""

from __future__ import annotations

import functools

import jax

from PtyLabX.utils.utils import fft2c, ifft2c


@functools.partial(jax.jit, static_argnums=(1,))
def propagate_fraunhofer(fields: jax.Array, fftshift_switch: bool) -> jax.Array:
    """Far-field (Fraunhofer) propagation: centred FFT.

    Parameters
    ----------
    fields : jax.Array
        Exit surface wave(s) to propagate.
    fftshift_switch : bool
        FFT centering convention (static for JIT).

    Returns
    -------
    jax.Array
        Detector-plane field.
    """
    return fft2c(fields, fftshift_switch)


@functools.partial(jax.jit, static_argnums=(1,))
def propagate_fraunhofer_inv(fields: jax.Array, fftshift_switch: bool) -> jax.Array:
    """Inverse far-field propagation: centred IFFT."""
    return ifft2c(fields, fftshift_switch)
