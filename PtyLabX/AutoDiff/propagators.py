"""Pure-function propagators for AD-based reconstruction.

These wrap the existing ``fft2c`` / ``ifft2c`` utilities as thin, JIT-friendly
functions that take only array + scalar arguments (no mutable Reconstruction or
Params objects), making them safe for ``jax.grad`` and ``jax.vmap``.
"""
# TODO: Add other propagators here if required and expose as options in GradientEngine.

from __future__ import annotations

from typing import Callable, cast

import functools

import jax

from PtyLabX.utils.utils import fft2c, ifft2c


def _propagate_fraunhofer_impl(fields: jax.Array, fftshift_switch: bool) -> jax.Array:
    """Core Fraunhofer propagation."""
    return fft2c(fields, fftshift_switch)


_propagate_fraunhofer_jit: Callable[..., jax.Array] = cast(
    Callable[..., jax.Array],
    functools.partial(jax.jit, static_argnums=(1,))(_propagate_fraunhofer_impl),
)


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
    return _propagate_fraunhofer_jit(fields, fftshift_switch)


def _propagate_fraunhofer_inv_impl(fields: jax.Array, fftshift_switch: bool) -> jax.Array:
    """Core inverse Fraunhofer propagation."""
    return ifft2c(fields, fftshift_switch)


_propagate_fraunhofer_inv_jit: Callable[..., jax.Array] = cast(
    Callable[..., jax.Array],
    functools.partial(jax.jit, static_argnums=(1,))(_propagate_fraunhofer_inv_impl),
)


def propagate_fraunhofer_inv(fields: jax.Array, fftshift_switch: bool) -> jax.Array:
    """Inverse far-field propagation: centred IFFT."""
    return _propagate_fraunhofer_inv_jit(fields, fftshift_switch)
