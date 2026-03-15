"""Shared JIT-compiled kernels for ptychographic engine update rules.

These pure functions are used across multiple engines (ePIE, mPIE, qNewton, etc.)
to avoid redundant code and enable JAX JIT compilation for performance.
"""

import functools

import jax
import jax.numpy as jnp

from PtyLabX._types import ExitWave, ObjectPatch, Probe
from PtyLabX.Regularizers import grad_TV


# ========================
# ePIE-style updates
# ========================


@jax.jit
def epie_object_update(objectPatch: ObjectPatch, probe: Probe, DELTA: ExitWave, betaObject: float) -> ObjectPatch:
    """Standard ePIE object patch update.

    Used by: ePIE, ePIE_mw, zPIE, aPIE, OPR, pcPIE
    """
    frac = probe.conj() / jnp.max(jnp.sum(jnp.abs(probe) ** 2, axis=(0, 1, 2, 3)))
    return objectPatch + betaObject * jnp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)


@jax.jit
def epie_probe_update(probe: Probe, objectPatch: ObjectPatch, DELTA: ExitWave, betaProbe: float) -> Probe:
    """Standard ePIE probe update.

    Used by: ePIE, ePIE_mw, zPIE, aPIE, OPR, pcPIE
    """
    frac = objectPatch.conj() / jnp.max(jnp.sum(jnp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3)))
    return probe + betaProbe * jnp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)


# ========================
# mPIE-style updates (regularized)
# ========================


@functools.partial(jax.jit, static_argnames=("fpm_mode",))
def mpie_object_update(
    objectPatch: ObjectPatch,
    probe: Probe,
    DELTA: ExitWave,
    betaObject: float,
    alphaObject: float,
    fpm_mode: bool = False,
) -> ObjectPatch:
    """Momentum-accelerated PIE object update with regularization.

    Used by: mPIE, mPIE_mw, mPIE_tv, pcPIE, mqNewton
    """
    absP2 = jnp.abs(probe) ** 2
    Pmax = jnp.max(jnp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
    if fpm_mode:
        frac = jnp.abs(probe) / Pmax * probe.conj() / (alphaObject * Pmax + (1 - alphaObject) * absP2)
    else:
        frac = probe.conj() / (alphaObject * Pmax + (1 - alphaObject) * absP2)
    return objectPatch + betaObject * jnp.sum(frac * DELTA, axis=2, keepdims=True)


@jax.jit
def mpie_probe_update(
    probe: Probe, objectPatch: ObjectPatch, DELTA: ExitWave, betaProbe: float, alphaProbe: float, weight: float
) -> Probe:
    """Momentum-accelerated PIE probe update with regularization.

    Used by: mPIE, mPIE_mw, mPIE_tv, pcPIE, mqNewton
    """
    absO2 = jnp.abs(objectPatch) ** 2
    Omax = jnp.max(jnp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
    frac = objectPatch.conj() / (alphaProbe * Omax + (1 - alphaProbe) * absO2)
    return probe + weight * betaProbe * jnp.sum(frac * DELTA, axis=1, keepdims=True)


# ========================
# Quasi-Newton updates
# ========================


@jax.jit
def qnewton_object_update(
    objectPatch: ObjectPatch, probe: Probe, DELTA: ExitWave, betaObject: float, regObject: float
) -> ObjectPatch:
    """Quasi-Newton object patch update.

    Used by: qNewton, mqNewton
    """
    Pmax = jnp.max(jnp.sum(jnp.abs(probe), axis=(0, 1, 2, 3)))
    frac = jnp.abs(probe) / Pmax * probe.conj() / (jnp.abs(probe) ** 2 + regObject)
    return objectPatch + betaObject * jnp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)


@jax.jit
def qnewton_probe_update(
    probe: Probe, objectPatch: ObjectPatch, DELTA: ExitWave, betaProbe: float, regProbe: float
) -> Probe:
    """Quasi-Newton probe update.

    Used by: qNewton, mqNewton
    """
    Omax = jnp.max(jnp.sum(jnp.abs(objectPatch), axis=(0, 1, 2, 3)))
    frac = jnp.abs(objectPatch) / Omax * objectPatch.conj() / (jnp.abs(objectPatch) ** 2 + regProbe)
    return probe + betaProbe * jnp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)


# ========================
# Momentum step
# ========================


@jax.jit
def momentum_step(
    current: jax.Array, buffer: jax.Array, momentum: jax.Array, frictionM: float, feedbackM: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply momentum gradient update. Returns (updated_current, updated_momentum, updated_buffer).

    Used by: mPIE, pcPIE, mqNewton for both object and probe momentum updates.
    """
    gradient = buffer - current
    momentum = gradient + frictionM * momentum
    current = current - feedbackM * momentum
    buffer = current  # JAX arrays are immutable; rebinding is sufficient inside JIT
    return current, momentum, buffer


# ========================
# TV-regularized updates
# ========================


@jax.jit
def epie_object_update_tv(
    objectPatch: ObjectPatch, probe: Probe, DELTA: ExitWave, betaObject: float, tv_step_size: float
) -> ObjectPatch:
    """ePIE object update with TV regularization.

    Used by: ePIE_TV, mPIE_tv
    """
    frac = probe.conj() / jnp.max(jnp.sum(jnp.abs(probe) ** 2, axis=(0, 1, 2, 3)))
    TV_update = grad_TV(objectPatch, epsilon=1e-2)
    return objectPatch + betaObject * jnp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True) + tv_step_size * TV_update
