"""Differentiable state containers for AD-based ptychographic reconstruction.

PtychographyState holds the optimizable parameters as a NamedTuple (automatic JAX pytree).
StaticConfig holds non-differentiable metadata (wavelength, pixel sizes, propagator type, etc.).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Params.Params import Params
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class PtychographyState(NamedTuple):
    """Differentiable reconstruction state — a JAX pytree by virtue of being a NamedTuple.

    All fields are JAX arrays that ``jax.grad`` can differentiate through.
    Set a field to ``None`` to freeze it (the optimizer will skip it).

    Attributes
    ----------
    object : jnp.ndarray
        Complex object estimate.  Shape ``(nlambda, nosm, 1, nslice, No, No)``.
    probe : jnp.ndarray | None
        Complex probe estimate.  Shape ``(nlambda, 1, npsm, nslice, Np, Np)``.
        ``None`` means probe is frozen (object-only reconstruction).
    """

    object: jnp.ndarray
    probe: jnp.ndarray | None = None


class StaticConfig(NamedTuple):
    """Non-differentiable metadata passed to forward models as constants.

    These values are treated as compile-time or static arguments by JAX and
    are never differentiated through.

    Attributes
    ----------
    positions : jnp.ndarray
        Scan positions in pixel coordinates, shape ``(num_frames, 2)``, dtype int32.
    ptychogram : jnp.ndarray
        Measured diffraction intensities, shape ``(num_frames, Nd, Nd)``.
    wavelength : float
        Illumination wavelength in metres.
    zo : float
        Sample-to-detector distance in metres.
    dxp : float
        Probe pixel size in metres.
    dxd : float
        Detector pixel size in metres.
    Np : int
        Probe/detector pixel count (one side).
    No : int
        Object pixel count (one side).
    fftshift_switch : bool
        Whether to use the fftshift convention for FFTs.
    propagator_type : str
        Which propagator to use (``'Fraunhofer'``, ``'Fresnel'``, ``'ASP'``).
    """

    positions: jnp.ndarray
    ptychogram: jnp.ndarray
    wavelength: float
    zo: float
    dxp: float
    dxd: float
    Np: int
    No: int
    fftshift_switch: bool
    propagator_type: str


def state_from_reconstruction(
    reconstruction: Reconstruction,
    *,
    optimize_probe: bool = False,
) -> PtychographyState:
    """Extract a ``PtychographyState`` from an initialised ``Reconstruction``.

    Parameters
    ----------
    reconstruction : Reconstruction
        Must have ``initializeObjectProbe()`` already called.
    optimize_probe : bool
        If ``True``, include the probe in the state (blind reconstruction).
        If ``False`` (default), probe is ``None`` (object-only).

    Returns
    -------
    PtychographyState
    """
    return PtychographyState(
        object=jnp.array(reconstruction.object, dtype=jnp.complex64),
        probe=jnp.array(reconstruction.probe, dtype=jnp.complex64) if optimize_probe else None,
    )


def static_from_reconstruction(
    reconstruction: Reconstruction,
    experimentalData: ExperimentalData,
    params: Params,
) -> StaticConfig:
    """Extract a ``StaticConfig`` from PtyLabX objects.

    Parameters
    ----------
    reconstruction : Reconstruction
        Initialised reconstruction holding geometry and pixel sizes.
    experimentalData : ExperimentalData
        Holds the ptychogram and scan positions.
    params : Params
        Configuration (propagator type, fftshift, etc.).

    Returns
    -------
    StaticConfig
    """
    return StaticConfig(
        positions=jnp.array(reconstruction.positions, dtype=jnp.int32),
        ptychogram=jnp.array(experimentalData.ptychogram, dtype=jnp.float32),
        wavelength=float(reconstruction.wavelength),
        zo=float(reconstruction.zo),
        dxp=float(reconstruction.dxp),
        dxd=float(reconstruction.dxd),
        Np=int(reconstruction.Np),
        No=int(reconstruction.No),
        fftshift_switch=bool(params.fftshiftSwitch),
        propagator_type=str(params.propagatorType),
    )


def state_to_reconstruction(state: PtychographyState, reconstruction: Reconstruction) -> None:
    """Write optimised arrays back into a ``Reconstruction`` for monitoring/saving.

    Parameters
    ----------
    state : PtychographyState
        Current optimised state.
    reconstruction : Reconstruction
        Target reconstruction object (mutated in place).
    """
    reconstruction.object = state.object
    if state.probe is not None:
        reconstruction.probe = state.probe
