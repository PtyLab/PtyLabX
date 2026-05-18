import functools
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

from PtyLabX import Params, Reconstruction

# Type alias for the return type of all propagator functions
PropagatorReturn: TypeAlias = tuple[jax.Array, jax.Array]
# Type alias for any propagator function
PropagatorFn: TypeAlias = Callable[..., PropagatorReturn]
from PtyLabX.Operators._propagation_kernels import _make_quad_phase
from PtyLabX.Operators.off_axis_sas import (
    _make_transferfunction_sas,
    propagate_sas,
    propagate_sas_inv,
)
from PtyLabX.Operators.propagator_utils import complexexp
from PtyLabX.utils.utils import circ, fft2c, ifft2c


def _asp_propagate_impl(
    u: jax.Array, transfer_function: jax.Array, fftshiftSwitch: bool
) -> jax.Array:
    """Core of ASP propagation: FFT, multiply by transfer function, IFFT."""
    return ifft2c(
        fft2c(u, fftshiftSwitch=fftshiftSwitch) * transfer_function,
        fftshiftSwitch=fftshiftSwitch,
    )


_asp_propagate_jit: Callable[..., jax.Array] = cast(
    Callable[..., jax.Array],
    functools.partial(jax.jit, static_argnums=(2,))(_asp_propagate_impl),
)


def _asp_propagate(
    u: jax.Array, transfer_function: jax.Array, fftshiftSwitch: bool
) -> jax.Array:
    """JIT-compiled core of ASP propagation: FFT, multiply by transfer function, IFFT."""
    return _asp_propagate_jit(u, transfer_function, fftshiftSwitch)


# how many kernels are kept in memory for every type of propagator? Higher can be faster but comes
# at the expense of (GPU) memory.
cache_size = 5


def propagate_fraunhofer(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Propagate using the fraunhofer approximation.

    Parameters
    ----------
    fields: jax.Array
        Electric field to propagate
    params: Params
        Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
        Reconstruction object.
    z: float
        propagation distance. Is ignored in this function.

    Returns
    -------

    A tuple of (reconstruction.esw, Propagated field)

    """
    return reconstruction.esw, fft2c(fields, params.fftshiftSwitch)


def propagate_fraunhofer_inv(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Inverse transform. See propagate_frauhofer for the arguments.

    Parameters
    ----------
    fields: jax.Array
        Electric field to propagate
    params: Params
        Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
        Reconstruction object.
    z: float
        propagation distance. Is ignored in this function.

    Returns
    -------
    A tuple of (reconstruction.esw, inverse transformed field)
    """
    return reconstruction.esw, ifft2c(fields, params.fftshiftSwitch)


def propagate_fresnel(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    # make the quad phase if it's not available yet
    """
    Propagate using the fresnel approximation.

    Parameters
    ----------
    fields: jax.Array
       Electric field to propagate
    params: Params
       Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
       Reconstruction object.
    z: float
       propagation distance in meter

    Returns
    -------

    A tuple of (reconstruction.esw, Propagated field)

    """
    if z is None:
        z = reconstruction.zo
    quadratic_phase = _make_quad_phase(
        z,
        reconstruction.wavelength,
        fields.shape[-1],
        reconstruction.dxp,
    )

    eswUpdate = fft2c(fields * quadratic_phase, params.fftshiftSwitch)
    # for legacy reasons, as far as I can see there's no reason to do this
    # esw = reconstruction.esw * quadratic_phase
    return reconstruction.esw, eswUpdate


def propagate_fresnel_inv(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Propagate using the inverse fresnel approximation.

    Parameters
    ----------
    fields: jax.Array
      Electric field to propagate
    params: Params
      Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
      Reconstruction object.
    z: float
      propagation distance in meter

    Returns
    -------

    A tuple of (reconstruction.esw, Propagated field)

    """
    # make the quad phase if it's not available yet
    if z is None:
        z = reconstruction.zo
    quadratic_phase = _make_quad_phase(
        z,
        reconstruction.wavelength,
        reconstruction.Np,
        reconstruction.dxp,
    ).conj()

    eswUpdate = ifft2c(fields, params.fftshiftSwitch) * quadratic_phase
    # esw = reconstruction.esw * quadratic_phase
    return reconstruction.esw, eswUpdate


def propagate_ASP(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    inverse: bool = False,
    z: float | None = None,
    fftflag: bool = True,
) -> PropagatorReturn:
    """
    Propagate using the angular spectrum method


    Parameters
    ----------
    fields: jax.Array
      Electric field to propagate
    params: Params
      Parameter object. The parameter params.fftshiftSwitch is inspected for the fourier transform
    reconstruction: Reconstruction
      Reconstruction object.
    z: float
      propagation distance in meter
    fftflag: bool
      Specified wether or not to use a centered fft internally. Set to false for debugging but should generally be turned on.

    Returns
    -------
    reconstruction.esw: jax.Array
        exit surface wave
    result: jax.Array
        propagated field
    """

    if params.fftshiftSwitch:
        raise ValueError("ASP propagator only works with fftshiftswitch == False")
    if reconstruction.nlambda > 1:
        raise ValueError("For multi-wavelength, set polychromeASP instead of ASP")
    if z is None:
        z = reconstruction.zo
    transfer_function = _make_transferfunction_ASP(
        params.fftshiftSwitch,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        reconstruction.wavelength,
        reconstruction.Lp,
        reconstruction.nlambda,
    )
    if fftflag:
        transfer_function = jnp.fft.ifftshift(transfer_function, axes=(-2, -1))
    if inverse:
        transfer_function = transfer_function.conj()
    result = _asp_propagate(fields, transfer_function, fftflag)
    return reconstruction.esw, result


def propagate_ASP_inv(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
    fftflag: bool = True,
) -> PropagatorReturn:
    """Inverse ASP propagation. See propagate_ASP."""
    return propagate_ASP(
        fields, params, reconstruction, z=z, fftflag=fftflag, inverse=True
    )


def propagate_twoStepPolychrome(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    inverse: bool = False,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Two-step polychrome propagation.

    Parameters
    ----------
    fields: jax.Array
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    inverse: bool
        Reverse propagation
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    """
    if z is None:
        z = reconstruction.zo
    assert reconstruction.spectralDensity is not None
    transfer_function, quadratic_phase = _make_cache_twoStepPolychrome(
        params.fftshiftSwitch,
        reconstruction.nlambda,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        # this has to be cast to a tuple to
        # make sure it is reused
        tuple(
            reconstruction.spectralDensity
        ),  # spectralDensity is asserted non-None before this call
        reconstruction.Lp,
        reconstruction.dxp,
    )
    if inverse:
        result = ifft2c(
            fft2c(fields * quadratic_phase.conj()) * transfer_function.conj()
        )
        return reconstruction.esw, result
    else:
        result = ifft2c(fft2c(fields) * transfer_function) * quadratic_phase
        result = fft2c(result, params.fftshiftSwitch)
        return reconstruction.esw, result


def propagate_twoStepPolychrome_inv(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    """
    See propagate_twoStepPolychrome.

    Parameters
    ----------
    fields
    params
    reconstruction
    z

    Returns
    -------

    """
    F = propagate_twoStepPolychrome(fields, params, reconstruction, inverse=True, z=z)[
        1
    ]
    G = propagate_twoStepPolychrome(
        reconstruction.ESW, params, reconstruction, inverse=True, z=z
    )[1]  # tODO: What is G here? Why are we not returning reconstruction.esw?
    return G, F


def propagate_scaledASP(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    inverse: bool = False,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Propagate using the scaled angular spectrum method.

    Parameters
    ----------
    fields
    params
    reconstruction
    inverse
    z

    Returns
    -------

    """
    if z is None:
        z = reconstruction.zo
    Q1, Q2 = _make_transferfunction_scaledASP(
        params.propagatorType,
        params.fftshiftSwitch,
        reconstruction.nlambda,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        reconstruction.wavelength,
        reconstruction.dxo,
        reconstruction.dxd,
    )
    if inverse:
        Q1, Q2 = Q1.conj(), Q2.conj()
        return reconstruction.esw, ifft2c(fft2c(fields) * Q2) * Q1
    return reconstruction.esw, ifft2c(fft2c(fields * Q1) * Q2)


def propagate_scaledASP_inv(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Reverse scaled angular spectrum propagation. See scaledASP for details.

    Parameters
    ----------
    fields: jax.Array
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    """
    return propagate_scaledASP(fields, params, reconstruction, inverse=True, z=z)


def propagate_scaledPolychromeASP(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    inverse: bool = False,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Scaled angular spectrum for multiple wavelengths.

    Parameters
    ----------
    fields: jax.Array
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    inverse: bool
        Reverse propagation
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    """
    if z is None:
        z = reconstruction.zo
    assert reconstruction.spectralDensity is not None
    Q1, Q2 = _make_transferfunction_scaledPolychromeASP(
        params.fftshiftSwitch,
        reconstruction.nlambda,
        reconstruction.nosm,
        reconstruction.npsm,
        z,
        reconstruction.Np,
        tuple(
            reconstruction.spectralDensity
        ),  # spectralDensity is asserted non-None before this call
        reconstruction.dxo,
        reconstruction.dxd,
    )
    if inverse:
        Q1, Q2 = Q1.conj(), Q2.conj()
        return reconstruction.esw, ifft2c(fft2c(fields) * Q2) * Q1
    return reconstruction.esw, ifft2c(fft2c(fields * Q1) * Q2)


def propagate_scaledPolychromeASP_inv(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Reverse Scaled angular spectrum for multiple wavelengths.

    Parameters
    ----------
    fields: jax.Array
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    inverse: bool
        Reverse propagation
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    Returns
    -------

    """
    return propagate_scaledPolychromeASP(
        fields, params, reconstruction, inverse=True, z=z
    )


def propagate_polychromeASP(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    inverse: bool = False,
    z: float | None = None,
) -> PropagatorReturn:
    """
    ASP propagation  for multiple wavelengths.

    Parameters
    ----------
    fields: jax.Array
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    inverse: bool
        Reverse propagation
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    Returns
    -------

    """
    if z is None:
        z = reconstruction.zo
    assert reconstruction.spectralDensity is not None
    transfer_function = _make_transferfunction_polychrome_ASP(
        params.propagatorType,
        params.fftshiftSwitch,
        reconstruction.nosm,
        reconstruction.npsm,
        reconstruction.Np,
        z,
        reconstruction.wavelength,
        reconstruction.Lp,
        reconstruction.nlambda,
        tuple(
            reconstruction.spectralDensity
        ),  # spectralDensity is asserted non-None before this call
    )

    if inverse:
        transfer_function = transfer_function.conj()
    result = ifft2c(fft2c(fields) * transfer_function)
    return reconstruction.esw, result


def propagate_identity(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    inverse: bool = False,
    z: float | None = None,
) -> PropagatorReturn:
    """
    Identity propagator (aka does nothing).

    Can probably be used to figure out orientation or to perform some kind of stitching.


    Parameters
    ----------
    fields
    params
    reconstruction
    inverse
    z

    Returns
    -------

    """
    transfer_function = _make_quad_phase(
        1e-3, 532e-9, reconstruction.Np, reconstruction.dxp
    )
    transfer_function = transfer_function * 0 + 1
    return reconstruction.esw, fields * transfer_function


def propagate_polychromeASP_inv(
    fields: jax.Array,
    params: Params,
    reconstruction: Reconstruction,
    z: float | None = None,
) -> PropagatorReturn:
    """
    inverse scaled angular spectrum for multiple wavelengths.

    Parameters
    ----------
    fields: jax.Array
        Field to propagate
    params: Params
        Parameters
    reconstruction: Reconstruction
    inverse: bool
        Reverse propagation
    z: float
        Propagation distance

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field

    """
    return propagate_polychromeASP(fields, params, reconstruction, inverse=True, z=z)


def detector2object(
    fields: jax.Array | None, params: Params, reconstruction: Reconstruction
) -> PropagatorReturn:
    """
    Implements detector2object.m. Returns a propagated version of the field.

    If field is not given, reconstruction.esw is taken
    :return: esw, updated esw
    """
    if fields is None:
        fields = reconstruction.ESW
    method: PropagatorFn = reverse_lookup_dictionary[params.propagatorType.lower()]
    return method(fields, params, reconstruction)


def object2detector(
    fields: jax.Array | None, params: Params, reconstruction: Reconstruction
) -> PropagatorReturn:
    """Propagate a field from the object to the detector. Return the new object, do not update in-place."""

    method: PropagatorFn = forward_lookup_dictionary[params.propagatorType.lower()]
    if fields is None:
        fields = reconstruction.esw
    return method(fields, params, reconstruction)


def aspw(
    u: jax.Array,
    z: float,
    wavelength: float,
    L: float,
    bandlimit: bool = True,
    is_FT: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """
    Angular spectrum plane wave propagation function.
    following: Matsushima et al., "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space
    Propagation in Far and Near Fields", Optics Express, 2009


    Parameters
    ----------
    u: jax.Array
        a 2D field distribution at z = 0 (u is assumed to be square, i.e. N x N)
    z: float
        propagation distance in meter
    wavelength: float
        Wavelength in meter
    L: float
        total size of the field in meter
    bandlimit: bool
        Wether or not to band limit the sample
    is_FT: bool
        If the field has already been fourier transformed.

    Returns
    -------
    U_prop, Q  (field distribution after propagation and the bandlimited transfer function)

    """
    N = u.shape[-1]
    phase_exp = _aspw_transfer_function(
        float(z),
        float(wavelength),
        int(N),
        float(L),
        bandlimit=bandlimit,
    )
    u_prop = _aspw_propagate_core(u, phase_exp, is_FT)
    return u_prop, phase_exp


def _aspw_propagate_core_impl(
    u: jax.Array, phase_exp: jax.Array, is_FT: bool
) -> jax.Array:
    """Core of aspw: FFT (if needed), multiply, IFFT."""
    U = u if is_FT else fft2c(u)
    return ifft2c(U * phase_exp)


_aspw_propagate_core_jit: Callable[..., jax.Array] = cast(
    Callable[..., jax.Array],
    functools.partial(jax.jit, static_argnums=(2,))(_aspw_propagate_core_impl),
)


def _aspw_propagate_core(u: jax.Array, phase_exp: jax.Array, is_FT: bool) -> jax.Array:
    """JIT-compiled core of aspw: FFT (if needed), multiply, IFFT."""
    return _aspw_propagate_core_jit(u, phase_exp, is_FT)


def scaledASP(
    u: jax.Array,
    z: float,
    wavelength: float,
    dx: float,
    dq: float,
    bandlimit: bool = True,
    exactSolution: bool = False,
) -> (
    tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """
    Angular spectrum propagation with customized grid spacing dq (within Fresnel(or paraxial) approximation)
    :param u: a 2D square input field
    :param z: propagation distance
    :param wavelength: propagation wavelength
    :param dx: grid spacing in original plane (u)
    :param dq: grid spacing in destination plane (Uout)
    :return: propagated field and two quadratic phases

    note: to be analytically correct, add Q3 (see below)
    if only intensities matter, leave it out
    """
    # optical wavenumber
    k = 2 * jnp.pi / wavelength
    # assume square grid
    N = u.shape[-1]
    # source plane coordinates (ogrid: avoid materializing full 2D grids)
    x1 = jnp.arange(-N // 2, N // 2) * dx
    X1, Y1 = x1.reshape(1, -1), x1.reshape(-1, 1)
    r1sq = X1**2 + Y1**2
    # spatial frequencies(of source plane)
    f = jnp.arange(-N // 2, N // 2) / (N * dx)
    FX, FY = f.reshape(1, -1), f.reshape(-1, 1)
    fsq = FX**2 + FY**2
    # scaling parameter
    m = dq / dx

    # quadratic phase factors
    Q1 = jnp.exp(1.0j * k / 2 * (1 - m) / z * r1sq)
    Q2 = jnp.exp(-1.0j * jnp.pi**2 * 2 * z / m / k * fsq)

    if bandlimit:
        if m != 1:
            r1sq_max = wavelength * z / (2 * dx * (1 - m))
            Wr = jnp.asarray(circ(X1, Y1, 2 * r1sq_max))
            Q1 = Q1 * Wr

        fsq_max = m / (2 * z * wavelength * (1 / (N * dx)))
        Wf = jnp.asarray(circ(FX, FY, 2 * fsq_max))
        Q2 = Q2 * Wf

    if exactSolution:  # if only intensities matter, leave it out
        # observation plane coordinates
        x2 = jnp.arange(-N // 2, N // 2) * dq
        X2, Y2 = x2.reshape(1, -1), x2.reshape(-1, 1)
        r2sq = X2**2 + Y2**2
        Q3 = jnp.exp(1.0j * k / 2 * (m - 1) / (m * z) * r2sq)
        # compute the propagated field
        Uout = Q3 * ifft2c(Q2 * fft2c(Q1 * u))
        return Uout, Q1, Q2, Q3
    else:  # ignore the phase part in the observation plane
        Uout = ifft2c(Q2 * fft2c(Q1 * u))
        return Uout, Q1, Q2


def scaledASPinv(
    u: jax.Array, z: float, wavelength: float, dx: float, dq: float
) -> jax.Array:
    """
    :param u:  a 2D square input field
    :param z:   propagation distance
    :param wavelength: wavelength
    :param dx:  grid spacing in original plane (u)
    :param dq:  grid spacing in destination plane (Uout)
    :return: propagated field

    note: to be analytically correct, add Q3 (see below)
    if only intensities matter, leave it out
    """
    # optical wavenumber
    k = 2 * jnp.pi / wavelength
    # assume square grid
    N = u.shape[-1]
    # source-plane coordinates (ogrid: avoid materializing full 2D grids)
    x1 = jnp.arange(-N / 2, N / 2) * dx
    X1, Y1 = x1.reshape(1, -1), x1.reshape(-1, 1)
    r1sq = X1**2 + Y1**2
    # spatial frequencies(of source plane)
    f = jnp.arange(-N / 2, N / 2) / (N * dx)
    FX, FY = f.reshape(1, -1), f.reshape(-1, 1)
    fsq = FX**2 + FY**2
    # scaling parameter
    m = dq / dx

    # quadratic phase factors
    Q1 = jnp.exp(1j * k / 2 * (1 - m) / z * r1sq)
    Q2 = jnp.exp(-1j * 2 * jnp.pi**2 * z / m / k * fsq)
    Uout = jnp.conj(Q1) * ifft2c(jnp.conj(Q2) * fft2c(u))

    # x2 = np.arange(-N / 2, N / 2) * dq
    # X2, Y2 = np.meshgrid(x2,x2)
    # r2sq = X2**2 + Y2**2
    # Q3 = np.exp(1.j * k / 2 * (m - 1) / (m * z) * r2sq)
    # # compute the propagated field
    # Uout = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u*np.conj(Q3)))

    return Uout


def fresnelPropagator(
    u: jax.Array, z: float, wavelength: float, L: float
) -> tuple[jax.Array, float, jax.Array, jax.Array]:
    """
    One-step Fresnel propagation, performing Fresnel-Kirchhoff integral.
    :param u:   field distribution at z = 0(u is assumed to be square, i.e.N x N)
    :param z:   propagation distance
    :param wavelength: wavelength
    :param L: total size[m] of the source plane
    :return: propagated field
    """
    k = 2 * jnp.pi / wavelength
    # source coordinates, assuming square grid
    N = u.shape[-1]
    dx = L / N  # source-plane pixel size
    x = jnp.arange(-N // 2, N // 2) * dx
    X, Y = x.reshape(1, -1), x.reshape(-1, 1)

    # observation coordinates
    dq = wavelength * z / L  # observation-plane pixel size
    q = jnp.arange(-N // 2, N // 2) * dq
    Qx, Qy = q.reshape(1, -1), q.reshape(-1, 1)

    # quadratic phase terms
    Q1 = jnp.exp(
        1j * k / (2 * z) * (X**2 + Y**2)
    )  # quadratic phase inside the integral
    Q2 = jnp.exp(1j * k / (2 * z) * (Qx**2 + Qy**2))

    # pre-factor
    A = 1 / (1j * wavelength * z)

    # Fresnel-Kirchhoff integral
    u_out = A * Q2 * fft2c(u * Q1)
    return u_out, dq, Q1, Q2


@lru_cache(cache_size)
def _aspw_transfer_function(
    z: float, wavelength: float, N: int, L: float, bandlimit: bool = True
) -> jax.Array:
    """
    Angular spectrum optical transfer function. You likely don't need to use this directly.

    The result of this call is cached so it can be reused and called as often as you need without having
    to worry about recalculating everything all the time.


    Parameters
    ----------
    z: float
        distance
    wavelength: float
        wavelength in meter
    N: int
        Number of pixels per side
    L: int
        Physical size
    bandlimit: bool
        If the transfer function should be band-limited.

    Returns
    -------

    """
    a_z = abs(z)
    k = 2 * jnp.pi / wavelength
    X = jnp.arange(-N / 2, N / 2) / L
    Fx, Fy = X.reshape(1, -1), X.reshape(-1, 1)
    f_max = L / (wavelength * jnp.sqrt(L**2 + 4 * a_z**2))
    # note: see the paper above if you are not sure what this bandlimit has to do here
    # W = rect(Fx/(2*f_max)) .* rect(Fy/(2*f_max));
    W = jnp.array(circ(Fx, Fy, 2 * f_max))
    # note: accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    exponent = 1 - (Fx * wavelength) ** 2 - (Fy * wavelength) ** 2
    # take out stuff that cannot exist
    mask = exponent > 0
    if not bandlimit:
        mask = 0 * mask + 1
    # put the out of range values to 0 so the square root can be taken
    exponent = jnp.clip(exponent, 0, jnp.inf)
    H = jnp.array(mask * complexexp(k * a_z * jnp.sqrt(exponent)))
    if z < 0:
        H = H.conj()
    phase_exp = H * W
    return phase_exp


@lru_cache(cache_size)
def _make_transferfunction_ASP(
    fftshiftSwitch: bool,
    nosm: int,
    npsm: int,
    Np: int,
    zo: float,
    wavelength: float,
    Lp: float,
    nlambda: int,
) -> jax.Array:
    if fftshiftSwitch:
        raise ValueError("ASP propagatorType works only with fftshiftSwitch = False!")
    if nlambda > 1:
        raise ValueError(
            "For multi-wavelength, polychromeASP needs to be used instead of ASP"
        )

    _transferFunction = jnp.array(
        [
            [
                [
                    [
                        _aspw_transfer_function(zo, wavelength, Np, Lp)
                        for nslice in range(1)
                    ]
                    for npsm in range(npsm)
                ]
                for nosm in range(nosm)
            ]
            for nlambda in range(nlambda)
        ],
        dtype=jnp.complex64,
    )

    return _transferFunction


def aspw_cached(u: jax.Array, z: float, wavelength: float, L: float) -> jax.Array:
    """Cached version of aspw."""
    transferFunction = _aspw_transfer_function(z, wavelength, u.shape[-1], L)
    U = fft2c(u)
    u_prime = ifft2c(U * transferFunction)
    return u_prime


@lru_cache(cache_size)
def _make_transferfunction_polychrome_ASP(
    propagatorType: str,
    fftshiftSwitch: bool,
    nosm: int,
    npsm: int,
    Np: int,
    zo: float,
    wavelength: float,
    Lp: float,
    nlambda: int,
    spectralDensity_as_tuple: tuple[float, ...],
) -> jax.Array:
    spectralDensity = np.array(spectralDensity_as_tuple)
    if fftshiftSwitch:
        raise ValueError("ASP propagatorType works only with fftshiftSwitch = False!")
    transferFunction = jnp.array(
        [
            [
                [
                    [
                        _aspw_transfer_function(
                            zo,
                            spectralDensity[nlambda],
                            Np,
                            Lp,
                        )
                        for nslice in range(1)
                    ]
                    for npsm in range(npsm)
                ]
                for nosm in range(nosm)
            ]
            for nlambda in range(nlambda)
        ]
    )
    return transferFunction


@lru_cache(cache_size)
def _make_transferfunction_scaledASP(
    propagatorType: str | None,
    fftshiftSwitch: bool,
    nlambda: int,
    nosm: int,
    npsm: int,
    Np: int,
    zo: float,
    wavelength: float,
    dxo: float,
    dxd: float,
) -> tuple[jax.Array, jax.Array]:
    if fftshiftSwitch:
        raise ValueError(
            "scaledASP propagatorType works only with fftshiftSwitch = False!"
        )
    if nlambda > 1:
        raise ValueError(
            "For multi-wavelength, scaledPolychromeASP needs to be used instead of scaledASP"
        )
    dummy = jnp.ones((1, nosm, npsm, 1, Np, Np), dtype=jnp.complex64)
    _Q1 = jnp.ones_like(dummy)
    _Q2 = jnp.ones_like(dummy)
    for i_nosm in range(nosm):
        for i_npsm in range(npsm):
            _, q1, q2 = cast(
                tuple[jax.Array, jax.Array, jax.Array],
                scaledASP(dummy[0, i_nosm, i_npsm, 0, :, :], zo, wavelength, dxo, dxd),
            )
            _Q1 = _Q1.at[0, i_nosm, i_npsm, 0, ...].set(q1)
            _Q2 = _Q2.at[0, i_nosm, i_npsm, 0, ...].set(q2)

    return _Q1, _Q2


@lru_cache(cache_size)
def _make_transferfunction_scaledPolychromeASP(
    fftshiftSwitch: bool,
    nlambda: int,
    nosm: int,
    npsm: int,
    zo: float,
    Np: int,
    spectralDensity_as_tuple: tuple[float, ...],
    dxo: float,
    dxd: float,
) -> tuple[jax.Array, jax.Array]:
    spectralDensity = np.array(spectralDensity_as_tuple)
    if fftshiftSwitch:
        raise ValueError(
            "scaledPolychromeASP propagatorType works only with fftshiftSwitch = False!"
        )
    dummy = jnp.ones((nlambda, nosm, npsm, 1, Np, Np), dtype="complex64")
    Q1 = jnp.ones_like(dummy)
    Q2 = jnp.ones_like(dummy)
    for nlambda in range(nlambda):
        Q1_candidate, Q2_candidate = _make_transferfunction_scaledASP(
            None,
            fftshiftSwitch,
            1,
            nosm,
            npsm,
            Np,
            zo,
            spectralDensity[nlambda],
            dxo,
            dxd,
        )
        Q1 = Q1.at[nlambda].set(Q1_candidate[0])
        Q2 = Q2.at[nlambda].set(Q2_candidate[0])
    return Q1, Q2


@lru_cache(cache_size)
def _make_cache_twoStepPolychrome(
    fftshiftSwitch: bool,
    nlambda: int,
    nosm: int,
    npsm: int,
    Np: int,
    zo: float,
    spectralDensity_as_tuple: tuple[float, ...],
    Lp: float,
    dxp: float,
) -> tuple[jax.Array, jax.Array]:
    spectralDensity = np.array(spectralDensity_as_tuple)
    if fftshiftSwitch:
        raise ValueError(
            "twoStepPolychrome propagatorType works only with fftshiftSwitch = False!"
        )
    transferFunction = jnp.array(
        [
            [
                [
                    [
                        _aspw_transfer_function(
                            z=zo * (1 - spectralDensity[0] / spectralDensity[nlambda]),
                            wavelength=spectralDensity[nlambda],
                            N=Np,
                            L=Lp,
                        )
                        for nslice in range(1)
                    ]
                    for npsm in range(npsm)
                ]
                for nosm in range(nosm)
            ]
            for nlambda in range(nlambda)
        ]
    )
    quadraticPhase = _make_quad_phase(zo, spectralDensity[0], Np, dxp)
    return transferFunction, quadraticPhase


def clear_cache(logger: logging.Logger | None = None) -> None:
    """Clear the cache of all cached functions in this module. Use if GPU memory is not available.

    IF logger is available, print some information about the methods being cleared.

    Returns nothing"""
    list_of_methods = [
        _aspw_transfer_function,
        _make_quad_phase,
        _make_transferfunction_ASP,
        _make_transferfunction_scaledASP,
        _make_cache_twoStepPolychrome,
        _make_transferfunction_polychrome_ASP,
        _make_transferfunction_scaledPolychromeASP,
        _make_transferfunction_sas,
    ]
    for method in list_of_methods:
        if logger is not None:
            logger.debug(method.cache_info())
            logger.info("clearing cache for %s", method)
        method.cache_clear()


forward_lookup_dictionary = {
    "fraunhofer": propagate_fraunhofer,
    "fresnel": propagate_fresnel,
    "asp": propagate_ASP,
    "polychromeasp": propagate_polychromeASP,
    "scaledasp": propagate_scaledASP,
    "scaledpolychromeasp": propagate_scaledPolychromeASP,
    "twosteppolychrome": propagate_twoStepPolychrome,
    "identity": propagate_identity,
    "sas": propagate_sas,
}


reverse_lookup_dictionary = {
    "fraunhofer": propagate_fraunhofer_inv,
    "fresnel": propagate_fresnel_inv,
    "asp": propagate_ASP_inv,
    "polychromeasp": propagate_polychromeASP_inv,
    "scaledasp": propagate_scaledASP_inv,
    "scaledpolychromeasp": propagate_scaledPolychromeASP_inv,
    "twosteppolychrome": propagate_twoStepPolychrome_inv,
    "identity": propagate_identity,
    "sas": propagate_sas_inv,
}
