from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from PtyLabX import Params, Reconstruction
from PtyLabX.Operators._propagation_kernels import __make_quad_phase
from PtyLabX.Operators.propagator_utils import (
    complexexp,
    convolve2d,
    gaussian2D,
    iterate_6d_fields,
)
from PtyLabX.utils.utils import fft2c, ifft2c

CACHE_SIZE = 5


def _to_tuple(theta):
    if theta is None:
        return (0.0, 0.0)
    if isinstance(theta, (int, float)):
        return (float(theta), 0.0)
    elif isinstance(theta, tuple) and len(theta) == 2:
        return (float(theta[0]), float(theta[1]))
    else:
        raise ValueError(
            "theta must be specified as None, a scalar, or a tuple of two numbers"
        )


def _pad_field(fields, pad_factor: int):
    rows, cols = fields.shape[-2:]
    rows_padded, cols_padded = pad_factor * rows, pad_factor * cols
    pad_rows = (rows_padded - rows) // 2
    pad_cols = (cols_padded - cols) // 2
    pad_width = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (pad_rows, pad_rows),
        (pad_cols, pad_cols),
    )
    fields_padded = jnp.pad(fields, pad_width, "constant")
    return fields_padded


def _unpad_field(fields_padded, pad_factor: int):
    rows_padded, cols_padded = fields_padded.shape[-2:]
    rows_unpadded, cols_unpadded = jnp.array(fields_padded.shape[-2:]) // pad_factor
    start_h, start_w = (
        (rows_padded - rows_unpadded) // 2,
        (cols_padded - cols_unpadded) // 2,
    )
    slicey, slicex = (
        slice(start_h, start_h + rows_unpadded),
        slice(start_w, start_w + cols_unpadded),
    )
    fields_unpadded = fields_padded[..., slicey, slicex]

    return fields_unpadded


def __sas_transfer_function(wavelength, Lp, Np, theta, z1, z2):
    """Precompensation transfer function for scalable off-axis transfer function.

    Parameters
    ----------
    wavelength : float
        wavelength
    Lp : float
        Physical size
    Np : float
        _description_
    theta : tuple / scalar
        Theta (angle in degrees) in the x-y plane.
    z1 : float
        propagation distance (ASPW)
    z2 : float
        propagation distance (Fresnel) - relaxing sampling requirements.

    Returns
    -------
    jnp.ndarray
        precompensated transfer function `H_precomp`
    """

    # Fourier grid
    df = 1 / Lp
    f = jnp.linspace(-Np / 2, Np / 2, int(Np)) * df
    Fx, Fy = jnp.meshgrid(f, f)

    # off-axis sines and tangents (theta in degrees)
    theta_x, theta_y = theta
    sx, sy = jnp.sin(jnp.radians(theta_x)), jnp.sin(jnp.radians(theta_y))
    tx, ty = jnp.tan(jnp.radians(theta_x)), jnp.tan(jnp.radians(theta_y))

    # transfer function
    # eq. 12 includes chi parameter under square root
    chi = (
        1 / wavelength**2
        - (Fx + (sx / wavelength)) ** 2
        - (Fy + (sy / wavelength)) ** 2
    )
    sqrt_chi = jnp.sqrt(jnp.maximum(0, chi))

    def _create_bandpass_filter(smooth_filter=True, eps=1e-10):
        """Creating a bandpass filter"""

        # for the field in x-direction
        Omegax = z1 * (tx - (Fx + sx / wavelength) / (sqrt_chi + eps))
        Omegax += wavelength * z2 * Fx

        # for the field in y-direction
        Omegay = z1 * (ty - (Fy + sy / wavelength) / (sqrt_chi + eps))
        Omegay += wavelength * z2 * Fy

        # Fourier Bandpass filter (W is a mask below)
        sampling_rate = 2
        W_mask = jnp.logical_and(
            df <= jnp.abs(1 / (sampling_rate * Omegax + eps)),
            df <= jnp.abs(1 / (sampling_rate * Omegay + eps)),
        )

        # smooth the bandpass filter corners with a Gaussian kernel
        if smooth_filter:
            kernel_gauss = gaussian2D(8, 2)
            bandpass_filter = convolve2d(W_mask, kernel_gauss, mode="same")
        else:
            bandpass_filter = W_mask

        return bandpass_filter

    # Pre-compensation transfer function

    # implements the angular spectrum transfer function (see eq. 23, part of the precompensation factor)
    # zo is z1 in the document.
    H_AS = complexexp(2 * jnp.pi * z1 * sqrt_chi)

    # Fresnel transfer function
    H_Fr = complexexp(
        -jnp.pi * z2 / wavelength * ((wavelength * Fx) ** 2 + (wavelength * Fy) ** 2)
    )

    # off-axis consideration of the transfer function
    H_offaxis = complexexp(2 * jnp.pi * z1 * (tx * Fx + ty * Fy))

    # precompensation with bandpass filter
    bandpass_filter = _create_bandpass_filter(smooth_filter=True, eps=1e-10)
    H_precomp = H_AS * jnp.conj(H_Fr) * H_offaxis * bandpass_filter

    return H_precomp


@lru_cache(CACHE_SIZE)
def __make_transferfunction_sas(
    params: Params,
    reconstruction: Reconstruction,
    Np: int,
    Lp: int,
    z1: float,
    z2: float,
):
    """
    Allows for a 6-dimensional (nlambda, nosm, npsm, nslice, Np, Np) array when computing the transfer function
    for a scalable off-axis angular spectrum propagator.

    Parameters
    ----------
    params: Params
        Instance of the Params class
    reconstruction: Reconstruction
        Instance of the Reconstruction class.
    Np: int
        Physical resolution
    Lp: int
        Length of the probe FOV
    z1: float
        Propagation distance for ASPW propagator
    z2: float
        Propagation distance for Fresnel propagator

    Returns
    -------
    jnp.ndarray
        The calculated transfer function with shape (nlambda, nosm, npsm, nslice, Np, Np).
    """

    # off axis theta tuple in degrees
    theta = _to_tuple(
        reconstruction.theta if hasattr(reconstruction, "theta") else None
    )
    fftshiftSwitch = params.fftshiftSwitch
    wavelength = reconstruction.wavelength  # Wavelength used in the scanning probe.
    nosm = reconstruction.nosm  # no. of spatial modes for the object.
    npsm = reconstruction.npsm  # no. of spatial modes for the probe.
    nlambda = reconstruction.nlambda  # no. of wavelengths for multi-wavelength.
    nslice = reconstruction.nslice  # no. of slices for multi-slice operation

    # ensuring some checks
    if fftshiftSwitch:
        raise ValueError("ASP propagatorType works only with fftshiftSwitch = False!")

    if nlambda > 1:
        raise ValueError("Currently for multi-wavelength, off-axis SAS does not work")

    if nslice > 1:
        raise ValueError(
            " Currently off-axis SAS not valid for multi-slice ptychography"
        )

    # transfer function over the entire 6D field (nlambda, nosm, npsm, nslice, Np, Np)
    transfer_function = jnp.zeros(
        (nlambda, nosm, npsm, nslice, Np, Np),
        dtype="complex64",
    )
    for inds in iterate_6d_fields(transfer_function):
        i_nlambda, j_nosm, k_npsm, l_nslice = inds
        transfer_function = transfer_function.at[
            i_nlambda, j_nosm, k_npsm, l_nslice
        ].set(
            __sas_transfer_function(wavelength, Lp, Np, theta, z1, z2)
        )

    return transfer_function


def _interface_sas(
    fields,
    params: Params,
    reconstruction: Reconstruction,
    z: float = None,
    with_quad_phase_Q2: bool = False,
):
    """Just an interface for the actual forward and backward off-axis sas propagator"""

    # ideally pad factor of 2 is supported as per the SAS publication, however can be modified by user.
    pad_factor = (
        reconstruction.pad_factor if hasattr(reconstruction, "pad_factor") else 2
    )
    fields_padded = _pad_field(fields, pad_factor)

    # modified Np and Lp with the pad factor
    wavelength = reconstruction.wavelength
    Np = reconstruction.Np * pad_factor
    Lp = Np * reconstruction.dxp

    # specifying z1 (aspw) and z2 (Fresnel) virtual distance for relaxing
    # sampling requirements. Similar issue as the NOTE on pad_factor above.
    z1 = reconstruction.zo if z is None else z

    # off-axis theta tuple in degrees and the calculated sines
    theta = _to_tuple(
        reconstruction.theta if hasattr(reconstruction, "theta") else None
    )
    theta_x, theta_y = theta
    sx, sy = jnp.sin(jnp.radians(theta_x)), jnp.sin(jnp.radians(theta_y))

    # prefactor_z for relaxing sampling, defaults to 1 / sqrt(1-sx**2-sy**2)
    prefactor_z = (
        reconstruction.prefactor_z
        if hasattr(reconstruction, "prefactor_z")
        else 1 / jnp.sqrt(1 - sx**2 - sy**2)
    )

    # z2 virtual distance (Fresnel) for relaxing sampling requirements
    z2 = float(prefactor_z * z1)

    # modify real-space resolution if required, however preferably kept at 1.0 (diffraction-limited)
    prefactor_dxp = (
        reconstruction.prefactor_dxp
        if hasattr(reconstruction, "prefactor_dxp")
        else 1.0
    )

    # probe FOV adjusted with the pad factor
    Ld = reconstruction.Ld * pad_factor
    reconstruction.dxp = float(prefactor_dxp * wavelength * z2 / Ld)

    # quadratic phase Q2 (currently zo, but this can be z2 and z1 separated)
    dxp = float(reconstruction.dxp)

    quad_phase_Q1 = __make_quad_phase(
        z2, wavelength, Np, dxp
    )

    # precompensated transfer function
    H_precomp = __make_transferfunction_sas(params, reconstruction, Np, Lp, z1, z2)

    if with_quad_phase_Q2:
        # quadratic phase Q2 (mostly okay to ignore it!)
        dxq = wavelength * z1 / Lp
        k = 2 * jnp.pi / wavelength
        x_q = jnp.linspace(-Np / 2, Np / 2, int(Np)) * dxq
        Xq, Yq = jnp.meshgrid(x_q, x_q)

        quad_phase_Q2 = jnp.exp(1j * k * z1) * jnp.exp(
            1.0j * k / (2 * z1) * (Xq**2 + Yq**2)
        )

    return_dict = {
        "fields_padded": fields_padded,
        "H_precomp": H_precomp,
        "quad_phase_Q1": quad_phase_Q1,
        "quad_phase_Q2": (
            quad_phase_Q2 if with_quad_phase_Q2 else jnp.ones_like(fields_padded)
        ),
        "pad_factor": pad_factor,
    }

    return return_dict


def propagate_sas(
    fields,
    params: Params,
    reconstruction: Reconstruction,
    z: float = None,
    with_quad_phase_Q2: bool = False,
):
    """
    Implementation of "scalable angular spectrum (SAS) propagation, Heintzmann et. al, 2024". This implementation also
    incorporates off-axis field shift by specifying the illumination angle `reconstruction.theta`.

    Parameters
    ----------
    fields: jnp.ndarray
        Field to propagate
    params: Params
        Instance of the Params class
    reconstruction: Reconstruction
        Instance of the Reconstruction class.
    z: float
        Propagation distance
    with_quad_phase_Q2: bool
        If True, the quadratic phase Q2 is applied

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field
    """

    interfaced_dict = _interface_sas(
        fields,
        params,
        reconstruction,
        z,
        with_quad_phase_Q2,
    )

    fields_padded = interfaced_dict["fields_padded"]
    H_precomp = interfaced_dict["H_precomp"]
    quad_phase_Q1 = interfaced_dict["quad_phase_Q1"]
    quad_phase_Q2 = interfaced_dict["quad_phase_Q2"]
    pad_factor = interfaced_dict["pad_factor"]

    # forward field propagation
    psi_precomp = ifft2c(H_precomp * fft2c(fields_padded))
    prop_fields = fft2c(quad_phase_Q1 * psi_precomp) * quad_phase_Q2

    # crop the field by the given pad-factor (default: 2)
    prop_fields_unpadded = _unpad_field(prop_fields, pad_factor)

    return reconstruction.esw, prop_fields_unpadded


def propagate_sas_inv(
    fields,
    params: Params,
    reconstruction: Reconstruction,
    z: float = None,
    with_quad_phase_Q2: bool = False,
):
    """
    Backward propagation (invertion) of the scalable angular spectrum (SAS) progator.

    Parameters
    ----------
    fields: jnp.ndarray
        Field to propagate
    params: Params
        Instance of the Params class
    reconstruction: Reconstruction
        Instance of the Reconstruction class.
    z: float
        Propagation distance
    with_quad_phase_Q2: bool
        If True, the quadratic phase Q2 is applied

    Returns
    -------
    reconstruction.esw, propagated field:
        Exit surface wave and the propagated field
    """

    interfaced_dict = _interface_sas(
        fields,
        params,
        reconstruction,
        z,
        with_quad_phase_Q2,
    )

    fields_padded = interfaced_dict["fields_padded"]
    H_precomp = interfaced_dict["H_precomp"]
    quad_phase_Q1 = interfaced_dict["quad_phase_Q1"]
    quad_phase_Q2 = interfaced_dict["quad_phase_Q2"]
    pad_factor = interfaced_dict["pad_factor"]

    # conjugates for the backward direction
    Q1_conj = jnp.conj(quad_phase_Q1)
    Q2_conj = jnp.conj(quad_phase_Q2)
    H_precomp_conj = jnp.conj(H_precomp)

    # backward field propagation
    psi_precomp = H_precomp_conj * fft2c(Q1_conj * ifft2c(Q2_conj * fields_padded))
    prop_fields = ifft2c(psi_precomp)

    # crop the field by the given pad-factor (default: 2)
    prop_fields_unpadded = _unpad_field(prop_fields, pad_factor)

    return reconstruction.esw, prop_fields_unpadded
