import functools
from typing import Callable, cast

import jax
import jax.numpy as jnp
import numpy as np


def _fft2c_impl(field: jax.Array, fftshiftSwitch: bool = False) -> jax.Array:
    if fftshiftSwitch:
        return jnp.fft.fft2(field, norm="ortho")
    else:
        axes = (-2, -1)
        return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(field, axes=axes), norm="ortho"), axes=axes)


_fft2c_jit: Callable[..., jax.Array] = cast(
    Callable[..., jax.Array],
    functools.partial(jax.jit, static_argnums=(1,))(_fft2c_impl),
)


def fft2c(field: jax.Array, fftshiftSwitch: bool = False) -> jax.Array:
    """
    performs 2 - dimensional unitary Fourier transformation, where energy is preserved sum( abs(g)**2 ) == sum( abs(fft2c(g))**2 )
    if g is two - dimensional, fft2c(g) yields the 2D DFT of g
    if g is multi - dimensional, fft2c(g) yields the 2D DFT of g along the last two axes
    :param array:
    :return:
    """
    return _fft2c_jit(field, fftshiftSwitch)


def _ifft2c_impl(field: jax.Array, fftshiftSwitch: bool = False) -> jax.Array:
    if fftshiftSwitch:
        return jnp.fft.ifft2(field, norm="ortho")
    else:
        axes = (-2, -1)
        return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(field, axes=axes), norm="ortho"), axes=axes)


_ifft2c_jit: Callable[..., jax.Array] = cast(
    Callable[..., jax.Array],
    functools.partial(jax.jit, static_argnums=(1,))(_ifft2c_impl),
)


def ifft2c(field: jax.Array, fftshiftSwitch: bool = False) -> jax.Array:
    """
    performs 2 - dimensional inverse Fourier transformation, where energy is preserved sum( abs(G)**2 ) == sum( abs(fft2c(g))**2 )
    if G is two - dimensional, fft2c(G) yields the 2D iDFT of G
    if G is multi - dimensional, fft2c(G) yields the 2D iDFT of G along the last two axes
    :param array:
    :return:
    """
    return _ifft2c_jit(field, fftshiftSwitch)


@jax.jit
def circ(x: jax.Array, y: jax.Array, D: float) -> jax.Array:
    """
    generate a binary array containing a circle on a 2D grid
    :param x: x coordinate array, may be 1D/2D (supports broadcasting from ogrid)
    :param y: y coordinate array, may be 1D/2D (supports broadcasting from ogrid)
    :param D: diameter
    :return: a binary 2D array
    """
    circle = (x**2 + y**2) < (D / 2) ** 2
    return circle


@jax.jit
def rect(arr: jax.Array, threshold: float = 0.5) -> jax.Array:
    """
    generate a binary array containing a rectangle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param threshold: threshold value to binarilize the input array, default value 0.5
    :return: a binary array
    """
    arr = abs(arr)
    return arr < threshold


@jax.jit
def posit(x: jax.Array) -> jax.Array:
    """
    returns 0 when x negative
    """
    r = (x + abs(x)) / 2
    # r[r<0]=0 #todo check which way is faster
    return r


@jax.jit
def fraccircshift(A: jax.Array, shiftsize: jax.Array) -> jax.Array:
    """
    fraccircshift expands jnp.roll to fractional shifts values, using linear interpolation.
    :param A: ndarray
    :param shiftsize: shift size in each dimension of A, len(shiftsize)==A.ndim.
    """
    integer = jnp.floor(shiftsize).astype(int)
    fraction = shiftsize - integer

    def _shift_along_axis(A, axis, int_shift, frac_shift):
        return (1 - frac_shift) * jnp.roll(A, int_shift, axis=axis) + frac_shift * jnp.roll(A, int_shift + 1, axis=axis)

    A = _shift_along_axis(A, 0, integer[0], fraction[0])
    A = _shift_along_axis(A, 1, integer[1], fraction[1])
    return A


def cart2pol(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform Cartesian to polar coordinates
    :param x:
    :param y:
    :return:
    """
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    return th, r


def gaussian2D(n: int, std: float) -> np.ndarray:
    # create the grid of (x,y) values
    n = (n - 1) // 2
    x, y = np.ogrid[-n : n + 1, -n : n + 1]
    # analytic function
    h = np.exp(-(x**2 + y**2) / (2 * std**2))
    # truncate very small values to zero
    mask = h < np.finfo(float).eps * np.max(h)
    h *= 1 - mask
    # normalize filter to unit L1 energy
    sumh = np.sum(h)
    if sumh != 0:
        h = h / sumh
    return h


def _orthogonalizeModes_impl(p: jax.Array, method: str | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Imposes orthogonality through singular value decomposition
    :return:
    """
    if method == "snapShots":
        p2D = p.reshape(p.shape[0], p.shape[1] * p.shape[2])
        w, V = jnp.linalg.eigh(jnp.dot(p2D.conj(), jnp.transpose(p2D)))
        s = jnp.sqrt(w).real
        U = jnp.dot(
            jnp.dot(jnp.transpose(p2D), V),
            jnp.linalg.inv(jnp.diag(s) + 1e-17 * jnp.eye(len(s))),
        )
        indices = jnp.argsort(s)[::-1]
        s = s[indices]
        U = U[:, indices]
        V = V[:, indices]
        p = jnp.transpose(jnp.dot(U, jnp.diag(s))).reshape(p.shape[0], p.shape[1], p.shape[2])
        normalizedEigenvalues = s**2 / jnp.sum(s**2)
        V = jnp.transpose(V)
        return p, normalizedEigenvalues, V

    else:
        U, s, V = cast(
            tuple[jax.Array, jax.Array, jax.Array],
            jnp.linalg.svd(p.reshape(p.shape[0], p.shape[1] * p.shape[2]), False),  # ty: ignore[too-many-positional-arguments,invalid-argument-type]
        )
        p = jnp.dot(jnp.diag(s), V).reshape(p.shape[0], p.shape[1], p.shape[2])
        normalizedEigenvalues = s**2 / jnp.sum(s**2)

        return p, normalizedEigenvalues, U.T.conj()


_orthogonalizeModes_jit: Callable[..., tuple[jax.Array, jax.Array, jax.Array]] = cast(
    Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    functools.partial(jax.jit, static_argnames=("method",))(_orthogonalizeModes_impl),
)


def orthogonalizeModes(p: jax.Array, method: str | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Imposes orthogonality through singular value decomposition
    :return:
    """
    return _orthogonalizeModes_jit(p, method)


def zernikeAberrations(Xp: np.ndarray, Yp: np.ndarray, D: float, z_coeff: np.ndarray) -> np.ndarray:
    """
    Compute the first 19 Zernike aberrations based on Zernike polynomials
    Based on https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices

    Xp,Yp - meshgrid coordinates
    D - radius within which to generate the zernike aberrations
    z_coeff - 19 element long list containing coefficients.

    minimal example:

        import matplotlib.pyplot as plt
        import numpy as np

        # create the circular dimensions which will define the size
        # of a unit circle used for zernike aberration calculations
        Xp,Yp = np.mgrid[-128:128, -128:128]
        D = 128

        # Get defocus aberration (4th index)
        z_coeff = np.zeros(19)
        z_coeff[4] = 3
        Z = zernikeAberrations(Xp,Yp,D,z_coeff)

        # plot the polynoial
        plt.figure(1)
        plt.imshow(np.angle(Z))
        plt.show()
    """

    aperture = circ(Xp, Yp, D)
    angle = np.double(np.arctan2(Yp, Xp)) * aperture
    p = np.double(np.hypot(Xp, Yp)) * aperture
    p = p / np.max(p)

    Z = dict()
    Z[0] = z_coeff[0]  # pistom
    Z[1] = z_coeff[1] * 4 ** (1 / 2.0) * p * np.cos(angle)
    # tip
    Z[2] = z_coeff[2] * 4 ** (1 / 2.0) * p * np.sin(angle)
    # tilt
    Z[3] = z_coeff[3] * 3 ** (1 / 2.0) * (2 * p**2 - 1)
    # defocus
    Z[4] = z_coeff[4] * 6 ** (1 / 2.0) * (p**2) * np.sin(2 * angle)
    # astigmatism
    Z[5] = z_coeff[5] * 6 ** (1 / 2.0) * (p**2) * np.cos(2 * angle)
    # astigmatism
    Z[6] = z_coeff[6] * 8 ** (1 / 2.0) * (3 * p**3 - 2 * p) * np.sin(angle)
    # coma
    Z[7] = z_coeff[7] * 8 ** (1 / 2.0) * (3 * p**3 - 2 * p) * np.cos(angle)
    # coma
    Z[8] = z_coeff[8] * 8 ** (1 / 2.0) * (p**3) * np.sin(3 * angle)
    # trefoil
    Z[9] = z_coeff[9] * 8 ** (1 / 2.0) * (p**3) * np.cos(3 * angle)
    # trefoil
    Z[10] = z_coeff[10] * 5 ** (1 / 2.0) * (6 * p**4 - 6 * p**2 + 1)
    # spherical
    Z[11] = z_coeff[11] * 10 ** (1 / 2.0) * (4 * p**4 - 3 * p**2) * np.cos(2.0 * angle)
    # 2nd astigmatism
    Z[12] = z_coeff[12] * 10 ** (1 / 2.0) * (4 * p**4 - 3 * p**2) * np.sin(2.0 * angle)
    # 2nd astigmatism
    Z[13] = z_coeff[13] * 10 ** (1 / 2.0) * (p**4) * np.cos(4.0 * angle)
    Z[14] = z_coeff[14] * 10 ** (1 / 2.0) * (p**4) * np.sin(4.0 * angle)
    Z[15] = z_coeff[15] * 12 ** (1 / 2.0) * (10 * p**5 - 12 * p**3 + 3 * p) * np.cos(angle)
    Z[16] = z_coeff[16] * 12 ** (1 / 2) * (10 * p**5 - 12 * p**3 + 3 * p) * np.sin(angle)
    Z[17] = z_coeff[17] * 12 ** (1 / 2) * (5 * p**5 - 4 * p**3) * np.cos(3 * angle)
    Z[18] = z_coeff[18] * 12 ** (1 / 2) * (5 * p**5 - 4 * p**3) * np.sin(3 * angle)

    return aperture * np.exp(1j * np.sum(list(Z.values())))


def p2bin(im: np.ndarray, binningFactor: int) -> tuple[np.ndarray, np.ndarray | range, range]:
    """
    perform binning at a factor of power of 2, return binned image and the indices for before and after binning.
    :Params im: input image for binning
    :Params binningFactor: must be power of 2 in the current implementation
    :return:
    """
    M, N = im.shape
    if np.mod(binningFactor, 2) != 0 and binningFactor != 1:
        raise ValueError("binning factor needs to be a power of 2")
    if np.mod(M, binningFactor) != 0 or np.mod(N, binningFactor) != 0:
        raise ValueError("#rows and #columns of reference need to be divided by binningFactor!")

    if binningFactor != 1:
        for k in range(1, int(np.log2(binningFactor))):
            im_binned = bin2(im)
        im_binned_ind = range(im_binned.size)
        im_ind = np.arange(M * N).reshape(M // binningFactor, N, binningFactor)
        im_ind = np.stack(im_ind, axis=1).reshape(M, N)
    else:
        im_binned = im
        im_binned_ind = range(im_binned.size)
        im_ind = im_binned_ind
    return im_binned, im_ind, im_binned_ind


def bin2(X: np.ndarray) -> np.ndarray:
    """
    perform 2-by-2 binning.
    :Params X: input 2D image for binning
    return: Y: output 2D image after 2-by-2 binning
    """
    # simple 2-fold binning
    m, n = X.shape
    Y = np.sum(X.reshape(2, m // 2, 2, n // 2), axis=(0, 2))
    return Y
