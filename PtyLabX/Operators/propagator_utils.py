
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def complexexp(angle):
    """
    Faster way of implementing np.exp(1j*something_unitary)

    Parameters
    ----------
    angle: np.ndarray
        Angle of the exponent

    Returns
    -------

    cos(angle) + 1j * sin(angle)

    """
    return jnp.cos(angle) + 1j * jnp.sin(angle)


def iterate_6d_fields(fields):
    """
    Iterate over the first four dimensions of a 6D field array. (nlambda, nosm, npsm, nslice, Np, Np)
    corresponding to multi-wavelengths, object modes, probe modes, multislice, diffraction pattern (2d)
    """
    for idx in np.ndindex(fields.shape[:4]):
        yield idx


def convolve2d(in1, in2, mode="same"):
    """2D convolution using FFT."""
    return _fft_convolve2d(in1, in2, mode=mode)


def gaussian2D(n, std):
    """Creates a 2D gaussian"""
    n = (n - 1) // 2
    x, y = jnp.meshgrid(jnp.arange(-n, n + 1), jnp.arange(-n, n + 1))
    h = jnp.exp(-(x**2 + y**2) / (2 * std**2))
    mask = h < jnp.finfo(float).eps * jnp.max(h)
    h = h * (1 - mask)
    sumh = jnp.sum(h)
    h = jnp.where(sumh != 0, h / sumh, h)
    return h


def _fft_convolve2d(x, y, mode="same"):
    """2D convolution using FFT."""
    s1 = x.shape
    s2 = y.shape
    size = s1[0] + s2[0] - 1, s1[1] + s2[1] - 1
    fx = jnp.fft.fft2(x, size)
    fy = jnp.fft.fft2(y, size)
    result = jnp.fft.ifft2(fx * fy)

    if mode == "same":
        return _centered(result, s1)
    elif mode == "valid":
        return _centered(result, (s1[0] - s2[0] + 1, s1[1] - s2[1] + 1))
    else:  # 'full'
        return result


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = jnp.asarray(newsize)
    currsize = jnp.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(int(startind[k]), int(endind[k])) for k in range(len(endind))]
    return arr[tuple(myslice)]
