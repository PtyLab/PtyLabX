from collections.abc import Generator

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def complexexp(angle: jax.Array) -> jax.Array:
    """
    Faster way of implementing np.exp(1j*something_unitary)

    Parameters
    ----------
    angle: jax.Array
        Angle of the exponent

    Returns
    -------

    cos(angle) + 1j * sin(angle)

    """
    return jnp.cos(angle) + 1j * jnp.sin(angle)


def iterate_6d_fields(fields: jax.Array) -> Generator[tuple[int, ...], None, None]:
    """
    Iterate over the first four dimensions of a 6D field array. (nlambda, nosm, npsm, nslice, Np, Np)
    corresponding to multi-wavelengths, object modes, probe modes, multislice, diffraction pattern (2d)
    """
    for idx in np.ndindex(fields.shape[:4]):
        yield idx


def convolve2d(in1: jax.Array, in2: jax.Array, mode: str = "same") -> jax.Array:
    """2D convolution using FFT."""
    return _fft_convolve2d(in1, in2, mode=mode)


def gaussian2D(n: int, std: float) -> jax.Array:
    """Creates a 2D gaussian"""
    n = (n - 1) // 2
    x, y = jnp.ogrid[-n : n + 1, -n : n + 1]
    h = jnp.exp(-(x**2 + y**2) / (2 * std**2))
    mask = h < jnp.finfo(float).eps * jnp.max(h)
    h = h * (1 - mask)
    sumh = jnp.sum(h)
    h = jnp.where(sumh != 0, h / sumh, h)
    return h


def _fft_convolve2d(x: jax.Array, y: jax.Array, mode: str = "same") -> jax.Array:
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


def _centered(arr: jax.Array, newsize: tuple[int, ...]) -> jax.Array:
    # Return the center newsize portion of the array.
    newsize_arr = jnp.asarray(newsize)
    currsize = jnp.array(arr.shape)
    startind = (currsize - newsize_arr) // 2
    endind = startind + newsize_arr
    myslice = [slice(int(startind[k]), int(endind[k])) for k in range(len(endind))]
    return arr[tuple(myslice)]
