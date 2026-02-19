from functools import lru_cache

import jax.numpy as jnp

cache_size = 30


@lru_cache(maxsize=cache_size)
def _make_quad_phase(zo, wavelength, Np, dxp):
    """
    Make a quadratic phase profile corresponding to distance zo at wavelength wl. The result is cached and can be
    called again with almost no time lost.
    :param wavelength:  wavelength in meters
    :param zo:
    :param Np:
    :param dxp:
    :return:
    """
    x_p = jnp.linspace(-Np / 2, Np / 2, int(Np)) * dxp
    Xp, Yp = jnp.meshgrid(x_p, x_p)

    quadraticPhase = jnp.exp(1.0j * jnp.pi / (wavelength * zo) * (Xp**2 + Yp**2))
    return quadraticPhase
