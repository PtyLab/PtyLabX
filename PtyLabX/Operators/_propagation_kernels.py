from functools import lru_cache

import jax
import jax.numpy as jnp

cache_size = 5


@lru_cache(maxsize=cache_size)
def _make_quad_phase(zo: float, wavelength: float, Np: int, dxp: float) -> jax.Array:
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
    Xp, Yp = x_p.reshape(1, -1), x_p.reshape(-1, 1)

    quadraticPhase = jnp.exp(1.0j * jnp.pi / (wavelength * zo) * (Xp**2 + Yp**2))
    return quadraticPhase
