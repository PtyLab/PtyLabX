import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JAX")
logging.getLogger("numexpr").setLevel(logging.WARNING)


def check_backend(verbose: bool = True) -> tuple[str, list[jax.Device]]:
    """Report which JAX backend is active."""
    backend = jax.default_backend()
    devices = jax.devices()
    return backend, devices


def asNumpyArray(ary: ArrayLike) -> np.ndarray:
    """Convert any array (JAX or numpy) to a numpy ndarray."""
    return np.asarray(ary)


def asJaxArray(field: ArrayLike, dtype: type | None = None) -> jax.Array:
    """Convert a numpy array to a JAX array with automatic dtype selection."""
    if dtype is None:
        if np.isrealobj(field):
            dtype = jnp.float32
        elif np.iscomplexobj(field):
            dtype = jnp.complex64
    return jnp.array(field, dtype=dtype)
