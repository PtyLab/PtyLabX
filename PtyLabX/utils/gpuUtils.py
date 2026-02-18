import logging

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JAX")
logging.getLogger("numexpr").setLevel(logging.WARNING)


def check_jax_backend(verbose=True):
    """Report which JAX backend is active."""
    backend = jax.default_backend()
    if verbose:
        logger.info("JAX backend: %s", backend)
    return backend


def asNumpyArray(ary) -> np.ndarray:
    """Convert any array (JAX or numpy) to a numpy ndarray."""
    return np.asarray(ary)


def asJaxArray(field, dtype=None):
    """Convert a numpy array to a JAX array with automatic dtype selection."""
    if dtype is None:
        if np.isrealobj(field):
            dtype = jnp.float32
        elif np.iscomplexobj(field):
            dtype = jnp.complex64
    return jnp.array(field, dtype=dtype)
