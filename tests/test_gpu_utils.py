"""Tests for PtyLabX.utils.gpuUtils — JAX array conversion utilities."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from PtyLabX.utils.gpuUtils import asJaxArray, asNumpyArray, check_jax_backend


def test_check_jax_backend():
    """Backend should be one of cpu, gpu, tpu."""
    backend = check_jax_backend(verbose=False)
    assert backend in ("cpu", "gpu", "tpu")


def test_asJaxArray_real():
    """Real numpy array should convert to float32 JAX array."""
    arr = np.array([1.0, 2.0, 3.0])
    jax_arr = asJaxArray(arr)
    assert jax_arr.dtype == jnp.float32
    assert_allclose(np.asarray(jax_arr), arr, atol=1e-6)


def test_asJaxArray_complex():
    """Complex numpy array should convert to complex64 JAX array."""
    arr = np.array([1.0 + 2j, 3.0 + 4j])
    jax_arr = asJaxArray(arr)
    assert jax_arr.dtype == jnp.complex64
    assert_allclose(np.asarray(jax_arr), arr, atol=1e-6)


def test_asNumpyArray_roundtrip():
    """JAX -> numpy -> JAX should preserve values."""
    jax_arr = jnp.array([1.0, 2.0, 3.0])
    np_arr = asNumpyArray(jax_arr)
    assert isinstance(np_arr, np.ndarray)
    assert_allclose(np_arr, np.array([1.0, 2.0, 3.0]), atol=1e-6)
