"""Shared pytest fixtures for PtyLabX test suite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="session")
def jax_key():
    """Provide a JAX PRNG key for reproducible random tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def random_complex_field():
    """Create a random complex field of a given shape."""

    def _make(shape=(64, 64), seed=0):
        rng = np.random.default_rng(seed)
        return jnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)

    return _make


@pytest.fixture
def random_real_field():
    """Create a random real field of a given shape."""

    def _make(shape=(64, 64), seed=0):
        rng = np.random.default_rng(seed)
        return jnp.array(rng.standard_normal(shape), dtype=jnp.float32)

    return _make
