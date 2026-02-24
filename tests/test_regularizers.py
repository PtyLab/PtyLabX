"""Tests for PtyLabX.Regularizers — TV, grad_TV, std, metric functions."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from PtyLabX.Regularizers import TV, _TV_jit, grad_TV, min_std, std


class TestTV:
    """Tests for Total Variation computation."""

    def test_tv_constant_field_near_zero(self):
        """TV of a constant field should be close to zero (only aleph contribution)."""
        field = jnp.ones((64, 64), dtype=jnp.complex64)
        tv_val = TV(field)
        # Gradient is zero everywhere, only aleph remains under sqrt
        assert tv_val > 0  # aleph contribution
        assert tv_val < 200  # but small (aleph contributes sqrt(aleph) per pixel)

    def test_tv_noisy_field_larger(self):
        """TV of a noisy field should be larger than a smooth field."""
        rng = np.random.default_rng(0)
        smooth = jnp.ones((64, 64), dtype=jnp.complex64)
        noisy = jnp.array(rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64)), dtype=jnp.complex64)
        assert TV(noisy) > TV(smooth)

    def test_tv_bug_fix_regression(self):
        """Regression test: grad_y should use axis=-2, not axis=-1 for both rolls.

        The old code had:
            grad_y = jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-1)  # BUG
        Fixed to:
            grad_y = jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-2)  # CORRECT
        """
        # Create a field with gradient only along axis=-2 (rows)
        field = jnp.arange(64, dtype=jnp.float32).reshape(64, 1) * jnp.ones((1, 64), dtype=jnp.float32)
        # Create a field with gradient only along axis=-1 (cols)
        field_t = field.T

        # Both should have similar TV since they have the same gradient magnitude
        # (just in different directions). With the bug, TV would differ.
        tv_rows = TV(field)
        tv_cols = TV(field_t)
        # They won't be exactly equal due to boundary effects from roll,
        # but should be in the same ballpark
        assert_allclose(tv_rows, tv_cols, rtol=0.1)

    def test_tv_jit_matches_wrapper(self):
        """JIT core and wrapper should give the same result."""
        rng = np.random.default_rng(1)
        field = jnp.array(rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32)), dtype=jnp.complex64)
        jit_val = float(np.asarray(_TV_jit(field)))
        wrapper_val = TV(field)
        assert_allclose(jit_val, wrapper_val, atol=1e-5)


class TestGradTV:
    """Tests for TV gradient computation."""

    def test_grad_tv_shape_preservation(self):
        """grad_TV should return same shape as input."""
        rng = np.random.default_rng(2)
        field = jnp.array(
            rng.standard_normal((1, 1, 1, 1, 32, 32)) + 1j * rng.standard_normal((1, 1, 1, 1, 32, 32)),
            dtype=jnp.complex64,
        )
        result = grad_TV(field)
        assert result.shape == field.shape

    def test_grad_tv_constant_field_near_zero(self):
        """Gradient of TV for a constant field should be near zero."""
        field = jnp.ones((1, 1, 1, 1, 32, 32), dtype=jnp.complex64)
        result = grad_TV(field)
        assert float(jnp.max(jnp.abs(result))) < 0.1


class TestStd:
    """Tests for std and min_std metrics."""

    def test_std_constant_zero(self):
        """Standard deviation of constant field is 0."""
        field = jnp.ones((32, 32))
        assert_allclose(std(field), 0.0, atol=1e-6)

    def test_min_std_negates(self):
        """min_std should return negative of std."""
        rng = np.random.default_rng(3)
        field = jnp.array(rng.standard_normal((32, 32)), dtype=jnp.float32)
        assert_allclose(min_std(field), -std(field), atol=1e-6)
