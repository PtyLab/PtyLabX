"""Tests for PtyLabX.Engines._jit_kernels — shared JIT-compiled update functions."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from PtyLabX.Engines._jit_kernels import (
    epie_object_update,
    epie_object_update_tv,
    epie_probe_update,
    momentum_step,
    mpie_object_update,
    mpie_probe_update,
    qnewton_object_update,
    qnewton_probe_update,
)


def _make_arrays(seed=0):
    """Create standard test arrays: (objectPatch, probe, DELTA) with 6D shapes."""
    rng = np.random.default_rng(seed)
    shape = (1, 1, 1, 1, 32, 32)  # nlambda, nosm, npsm, nslice, Np, Np
    objectPatch = jnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)
    probe = jnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)
    DELTA = jnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)
    return objectPatch, probe, DELTA


class TestEpieKernels:
    """Tests for ePIE-style update kernels."""

    def test_epie_object_update_zero_delta(self):
        """With zero DELTA, object patch should be unchanged."""
        objectPatch, probe, _ = _make_arrays()
        DELTA = jnp.zeros_like(objectPatch)
        result = epie_object_update(objectPatch, probe, DELTA, 0.25)
        assert_allclose(np.asarray(result), np.asarray(objectPatch), atol=1e-6)

    def test_epie_object_update_shape(self):
        """Output shape should match input objectPatch shape."""
        objectPatch, probe, DELTA = _make_arrays()
        result = epie_object_update(objectPatch, probe, DELTA, 0.25)
        assert result.shape == objectPatch.shape

    def test_epie_probe_update_shape(self):
        """Output shape should match input probe shape."""
        objectPatch, probe, DELTA = _make_arrays()
        result = epie_probe_update(probe, objectPatch, DELTA, 0.25)
        assert result.shape == probe.shape

    def test_epie_probe_update_zero_delta(self):
        """With zero DELTA, probe should be unchanged."""
        objectPatch, probe, _ = _make_arrays()
        DELTA = jnp.zeros_like(probe)
        result = epie_probe_update(probe, objectPatch, DELTA, 0.25)
        assert_allclose(np.asarray(result), np.asarray(probe), atol=1e-6)


class TestMpieKernels:
    """Tests for mPIE-style update kernels."""

    def test_mpie_object_update_shape(self):
        objectPatch, probe, DELTA = _make_arrays()
        result = mpie_object_update(objectPatch, probe, DELTA, 0.25, 0.1)
        assert result.shape == objectPatch.shape

    def test_mpie_object_update_fpm_mode(self):
        """FPM mode should produce different result than CPM mode."""
        objectPatch, probe, DELTA = _make_arrays()
        cpm = mpie_object_update(objectPatch, probe, DELTA, 0.25, 0.1, fpm_mode=False)
        fpm = mpie_object_update(objectPatch, probe, DELTA, 0.25, 0.1, fpm_mode=True)
        # They should differ
        assert not jnp.allclose(cpm, fpm)

    def test_mpie_probe_update_shape(self):
        objectPatch, probe, DELTA = _make_arrays()
        result = mpie_probe_update(probe, objectPatch, DELTA, 0.25, 0.1, 1.0)
        assert result.shape == probe.shape


class TestQNewtonKernels:
    """Tests for quasi-Newton update kernels."""

    def test_qnewton_object_update_shape(self):
        objectPatch, probe, DELTA = _make_arrays()
        result = qnewton_object_update(objectPatch, probe, DELTA, 1.0, 1.0)
        assert result.shape == objectPatch.shape

    def test_qnewton_probe_update_shape(self):
        objectPatch, probe, DELTA = _make_arrays()
        result = qnewton_probe_update(probe, objectPatch, DELTA, 1.0, 1.0)
        assert result.shape == probe.shape

    def test_qnewton_high_regularization_limits_update(self):
        """High regularization should make the update small."""
        objectPatch, probe, DELTA = _make_arrays()
        low_reg = qnewton_object_update(objectPatch, probe, DELTA, 1.0, 0.01)
        high_reg = qnewton_object_update(objectPatch, probe, DELTA, 1.0, 1000.0)
        diff_low = float(jnp.max(jnp.abs(low_reg - objectPatch)))
        diff_high = float(jnp.max(jnp.abs(high_reg - objectPatch)))
        assert diff_high < diff_low


class TestMomentumStep:
    """Tests for momentum gradient update."""

    def test_momentum_zero_friction(self):
        """With zero friction, momentum should equal the gradient."""
        rng = np.random.default_rng(5)
        shape = (1, 1, 1, 1, 16, 16)
        current = jnp.array(rng.standard_normal(shape), dtype=jnp.float32)
        buffer = jnp.array(rng.standard_normal(shape), dtype=jnp.float32)
        momentum = jnp.zeros(shape, dtype=jnp.float32)
        new_current, new_momentum, new_buffer = momentum_step(current, buffer, momentum, 0.0, 0.3)
        expected_gradient = buffer - current
        assert_allclose(np.asarray(new_momentum), np.asarray(expected_gradient), atol=1e-6)

    def test_momentum_no_change_when_converged(self):
        """When buffer equals current, gradient is zero and momentum decays."""
        shape = (1, 1, 1, 1, 16, 16)
        current = jnp.ones(shape, dtype=jnp.float32)
        buffer = current.copy()
        momentum = jnp.zeros(shape, dtype=jnp.float32)
        new_current, new_momentum, _ = momentum_step(current, buffer, momentum, 0.7, 0.3)
        assert_allclose(np.asarray(new_current), np.asarray(current), atol=1e-6)


class TestTVKernels:
    """Tests for TV-regularized update kernels."""

    def test_epie_tv_update_includes_regularization(self):
        """TV update should differ from standard ePIE update."""
        objectPatch, probe, DELTA = _make_arrays()
        standard = epie_object_update(objectPatch, probe, DELTA, 0.25)
        tv = epie_object_update_tv(objectPatch, probe, DELTA, 0.25, 0.01)
        # They should differ due to TV term
        assert not jnp.allclose(standard, tv)

    def test_epie_tv_update_shape(self):
        objectPatch, probe, DELTA = _make_arrays()
        result = epie_object_update_tv(objectPatch, probe, DELTA, 0.25, 0.01)
        assert result.shape == objectPatch.shape


class TestJITCompilation:
    """Verify that kernels are actually JIT-compiled."""

    def test_all_kernels_are_jit_wrapped(self):
        """All exported kernels should be JIT-wrapped functions."""
        from PtyLabX.Engines._jit_kernels import (
            _epie_object_update_jit,
            _epie_probe_update_jit,
            _mpie_object_update_jit,
            _mpie_probe_update_jit,
            _momentum_step_jit,
            _qnewton_object_update_jit,
            _qnewton_probe_update_jit,
            _epie_object_update_tv_jit,
        )

        kernels = [
            _epie_object_update_jit,
            _epie_probe_update_jit,
            _mpie_object_update_jit,
            _mpie_probe_update_jit,
            _qnewton_object_update_jit,
            _qnewton_probe_update_jit,
            _momentum_step_jit,
            _epie_object_update_tv_jit,
        ]
        for kernel in kernels:
            # JIT-wrapped functions are instances of jax stages
            assert hasattr(kernel, "lower") or hasattr(kernel, "_fun"), f"{kernel.__name__} is not JIT-wrapped"
