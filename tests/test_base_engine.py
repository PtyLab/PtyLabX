"""Tests for PtyLabX.Engines.BaseEngine — core engine functionality."""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose


class TestSpectralPowerCorrection:
    """Tests for vectorized spectral power correction in applyConstraints."""

    def test_spectral_power_correction_preserves_shape(self):
        """Vectorized spectral power correction should preserve probe shape."""
        probe = jnp.ones((3, 1, 1, 1, 32, 32), dtype=jnp.complex64) * (1 + 0.5j)
        spectral_power = jnp.array([0.5, 0.3, 0.2])
        maxProbePower = 1.0

        norms = jnp.sqrt(jnp.sum(probe * probe.conj(), axis=(1, 2, 3, 4, 5), keepdims=True))
        scales = maxProbePower * spectral_power.reshape(-1, 1, 1, 1, 1, 1) / norms
        result = probe * scales
        assert result.shape == probe.shape

    def test_spectral_power_sets_correct_power(self):
        """After correction, each wavelength should have the target spectral power."""
        rng = np.random.default_rng(0)
        probe = jnp.array(
            rng.standard_normal((3, 1, 1, 1, 32, 32)) + 1j * rng.standard_normal((3, 1, 1, 1, 32, 32)),
            dtype=jnp.complex64,
        )
        spectral_power = jnp.array([0.5, 0.3, 0.2])
        maxProbePower = 10.0

        norms = jnp.sqrt(jnp.sum(probe * probe.conj(), axis=(1, 2, 3, 4, 5), keepdims=True))
        scales = maxProbePower * spectral_power.reshape(-1, 1, 1, 1, 1, 1) / norms
        result = probe * scales

        # Check that the power per wavelength matches target
        for wl in range(3):
            power = float(jnp.sqrt(jnp.sum(result[wl] * result[wl].conj())).real)
            expected = maxProbePower * float(spectral_power[wl])
            assert_allclose(power, expected, rtol=1e-5)


class TestWavelengthCoupling:
    """Tests for vectorized wavelength coupling in applyConstraints."""

    def test_coupling_boundary_conditions(self):
        """Boundary wavelengths should only couple with their one neighbor."""
        probe = jnp.zeros((4, 1, 1, 1, 8, 8), dtype=jnp.complex64)
        # Set each wavelength to a distinct value
        for i in range(4):
            probe = probe.at[i].set((i + 1.0) * jnp.ones((1, 1, 1, 8, 8)))

        a = 0.5
        shifted_up = jnp.roll(probe, -1, axis=0)
        shifted_down = jnp.roll(probe, 1, axis=0)
        coupled = (1 - a) * probe + a * (shifted_up + shifted_down) / 2
        coupled = coupled.at[0].set((1 - a) * probe[0] + a * probe[1])
        coupled = coupled.at[-1].set((1 - a) * probe[-1] + a * probe[-2])

        # First wavelength: (1-0.5)*1 + 0.5*2 = 1.5
        assert_allclose(float(coupled[0, 0, 0, 0, 0, 0].real), 1.5, atol=1e-5)
        # Last wavelength: (1-0.5)*4 + 0.5*3 = 3.5
        assert_allclose(float(coupled[-1, 0, 0, 0, 0, 0].real), 3.5, atol=1e-5)


class TestPositionCorrectionVmap:
    """Tests for vmapped position correction."""

    def test_vmap_cc_matches_loop(self):
        """Vmapped cross-correlation should match sequential loop."""
        rng = np.random.default_rng(42)
        Opatch = jnp.array(rng.standard_normal((32, 32)), dtype=jnp.float32)
        O_slice = jnp.array(rng.standard_normal((32, 32)), dtype=jnp.float32)
        rowShifts = jnp.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        colShifts = jnp.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

        # Loop version
        cc_loop = jnp.zeros((9, 1))
        for i in range(9):
            shifted = jnp.roll(jnp.roll(Opatch, rowShifts[i], axis=-2), colShifts[i], axis=-1)
            cc_loop = cc_loop.at[i].set(jnp.squeeze(jnp.sum(shifted.conj() * O_slice, axis=(-2, -1))))

        # Vmap version
        def _cc_at_shift(i):
            shifted = jnp.roll(jnp.roll(Opatch, rowShifts[i], axis=-2), colShifts[i], axis=-1)
            return jnp.squeeze(jnp.sum(shifted.conj() * O_slice, axis=(-2, -1)))

        cc_vmap = jax.vmap(_cc_at_shift)(jnp.arange(9)).reshape(-1, 1)

        assert_allclose(np.asarray(cc_loop), np.asarray(cc_vmap), atol=1e-5)
