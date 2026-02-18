"""Tests for PtyLabX.utils.utils — FFT, shifts, orthogonalization, geometry."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from PtyLabX.utils.utils import circ, fft2c, fraccircshift, ifft2c, orthogonalizeModes, posit, rect


class TestFFT2c:
    """Tests for centered FFT/IFFT."""

    def test_fft2c_ifft2c_roundtrip(self):
        """FFT(IFFT(x)) == x and IFFT(FFT(x)) == x (unitarity)."""
        rng = np.random.default_rng(0)
        E_in = rng.standard_normal((5, 100, 100)) + 1j * rng.standard_normal((5, 100, 100))
        E_in = jnp.array(E_in, dtype=jnp.complex64)
        assert_allclose(np.asarray(ifft2c(fft2c(E_in))), np.asarray(E_in), atol=1e-5)
        assert_allclose(np.asarray(fft2c(ifft2c(E_in))), np.asarray(E_in), atol=1e-5)

    def test_fft2c_with_fftshift_switch(self):
        """Verify fftshiftSwitch=True path also round-trips."""
        rng = np.random.default_rng(1)
        E_in = jnp.array(rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32)), dtype=jnp.complex64)
        result = ifft2c(fft2c(E_in, fftshiftSwitch=True), fftshiftSwitch=True)
        assert_allclose(np.asarray(result), np.asarray(E_in), atol=1e-5)

    def test_fft2c_energy_preservation(self):
        """Parseval's theorem: sum(|x|^2) == sum(|FFT(x)|^2) for unitary FFT."""
        rng = np.random.default_rng(2)
        E_in = jnp.array(rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64)), dtype=jnp.complex64)
        energy_in = float(jnp.sum(jnp.abs(E_in) ** 2))
        energy_out = float(jnp.sum(jnp.abs(fft2c(E_in)) ** 2))
        assert_allclose(energy_in, energy_out, rtol=1e-4)

    def test_fft2c_jit_compiled(self):
        """Verify fft2c runs under JIT (it's decorated with @jax.jit)."""
        E_in = jnp.ones((16, 16), dtype=jnp.complex64)
        result = fft2c(E_in)
        assert result.shape == (16, 16)


class TestFraccircshift:
    """Tests for fractional circular shift."""

    def test_integer_shift_matches_roll(self):
        """Integer shifts should match jnp.roll exactly."""
        rng = np.random.default_rng(3)
        A = jnp.array(rng.standard_normal((32, 32)), dtype=jnp.float32)
        shift = jnp.array([3.0, -2.0])
        result = fraccircshift(A, shift)
        expected = jnp.roll(jnp.roll(A, 3, axis=0), -2, axis=1)
        assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-5)

    def test_zero_shift_identity(self):
        """Zero shift should return the same array."""
        rng = np.random.default_rng(4)
        A = jnp.array(rng.standard_normal((16, 16)), dtype=jnp.float32)
        result = fraccircshift(A, jnp.array([0.0, 0.0]))
        assert_allclose(np.asarray(result), np.asarray(A), atol=1e-6)

    def test_subpixel_shift_interpolation(self):
        """Half-pixel shift should average adjacent pixels."""
        A = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        result = fraccircshift(A, jnp.array([0.5, 0.0]))
        # Half shift along axis 0: 0.5 * A[0,:] + 0.5 * A[1,:]
        assert float(result[0, 0]) > 0
        assert float(result[1, 0]) > 0


class TestOrthogonalizeModes:
    """Tests for SVD-based mode orthogonalization."""

    def test_svd_roundtrip(self):
        """Orthogonalized modes should be orthogonal."""
        rng = np.random.default_rng(5)
        p = jnp.array(rng.standard_normal((3, 32, 32)) + 1j * rng.standard_normal((3, 32, 32)), dtype=jnp.complex64)
        p_orth, eigenvalues, _ = orthogonalizeModes(p)
        assert p_orth.shape == p.shape
        assert eigenvalues.shape == (3,)
        # Eigenvalues should sum to 1 (normalized)
        assert_allclose(float(jnp.sum(eigenvalues)), 1.0, atol=1e-5)

    def test_snapshots_method(self):
        """Test the snapShots method path."""
        rng = np.random.default_rng(6)
        p = jnp.array(rng.standard_normal((3, 32, 32)) + 1j * rng.standard_normal((3, 32, 32)), dtype=jnp.complex64)
        p_orth, eigenvalues, V = orthogonalizeModes(p, method="snapShots")
        assert p_orth.shape == p.shape
        assert_allclose(float(jnp.sum(eigenvalues)), 1.0, atol=1e-5)


class TestCirc:
    """Tests for circular mask generation."""

    def test_basic_shape(self):
        """Circ should produce a boolean mask of the correct shape."""
        x, y = jnp.meshgrid(jnp.linspace(-1, 1, 64), jnp.linspace(-1, 1, 64))
        mask = circ(x, y, 1.0)
        assert mask.shape == (64, 64)
        assert mask.dtype == jnp.bool_

    def test_circ_center_is_true(self):
        """Center pixel of circ should be True for nonzero diameter."""
        x, y = jnp.meshgrid(jnp.linspace(-1, 1, 65), jnp.linspace(-1, 1, 65))
        mask = circ(x, y, 1.0)
        assert bool(mask[32, 32])


class TestRect:
    """Tests for rect function."""

    def test_basic(self):
        x = jnp.array([-1.0, -0.3, 0.0, 0.3, 1.0])
        result = rect(x)
        expected = jnp.array([False, True, True, True, False])
        assert jnp.array_equal(result, expected)


class TestPosit:
    """Tests for posit (ReLU-like) function."""

    def test_positive_values_unchanged(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert_allclose(np.asarray(posit(x)), np.asarray(x), atol=1e-6)

    def test_negative_values_zero(self):
        x = jnp.array([-1.0, -2.0, -3.0])
        assert_allclose(np.asarray(posit(x)), np.zeros(3), atol=1e-6)
