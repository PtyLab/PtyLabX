"""Tests for PtyLabX.Operators — wave propagation operators."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from PtyLabX.Operators.Operators import aspw, scaledASP, scaledASPinv, fresnelPropagator
from PtyLabX.Operators.propagator_utils import complexexp, convolve2d
from PtyLabX.utils.utils import circ


class TestASPW:
    """Tests for angular spectrum plane wave propagation."""

    def setup_method(self):
        self.dx = 5e-6
        N = 100
        x = jnp.arange(-N / 2, N / 2) * self.dx
        X, Y = jnp.meshgrid(x, x)
        self.E_in = circ(X, Y, N / 2 * self.dx).astype(jnp.complex64)
        self.wavelength = 600e-9
        self.z = 1e-4
        self.L = self.dx * N

    def test_aspw_zero_propagation(self):
        """Propagating z=0 should return the input field."""
        E_out, _ = aspw(self.E_in, 0, self.wavelength, self.L, is_FT=False)
        assert_allclose(np.asarray(jnp.abs(E_out)), np.asarray(jnp.abs(self.E_in)), atol=1e-5)

    def test_aspw_roundtrip(self):
        """Forward then backward propagation should recover magnitude."""
        E_1, _ = aspw(self.E_in, self.z, self.wavelength, self.L, is_FT=False)
        E_2, _ = aspw(E_1, -self.z, self.wavelength, self.L, is_FT=False)
        assert_allclose(np.asarray(jnp.abs(E_2)), np.asarray(jnp.abs(self.E_in)), atol=1e-4)

    def test_aspw_energy_conservation(self):
        """Total energy should be preserved through propagation."""
        energy_in = float(jnp.sum(jnp.abs(self.E_in) ** 2))
        E_out, _ = aspw(self.E_in, self.z, self.wavelength, self.L, is_FT=False)
        energy_out = float(jnp.sum(jnp.abs(E_out) ** 2))
        assert_allclose(energy_in, energy_out, rtol=1e-3)


class TestScaledASP:
    """Tests for scaled angular spectrum propagation."""

    def setup_method(self):
        self.dx = 5e-6
        N = 100
        x = jnp.arange(-N / 2, N / 2) * self.dx
        X, Y = jnp.meshgrid(x, x)
        self.E_in = circ(X, Y, N / 2 * self.dx).astype(jnp.complex64)
        self.wavelength = 600e-9
        self.z = 1e-4

    def test_scaledASP_roundtrip(self):
        """Forward then backward scaled ASP should recover magnitude."""
        E_1, _, _ = scaledASP(self.E_in, self.z, self.wavelength, self.dx, self.dx)
        E_2, _, _ = scaledASP(E_1, -self.z, self.wavelength, self.dx, self.dx)
        assert_allclose(np.asarray(jnp.abs(E_2)), np.asarray(jnp.abs(self.E_in)), atol=1e-4)


class TestComplexexp:
    """Tests for fast complex exponential."""

    def test_matches_exp(self):
        """complexexp(x) should match jnp.exp(1j*x)."""
        angles = jnp.linspace(-jnp.pi, jnp.pi, 100)
        result = complexexp(angles)
        expected = jnp.exp(1j * angles)
        assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-6)


class TestConvolve2d:
    """Tests for FFT-based 2D convolution."""

    def test_identity_kernel(self):
        """Convolving with a delta function should return the input."""
        x = jnp.ones((8, 8), dtype=jnp.float32)
        kernel = jnp.zeros((3, 3), dtype=jnp.float32).at[1, 1].set(1.0)
        result = convolve2d(x, kernel, mode="same")
        assert_allclose(np.asarray(jnp.abs(result)), np.asarray(x), atol=1e-5)
