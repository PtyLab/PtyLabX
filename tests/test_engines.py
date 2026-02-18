"""Integration tests for individual reconstruction engines.

Tests verify that engines can be instantiated and that their update methods
produce outputs with correct shapes and properties when using the shared JIT kernels.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from PtyLabX.Engines._jit_kernels import (
    epie_object_update,
    epie_probe_update,
    momentum_step,
    mpie_object_update,
    mpie_probe_update,
    qnewton_object_update,
    qnewton_probe_update,
)

SIMU_DATA = os.path.join(os.path.dirname(__file__), "..", "example_data", "simu.hdf5")
has_simu_data = os.path.exists(SIMU_DATA)
requires_simu_data = pytest.mark.skipif(not has_simu_data, reason="Simulation data not available")


@pytest.fixture(scope="module")
def simulation_cpm():
    """Load simulation data once for all engine tests."""
    import PtyLabX

    return PtyLabX.easyInitialize("example:simulation_cpm", operationMode="CPM")


@pytest.fixture(scope="module")
def cpm_components(simulation_cpm):
    experimentalData, reconstruction, params, monitor, engine = simulation_cpm
    return experimentalData, reconstruction, params, monitor


class TestEngineUpdates:
    """Test that engine update methods produce correct outputs via shared kernels."""

    @pytest.fixture
    def patch_data(self):
        """Create random object patch, probe, and DELTA for testing."""
        rng = np.random.default_rng(42)
        shape = (1, 1, 1, 1, 64, 64)
        objectPatch = jnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)
        probe = jnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)
        DELTA = jnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)
        return objectPatch, probe, DELTA

    def test_epie_object_update_shape(self, patch_data):
        objectPatch, probe, DELTA = patch_data
        result = epie_object_update(objectPatch, probe, DELTA, 0.25)
        assert result.shape == objectPatch.shape

    def test_epie_probe_update_shape(self, patch_data):
        objectPatch, probe, DELTA = patch_data
        result = epie_probe_update(probe, objectPatch, DELTA, 0.25)
        assert result.shape == probe.shape

    def test_mpie_object_update_shape(self, patch_data):
        objectPatch, probe, DELTA = patch_data
        result = mpie_object_update(objectPatch, probe, DELTA, 0.25, 0.1)
        assert result.shape == objectPatch.shape

    def test_mpie_probe_update_shape(self, patch_data):
        objectPatch, probe, DELTA = patch_data
        result = mpie_probe_update(probe, objectPatch, DELTA, 0.25, 0.1, 1.0)
        assert result.shape == probe.shape

    def test_qnewton_object_update_shape(self, patch_data):
        objectPatch, probe, DELTA = patch_data
        result = qnewton_object_update(objectPatch, probe, DELTA, 1.0, 1.0)
        assert result.shape == objectPatch.shape

    def test_qnewton_probe_update_shape(self, patch_data):
        objectPatch, probe, DELTA = patch_data
        result = qnewton_probe_update(probe, objectPatch, DELTA, 1.0, 1.0)
        assert result.shape == probe.shape

    def test_momentum_step_returns_triple(self, patch_data):
        objectPatch, probe, DELTA = patch_data
        current, momentum, buffer = momentum_step(objectPatch, objectPatch, jnp.zeros_like(objectPatch), 0.7, 0.3)
        assert current.shape == objectPatch.shape
        assert momentum.shape == objectPatch.shape
        assert buffer.shape == objectPatch.shape

    def test_zero_delta_no_object_change(self, patch_data):
        """With zero DELTA, object should not change."""
        objectPatch, probe, _ = patch_data
        DELTA_zero = jnp.zeros_like(objectPatch)
        result = epie_object_update(objectPatch, probe, DELTA_zero, 0.25)
        np.testing.assert_allclose(np.asarray(result), np.asarray(objectPatch), atol=1e-6)

    def test_zero_delta_no_probe_change(self, patch_data):
        """With zero DELTA, probe should not change."""
        objectPatch, probe, _ = patch_data
        DELTA_zero = jnp.zeros_like(probe)
        result = epie_probe_update(probe, objectPatch, DELTA_zero, 0.25)
        np.testing.assert_allclose(np.asarray(result), np.asarray(probe), atol=1e-6)

    def test_mpie_fpm_mode_differs_from_cpm(self, patch_data):
        """FPM mode should produce different results than CPM mode."""
        objectPatch, probe, DELTA = patch_data
        result_cpm = mpie_object_update(objectPatch, probe, DELTA, 0.25, 0.1, fpm_mode=False)
        result_fpm = mpie_object_update(objectPatch, probe, DELTA, 0.25, 0.1, fpm_mode=True)
        assert not jnp.allclose(result_cpm, result_fpm)


@requires_simu_data
class TestEngineInstantiation:
    """Test that all engines can be instantiated without errors."""

    def test_epie_init(self, cpm_components):
        from PtyLabX.Engines.ePIE import ePIE

        ed, recon, params, monitor = cpm_components
        engine = ePIE(recon, ed, params, monitor)
        assert engine is not None
        assert hasattr(engine, "betaObject")
        assert hasattr(engine, "betaProbe")

    def test_mpie_init(self, cpm_components):
        from PtyLabX.Engines.mPIE import mPIE

        ed, recon, params, monitor = cpm_components
        engine = mPIE(recon, ed, params, monitor)
        assert engine is not None
        assert hasattr(engine, "frictionM")
        assert hasattr(engine, "feedbackM")

    def test_qnewton_init(self, cpm_components):
        from PtyLabX.Engines.qNewton import qNewton

        ed, recon, params, monitor = cpm_components
        engine = qNewton(recon, ed, params, monitor)
        assert engine is not None
        assert hasattr(engine, "regObject")
        assert hasattr(engine, "regProbe")

    def test_pcpie_init(self, cpm_components):
        from PtyLabX.Engines.pcPIE import pcPIE

        ed, recon, params, monitor = cpm_components
        engine = pcPIE(recon, ed, params, monitor)
        assert engine is not None

    def test_epie_tv_init(self, cpm_components):
        from PtyLabX.Engines.ePIE_TV import ePIE_TV

        ed, recon, params, monitor = cpm_components
        engine = ePIE_TV(recon, ed, params, monitor)
        assert engine is not None


@requires_simu_data
class TestEngineOneIteration:
    """Test that engines can run a single position update without crashing."""

    def test_epie_single_position_update(self, cpm_components):
        from PtyLabX.Engines.ePIE import ePIE

        ed, recon, params, monitor = cpm_components
        engine = ePIE(recon, ed, params, monitor)
        engine._prepareReconstruction()

        row, col = recon.positions[0]
        sy = slice(row, row + recon.Np)
        sx = slice(col, col + recon.Np)
        objectPatch = recon.object[..., sy, sx].copy()

        recon.esw = objectPatch * recon.probe
        engine.intensityProjection(0)
        DELTA = recon.eswUpdate - recon.esw

        updated_patch = engine.objectPatchUpdate(objectPatch, DELTA)
        assert updated_patch.shape == objectPatch.shape
        assert jnp.isfinite(updated_patch).all()

        updated_probe = engine.probeUpdate(objectPatch, DELTA)
        assert updated_probe.shape == recon.probe.shape
        assert jnp.isfinite(updated_probe).all()

    def test_qnewton_single_position_update(self, cpm_components):
        from PtyLabX.Engines.qNewton import qNewton

        ed, recon, params, monitor = cpm_components
        engine = qNewton(recon, ed, params, monitor)
        engine._prepareReconstruction()

        row, col = recon.positions[0]
        sy = slice(row, row + recon.Np)
        sx = slice(col, col + recon.Np)
        objectPatch = recon.object[..., sy, sx].copy()

        recon.esw = objectPatch * recon.probe
        engine.intensityProjection(0)
        DELTA = recon.eswUpdate - recon.esw

        updated_patch = engine.objectPatchUpdate(objectPatch, DELTA)
        assert updated_patch.shape == objectPatch.shape
        assert jnp.isfinite(updated_patch).all()
