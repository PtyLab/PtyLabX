"""Tests for Reconstruction module.

Migrated from PtyLabX/Reconstruction/test/test_optimizable.py
"""

import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

SIMU_DATA = os.path.join(os.path.dirname(__file__), "..", "example_data", "simu.hdf5")
has_simu_data = os.path.exists(SIMU_DATA)


@pytest.mark.skipif(not has_simu_data, reason="Simulation data not available")
class TestReconstruction:
    @pytest.fixture
    def reconstruction_with_data(self):
        import PtyLabX

        experimentalData, reconstruction, params, monitor, engine = PtyLabX.easyInitialize(
            "example:simulation_cpm", operationMode="CPM"
        )
        return experimentalData, reconstruction

    def test_scalar_property_copy(self, reconstruction_with_data):
        """Check that scalar properties are properly copied from data."""
        data, optimizable = reconstruction_with_data
        assert optimizable.wavelength == data.wavelength

        # Changing optimizable should not affect data
        original = data.wavelength
        optimizable.wavelength = 4321
        assert optimizable.wavelength != original

    def test_array_property_copy(self, reconstruction_with_data):
        """Positions should be independently modifiable."""
        data, optimizable = reconstruction_with_data
        original_positions = data.positions.copy()
        optimizable.positions = optimizable.positions + 1
        assert_array_almost_equal(optimizable.positions - 1, original_positions)
