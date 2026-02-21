"""Tests for I/O module.

Migrated from:
- PtyLabX/io/test/test_example_loader.py
- PtyLabX/io/test/test_get_example_data_folder.py
- PtyLabX/io/test/test_loadInputData.py
"""

import os

import pytest

from PtyLabX.io import getExampleDataFolder

SIMU_DATA = os.path.join(os.path.dirname(__file__), "..", "example_data", "simu.hdf5")
has_simu_data = os.path.exists(SIMU_DATA)


class TestExampleDataFolder:
    def test_example_folder_exists(self):
        """Test that the path returned by getExampleDataFolder exists."""
        example_data_folder = getExampleDataFolder()
        assert example_data_folder.exists(), "example data folder does not exist"


class TestExampleLoader:
    """Tests for reading example datasets."""

    def test_list_examples(self):
        """Check that listExamples runs without error."""
        from PtyLabX.io import readExample

        readExample.listExamples()

    @pytest.mark.skipif(not has_simu_data, reason="Simulation data not available")
    def test_load_simulation_cpm(self):
        """Check that simulation_cpm example can be loaded."""
        from PtyLabX.io import readExample

        archive = readExample.loadExample("example:simulation_cpm")
        assert "Nd" in archive or "ptychogram" in archive

    @pytest.mark.skipif(
        not (getExampleDataFolder() / "fourier_simulation.hdf5").exists(),
        reason="fourier_simulation.hdf5 not available",
    )
    def test_load_fourier_simulation(self):
        """Load Fourier simulation data and verify shape."""
        from PtyLabX.io.readHdf5 import loadInputData

        filename = getExampleDataFolder() / "fourier_simulation.hdf5"
        result = loadInputData(filename)
        assert result["ptychogram"].shape == (49, 256, 256)
