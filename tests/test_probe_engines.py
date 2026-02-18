"""Tests for ProbeEngines module.

Migrated from:
- PtyLabX/ProbeEngines/test_OPRP.py
- PtyLabX/ProbeEngines/test_StandardProbe.py

Note: ProbeEngines/__init__.py intentionally raises ValueError ("not ready yet"),
so these tests are skipped until the module is ready.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

try:
    from PtyLabX.ProbeEngines.OPRP import OPRP_storage

    HAS_PROBE_ENGINES = True
except (ValueError, ImportError):
    HAS_PROBE_ENGINES = False

pytestmark = pytest.mark.skipif(not HAS_PROBE_ENGINES, reason="ProbeEngines not ready yet")


class TestOPRPStorage:
    """Tests for OPRP_storage with TSVD and probe centering."""

    @pytest.fixture
    def storage_setup(self):
        N = 100
        npix = 128
        probes = np.random.rand(N // 10, 4, npix, npix)
        probes = np.repeat(probes, axis=0, repeats=10)
        storage = OPRP_storage(5)
        return storage, probes, N, npix

    def test_push_and_get(self, storage_setup):
        storage, probes, N, npix = storage_setup
        for tsvd in [False, True]:
            storage.push(probes[0], 0, N)
            if tsvd:
                storage.tsvd()
            p1 = storage.get(0)
            assert_almost_equal(p1, probes[0], decimal=5)
            p1 = storage.get(1)
            assert_almost_equal(p1, probes[0], decimal=5)
            storage.clear()

    def test_tsvd_roundtrip(self, storage_setup):
        storage, probes, N, npix = storage_setup
        for i, p in enumerate(probes):
            storage.push(p, i, N)

        storage.N_probes = N
        storage.tsvd()

        p1 = storage.get(0)
        assert_almost_equal(p1, probes[0], decimal=5)

    def test_center_probe_roundtrip(self, storage_setup):
        storage, probes, N, npix = storage_setup
        probes[:, :, : npix // 3, :] = 0
        storage.push(probes[0], 0, N)
        for i, p in enumerate(probes[:5]):
            p_inout, _ = storage.uncenter_probe(storage.center_probe(p, i)[0], i)
            assert_almost_equal(p, p_inout)
