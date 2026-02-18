"""Tests for Monitor module.

Migrated from PtyLabX/Monitor/test/test_matplotlib_monitor.py

These are visual tests that require a display. Skipped by default.
"""

import numpy as np
import pytest

# Visual tests are skipped by default
pytestmark = pytest.mark.skip(reason="Visual tests require display, run manually")


class TestMatplotlibMonitor:
    def test_live_update(self):
        from PtyLabX.Monitor.Plots import ObjectProbeErrorPlot

        monitor = ObjectProbeErrorPlot()
        error_metrics = []
        for k in range(10):  # reduced from 100 for speed
            error_metrics.append(np.random.rand())
            monitor.updateObject(np.random.rand(100, 100))
            monitor.updateError(error_metrics)
            monitor.drawNow()
