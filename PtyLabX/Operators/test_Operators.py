import time
from unittest import TestCase

import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose

from PtyLabX import easyInitialize
from PtyLabX.Operators.Operators import (
    aspw,
    aspw_cached,
    forward_lookup_dictionary,
    object2detector,
    propagate_ASP,
    propagate_fresnel,
    propagate_scaledASP,
    propagate_scaledPolychromeASP,
    propagate_twoStepPolychrome,
)


def test_caching_aspw():
    E = jnp.array(np.random.rand(10, 1, 3, 512, 512))
    z = 1e-3
    wl = 512e-9
    pixel_pitch = 10e-6
    L = pixel_pitch * E.shape[-1]

    t0 = time.time()
    for i in range(100):
        E_prop = aspw_cached(E, z, wl, L)
    E_prop = np.asarray(E_prop)
    t1 = time.time()
    t_cached = t1 - t0
    for i in range(100):
        E_prop2 = aspw(E, z, wl, L)[0]
    E_prop2 = np.asarray(E_prop2)
    t2 = time.time()
    t_noncached = t2 - t1
    print(f"\n\nNon-cached took: {t_noncached}", f"Cached took {t_cached}s")

    assert_allclose(E_prop, E_prop2)


def test_object2detector():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    _doit(reconstruction, params)


def _doit(reconstruction, params):
    for operator_name in forward_lookup_dictionary:
        params.propagatorType = operator_name
        reconstruction.esw = reconstruction.probe
        reconstruction.theta = (40, 0)  # for off-axis sas
        print("\n")

        for i in range(3):
            t0 = time.time()
            object2detector(reconstruction.esw, params, reconstruction)
            t1 = time.time()
            print(operator_name, i, 1e3 * (t1 - t0), "ms")


def test_propagate_fresnel(nruns: int = 10):
    """Checks if Fresnel based propagators are bug-free.

    Parameters
    ----------
    nruns : int, optional
        No. of runs for each propagator
    """
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )

    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    for operator in [
        propagate_fresnel,
        propagate_ASP,
        propagate_scaledASP,
        propagate_twoStepPolychrome,
        propagate_scaledPolychromeASP,
    ]:
        print("\n---------------")
        print(f"{operator.__name__}\n")
        for i in range(nruns):
            t0 = time.time()
            _ = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(f"Run {i}: {1e3 * (t1 - t0):.3f} ms")


def test_aspw_cached():
    assert False


class TestASP(TestCase):
    def test_propagate_asp(self):
        experimentalData, reconstruction, params, monitor, engine = easyInitialize(
            "example:simulation_cpm"
        )
        reconstruction.esw = None
        a = reconstruction.probe
        P1 = propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=False)[1]
        P2 = propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=True)[1]
        assert_allclose(P1, P2)
