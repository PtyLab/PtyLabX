import time

import jax
import jax.numpy as jnp

from PtyLabX import easyInitialize
from PtyLabX.Operators.off_axis_sas import propagate_sas


def load_data(path="example:simulation_cpm"):
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(path)
    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    reconstruction.theta = (40, 0)
    return reconstruction, params


def benchmark_runs(nruns: int = 10):
    """Benchmarks off-axis SAS propagator using JAX.

    Parameters
    ----------
    nruns : int, optional
        No. of runs for each propagator
    """

    # load data
    reconstruction, params = load_data(path="example:simulation_cpm")
    reconstruction.pad_factor = 2  # can be modified by user.

    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    def run_propagator_func():
        return propagate_sas(reconstruction.probe, params, reconstruction)

    # Warm-up run
    t0 = time.time()
    _ = run_propagator_func()
    t1 = time.time()
    print(f"\nWarm-up run time: {1e3 * (t1 - t0):.3f} ms")

    # Timed runs
    print(f"\nBenchmark run times ({nruns} runs):")
    times = []
    for i in range(nruns):
        t0 = time.time()
        result = run_propagator_func()
        # Block until computation is complete
        jax.block_until_ready(result)
        t1 = time.time()
        elapsed = 1e3 * (t1 - t0)
        times.append(elapsed)
        print(f"Run {i}: {elapsed:.3f} ms")

    import numpy as np
    times = np.array(times)
    print(f"\nMean: {times.mean():.3f} ms, Std: {times.std():.3f} ms")


if __name__ == "__main__":
    benchmark_runs(nruns=10)
