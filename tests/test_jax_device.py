import jax
import jax.numpy as jnp
import pytest


def test_jax_default_backend():
    """Check that JAX reports a valid backend (cpu, gpu, or tpu)."""
    backend = jax.default_backend()
    print(f"JAX default backend: {backend}")
    assert backend in ("cpu", "gpu", "tpu"), f"Unexpected JAX backend: {backend}"


def test_jax_devices_available():
    """At least one device must be available."""
    devices = jax.devices()
    print(f"Available devices ({len(devices)}):")
    for d in devices:
        print(f"  - {d}")
    assert len(devices) >= 1, "No JAX devices found"


def test_jax_gpu_available():
    """Check whether a GPU device is available. Skips instead of failing on CPU-only machines."""
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []

    if not gpu_devices:
        print("No GPU detected — running on CPU only")
        pytest.skip("No GPU devices available — running on CPU only")

    for d in gpu_devices:
        print(f"GPU found: {d}")
    assert gpu_devices[0].platform == "gpu"


def test_jax_computation_runs_on_expected_device():
    """Run a small computation and verify it lands on the default device."""
    x = jnp.ones((128, 128))
    y = jnp.fft.fft2(x)
    device = y.devices().pop()
    print(f"Computation ran on: {device} (platform={device.platform})")
    assert device.platform == jax.default_backend()


def test_jax_gpu_accelerated():
    """If a GPU is present, verify a computation actually runs on it. Skips on CPU-only machines."""
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        print("No GPU backend available")
        pytest.skip("No GPU backend available")

    if not gpu_devices:
        print("No GPU devices available — running on CPU only")
        pytest.skip("No GPU devices available — running on CPU only")

    x = jax.device_put(jnp.ones((256, 256)), gpu_devices[0])
    y = jnp.fft.fft2(x)
    device = y.devices().pop()
    print(f"GPU-accelerated computation ran on: {device} (platform={device.platform})")
    assert device.platform == "gpu", "Computation did not run on GPU"
