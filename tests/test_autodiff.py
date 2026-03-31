"""Tests for the AutoDiff subpackage.

Tests cover:
- State containers (PtychographyState, StaticConfig) round-trip
- Forward model output shapes and differentiability
- Loss functions (finite, non-negative, differentiable)
- Regularizer (TV)
- Optimizer construction and update
- End-to-end: gradient step reduces loss on synthetic data
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from PtyLabX.AutoDiff import build_loss
from PtyLabX.AutoDiff.forward_models import single_slice_forward
from PtyLabX.AutoDiff.losses import amplitude_loss, mad_amplitude_loss, poisson_loss
from PtyLabX.AutoDiff.optimizers import build_optimizer
from PtyLabX.AutoDiff.reconstructor import GradientReconstructor
from PtyLabX.AutoDiff.regularizers import object_tv
from PtyLabX.AutoDiff.state import PtychographyState, StaticConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

Np = 32
No = 64
NUM_FRAMES = 10


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def synthetic_data(rng):
    """Create minimal synthetic ptychography data for testing."""
    # Object: random complex (6D convention)
    obj = jnp.array(
        rng.standard_normal((1, 1, 1, 1, No, No)) + 1j * rng.standard_normal((1, 1, 1, 1, No, No)),
        dtype=jnp.complex64,
    )
    # Probe: random complex (6D convention)
    probe = jnp.array(
        rng.standard_normal((1, 1, 1, 1, Np, Np)) + 1j * rng.standard_normal((1, 1, 1, 1, Np, Np)),
        dtype=jnp.complex64,
    )
    # Positions: random valid positions within object bounds
    max_pos = No - Np
    positions = jnp.array(rng.integers(0, max_pos, size=(NUM_FRAMES, 2)), dtype=jnp.int32)

    # Generate ptychogram from forward model
    from PtyLabX.utils.utils import fft2c

    ptychogram = np.zeros((NUM_FRAMES, Np, Np), dtype=np.float32)
    for i in range(NUM_FRAMES):
        row, col = int(positions[i, 0]), int(positions[i, 1])
        patch = obj[..., row : row + Np, col : col + Np]
        esw = patch * probe
        ESW = fft2c(esw)
        ptychogram[i] = np.asarray(jnp.sum(jnp.abs(ESW) ** 2, axis=(0, 1, 2, 3)))

    ptychogram = jnp.array(ptychogram)

    static = StaticConfig(
        positions=positions,
        ptychogram=ptychogram,
        wavelength=632.8e-9,
        zo=5e-2,
        dxp=1e-6,
        dxd=1e-5,
        Np=Np,
        No=No,
        fftshift_switch=False,
        propagator_type="Fraunhofer",
    )

    return obj, probe, static


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


class TestState:
    def test_ptychography_state_is_pytree(self):
        """PtychographyState should be a valid JAX pytree."""
        state = PtychographyState(
            object=jnp.ones((1, 1, 1, 1, 4, 4), dtype=jnp.complex64),
            probe=None,
        )
        leaves = jax.tree.leaves(state)
        assert len(leaves) == 1  # only object, probe is None

    def test_state_with_probe(self):
        state = PtychographyState(
            object=jnp.ones((1, 1, 1, 1, 4, 4), dtype=jnp.complex64),
            probe=jnp.ones((1, 1, 1, 1, 4, 4), dtype=jnp.complex64),
        )
        leaves = jax.tree.leaves(state)
        assert len(leaves) == 2


# ---------------------------------------------------------------------------
# Forward model tests
# ---------------------------------------------------------------------------


class TestForwardModel:
    def test_output_shape(self, synthetic_data):
        """Forward model should return (batch_size, Np, Np) intensities."""
        obj, probe, static = synthetic_data
        state = PtychographyState(object=obj, probe=None)
        batch_indices = jnp.arange(3, dtype=jnp.int32)
        I_pred = single_slice_forward(state, batch_indices, static, known_probe=probe)
        assert I_pred.shape == (3, Np, Np)

    def test_output_nonnegative(self, synthetic_data):
        """Intensities must be non-negative."""
        obj, probe, static = synthetic_data
        state = PtychographyState(object=obj, probe=None)
        batch_indices = jnp.arange(5, dtype=jnp.int32)
        I_pred = single_slice_forward(state, batch_indices, static, known_probe=probe)
        assert jnp.all(I_pred >= 0)

    def test_gradient_flows(self, synthetic_data):
        """jax.grad should not error through the forward model."""
        obj, probe, static = synthetic_data
        batch_indices = jnp.arange(3, dtype=jnp.int32)

        def _loss(obj_arr):
            s = PtychographyState(object=obj_arr, probe=None)
            I_pred = single_slice_forward(s, batch_indices, static, known_probe=probe)
            return jnp.mean(I_pred)

        grad = jax.grad(_loss)(obj)
        assert grad.shape == obj.shape
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestLosses:
    @pytest.mark.parametrize("loss_fn", [amplitude_loss, poisson_loss, mad_amplitude_loss])
    def test_loss_scalar_finite(self, loss_fn):
        """Each loss should return a finite scalar."""
        I_meas = jnp.ones((5, 32, 32)) * 2.0
        I_pred = jnp.ones((5, 32, 32)) * 1.5
        val = loss_fn(I_meas, I_pred)
        assert val.shape == ()
        assert jnp.isfinite(val)

    @pytest.mark.parametrize("loss_fn", [amplitude_loss, poisson_loss, mad_amplitude_loss])
    def test_loss_zero_at_match(self, loss_fn):
        """Loss should be minimal (near zero for amplitude/MAD) when prediction matches measurement."""
        intensity = jnp.ones((5, 32, 32)) * 3.0
        val = loss_fn(intensity, intensity)
        # Poisson loss is not exactly zero at match (it's I - I*log(I)), but amplitude/MAD should be ~0
        if loss_fn is not poisson_loss:
            assert val < 1e-6

    @pytest.mark.parametrize("loss_fn", [amplitude_loss, poisson_loss, mad_amplitude_loss])
    def test_loss_differentiable(self, loss_fn):
        """jax.grad should work through each loss."""
        I_meas = jnp.ones((5, 32, 32)) * 2.0
        I_pred = jnp.ones((5, 32, 32)) * 1.5
        grad = jax.grad(loss_fn, argnums=1)(I_meas, I_pred)
        assert grad.shape == I_pred.shape
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# Regularizer tests
# ---------------------------------------------------------------------------


class TestRegularizers:
    def test_tv_scalar_finite(self):
        state = PtychographyState(
            object=jnp.ones((1, 1, 1, 1, 16, 16), dtype=jnp.complex64),
            probe=None,
        )
        # StaticConfig is not used by object_tv, but we need a placeholder
        static = StaticConfig(
            positions=jnp.zeros((1, 2), dtype=jnp.int32),
            ptychogram=jnp.zeros((1, 8, 8)),
            wavelength=1.0,
            zo=1.0,
            dxp=1.0,
            dxd=1.0,
            Np=8,
            No=16,
            fftshift_switch=False,
            propagator_type="Fraunhofer",
        )
        val = object_tv(state, static, weight=1e-4)
        assert val.shape == ()
        assert jnp.isfinite(val)

    def test_tv_differentiable(self):
        obj = jnp.ones((1, 1, 1, 1, 16, 16), dtype=jnp.complex64)
        static = StaticConfig(
            positions=jnp.zeros((1, 2), dtype=jnp.int32),
            ptychogram=jnp.zeros((1, 8, 8)),
            wavelength=1.0,
            zo=1.0,
            dxp=1.0,
            dxd=1.0,
            Np=8,
            No=16,
            fftshift_switch=False,
            propagator_type="Fraunhofer",
        )

        def _tv_loss(obj_arr):
            s = PtychographyState(object=obj_arr, probe=None)
            return object_tv(s, static, weight=1.0)

        grad = jax.grad(_tv_loss)(obj)
        assert grad.shape == obj.shape


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_build_optimizer_init(self):
        """Optimizer should initialise on a PtychographyState."""
        opt = build_optimizer(object_lr=1e-3, probe_lr=0.0)
        state = PtychographyState(
            object=jnp.ones((1, 1, 1, 1, 8, 8), dtype=jnp.complex64),
            probe=None,
        )
        opt_state = opt.init(state)
        assert opt_state is not None

    def test_optimizer_update(self):
        """Optimizer should produce updates of correct shape."""
        opt = build_optimizer(object_lr=1e-3, probe_lr=0.0)
        state = PtychographyState(
            object=jnp.ones((1, 1, 1, 1, 8, 8), dtype=jnp.complex64),
            probe=None,
        )
        grads = PtychographyState(
            object=jnp.ones((1, 1, 1, 1, 8, 8), dtype=jnp.complex64) * 0.1,
            probe=None,
        )
        opt_state = opt.init(state)
        updates, new_opt_state = opt.update(grads, opt_state, state)
        new_state = optax.apply_updates(state, updates)
        assert new_state.object.shape == state.object.shape


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_gradient_step_reduces_loss(self, synthetic_data):
        """A single optimisation epoch should reduce the loss on synthetic data."""
        obj_gt, probe, static = synthetic_data

        # Start from a perturbed object (all ones)
        obj_init = jnp.ones_like(obj_gt)
        state = PtychographyState(object=obj_init, probe=None)

        loss_fn = build_loss(single_slice_forward, amplitude_loss)
        optimizer = build_optimizer(object_lr=0.1)

        reconstructor = GradientReconstructor(
            loss_fn=loss_fn,
            optimizer=optimizer,
            state=state,
            static=static,
            known_probe=probe,
            batch_size=5,
        )

        losses = []
        for epoch, mean_loss in reconstructor.reconstruct(num_iterations=5):
            losses.append(mean_loss)

        # Loss should decrease over 5 epochs
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_build_loss_with_regularizer(self, synthetic_data):
        """build_loss with a TV regularizer should still be differentiable."""
        obj_gt, probe, static = synthetic_data
        state = PtychographyState(object=obj_gt, probe=None)

        loss_fn = build_loss(
            single_slice_forward,
            amplitude_loss,
            regularizers=[lambda s, c: object_tv(s, c, weight=1e-4)],
        )

        batch_indices = jnp.arange(3, dtype=jnp.int32)
        batch_I_meas = static.ptychogram[:3]

        val = loss_fn(state, batch_I_meas, batch_indices, static, probe)
        assert jnp.isfinite(val)
