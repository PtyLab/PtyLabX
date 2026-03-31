"""AutoDiff — Automatic differentiation based ptychographic reconstruction.

This subpackage provides a composable, JAX-native framework for gradient-descent
ptychographic reconstruction.  It coexists with the existing iterative engines
(ePIE, mPIE, etc.) without modifying them.

Quick start
-----------
>>> from PtyLabX.AutoDiff import build_loss, GradientReconstructor
>>> from PtyLabX.AutoDiff.forward_models import single_slice_forward
>>> from PtyLabX.AutoDiff.losses import amplitude_loss
>>> from PtyLabX.AutoDiff.optimizers import build_optimizer
>>>
>>> loss_fn = build_loss(single_slice_forward, amplitude_loss)
>>> optimizer = build_optimizer(object_lr=1e-3)
>>> reconstructor = GradientReconstructor(loss_fn, optimizer, state, static, known_probe=probe)
>>> for epoch, loss in reconstructor.reconstruct(num_iterations=100):
...     print(f"Epoch {epoch}: loss={loss:.6f}")

See Also
--------
.claude/DESIGN.md : Architecture plan and development roadmap.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from PtyLabX.AutoDiff.optimizers import build_optimizer as build_optimizer
from PtyLabX.AutoDiff.reconstructor import GradientReconstructor as GradientReconstructor
from PtyLabX.AutoDiff.state import PtychographyState as PtychographyState
from PtyLabX.AutoDiff.state import StaticConfig as StaticConfig
from PtyLabX.AutoDiff.state import state_from_reconstruction as state_from_reconstruction
from PtyLabX.AutoDiff.state import state_to_reconstruction as state_to_reconstruction
from PtyLabX.AutoDiff.state import static_from_reconstruction as static_from_reconstruction


def build_loss(
    forward_model: Callable,
    data_loss: Callable,
    regularizers: list[Callable] | None = None,
) -> Callable:
    """Compose a forward model, data loss, and optional regularizers into a single loss function.

    The returned function has signature::

        loss_fn(state, batch_I_meas, position_indices, static, known_probe) -> scalar

    and is suitable for ``jax.value_and_grad``.

    Parameters
    ----------
    forward_model : callable
        ``(state, position_indices, static, known_probe) -> I_predicted``
    data_loss : callable
        ``(I_measured, I_predicted) -> scalar``
    regularizers : list of callable, optional
        Each: ``(state, static) -> scalar``

    Returns
    -------
    callable
        Composed loss function.
    """

    def loss_fn(
        state: PtychographyState,
        batch_I_meas: jnp.ndarray,
        position_indices: jnp.ndarray,
        static: StaticConfig,
        known_probe: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        I_pred = forward_model(state, position_indices, static, known_probe)
        loss = data_loss(batch_I_meas, I_pred)
        if regularizers:
            for reg in regularizers:
                loss = loss + reg(state, static)
        return loss

    return loss_fn
