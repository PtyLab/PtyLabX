"""GradientEngine — BaseEngine adapter for AD-based reconstruction.

Provides a familiar ``easyInitialize``-compatible interface while internally
using the ``AutoDiff`` subpackage for gradient-descent reconstruction.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

from tqdm.auto import trange

from PtyLabX.AutoDiff import GradientReconstructor, build_loss, build_optimizer
from PtyLabX.AutoDiff._state import (
    state_from_reconstruction,
    state_to_reconstruction,
    static_from_reconstruction,
)
from PtyLabX.AutoDiff.forward_models import single_slice_forward
from PtyLabX.AutoDiff.losses import amplitude_loss
from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class GradientEngine(BaseEngine):
    """AD-based reconstruction engine compatible with ``easyInitialize()``.

    Uses ``PtyLabX.AutoDiff`` internally while presenting the same constructor
    signature and ``reconstruct()`` generator pattern as other engines.

    Attributes
    ----------
    numIterations : int
        Number of epochs (default 100).
    object_lr : float
        Object learning rate for Adam (default 1e-2).
    probe_lr : float
        Probe learning rate for Adam. ``0`` = frozen (default 0, object-only).
    batch_size : int
        Scan positions per mini-batch (default 10).
    data_loss : callable
        Loss function (default ``amplitude_loss``).
    forward_model : callable
        Forward model (default ``single_slice_forward``).
    regularizers : list of callable or None
        Optional regularizers.
    preconditioning : bool
        Apply rPIE-like gradient preconditioning (default True).
    """

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ) -> None:
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("GradientEngine")
        self.logger.info("Created GradientEngine")

        # User-facing hyperparameters
        self.numIterations = 100
        self.object_lr = 1e-2
        self.probe_lr = 0.0
        self.batch_size = 10
        self.data_loss = amplitude_loss
        self.forward_model = single_slice_forward
        self.regularizers = None
        self.preconditioning = True

    def reconstruct(self) -> Generator[tuple[int, float], None, None]:
        """Run AD-based reconstruction.

        Yields ``(epoch, mean_loss)`` after each epoch — same pattern as
        other engines, enabling the standard ``for loop, _ in engine.reconstruct()``
        usage.
        """
        # Build components from current settings
        optimize_probe = self.probe_lr > 0
        state = state_from_reconstruction(self.reconstruction, optimize_probe=optimize_probe)
        static = static_from_reconstruction(self.reconstruction, self.experimentalData, self.params)

        known_probe = self.reconstruction.probe if not optimize_probe else None

        loss_fn = build_loss(self.forward_model, self.data_loss, self.regularizers)
        optimizer = build_optimizer(
            object_lr=self.object_lr,
            probe_lr=self.probe_lr,
        )

        reconstructor = GradientReconstructor(
            loss_fn=loss_fn,
            optimizer=optimizer,
            state=state,
            static=static,
            known_probe=known_probe,
            batch_size=self.batch_size,
            preconditioning=self.preconditioning,
        )

        self.pbar = trange(self.numIterations, desc="GradientEngine", leave=True)
        for epoch, mean_loss in reconstructor.reconstruct(self.numIterations):
            # Sync state back to Reconstruction for monitor/save
            state_to_reconstruction(reconstructor.state, self.reconstruction)
            self.reconstruction.error = reconstructor.error

            self.pbar.update(1)
            self.pbar.set_postfix(loss=f"{mean_loss:.6f}")
            yield epoch, mean_loss

        self.pbar.close()
