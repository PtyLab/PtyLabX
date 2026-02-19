import numpy as np
import jax.numpy as jnp


import logging

from tqdm.auto import trange

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params

# PtyLab imports
from PtyLabX.Engines._jit_kernels import momentum_step, mpie_object_update
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class multiPIE(BaseEngine):
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("multiPIE")
        self.logger.info("Sucesfully created multiPIE multiPIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        # initialize multiPIE Params
        self.initializeReconstructionParams()
        self.params.momentumAcceleration = True

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the multiPIE settings.
        :return:
        """
        # self.eswUpdate = self.reconstruction.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1  # probe regularization
        self.alphaObject = 0.1  # object regularization
        self.betaM = 0.3  # feedback
        self.stepM = 0.7  # friction
        # self.reconstruction.probeWindow = np.abs(self.reconstruction.probe)
        self.numIterations = 50

        # initialize momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()
        # set object and probe buffers
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

    def reconstruct(self):
        self._prepareReconstruction()

        self.pbar = trange(self.numIterations, desc="multiPIE", leave=True)

        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()

                # make exit surface wave
                self.reconstruction.esw = objectPatch * self.reconstruction.probe

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                # object update
                self.reconstruction.object = self.reconstruction.object.at[..., sy, sx].set(
                    self.objectPatchUpdate(objectPatch, DELTA)
                )

                # probe update
                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)

                # momentum updates
                if np.random.rand(1) > 0.95:
                    self.objectMomentumUpdate()
                    self.probeMomentumUpdate()

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

            # todo clearMemory implementation

    def objectMomentumUpdate(self):
        self.reconstruction.object, self.reconstruction.objectMomentum, self.reconstruction.objectBuffer = (
            momentum_step(
                self.reconstruction.object,
                self.reconstruction.objectBuffer,
                self.reconstruction.objectMomentum,
                self.stepM,
                self.betaM,
            )
        )

    def probeMomentumUpdate(self):
        self.reconstruction.probe, self.reconstruction.probeMomentum, self.reconstruction.probeBuffer = momentum_step(
            self.reconstruction.probe,
            self.reconstruction.probeBuffer,
            self.reconstruction.probeMomentum,
            self.stepM,
            self.betaM,
        )

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return mpie_object_update(
            objectPatch,
            self.reconstruction.probe,
            DELTA,
            self.betaObject,
            self.alphaObject,
            fpm_mode=(self.experimentalData.operationMode == "FPM"),
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        absO2 = jnp.abs(objectPatch) ** 2
        Omax = jnp.max(jnp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        frac = objectPatch.conj() / (self.alphaProbe * Omax + (1 - self.alphaProbe) * absO2)
        r = self.reconstruction.probe + self.betaProbe * jnp.sum(frac * DELTA, axis=(0, 1), keepdims=True)
        return r
