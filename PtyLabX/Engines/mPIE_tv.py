import numpy as np
import jax.numpy as jnp


import logging
import sys

import tqdm

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params

# fracPy imports
from PtyLabX.Engines._jit_kernels import momentum_step, mpie_object_update, mpie_probe_update
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class mPIE_tv(BaseEngine):
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
        self.logger = logging.getLogger("mPIE")
        self.logger.info("Sucesfully created mPIE mPIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        # initialize mPIE Params
        self.initializeReconstructionParams()
        self.params.momentumAcceleration = True

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the mPIE settings.
        :return:
        """
        # self.eswUpdate = self.reconstruction.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1  # probe regularization
        self.alphaObject = 0.1  # object regularization
        self.feedbackM = 0.3  # feedback
        self.frictionM = 0.7  # friction
        self.numIterations = 50

        # initialize momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()
        # set object and probe buffers
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

        self.reconstruction.probeWindow = np.abs(self.reconstruction.probe)

    def reconstruct(self):
        self._prepareReconstruction()

        # actual reconstruction MPIE_engine
        self.pbar = tqdm.trange(self.numIterations, desc="mPIE", file=sys.stdout, leave=True)
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

                tv_freq = 1
                if loop % tv_freq == 0:
                    # object update
                    self.reconstruction.object = self.reconstruction.object.at[..., sy, sx].set(
                        self.objectPatchUpdate_TV(objectPatch, DELTA)
                    )
                else:
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
                self.frictionM,
                self.feedbackM,
            )
        )

    def probeMomentumUpdate(self):
        self.reconstruction.probe, self.reconstruction.probeMomentum, self.reconstruction.probeBuffer = momentum_step(
            self.reconstruction.probe,
            self.reconstruction.probeBuffer,
            self.reconstruction.probeMomentum,
            self.frictionM,
            self.feedbackM,
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

    def objectPatchUpdate_TV(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """

        def divergence(f):
            return jnp.gradient(f[0], axis=(4, 5))[0] + jnp.gradient(f[1], axis=(4, 5))[1]

        frac = self.reconstruction.probe.conj() / jnp.max(
            jnp.sum(jnp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3))
        )

        epsilon = 1e-2
        gradient = jnp.gradient(objectPatch, axis=(4, 5))
        # norm = jnp.abs(gradient[0] + gradient[1]) ** 2
        norm = (gradient[0] + gradient[1]) ** 2
        temp = [
            gradient[0] / jnp.sqrt(norm + epsilon),
            gradient[1] / jnp.sqrt(norm + epsilon),
        ]
        TV_update = divergence(temp)
        """
        plt.figure()
        plt.imshow(np.abs(TV_update.get()[0, 0, 0, 0, :, :]))
        plt.figure()
        plt.imshow(np.angle(TV_update.get()[0, 0, 0, 0, :, :]))
        plt.show()
        """
        lam = self.params.TV_lam
        return (
            objectPatch
            + self.betaObject * jnp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)
            + lam * self.betaObject * TV_update
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return mpie_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe, self.alphaProbe, 1.0)
