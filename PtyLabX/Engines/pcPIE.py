import logging
import sys

import jax.numpy as jnp
import numpy as np
import tqdm
from matplotlib import pyplot as plt

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params

# PtyLab imports
from PtyLabX.Reconstruction.Reconstruction import Reconstruction
from PtyLabX.utils.utils import fft2c, ifft2c


class pcPIE(BaseEngine):
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
        self.logger = logging.getLogger("pcPIE")
        self.logger.info("Successfully created pcPIE pcPIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        # initialize pcPIE Params
        self.initializeReconstructionParams()
        # initialize momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()
        # set object and probe buffers
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

        self.params.momentumAcceleration = True

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the pcPIE settings.
        :return:
        """
        # these are same as mPIE
        # self.eswUpdate = self.reconstruction.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1  # probe regularization
        self.alphaObject = 0.1  # object regularization
        self.betaM = 0.3  # feedback
        self.stepM = 0.7  # friction
        # self.probeWindow = np.abs(self.reconstruction.probe)
        self.numIterations = 50

    def reconstruct(self):
        self._prepareReconstruction()

        # actual reconstruction ePIE_engine

        self.pbar = tqdm.trange(
            self.numIterations, desc="pcPIE", file=sys.stdout, leave=True
        )  # in order to change description to the tqdm progress bar
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
                if self.params.positionCorrectionSwitch:
                    self.positionCorrection(objectPatch, positionIndex, sy, sx)

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
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
        """
        gradient = self.reconstruction.objectBuffer - self.reconstruction.object
        self.reconstruction.objectMomentum = (
            gradient + self.stepM * self.reconstruction.objectMomentum
        )
        self.reconstruction.object = (
            self.reconstruction.object - self.betaM * self.reconstruction.objectMomentum
        )
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()

    def probeMomentumUpdate(self):
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
        """
        gradient = self.reconstruction.probeBuffer - self.reconstruction.probe
        self.reconstruction.probeMomentum = (
            gradient + self.stepM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probe = (
            self.reconstruction.probe - self.betaM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        absP2 = jnp.abs(self.reconstruction.probe) ** 2
        Pmax = jnp.max(jnp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        if self.experimentalData.operationMode == "FPM":
            frac = (
                abs(self.reconstruction.probe)
                / Pmax
                * self.reconstruction.probe.conj()
                / (self.alphaObject * Pmax + (1 - self.alphaObject) * absP2)
            )
        else:
            frac = self.reconstruction.probe.conj() / (
                self.alphaObject * Pmax + (1 - self.alphaObject) * absP2
            )
        return objectPatch + self.betaObject * jnp.sum(
            frac * DELTA, axis=2, keepdims=True
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
        frac = objectPatch.conj() / (
            self.alphaProbe * Omax + (1 - self.alphaProbe) * absO2
        )
        r = self.reconstruction.probe + self.betaProbe * jnp.sum(
            frac * DELTA, axis=1, keepdims=True
        )
        return r
