import numpy as np
import jax.numpy as jnp

# PtyLab imports
import logging

from tqdm.auto import trange

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params
from PtyLabX.Engines._jit_kernels import qnewton_object_update, qnewton_probe_update
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class mqNewton(BaseEngine):
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
        self.logger = logging.getLogger("mqNewton")
        self.logger.info("Sucesfully created momentum accelerated qNewton mqNewton")

        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
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
        Set parameters that are specific to the qNewton settings.
        :return:
        """
        self.betaProbe = 1
        self.betaObject = 1
        self.regObject = 1
        self.regProbe = 1
        self.beta1 = 0.5
        self.beta2 = 0.5
        self.betaProbe_m = 0.25
        self.betaObject_m = 0.25
        self.feedbackM = 0.3  # feedback
        self.frictionM = 0.7  # friction
        self.momentum_method = "ADAM"  # which optimizer to use for momentum updates
        self.numIterations = 50

    def initializeAdaptiveMomentum(self):
        self.momentum_engine = getattr(mqNewton, self.momentum_method)
        print("Momentum Engines implemented: momentum, ADAM, NADAM")
        print("Momentum mqNewton used: {}".format(self.momentum_method))
        if self.momentum_method in ["ADAM", "NADAM"]:
            # 2nd order momentum terms
            self.reconstruction.objectMomentum_v = self.reconstruction.objectMomentum.copy()
            self.reconstruction.probeMomentum_v = self.reconstruction.probeMomentum.copy()

    def reconstruct(self, experimentalData: ExperimentalData = None):
        if experimentalData is not None:
            self.experimentalData = experimentalData
            self.reconstruction.data = experimentalData
        self._prepareReconstruction()
        self.initializeAdaptiveMomentum()

        self.pbar = trange(self.numIterations, desc="mqNewton", leave=True)
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
                self.objectMomentumUpdate(loop)
                self.probeMomentumUpdate(loop)

                if self.params.positionCorrectionSwitch:
                    self.positionCorrection(objectPatch, positionIndex, sy, sx)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

            # todo clearMemory implementation

    def ADAM(self, grad, mt, vt, itr):
        beta1_scale = 1 - self.beta1**itr
        beta2_scale = 1 - self.beta2**itr
        mt = self.beta1 * mt + (1 - self.beta1) * grad
        vt = self.beta2 * vt + (1 - self.beta2) * jnp.linalg.norm(grad.flatten().squeeze(), 2) ** 2
        m_hat = mt / beta1_scale
        v_hat = vt / beta2_scale
        return m_hat / (v_hat**0.5 + 1e-8), mt, vt

    def NADAM(self, grad, mt, vt, itr):
        """
        NADAM optimizer uses adaptive momentum updates (ADAM) with Nesterov
        momentum acceleration
        :return:
        """

        beta1_scale = 1 - self.beta1**itr
        beta2_scale = 1 - self.beta2**itr

        norm_sq = jnp.linalg.norm(grad.flatten(), 2) ** 2
        mt = self.beta1 * mt + (1 - self.beta1) * grad
        vt = self.beta2 * vt + (1 - self.beta2) * norm_sq
        m_hat = mt / beta1_scale
        v_hat = vt / beta2_scale
        update = (self.beta1 * m_hat + grad * (1 - self.beta1) / beta1_scale) / (v_hat**0.5 + 1e-8)
        return update, mt, vt

    def momentum(self, grad, mt, vt, itr):
        """
        standard momentum update
        :return:
        """
        mt = grad + self.frictionM * mt
        return mt, mt, vt

    def objectMomentumUpdate(self, loop):
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
        """
        gradient = self.reconstruction.objectBuffer - self.reconstruction.object
        (
            update,
            self.reconstruction.objectMomentum,
            self.reconstruction.objectMomentum_v,
        ) = self.momentum_engine(
            self,
            gradient,
            self.reconstruction.objectMomentum,
            self.reconstruction.objectMomentum_v,
            loop + 1,
        )

        self.reconstruction.object = self.reconstruction.object - self.betaObject_m * update
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()

    def probeMomentumUpdate(self, loop):
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
        """
        gradient = self.reconstruction.probeBuffer - self.reconstruction.probe
        (
            update,
            self.reconstruction.probeMomentum,
            self.reconstruction.probeMomentum_v,
        ) = self.momentum_engine(
            self,
            gradient,
            self.reconstruction.probeMomentum,
            self.reconstruction.probeMomentum_v,
            loop + 1,
        )

        self.reconstruction.probe = self.reconstruction.probe - self.betaProbe_m * update
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return qnewton_object_update(objectPatch, self.reconstruction.probe, DELTA, self.betaObject, self.regObject)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return qnewton_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe, self.regProbe)
