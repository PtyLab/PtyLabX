import jax.numpy as jnp


import logging

from tqdm.auto import trange

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params

# fracPy imports
from PtyLabX._types import ExitWave, ObjectPatch, Probe
from PtyLabX.Engines._jit_kernels import epie_object_update, epie_probe_update
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class ePIE_TV(BaseEngine):
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ) -> None:
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("ePIE")
        self.logger.info("Sucesfully created ePIE ePIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self) -> None:
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50

    def reconstruct(self):
        self._prepareReconstruction()

        # actual reconstruction ePIE_engine
        self.pbar = trange(self.numIterations, desc="ePIE", leave=True)
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
                if loop % 5 == 0:
                    # object update
                    self.reconstruction.object = self.reconstruction.object.at[..., sy, sx].set(
                        self.objectPatchUpdate_TV(objectPatch, DELTA)
                    )
                else:
                    self.reconstruction.object = self.reconstruction.object.at[..., sy, sx].set(
                        self.objectPatchUpdate(objectPatch, DELTA)
                    )

                # probe update
                # self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)
                self.probeUpdate_new(objectPatch, DELTA)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

    def objectPatchUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave) -> ObjectPatch:
        return epie_object_update(objectPatch, self.reconstruction.probe, DELTA, self.betaObject)

    def objectPatchUpdate_TV(self, objectPatch: ObjectPatch, DELTA: ExitWave) -> ObjectPatch:
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
        lam = 5e-4
        return (
            objectPatch
            + self.betaObject * jnp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)
            + lam * self.betaObject * TV_update
        )

    def probeUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave) -> Probe:
        return epie_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe)

    def probeUpdate_new(self, objectPatch: ObjectPatch, DELTA: ExitWave) -> None:
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        self.reconstruction.probe = self.reconstruction.probe + self.betaProbe * jnp.sum(
            objectPatch.conj() / jnp.max(jnp.sum(jnp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3))) * DELTA,
            axis=(0, 1, 3),
            keepdims=True,
        )
