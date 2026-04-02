import numpy as np
import jax.numpy as jnp
from tqdm.auto import trange


import logging

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Operators.Operators import aspw
from PtyLabX.Params.Params import Params

# fracPy imports
from PtyLabX._types import ExitWave, ObjectPatch, Probe
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class e3PIE(BaseEngine):
    betaProbe: float
    betaObject: float
    numIterations: int

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ) -> None:
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to e3PIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("e3PIE")
        self.logger.info("Sucesfully created e3PIE e3PIE_engine")

        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)

        self.initializeReconstructionParams()

    def initializeReconstructionParams(self) -> None:
        """
        Set parameters that are specific to the e3PIE settings.
        :return:
        """
        self.params.betaProbe = 0.25
        self.params.betaObject = 0.25
        self.numIterations = 50
        self.betaProbe = 0.25

        if False:
            # preallocate transfer function
            self.reconstruction.H = aspw(
                np.squeeze(self.reconstruction.probe[0, 0, 0, 0, ...]),
                self.reconstruction.dz,
                self.reconstruction.wavelength / self.reconstruction.refrIndex,
                self.reconstruction.Lp,
            )[1]
            # shift transfer function to avoid fftshifts for FFTS
            # self.reconstruction.H = np.fft.ifftshift(self.optimizableH)
            self.reconstruction.H = np.fft.ifftshift(self.reconstruction.H)

        if True:
            # preallocate transfer function
            dz = self.reconstruction.dz
            wavelength = self.reconstruction.wavelength
            refrIndex = self.reconstruction.refrIndex
            assert dz is not None
            assert wavelength is not None
            assert refrIndex is not None
            self.reconstruction.H = aspw(
                jnp.squeeze(self.reconstruction.probe[0, 0, 0, 0, ...]),
                float(dz),
                wavelength / refrIndex,
                self.reconstruction.Lp,
            )[1]
            # shift transfer function to avoid fftshifts for FFTS
            # self.reconstruction.H = np.fft.ifftshift(self.optimizableH)
            self.reconstruction.H = jnp.fft.ifftshift(self.reconstruction.H)

    def reconstruct(self):
        self._prepareReconstruction()

        # initialize esw
        self.reconstruction.esw = self.reconstruction.probe.copy()
        # get module

        self.pbar = trange(self.numIterations, desc="e3PIE", leave=True)

        # self.pbar = (1, 2)

        for loop in self.pbar:
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()
                # objectPatch2 = self.reconstruction.object[..., :, :].copy()

                # form first slice esw (exit surface wave)
                self.reconstruction.esw = self.reconstruction.esw.at[:, :, :, 0, ...].set(
                    objectPatch[:, :, :, 0, ...] * self.reconstruction.probe[:, :, :, 0, ...]
                )

                # propagate through object
                for sliceLoop in range(1, self.reconstruction.nslice):
                    self.reconstruction.probe = self.reconstruction.probe.at[:, :, :, sliceLoop, ...].set(
                        jnp.fft.ifft2(
                            jnp.fft.fft2(self.reconstruction.esw[:, :, :, sliceLoop - 1, ...]) * self.reconstruction.H
                        )
                    )
                    self.reconstruction.esw = self.reconstruction.esw.at[:, :, :, sliceLoop, ...].set(
                        self.reconstruction.probe[:, :, :, sliceLoop, ...] * objectPatch[:, :, :, sliceLoop, ...]
                    )

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = (self.reconstruction.eswUpdate - self.reconstruction.esw)[:, :, :, -1, ...]
                # update object slice
                for loopTemp in range(self.reconstruction.nslice - 1):
                    sliceLoop = self.reconstruction.nslice - 1 - loopTemp

                    # temp_delta = self.reconstruction.esw[..., sliceLoop, sy, sx]

                    # compute and update current object slice
                    self.reconstruction.object = self.reconstruction.object.at[..., sliceLoop, sy, sx].set(
                        self.objectPatchUpdate(
                            objectPatch[:, :, :, sliceLoop, ...],
                            DELTA,
                            self.reconstruction.probe[:, :, :, sliceLoop, ...],
                        )
                    )
                    # eswTemp update (here probe incident on last slice)
                    beth = 0.9  # todo, why need beth, not betaProbe, changable?
                    self.reconstruction.probe = self.reconstruction.probe.at[:, :, :, sliceLoop, ...].set(
                        self.probeUpdate(
                            objectPatch[:, :, :, sliceLoop, ...],
                            DELTA,
                            self.reconstruction.probe[:, :, :, sliceLoop, ...],
                            beth,
                        )
                    )

                    # back-propagate and calculate gradient term
                    DELTA = (
                        jnp.fft.ifft2(
                            jnp.fft.fft2(self.reconstruction.probe[:, :, :, sliceLoop, ...])
                            * self.reconstruction.H.conj()
                        )
                        - self.reconstruction.esw[:, :, :, sliceLoop - 1, ...]
                    )

                # update last object slice
                self.reconstruction.object = self.reconstruction.object.at[..., 0, sy, sx].set(
                    self.objectPatchUpdate(
                        objectPatch[:, :, :, 0, ...],
                        DELTA,
                        self.reconstruction.probe[:, :, :, 0, ...],
                    )
                )
                # update probe
                self.reconstruction.probe = self.reconstruction.probe.at[:, :, :, 0, ...].set(
                    self.probeUpdate(
                        objectPatch[:, :, :, 0, ...],
                        DELTA,
                        self.reconstruction.probe[:, :, :, 0, ...],
                        self.betaProbe,
                    )
                )

            # set porduct of all object slices
            self.reconstruction.objectProd = np.prod(self.reconstruction.object, 3)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints todo uncomment orthogonalization? check object smootheness regularization
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

    def objectPatchUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave, localProbe: Probe) -> ObjectPatch:
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        frac = localProbe.conj() / jnp.max(jnp.sum(jnp.abs(localProbe) ** 2, axis=(0, 1, 2)))
        return objectPatch + self.betaObject * jnp.sum(frac * DELTA, axis=(0, 2), keepdims=True)

    def probeUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave, localProbe: Probe, beth: float) -> Probe:
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        frac = objectPatch.conj() / jnp.max(jnp.sum(jnp.abs(objectPatch) ** 2, axis=(0, 1, 2)))
        r = localProbe + beth * jnp.sum(frac * DELTA, axis=(0, 1), keepdims=True)
        return r
