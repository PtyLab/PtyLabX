import logging

from tqdm.auto import trange

from PtyLabX._types import ExitWave, ObjectPatch, Probe
from PtyLabX.Engines._jit_kernels import epie_object_update, epie_probe_update
from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params

# PtyLab imports
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class ePIE(BaseEngine):
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

    def reconstruct(self, experimentalData: ExperimentalData | None = None) -> None:
        if experimentalData is not None:
            self.reconstruction.data = experimentalData
            self.experimentalData = experimentalData
        self._prepareReconstruction()

        # actual reconstruction ePIE_engine
        self.pbar = trange(self.numIterations, desc="ePIE", leave=True)
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            if self.params.OPRP:
                # make the initial guess the default storage
                self.reconstruction.probe_storage.push(
                    self.reconstruction.probe,
                    0,
                    self.experimentalData.ptychogram.shape[0],
                )
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                if self.params.OPRP:
                    self.reconstruction.probe = self.reconstruction.probe_storage.get(positionIndex)
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
                if self.params.OPRP:
                    self.reconstruction.probe_storage.push(
                        self.reconstruction.probe,
                        positionIndex,
                        self.experimentalData.ptychogram.shape[0],
                    )
                yield loop, positionLoop

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

    def objectPatchUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave) -> ObjectPatch:
        return epie_object_update(objectPatch, self.reconstruction.probe, DELTA, self.betaObject)

    def probeUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave) -> Probe:
        return epie_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe)
