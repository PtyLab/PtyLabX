import numpy as np

# PtyLab imports
import logging
import sys

import tqdm

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params
from PtyLabX.Engines._jit_kernels import qnewton_object_update, qnewton_probe_update
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class qNewton(BaseEngine):
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
        self.logger = logging.getLogger("qNewton")
        self.logger.info("Sucesfully created qNewton qNewton_engine")

        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the qNewton settings.
        :return:
        """
        self.betaProbe = 1
        self.betaObject = 1
        self.regObject = 1
        self.regProbe = 1
        self.numIterations = 50

    def reconstruct(self, experimentalData: ExperimentalData = None):
        if experimentalData is not None:
            self.reconstruction.data = experimentalData
            self.experimentalData = experimentalData
        self._prepareReconstruction()

        self.pbar = tqdm.trange(self.numIterations, desc="qNewton", file=sys.stdout, leave=True)
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

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

            # todo clearMemory implementation

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return qnewton_object_update(objectPatch, self.reconstruction.probe, DELTA, self.betaObject, self.regObject)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return qnewton_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe, self.regProbe)
