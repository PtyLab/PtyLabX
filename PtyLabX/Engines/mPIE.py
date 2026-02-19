import numpy as np


import logging
import sys

import tqdm
from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Params.Params import Params

# PtyLab imports
from PtyLabX.Engines._jit_kernels import momentum_step, mpie_object_update, mpie_probe_update
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class mPIE(BaseEngine):
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
        self.name = "mPIE"

    @property
    def keepPatches(self):
        """Wether or not to keep track of the individual object update patches.

        This strongly increases the amount of memory required, only use when absolutely required.

        """
        return hasattr(self, "patches")

    @keepPatches.setter
    def keepPatches(self, keep_them):

        if keep_them:
            self.logger.info("Keeping patches!")
            self.patches = np.zeros(
                (
                    self.experimentalData.ptychogram.shape[0],
                    *self.reconstruction.shape_O,
                ),
                np.complex64,
            )
        else:
            self.logger.info("Not keeping patches")
            if hasattr(self, "patches"):
                del self.patches

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

    def reconstruct(
        self,
        experimentalData: ExperimentalData = None,
        reconstruction: Reconstruction = None,
        probe_update_switch: bool = True,
        vis_after_each_iteration=None,
    ):
        """Reconstruct object. If experimentalData is given, it replaces the current data. Idem for reconstruction."""

        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)

        self._prepareReconstruction()
        # set object and probe buffers, in case object and probe are changed in the _prepareReconstruction() step
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()
        # actual reconstruction MPIE_engine
        self.pbar = tqdm.trange(self.numIterations, desc="mPIE", file=sys.stdout, leave=True)
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            self.pbar_pos = tqdm.tqdm(self.positionIndices, leave=False, desc="ptychogram", file=sys.stdout)
            for positionLoop, positionIndex in enumerate(self.pbar_pos):
                # get object patch, stored as self.probe
                # self.reconstruction.make_probe(positionIndex)

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
                # self.viewer.layers['update'].data[positionIndex] = abs(DELTA ** 2).get()
                # import pyqtgraph as pg
                # pg.QtGui.QGuiApplication.processEvents()

                # object update
                if self.params.objectTVregSwitch and loop % self.params.objectTVfreq == 0:
                    object_patch = self.objectPatchUpdate_TV(objectPatch, DELTA)
                else:
                    object_patch = self.objectPatchUpdate(objectPatch, DELTA)

                if self.keepPatches:
                    self.patches[positionIndex, ..., sy, sx] = np.asarray(abs(object_patch) ** 2)
                else:
                    self.reconstruction.object = self.reconstruction.object.at[..., sy, sx].set(object_patch)

                # probe update
                if probe_update_switch:
                    weight = 1
                    if self.params.weigh_probe_updates_by_intensity:
                        weight = self.experimentalData.relative_intensity(positionIndex)
                        # print(f'for position {positionIndex}, using weight {weight}')

                    self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA, weight)
                    # self.reconstruction.push_probe_update(self.reconstruction.probe, positionIndex, self.experimentalData.ptychogram.shape[0])

                if self.params.positionCorrectionSwitch:
                    self.positionCorrection(objectPatch, positionIndex, sy, sx)
                    # self.pbar_pos.write(f'Corr: ...')

                # momentum updates
                if np.random.rand(1) > 0.95:
                    self.objectMomentumUpdate()
                    if probe_update_switch:
                        self.probeMomentumUpdate()
                # yield positionLoop, positionIndex

            # get error metric
            self.getErrorMetrics()
            # yield 1,1

            # apply Constraints
            self.applyConstraints(loop)
            # yield 1, 1

            # show reconstruction
            self.showReconstruction(loop)

            if callable(vis_after_each_iteration):
                vis_after_each_iteration(loop, self.reconstruction)

        pass

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

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray, weight: float):
        return mpie_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe, self.alphaProbe, weight)
