import numpy as np
import tqdm
from matplotlib import pyplot as plt

import jax.numpy as jnp
import logging
import sys

from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Operators.Operators import aspw
from PtyLabX.Params.Params import Params

# PtyLab imports
from PtyLabX.Engines._jit_kernels import epie_object_update, epie_probe_update
from PtyLabX.Reconstruction.Reconstruction import Reconstruction


class zPIE(BaseEngine):
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
        self.logger = logging.getLogger("zPIE")
        self.logger.info("Sucesfully created zPIE zPIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()
        self.name = "zPIE"

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50
        self.DoF = self.reconstruction.DoF
        self.zPIEgradientStepSize = 100  # gradient step size for axial position correction (typical range [1, 100])
        self.zPIEfriction = 0.7
        self.focusObject = True
        self.zMomentun = 0

    def show_defocus(self, scanrange_times_dof=1000, N_points=10):
        from PtyLabX.utils.visualisation import plot_defocus_stack

        z = np.linspace(-1, 1, N_points) * scanrange_times_dof * self.reconstruction.DoF
        reconstruction = self.reconstruction
        defocii = np.abs(
            np.array(
                [
                    aspw(
                        reconstruction.object,
                        dz,
                        reconstruction.wavelength,
                        reconstruction.Lo,
                    )[0]
                    for dz in z
                ]
            )
            ** 2
        )
        plot_defocus_stack(defocii, z)
        plt.show()

    def reconstruct(self, experimentalData=None, reconstruction=None):
        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)
        self._prepareReconstruction()

        ###################################### actual reconstruction zPIE_engine #######################################

        if not hasattr(self.reconstruction, "zHistory"):
            self.reconstruction.zHistory = []

        # preallocate grids
        if self.params.propagatorType == "ASP":
            n = self.reconstruction.Np * 1
        else:
            n = 2 * self.reconstruction.Np

        if not self.focusObject:
            n = self.reconstruction.Np

        self.pbar = tqdm.trange(
            self.numIterations, desc="zPIE", file=sys.stdout, leave=True
        )  # in order to change description to the tqdm progress bar
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            imProps = []

            # get positions
            if loop == 1:
                zNew = self.reconstruction.zo.copy()
            else:
                d = 10

                dz = np.linspace(-1, 1, 11) * d * self.DoF
                self.dz = dz

                merit = []
                # todo, mixed states implementation, check if more need to be put on GPU to speed up
                for k in np.arange(len(dz)):
                    imProp = None
                    if self.focusObject:
                        roi = slice(
                            self.reconstruction.No // 2 - n // 2,
                            self.reconstruction.No // 2 + n // 2,
                        )
                        imProp, _ = aspw(
                            u=jnp.squeeze(self.reconstruction.object[..., roi, roi]),
                            z=dz[k],
                            wavelength=self.reconstruction.wavelength,
                            L=self.reconstruction.dxo * n,
                            bandlimit=False,
                        )
                    else:
                        if self.reconstruction.nlambda == 1:
                            imProp, _ = aspw(
                                u=jnp.squeeze(self.reconstruction.probe[..., :, :]),
                                z=dz[k],
                                wavelength=self.reconstruction.wavelength,
                                L=self.reconstruction.Lp,
                            )
                        else:
                            nlambda = self.reconstruction.nlambda // 2
                            imProp, _ = aspw(
                                jnp.squeeze(
                                    self.reconstruction.probe[nlambda, ..., :, :]
                                ),
                                dz[k],
                                self.reconstruction.spectralDensity[nlambda],
                                self.reconstruction.Lp,
                            )
                    imProps.append(np.asarray(imProp))
                    # TV approach
                    aleph = 1e-2
                    gradx = jnp.roll(imProp, -1, axis=-1) - jnp.roll(imProp, 1, axis=-1)
                    grady = jnp.roll(imProp, -1, axis=-2) - jnp.roll(imProp, 1, axis=-2)
                    merit.append(
                        jnp.sum(jnp.sqrt(abs(gradx) ** 2 + abs(grady) ** 2 + aleph))
                    )
                    # take a tiny break, we may overask the GPU
                    # yield 0, 0

                merit = jnp.array(merit)
                if not hasattr(self.reconstruction, "TV_history"):
                    self.reconstruction.TV_history = []

                self.reconstruction.TV_history.append(
                    float(merit[len(merit) // 2])
                )
                merit = np.asarray(merit)
                feedback = np.sum(dz * merit) / np.sum(
                    merit
                )  # at optimal z, feedback term becomes 0

                print("Step size: ", feedback)
                self.zMomentun = (
                    self.zPIEfriction * self.zMomentun
                    + self.zPIEgradientStepSize * feedback
                )
                zNew = self.reconstruction.zo + self.zMomentun

                # asdlkcmasldk

            self.reconstruction.zHistory.append(self.reconstruction.zo)

            # print updated z
            self.pbar.set_description(
                "zPIE: update z = %.3f mm (dz = %.1f um)"
                % (self.reconstruction.zo * 1e3, self.zMomentun * 1e6)
            )

            # reset coordinates
            self.reconstruction.zo = zNew

            # re-sample is automatically done by using @property
            if self.params.propagatorType != "ASP":
                self.reconstruction.dxp = (
                    self.reconstruction.wavelength
                    * self.reconstruction.zo
                    / self.reconstruction.Ld
                )
                # reset propagatorType
                # self.reconstruction.quadraticPhase = jnp.array(np.exp(1.j * np.pi / (self.reconstruction.wavelength * self.reconstruction.zo)
                #                                                      * (self.reconstruction.Xp ** 2 + self.reconstruction.Yp ** 2)))
            ##################################################################################################################

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # print('Starting normal reconstruction loop')
                ### patch1 ###
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
            # yield positionLoop, positionIndex

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)
            # display it
            # self.showReconstruction(loop)

            self.merit = merit
            self.zNew = zNew
            self.reconstruction.merit = merit
            self.reconstruction.dz = dz

            self.reconstruction.make_alignment_plot(True)
            # show reconstruction
            if False:
                if loop == 0:
                    figure, axes = plt.subplots(
                        1, 3, num=666, squeeze=True, clear=True, figsize=(5, 5)
                    )
                    ax = axes[0]
                    ax_score = axes[1]
                    ax.set_title("Estimated distance (object-camera)")
                    ax.set_xlabel("iteration")
                    ax.set_ylabel("estimated z (mm)")
                    ax.set_xscale("symlog")

                    ax_score.set_title("TV score")
                    ax_score.set_xlabel("Distance [um]")
                    ax_score.set_ylabel("TV")
                    (score_line,) = ax_score.plot(dz * 1e6, merit)
                    (line,) = ax.plot(0, zNew, "o-")
                    plt.tight_layout()
                    plt.show(block=False)

                elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                    idx = np.linspace(
                        0,
                        np.log10(len(self.reconstruction.zHistory) - 1),
                        np.minimum(len(self.reconstruction.zHistory), 100),
                    )
                    idx = np.rint(10**idx).astype("int")

                    line.set_xdata(idx)
                    line.set_ydata(np.array(self.reconstruction.zHistory)[idx] * 1e3)

                    score_line.set_ydata(merit)
                    ax_score.set_ylim(merit.min() - 1, merit.max() + 1)
                    ax.set_xlim(0, np.max(idx))
                    ax.set_ylim(
                        np.min(self.reconstruction.zHistory) * 1e3,
                        np.max(self.reconstruction.zHistory) * 1e3,
                    )

                    figure.canvas.draw()
                    figure.canvas.flush_events()
            self.showReconstruction(loop)


    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return epie_object_update(objectPatch, self.reconstruction.probe, DELTA, self.betaObject)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        return epie_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe)

    #
    #
