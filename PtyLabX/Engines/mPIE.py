import logging

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm, trange

from PtyLabX._types import ExitWave, ObjectPatch, Probe
from PtyLabX.Engines._jit_kernels import momentum_step, mpie_object_update, mpie_probe_update
from PtyLabX.Engines.BaseEngine import BaseEngine
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLabX.Monitor.Monitor import Monitor
from PtyLabX.Operators._propagation_kernels import _make_quad_phase
from PtyLabX.Operators.Operators import (
    _make_transferfunction_ASP,
    _make_transferfunction_polychrome_ASP,
    _make_transferfunction_scaledASP,
    _make_transferfunction_scaledPolychromeASP,
)
from PtyLabX.Params.Params import Params
from PtyLabX.Reconstruction.Reconstruction import Reconstruction
from PtyLabX.Regularizers import grad_TV
from PtyLabX.utils.utils import fft2c, ifft2c


class mPIE(BaseEngine):
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

    def initializeReconstructionParams(self) -> None:
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

        self.reconstruction.probeWindow = jnp.abs(self.reconstruction.probe)

    def _can_use_fused_kernel(self):
        """Check if the closure-based fused JIT kernel can handle this configuration.

        Only falls back to the slow path for truly exotic features that can't be
        expressed as pure JAX operations within the kernel.
        """
        constraint = self.params.intensityConstraint.lower()
        prop_type = self.params.propagatorType.lower()
        return (
            not self.keepPatches
            and constraint != "interferometric"
            and not self.params.CPSCswitch
            and prop_type not in ("twosteppolychrome", "sas")
        )

    def _build_fused_step(self, probe_update_switch):
        """Build a specialized fused JIT kernel for the current reconstruction configuration.

        Captures all configuration as Python values in the closure, so JAX resolves
        if-branches at trace time and compiles a single optimized XLA kernel per
        unique configuration. Pre-fetches propagator transfer functions from the
        @lru_cache'd factories in Operators.py.

        Returns a @jax.jit function: step(object_array, probe, ...) -> (object_array, probe, ...)
        """
        # --- Capture config as Python values (resolved at JIT trace time) ---
        prop_type = self.params.propagatorType.lower()
        constraint = self.params.intensityConstraint.lower()
        fftshift = self.params.fftshiftSwitch
        fpm_mode = self.experimentalData.operationMode == "FPM"
        has_fourier_mask = self.params.FourierMaskSwitch
        has_background = self.params.backgroundModeSwitch
        has_denoising = self.params.adaptiveDenoisingSwitch
        has_tv = self.params.objectTVregSwitch
        tv_step_size = getattr(self.params, "objectTVregStepSize", 0.0) if has_tv else 0.0
        do_probe_update = probe_update_switch
        has_intensity_mask = hasattr(self.params, "intensityMask") and self.params.intensityMask
        has_position_correction = self.params.positionCorrectionSwitch
        num_frames = self.experimentalData.numFrames

        recon = self.reconstruction

        # --- Pre-fetch propagator-specific data (already @lru_cache'd, no recomputation) ---
        # These arrays are captured in the closure and compiled into the XLA kernel.
        if prop_type == "fresnel":
            quad_phase_fwd = _make_quad_phase(recon.zo, recon.wavelength, recon.Np, recon.dxp)
            quad_phase_inv = quad_phase_fwd.conj()
        elif prop_type == "asp":
            tf = _make_transferfunction_ASP(
                fftshift, recon.nosm, recon.npsm, recon.Np, recon.zo, recon.wavelength, recon.Lp, recon.nlambda
            )
            tf_fwd = jnp.fft.ifftshift(tf, axes=(-2, -1))
            tf_inv = tf_fwd.conj()
        elif prop_type == "polychromeasp":
            tf = _make_transferfunction_polychrome_ASP(
                self.params.propagatorType,
                fftshift,
                recon.nosm,
                recon.npsm,
                recon.Np,
                recon.zo,
                recon.wavelength,
                recon.Lp,
                recon.nlambda,
                tuple(recon.spectralDensity),  # ty: ignore[invalid-argument-type]
            )
            tf_fwd = tf
            tf_inv = tf.conj()
        elif prop_type == "scaledasp":
            Q1_fwd, Q2_fwd = _make_transferfunction_scaledASP(
                self.params.propagatorType,
                fftshift,
                recon.nlambda,
                recon.nosm,
                recon.npsm,
                recon.Np,
                recon.zo,
                recon.wavelength,
                recon.dxo,
                recon.dxd,
            )
            Q1_inv, Q2_inv = Q1_fwd.conj(), Q2_fwd.conj()
        elif prop_type == "scaledpolychromeasp":
            Q1_fwd, Q2_fwd = _make_transferfunction_scaledPolychromeASP(
                fftshift,
                recon.nlambda,
                recon.nosm,
                recon.npsm,
                recon.zo,
                recon.Np,
                tuple(recon.spectralDensity),  # ty: ignore[invalid-argument-type]
                recon.dxo,
                recon.dxd,
            )
            Q1_inv, Q2_inv = Q1_fwd.conj(), Q2_fwd.conj()

        # Pre-fetch intensity mask if needed
        if has_intensity_mask:
            intensity_mask = recon.intensity_mask

        @jax.jit
        def step(
            object_array,
            probe,
            ptychogram_frame,
            detector_error,
            row,
            col,
            positionIndex,
            betaObject,
            alphaObject,
            betaProbe,
            alphaProbe,
            weight,
            background,
            W,
            use_tv_this_iter,
        ):
            gimmel = 1e-10

            # 1. Extract object patch via dynamic slicing (JIT-compatible)
            patch_shape = object_array.shape[:4] + probe.shape[-2:]
            objectPatch = jax.lax.dynamic_slice(object_array, (0, 0, 0, 0, row, col), patch_shape)

            # 2. Exit surface wave
            esw = objectPatch * probe

            # 3. Forward propagation (branch resolved at trace time)
            if prop_type == "fraunhofer":
                ESW = fft2c(esw, fftshift)
            elif prop_type == "fresnel":
                ESW = fft2c(esw * quad_phase_fwd, fftshift)
            elif prop_type in ("asp", "polychromeasp"):
                ESW = ifft2c(fft2c(esw, fftshift) * tf_fwd, fftshift)
            elif prop_type in ("scaledasp", "scaledpolychromeasp"):
                ESW = ifft2c(fft2c(esw * Q1_fwd) * Q2_fwd)
            elif prop_type == "identity":
                ESW = esw

            # 4. Intensity estimate
            Iestimated = jnp.sum(jnp.abs(ESW) ** 2, axis=(0, 1, 2))[-1]

            # 5. Background addition
            if has_background:
                Iestimated = Iestimated + background

            # 6. Measured intensity
            Imeasured = ptychogram_frame

            # 7. Detector error (RMSD)
            currentDetectorError = jnp.abs(Imeasured - Iestimated)
            detector_error = detector_error.at[positionIndex].set(currentDetectorError)

            # 8. Adaptive denoising
            if has_denoising:
                Ameasured = Imeasured**0.5
                Aestimated = jnp.abs(Iestimated) ** 0.5
                noise = jnp.abs(jnp.mean(Ameasured - Aestimated))
                Ameasured = jnp.where(Ameasured - noise < 0, 0.0, Ameasured - noise)
                Imeasured = Ameasured**2

            # 9. Intensity constraint (branch resolved at trace time)
            if constraint == "standard":
                frac = jnp.sqrt(Imeasured / (Iestimated + gimmel))
            elif constraint == "fluctuation":
                if has_fourier_mask:
                    aleph = jnp.sum(Imeasured * Iestimated * W) / jnp.sum(Imeasured * Imeasured * W)
                else:
                    aleph = jnp.sum(Imeasured * Iestimated) / jnp.sum(Imeasured * Imeasured)
                frac = (1 + aleph) / 2 * Imeasured / (Iestimated + gimmel)
            elif constraint == "exponential":
                x = currentDetectorError / (Iestimated + gimmel)
                W_exp = jnp.exp(-0.05 * x)
                frac = W_exp * jnp.sqrt(Imeasured / (Iestimated + gimmel)) + (1 - W_exp)
            elif constraint in ("poisson", "poission"):
                frac = (Imeasured + gimmel) / (Iestimated + gimmel)
            elif constraint == "proxpoisson":
                lam = 1.0
                frac = (Imeasured + lam * Iestimated + gimmel) / ((1 + lam) * Iestimated + gimmel)

            # 10. Fourier mask application
            if has_fourier_mask:
                frac = W * frac + (1 - W)

            # 11. Update ESW
            if has_intensity_mask:
                ESW = ESW * (frac * intensity_mask + (intensity_mask - 1))
            else:
                ESW = ESW * frac

            # 12. Background update
            if has_background:
                if has_fourier_mask:
                    background = background * (1 + 1 / num_frames * (jnp.sqrt(frac) - 1)) ** 2 * W
                else:
                    background = background * (1 + 1 / num_frames * (jnp.sqrt(frac) - 1)) ** 2

            # 13. Backward propagation (branch resolved at trace time)
            if prop_type == "fraunhofer":
                eswUpdate = ifft2c(ESW, fftshift)
            elif prop_type == "fresnel":
                eswUpdate = ifft2c(ESW, fftshift) * quad_phase_inv
            elif prop_type in ("asp", "polychromeasp"):
                eswUpdate = ifft2c(fft2c(ESW, fftshift) * tf_inv, fftshift)
            elif prop_type in ("scaledasp", "scaledpolychromeasp"):
                eswUpdate = ifft2c(fft2c(ESW) * Q2_inv) * Q1_inv
            elif prop_type == "identity":
                eswUpdate = ESW

            # 14. DELTA
            DELTA = eswUpdate - esw

            # 15. Object update
            if has_tv:
                # When TV is active, conditionally apply TV regularization based on iteration
                def _obj_update_with_tv(_):
                    # ePIE-style update + TV term (matches BaseEngine.objectPatchUpdate_TV)
                    frac_obj_tv = probe.conj() / jnp.max(jnp.sum(jnp.abs(probe) ** 2, axis=(0, 1, 2, 3)))
                    TV_update = grad_TV(objectPatch, epsilon=1e-2)
                    return (
                        objectPatch
                        + betaObject * jnp.sum(frac_obj_tv * DELTA, axis=(0, 2, 3), keepdims=True)
                        + tv_step_size * betaObject * TV_update
                    )

                def _obj_update_standard(_):
                    absP2_ = jnp.abs(probe) ** 2
                    Pmax_ = jnp.max(jnp.sum(absP2_, axis=(0, 1, 2, 3)), axis=(-1, -2))
                    if fpm_mode:
                        frac_obj_ = (
                            jnp.abs(probe) / Pmax_ * probe.conj() / (alphaObject * Pmax_ + (1 - alphaObject) * absP2_)
                        )
                    else:
                        frac_obj_ = probe.conj() / (alphaObject * Pmax_ + (1 - alphaObject) * absP2_)
                    return objectPatch + betaObject * jnp.sum(frac_obj_ * DELTA, axis=2, keepdims=True)

                object_patch_new = jax.lax.cond(use_tv_this_iter, _obj_update_with_tv, _obj_update_standard, None)
            else:
                # Standard mPIE object update (no TV)
                absP2 = jnp.abs(probe) ** 2
                Pmax = jnp.max(jnp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
                if fpm_mode:
                    frac_obj = jnp.abs(probe) / Pmax * probe.conj() / (alphaObject * Pmax + (1 - alphaObject) * absP2)
                else:
                    frac_obj = probe.conj() / (alphaObject * Pmax + (1 - alphaObject) * absP2)
                object_patch_new = objectPatch + betaObject * jnp.sum(frac_obj * DELTA, axis=2, keepdims=True)

            # 16. Probe update
            if do_probe_update:
                absO2 = jnp.abs(objectPatch) ** 2
                Omax = jnp.max(jnp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
                frac_probe = objectPatch.conj() / (alphaProbe * Omax + (1 - alphaProbe) * absO2)
                probe = probe + weight * betaProbe * jnp.sum(frac_probe * DELTA, axis=1, keepdims=True)

            # 17. Write back object patch
            object_array = jax.lax.dynamic_update_slice(object_array, object_patch_new, (0, 0, 0, 0, row, col))

            if has_position_correction:
                # Return original objectPatch for position correction
                return object_array, probe, detector_error, background, objectPatch
            else:
                return object_array, probe, detector_error, background

        return step

    def reconstruct(
        self,
        experimentalData: ExperimentalData | None = None,
        reconstruction: Reconstruction | None = None,
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

        use_fused = self._can_use_fused_kernel()
        if use_fused:
            step_fn = self._build_fused_step(probe_update_switch)
            self.logger.info(
                "Using fused JIT kernel (propagator=%s, constraint=%s)",
                self.params.propagatorType,
                self.params.intensityConstraint,
            )

        has_position_correction = self.params.positionCorrectionSwitch
        has_tv = self.params.objectTVregSwitch
        tv_freq = getattr(self.params, "objectTVfreq", 1) if has_tv else 1
        has_background = self.params.backgroundModeSwitch
        has_fourier_mask = self.params.FourierMaskSwitch
        weigh_by_intensity = self.params.weigh_probe_updates_by_intensity

        # Pre-compute relative intensity weights if needed
        if weigh_by_intensity:
            rel_intensity = jnp.array(
                [self.experimentalData.relative_intensity(i) for i in range(self.experimentalData.numFrames)]
            )

        # Cache positions array to avoid recomputing the property every position
        positions = self.reconstruction.positions

        # Prepare Fourier mask (W array) — dummy if not used
        W = self.experimentalData.W if has_fourier_mask else jnp.zeros(1)

        # actual reconstruction MPIE_engine
        self.pbar = trange(self.numIterations, desc="mPIE", leave=True)
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            self.pbar_pos = tqdm(self.positionIndices, leave=False, desc="ptychogram", mininterval=0.5)

            # Determine TV flag for this iteration
            use_tv_this_iter = has_tv and (loop % tv_freq == 0)

            # Get current background (may be updated per position)
            background = self.reconstruction.background if has_background else jnp.zeros(1)

            if use_fused:
                for positionLoop, positionIndex in enumerate(self.pbar_pos):
                    row, col = int(positions[positionIndex, 0]), int(positions[positionIndex, 1])

                    weight = float(rel_intensity[positionIndex]) if weigh_by_intensity else 1.0

                    result = step_fn(
                        self.reconstruction.object,
                        self.reconstruction.probe,
                        self.experimentalData.ptychogram[positionIndex],
                        self.reconstruction.detectorError,
                        row,
                        col,
                        positionIndex,
                        self.betaObject,
                        self.alphaObject,
                        self.betaProbe,
                        self.alphaProbe,
                        weight,
                        background,
                        W,
                        use_tv_this_iter,
                    )

                    if has_position_correction:
                        (
                            self.reconstruction.object,
                            self.reconstruction.probe,
                            self.reconstruction.detectorError,
                            background,
                            objectPatch,
                        ) = result
                        sy = slice(row, row + self.reconstruction.Np)
                        sx = slice(col, col + self.reconstruction.Np)
                        self.positionCorrection(objectPatch, positionIndex, sy, sx)
                    else:
                        (
                            self.reconstruction.object,
                            self.reconstruction.probe,
                            self.reconstruction.detectorError,
                            background,
                        ) = result

                    # momentum updates
                    if np.random.rand(1) > 0.95:
                        self.objectMomentumUpdate()
                        if probe_update_switch:
                            self.probeMomentumUpdate()

                # Write back updated background
                if has_background:
                    self.reconstruction.background = background

            else:
                for positionLoop, positionIndex in enumerate(self.pbar_pos):
                    # get object patch, stored as self.probe
                    # self.reconstruction.make_probe(positionIndex)

                    row, col = positions[positionIndex]
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

                        self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA, weight)

                    if self.params.positionCorrectionSwitch:
                        self.positionCorrection(objectPatch, positionIndex, sy, sx)

                    # momentum updates
                    if np.random.rand(1) > 0.95:
                        self.objectMomentumUpdate()
                        if probe_update_switch:
                            self.probeMomentumUpdate()

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

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

    def objectPatchUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave) -> ObjectPatch:
        return mpie_object_update(
            objectPatch,
            self.reconstruction.probe,
            DELTA,
            self.betaObject,
            self.alphaObject,
            fpm_mode=(self.experimentalData.operationMode == "FPM"),
        )

    def probeUpdate(self, objectPatch: ObjectPatch, DELTA: ExitWave, weight: float) -> Probe:
        return mpie_probe_update(self.reconstruction.probe, objectPatch, DELTA, self.betaProbe, self.alphaProbe, weight)
