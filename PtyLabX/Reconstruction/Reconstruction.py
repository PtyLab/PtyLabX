from __future__ import annotations

import logging
import time
from copy import copy
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from PtyLabX import Params
from PtyLabX.ExperimentalData.ExperimentalData import ExperimentalData

# logging.basicConfig(level=logging.DEBUG)
from PtyLabX._types import ExitWave, ObjectArray, ObjectPatch, Probe
from PtyLabX.Regularizers import TV, metric_at
from PtyLabX.utils.initializationFunctions import initialProbeOrObject
from PtyLabX.utils.visualisation import plot_alignment


def calculate_pixel_positions(
    encoder_corrected: np.ndarray, dxo: float, No: int, Np: int, asint: bool
) -> np.ndarray:
    """
    Calculate the pixel positions.
    """
    positions = np.round(encoder_corrected / dxo)  # encoder is in m, positions0 and positions are in pixels
    positions = positions + No // 2 - Np // 2
    if asint:
        positions = positions.astype(int)
    return positions


class Reconstruction(object):
    """
    This object will contain all the things that can be modified by a reconstruction.

    In itself, it's little more than a data holder. It is initialized with an ExperimentalData object.

    Some parameters which are "immutable" within the ExperimentalData can be modified
    (e.g. zo modification by zPIE during the reconstruction routine). All of them
    are defined in the listOfReconstructionProperties
    """

    _Nd: int | None = None

    # --- Scalar reconstruction state ---
    zMomentum: float
    wavelength: float | None
    _zo: float | None
    dxd: float | None
    dxp: float
    theta: float | None
    spectralDensity: np.ndarray | None
    entrancePupilDiameter: float | None
    zled: float | None
    NA: float | None
    No: int
    # --- Incoherent mode counts ---
    nlambda: int
    nosm: int
    npsm: int
    nslice: int
    # --- Purity ---
    purityProbe: float
    purityObject: float
    purityProbeHist: list[float]
    # --- Positions ---
    positions0: np.ndarray
    encoder_corrected: np.ndarray | None
    # --- Init type ---
    initialObject: str
    initialProbe: str
    # --- 6D shapes ---
    shape_O: tuple[int, int, int, int, int, int]
    shape_P: tuple[int, int, int, int, int, int]
    initialGuessObject: jax.Array
    initialGuessProbe: jax.Array | None
    # --- JAX arrays (set by initializeObjectProbe) ---
    object: ObjectArray      # shape: (nlambda, nosm, 1, nslice, No, No)
    probe: Probe             # shape: (nlambda, 1, npsm, nslice, Np, Np)
    objectMomentum: ObjectArray
    probeMomentum: Probe
    # --- Engine working arrays (set during reconstruction) ---
    esw: ExitWave            # current exit surface wave
    ESW: ExitWave            # current detector-plane field (after forward propagation)
    eswUpdate: ExitWave      # updated exit surface wave
    objectBuffer: ObjectArray
    probeBuffer: Probe
    errorAtPos: jax.Array
    detectorError: jax.Array
    beamWidthX: float
    beamWidthY: float
    linearOverlap: float
    areaOverlap: float | jax.Array
    # --- Engine-set working arrays ---
    Iestimated: jax.Array      # estimated detector intensity
    Imeasured: jax.Array       # measured detector intensity
    background: jax.Array      # background additive term (backgroundModeSwitch)
    reference: jax.Array       # reference field (used by pcPIE/zPIE)
    intensity_mask: jax.Array  # custom intensity masking
    probe_stack: jax.Array     # OPR per-position probe stack (set by OPR engine)
    probe_storage: Any         # OPRP probe storage (set by OPRP engine)
    thetaHistory: jax.Array    # illumination angle history (set by aPIE)
    initialProbe_filename: str # path to probe/object init file
    # --- Engine-specific optional state ---
    thetaMomentum: float       # aPIE illumination momentum
    zHistory: list[float]      # zPIE z-distance history
    TV_history: list[float]    # zPIE TV metric history
    merit: np.ndarray          # zPIE merit function value
    dz: np.ndarray             # e3PIE/zPIE slice thickness
    refrIndex: float           # e3PIE refractive index
    H: jax.Array               # e3PIE propagation kernel
    objectProd: jax.Array      # e3PIE intermediate product
    objectMomentum_v: jax.Array   # qNewton object momentum
    probeMomentum_v: jax.Array    # qNewton probe momentum
    probeWindow: np.ndarray | jax.Array  # absorbing boundary window (also stored on BaseEngine)
    # --- Error metric ---
    error: list[float]
    # --- Logger ---
    logger: logging.Logger

    # Note: zo, the sample-detector distance, is always read.
    listOfReconstructionPropertiesCPM = [
        "wavelength",
        # 'zo',
        "dxd",
        "theta",
        "spectralDensity",
        "entrancePupilDiameter",
    ]
    listOfReconstructionPropertiesFPM = [
        "wavelength",
        # 'zo',
        "dxd",
        "zled",
        "NA",
    ]

    def __init__(self, data: ExperimentalData, params: Params) -> None:

        self.zMomentum = 0
        self.wavelength = None
        self._zo = None
        self.dxd = None
        self.theta = None

        # positions including possible misalignment correction
        self.encoder_corrected = None

        self.logger = logging.getLogger("Reconstruction")
        self.data = data
        self.params = params
        self.copyAttributesFromExperiment(data)
        self.computeParameters()
        self.initializeSettings()

        pass

    # @property
    # def probe(self):
    #     # convenience function. Updates the temporary probe. Nothing in probe is updated
    #     # return self._probe
    #     return self.probe_storage.get_temporary()#_probe_storage.get(None)
    #
    # @probe.setter
    # def probe(self, new_probe):
    #     # ignore this for now
    #     # self._probe = new_probe
    #     # self.probe_storage.set_temporary(new_probe)

    def copyAttributesFromExperiment(self, data: ExperimentalData) -> None:
        """
        Copy all the attributes from the experiment that are in listOfReconstructionProperties (CPM or FPM)
        """
        self.logger.debug("Copying attributes from Experimental Data")
        if self.data.operationMode == "CPM":
            listOfReconstructionProperties = self.listOfReconstructionPropertiesCPM
        elif self.data.operationMode == "FPM":
            listOfReconstructionProperties = self.listOfReconstructionPropertiesFPM
        for key in listOfReconstructionProperties:
            self.logger.info("Copying attribute %s", key)
            # setattr(self, key, copy(np.array(getattr(data, key))))
            setattr(self, key, copy(getattr(data, key)))

        # set the distance, this has to be last
        # In FPM the sample to detector distance is irrelevant
        # LED-to-sample distance is the more important factor that affects
        # wave propagation and illumination angle
        if self.data.operationMode == "CPM":
            self.zo = getattr(data, "zo")

        # set the original positions
        if self.encoder_corrected is None:
            self.encoder_corrected = data.encoder.copy()

    def reset_positioncorrection(self) -> None:
        """Reset the position corrections."""
        self.encoder_corrected = self.data.encoder.copy()

    @property
    def zo(self) -> float | None:
        """Distance from sample to detector. Also updates all derived qualities."""
        return self._zo

    @zo.setter
    def zo(self, new_value: float) -> None:
        self._zo = new_value
        if self.data.operationMode == "CPM":
            self.logger.debug(f"Changing sample-detector distance to {new_value}")
            assert self.wavelength is not None
            self.dxp = self.wavelength * self._zo / self.Ld
        elif self.data.operationMode == "FPM":
            self.logger.debug(f"Changing illumination-to-sample distance to {new_value}")
            self.zled = self._zo

    def computeParameters(self) -> None:
        """
        compute parameters that can be altered by the user later.
        """

        if self.data.operationMode == "CPM":
            # CPM dxp (depending on the propagatorType, if none given, assum Fraunhofer/Fresnel)
            # self.dxp = self.wavelength * self._zo / self.Ld
            # if entrancePupilDiameter is not provided in the hdf5 file, set it to be one third of the probe FoV.
            if self.data.entrancePupilDiameter is None:
                self.data.entrancePupilDiameter = self.Lp / 3
            # if spectralDensity is not provided in the hdf5 file, set it to be a 1d array of the wavelength
            if isinstance(self.spectralDensity, type(None)):
                # this is a confusing name, it should be the wavelengths, not the intensity of the different
                # wavelengths
                assert self.wavelength is not None
                self.spectralDensity = np.atleast_1d(float(self.wavelength))

        elif self.data.operationMode == "FPM":
            # FPM dxp (different from CPM due to lens-based systems)
            assert self.dxd is not None
            assert self.data.magnification is not None
            self.dxp = self.dxd / self.data.magnification
            # the propagation distance that is meaningful in this context is the
            # illumination to sample distance for LED array based microscopes
            assert self.zled is not None
            self.zo = float(self.zled)
            # if NA is not provided in the hdf5 file, set Fourier pupil entrance diameter it to be half of the Fourier space FoV.
            # then estimate the NA from the pupil diameter in the Fourier plane
            assert self.wavelength is not None
            if isinstance(self.NA, type(None)):
                self.data.entrancePupilDiameter = self.Lp / 2
                assert self.data.entrancePupilDiameter is not None
                self.NA = self.data.entrancePupilDiameter * self.wavelength / (2 * self.dxp**2 * self.Np)
            else:
                # compute the pupil radius in the Fourier plane
                self.data.entrancePupilDiameter = 2 * self.dxp**2 * self.Np * self.NA / self.wavelength

        # set object pixel numbers
        if not hasattr(self, "No"):
            self.No = self.Np * 2**2  # unimportant but leave it here as it's required for self.positions
            # we need space for the probe as well, on both sides that would be half the probe
            range_pixels = np.max(self.positions, axis=0) - np.min(self.positions, axis=0)
            # print(range_pixels)
            range_pixels = np.max(range_pixels) + self.Np * 2
            if range_pixels % 2 == 1:
                range_pixels += 1
            self.No = np.max([self.Np, range_pixels])

    def make_alignment_plot(self, saveit: bool = False):
        return plot_alignment(self, saveit=saveit)

    def initializeSettings(self) -> None:
        """
        Initialize the attributes that have to do with a reconstruction
        or experimentalData fields which will become "reconstruction"

        This method just sets the settings. It sets the what kind of initial guess should be used for initialObject
        and initialProbe but it does not compute them yet. That will be done by calling initializeObjectProbe()

        :return:
        """
        # create a 6D object where which allows to have:
        # 1. polychromatic = nlambda
        # 2. mixed state object - nosm
        # 3. mixed state probe - npsm
        # 4. multislice object (thick) - nslice
        self.nlambda = 1
        self.nosm = 1
        self.npsm = 1
        self.nslice = 1

        # beam and object purity (# default initial value for plots.)
        self.purityProbe = 1
        self.purityObject = 1
        self.purityProbeHist = []

        self.positions0 = self.positions.copy()

        if self.data.operationMode == "FPM":
            self.initialObject = "upsampled"
            self.initialProbe = "circ"
        elif self.data.operationMode == "CPM":
            self.initialProbe = "circ"
            self.initialObject = "ones"
        else:
            self.initialProbe = "circ"
            self.initialObject = "ones"

    def prepare_probe(self, i: int) -> None:
        """Replace probe with the i-th TSVD estimate.

        This function is used in OPRP
        """
        raise NotImplementedError()

    def initializeObjectProbe(self, force: bool = True) -> None:

        # initialize object and probe
        self.initializeObject(force=force)
        self.initializeProbe(force=force)

        # set object and probe objects as JAX arrays (required for .at[].set() in engines)
        self.object = jnp.array(self.initialGuessObject)
        self.probe = jnp.array(self.initialGuessProbe)

    def initializeObject(self, type_of_init: str | None = None, force: bool = True) -> None:
        if not force:
            raise NotImplementedError()
        if type_of_init is not None:
            self.initialObject = type_of_init
        self.logger.info("Initial object set to %s", self.initialObject)
        self.shape_O = (
            self.nlambda,
            self.nosm,
            1,
            self.nslice,
            self.No,
            self.No,
        )
        if self.initialObject == "recon":
            # Load the object from an existing reconstruction
            self.initialGuessObject = jnp.array(self.loadResults(self.initialProbe_filename, datatype="object"), dtype=jnp.complex64)
        else:
            self.initialGuessObject = jnp.array(
                initialProbeOrObject(self.shape_O, self.initialObject, self, self.logger), dtype=jnp.complex64
            )

        # self.initialGuessObject *= 1e-2

    @staticmethod
    def loadResults(fileName: str | Path, datatype: str = "probe") -> np.ndarray:
        """
        Loads data from a ptylab reconstruction file.
        """
        with h5py.File(fileName) as archive:
            data = np.copy(np.array(archive[datatype]))
        return data

    def initializeProbe(self, force: bool = False) -> None:
        if self.data.entrancePupilDiameter is None:
            # if it is not set, set it to something reasonable
            self.logger.warning("entrancePupilDiameter not set. Setting to one third of the FoV of the probe.")
            self.data.entrancePupilDiameter = self.Lp / 3
        self.logger.info("Initial probe set to %s", self.initialProbe)
        self.shape_P = (
            self.nlambda,
            1,
            self.npsm,
            self.nslice,
            int(self.Np),
            int(self.Np),
        )

        if self.initialProbe == "recon":
            self.initialGuessProbe = jnp.array(self.loadResults(self.initialProbe_filename, datatype="probe"), dtype=jnp.complex64)
        else:
            if force:
                self.initialGuessProbe = None
            # if force:
            #     self.initialProbe = "circ"
            self.initialGuessProbe = jnp.array(initialProbeOrObject(self.shape_P, self.initialProbe, self), dtype=jnp.complex64)

    # initialize momentum, called in specific engines with momentum accelaration
    def initializeObjectMomentum(self) -> None:
        self.objectMomentum = jnp.zeros(self.initialGuessObject.shape, dtype=jnp.complex64)

    def initializeProbeMomentum(self) -> None:
        assert self.initialGuessProbe is not None
        self.probeMomentum = jnp.zeros(self.initialGuessProbe.shape, dtype=jnp.complex64)

    def load_object(self, filename: str | Path) -> None:
        """
        Load the object from a previous reconstruction

        Parameters
        ----------
        filename: .hdf5 file
            Filenamne of the reconstruction whose object should be loaded.

        Returns
        -------

        """
        with h5py.File(filename, "r") as archive:
            obj = np.array(archive["object"])
            obj = obj[
                : self.shape_O[0],
                : self.shape_O[1],
                : self.shape_O[2],
                : self.shape_O[3],
                : self.shape_O[4],
                : self.shape_O[5],
            ]
            if np.all(np.array(obj.shape) == np.array(self.shape_O)):
                self.object = jnp.array(obj)
            else:
                raise RuntimeError(
                    f"Shape of saved probe cannot be extended to shape of required probe. File: {archive['object'].shape}. Need: {self.shape_O}"
                )

    def load_probe(self, filename: str | Path, expand_npsm: bool = False, center_phase: bool = False) -> None:
        """
        Load the probe from a previous reconstruction.

        Parameters
        ----------
        filename: .hdf5 file
            The filename of the reconstruction whose probe should be loaded.

        Returns
        -------

        """
        with h5py.File(filename, "r") as archive:
            probe = np.array(archive["probe"])
            N_probe_read = probe.shape[-1]
            # roughly extract the center
            ss = slice(
                np.clip(N_probe_read // 2 - self.Np // 2, 0, None),
                np.clip(N_probe_read // 2 - self.Np // 2 + int(self.Np), 0, N_probe_read),
            )
            probe = probe[: self.nlambda, :1, : self.npsm, : self.nslice, ss, ss]
            if np.all(np.array(probe.shape) == np.array(self.shape_P)):
                self.probe = jnp.array(probe)
            else:
                raise RuntimeError(
                    f"Shape of saved probe cannot be extended to shape of required probe. File: {archive['probe'].shape}. Need: {self.shape_P}"
                )
        if center_phase:
            self._center_probe_angle()

    def _center_probe_angle(self) -> None:
        """Center the angle of propagation for the probe."""
        from scipy.ndimage import fourier_shift
        from skimage.registration import phase_cross_correlation

        p0 = np.squeeze(self.probe)[0]
        shift = phase_cross_correlation(p0, 0 * p0 + 1, normalization=None, space="fourier")[0]
        phexp = np.fft.fftshift(fourier_shift(0 * p0 + 1j, -shift / 2))
        self.probe *= phexp

    def load(self, filename: str | Path) -> None:
        """Load the results given by saveResults."""
        with h5py.File(filename, "r") as archive:
            self.probe = jnp.array(archive["probe"])
            self.object = jnp.array(archive["object"])
            self.error = [float(x) for x in np.asarray(archive["error"])]
            self.wavelength = float(np.array(archive["wavelength"]))
            self.dxp = float(np.array(archive["dxp"]))
            self.purityProbe = float(np.array(archive["purityProbe"]))
            self.purityObject = float(np.array(archive["purityObject"]))
            self.zo = float(np.array(archive["zo"]))
            if "theta" in archive.keys():
                self.theta = float(np.array(archive["theta"]))

    def saveResults(self, fileName: str | Path = "recent", type: str = "all", squeeze: bool = False) -> None:
        """
        Save reconstruction results.


        Parameters
        ----------
        fileName
        type
        squeeze


        Returns
        -------

        """

        allowed_save_types = ["all", "object", "probe", "probe_stack"]
        if type not in allowed_save_types:
            raise NotImplementedError(f"Only {allowed_save_types} are allowed keywords for type")
        if not squeeze:

            def squeezefun(x):
                return x
        else:
            squeezefun = np.squeeze
        if type == "all":
            if self.data.operationMode == "CPM":
                with h5py.File(fileName, "w") as hf:
                    hf.create_dataset("probe", data=self.probe, dtype="complex64")
                    hf.create_dataset("object", data=self.object, dtype="complex64")
                    hf.create_dataset("error", data=self.error, dtype="f")
                    hf.create_dataset("zo", data=self._zo, dtype="f")
                    hf.create_dataset("wavelength", data=self.wavelength, dtype="f")
                    hf.create_dataset("dxp", data=self.dxp, dtype="f")
                    hf.create_dataset("purityProbe", data=self.purityProbe, dtype="f")
                    hf.create_dataset("purityObject", data=self.purityObject, dtype="f")
                    hf.create_dataset("I object", data=abs(self.object), dtype="f")
                    hf.create_dataset("I probe", data=abs(self.probe), dtype="f")
                    hf.create_dataset("encoder_corrected", data=self.encoder_corrected)

                    if hasattr(self, "theta"):
                        if self.theta is not None:
                            hf.create_dataset("theta", data=self.theta, dtype="f")

            if self.data.operationMode == "FPM":
                hf = h5py.File(fileName, "w")
                hf.create_dataset("probe", data=self.probe, dtype="complex64")
                hf.create_dataset("object", data=self.object, dtype="complex64")
                hf.create_dataset("error", data=self.error, dtype="f")
                hf.create_dataset("zled", data=self.zled, dtype="f")
                hf.create_dataset("wavelength", data=self.wavelength, dtype="f")
                hf.create_dataset("dxp", data=self.dxp, dtype="f")
        elif type == "probe":
            with h5py.File(fileName, "w") as hf:
                hf.create_dataset("probe", data=squeezefun(self.probe), dtype="complex64")
        elif type == "object":
            with h5py.File(fileName, "w") as hf:
                hf.create_dataset("object", data=squeezefun(self.object), dtype="complex64")
        elif type == "probe_stack":
            hf = h5py.File(str(fileName) + "_probe_stack.hdf5", "w")
            hf.create_dataset("probe_stack", data=np.asarray(self.probe_stack), dtype="complex64")
        print("The reconstruction results (%s) have been saved" % type)

    # detector coordinates
    @property
    def Nd(self) -> int:
        return self.data.ptychogram.shape[1]

    @property
    def xd(self) -> np.ndarray:
        """Detector coordinates 1D"""
        assert self.dxd is not None
        return np.linspace(-self.Nd / 2, self.Nd / 2, int(self.Nd)) * self.dxd

    @property
    def Xd(self) -> np.ndarray:
        """Detector coordinates — shape (1, Nd), broadcasts with Yd."""
        return self.xd.reshape(1, -1)

    @property
    def Yd(self) -> np.ndarray:
        """Detector coordinates — shape (Nd, 1), broadcasts with Xd."""
        return self.xd.reshape(-1, 1)

    @property
    def Ld(self) -> float:
        """Detector size in SI units."""
        assert self.dxd is not None
        return self.Nd * self.dxd

    # probe coordinates
    @property
    def Np(self) -> int:
        """Probe pixel numbers"""
        Np = self.Nd
        return Np

    @property
    def Lp(self) -> float:
        """probe size in SI units"""
        Lp = self.Np * self.dxp
        return Lp

    @property
    def xp(self) -> np.ndarray:
        """Probe coordinates 1D"""
        try:
            return np.linspace(-self.Np / 2, self.Np / 2, int(self.Np)) * self.dxp
        except AttributeError as e:
            raise AttributeError(e, 'probe pixel number "Np" and/or probe sampling "dxp" not defined yet')

    @property
    def Xp(self) -> np.ndarray:
        """Probe coordinates — shape (1, Np), broadcasts with Yp."""
        return self.xp.reshape(1, -1)

    @property
    def Yp(self) -> np.ndarray:
        """Probe coordinates — shape (Np, 1), broadcasts with Xp."""
        return self.xp.reshape(-1, 1)

    # Object coordinates
    @property
    def dxo(self) -> float:
        """object pixel size, always equal to probe pixel size."""
        dxo = self.dxp
        return dxo

    @property
    def Lo(self) -> float:
        """Field of view (entrance pupil plane)"""
        return self.No * self.dxo

    @property
    def xo(self) -> np.ndarray:
        """object coordinates 1D"""
        try:
            return np.linspace(-self.No / 2, self.No / 2, int(self.No)) * self.dxo
        except AttributeError as e:
            raise AttributeError(e, 'object pixel number "No" and/or pixel size "dxo" not defined yet')

    @property
    def Xo(self) -> np.ndarray:
        """Object coordinates — shape (1, No), broadcasts with Yo."""
        return self.xo.reshape(1, -1)

    @property
    def Yo(self) -> np.ndarray:
        """Object coordinates — shape (No, 1), broadcasts with Xo."""
        return self.xo.reshape(-1, 1)

    # scan positions in pixel
    @property
    def positions(self) -> np.ndarray:
        """estimated positions in pixel numbers(real space for CPM, Fourier space for FPM)
        note: Positions are given in row-column order and refer to the
        pixel in the upper left corner of the respective data matrix;
        -1st example: suppose the 2nd row of positions0 is [3, 4] and the
        operation mode is 'CPM'. That implies that the second intensity
        in the spectrogram updates an object patch that has
        its left uppper corner pixel at the pixel coordinates [3, 4]
        -2nd example: suppose the 2nd row of positions0 is [3, 4] and the
        operation mode is 'FPM'. That implies that the second intensity
        in the spectrogram is updates a patch which has pixel coordinates
        [3,4] in the high-resolution Fourier transform
        """
        assert self.encoder_corrected is not None
        if self.data.operationMode == "FPM":
            assert self.wavelength is not None
            assert self.zled is not None
            conv = -(1 / self.wavelength) * self.dxo * self.Np
            positions = np.round(
                conv
                * self.encoder_corrected
                / np.sqrt(self.encoder_corrected[:, 0] ** 2 + self.encoder_corrected[:, 1] ** 2 + self.zled**2)[
                    ..., None
                ]
            )

            try:
                positions = positions + self.No // 2 - self.Np // 2
            except Exception:
                pass

            return positions.astype(int)
        else:
            return calculate_pixel_positions(self.encoder_corrected, self.dxo, self.No, self.Np, asint=True)

    # system property list
    @property
    def NAd(self) -> float:
        """Detection NA"""
        assert self.zo is not None
        NAd = self.Ld / (2 * self.zo)
        return NAd

    @property
    def DoF(self) -> float:
        """expected Depth of field"""
        assert self.wavelength is not None
        DoF = self.wavelength / self.NAd**2
        # self.Dof2 = 5.2 *self.dxp**2 /self.wavelength
        return DoF

    def describe_reconstruction(self) -> str:
        minmax_tv = ""
        if self.params.TV_autofocus_min_z is not None and self.params.TV_autofocus_max_z is not None:
            minmax_tv = f"(min: {self.params.TV_autofocus_min_z * 1e3}, max: {self.params.TV_autofocus_max_z * 1e3}.)"
        info = f"""
        Experimental data:
        - Number of ptychograms: {self.data.ptychogram.shape}
        - Number of pixels ptychogram: {self.data.Nd}
        - Ptychogram size: {self.data.Ld * 1e3} mm
        - Pixel pitch: {self.data.dxd * 1e6} um
        - Scan size: {1e3 * (self.data.encoder.max(axis=0) - self.data.encoder.min(axis=0))} mm 
        
        Reconstruction:
        - number of pixels: {self.No}
        - Pixel pitch: {self.dxo * 1e6} um
        - Field of view: {self.Lo * 1e3} mm
        - Scan size in pixels: {self.positions.max(axis=0) - self.positions.min(axis=0)}
        - Propagation distance: {self.zo * 1e3 if self.zo is not None else 'N/A'} mm {minmax_tv}
        - Probe FoV: {self.Lp * 1e3} mm
        
        Derived parameters:
        - NA detector: {self.NAd}
        - DOF: {self.DoF * 1e6} um
        
        """
        self.logger.info(info)
        return info

    @property
    def quadraticPhase(self):
        """These functions are cached internally in Python and therefore no longer required."""
        raise NotImplementedError("Quadratic phase is no longer cached. ")

    @property
    def transferFunction(self):
        raise NotImplementedError("Quad phase is not longer cached")

    @property
    def Q1(self):
        raise NotImplementedError("Q1 is no longer available")

    @property
    def Q2(self):
        raise NotImplementedError("Q2 is no longer available")

    def TV_autofocus(self, params: Params, loop: int | None) -> tuple[float | None, np.ndarray | None, tuple | None]:
        """Perform an autofocusing step based on optimizing the total variation.

        If not required, returns none. Otherwise, returns the value of the TV at the current z0."""
        start_time = time.time()

        if self.data.operationMode == "FPM":
            raise NotImplementedError(
                f"Not implemented/tested for FPM. Set params.TV_autofocus to False. Got {params.TV_autofocus}"
            )
        if not params.TV_autofocus:
            return None, None, None
        if loop is not None:
            if loop % params.TV_autofocus_run_every != 0:
                return None, None, None

        if params.l2reg:
            self.logger.warning(
                "Both TV_autofocus and L2reg are turned on. This usually leads to poor performance. Consider disabling l2reg if the probe collapses to focal points"
            )

        d = params.TV_autofocus_range_dof
        nplanes = params.TV_autofocus_nplanes
        dz = np.linspace(-1, 1, nplanes) * d * self.DoF

        if params.TV_autofocus_what == "object":
            field = self.object[self.nlambda // 2, 0, 0, self.nslice // 2, :, :]
        elif params.TV_autofocus_what == "probe":
            field = self.probe[self.nlambda // 2, 0, 0, self.nslice // 2, :, :]
        else:
            raise NotImplementedError(
                f"So far, only object and probe are valid options for params.T_autofocus_what. Got {params.TV_autofocus_what}"
            )

        ss = params.TV_autofocus_roi
        if isinstance(ss, list):
            # semi-smart way to set up an AOI.
            # if the coordinates are a list, expand the list for y and x
            ss = np.array(ss)
            if ss.ndim == 1:
                ss = np.repeat(ss[None], axis=0, repeats=2)

            N = field.shape[-1]
            sy, sx = [slice(int(s[0] * N), int(s[1] * N)) for s in ss]
            # make them the same size if they're not
            sy = slice(sy.start, sy.start + sx.stop - sx.start)
        else:
            sy, sx = ss, ss

        assert self.wavelength is not None
        merit, OEs = metric_at(
            field,
            dz,
            self.dxo,  # same as dxp
            self.wavelength,
            (sy, sx),
            intensity_only=self.params.TV_autofocus_intensityonly,
            metric=self.params.TV_autofocus_metric,
            return_propagated=True,
        )
        # from here on we are looking at 11 data points, work on CPU
        # as it's much more convenient and faster
        feedback = np.sum(dz * merit) / np.sum(merit)

        scores = np.vstack([self.zo + dz, merit])

        self.zMomentum *= params.TV_autofocus_friction
        self.zMomentum += params.TV_autofocus_stepsize * feedback
        # now, clip it to the bounds
        delta_z = self.zo - np.clip(
            self.zo + self.zMomentum,
            self.params.TV_autofocus_min_z,
            self.params.TV_autofocus_max_z,
        )
        self.zo -= delta_z
        end_time = time.time()
        self.logger.info(
            f"TV autofocus took {end_time - start_time} seconds, and moved focus by {-delta_z * 1e6} micron"
        )
        indices = np.array([nplanes // 2, np.argmax(merit)])
        OEs = OEs[indices]
        phexp = OEs.sum((-2, -1), keepdims=True).conj()
        phexp = phexp / abs(phexp)
        OEs *= phexp
        return (
            merit[nplanes // 2] / float(np.asarray(abs(self.object[..., sy, sx]).mean())),
            np.hstack(OEs),
            (scores, self.zo),
        )

    def reset_TV_autofocus(self) -> None:
        """Reset the settings of TV autofocus. Can be useful to reset the memory effect if the steps are getting really large."""
        self.zMomentum = 0

    @property
    def TV(self) -> float:
        """Return the TV of the object"""
        return TV(self.object, 1e-2)
