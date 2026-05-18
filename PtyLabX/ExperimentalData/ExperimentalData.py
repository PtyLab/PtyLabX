import logging
from pathlib import Path

import jax
import numpy as np

from PtyLabX.io import readHdf5
from PtyLabX.utils.visualisation import show3Dslider


class ExperimentalData:
    """
    This is a container class for all the data associated with the ptychography reconstruction.
    It only holds attributes that are the same for every type of reconstruction.
    """

    # --- Fields loaded from HDF5 (required for CPM and FPM) ---
    ptychogram: np.ndarray | jax.Array  # 3D image stack of diffraction patterns (numpy at load, JAX after GPU transfer)
    wavelength: float  # illumination wavelength (meters)
    encoder: np.ndarray  # scan positions

    # --- CPM-specific fields ---
    dxd: float  # detector pixel size (meters)
    zo: float  # sample-to-detector distance (meters)
    entrancePupilDiameter: float | None  # probe aperture diameter (meters)
    spectralDensity: np.ndarray | None  # wavelength spectrum for polychromatic ptychography
    theta: float | None  # tilt angle for reflection geometry (radians)
    emptyBeam: np.ndarray | jax.Array | None  # reference image of the probe beam

    # --- FPM-specific fields ---
    zled: float | None  # LED-to-sample distance (meters)
    magnification: float | None  # objective magnification
    NA: float | None  # numerical aperture of the objective

    # --- Derived fields set by _setData() ---
    Nd: int  # detector array size (pixels)
    xd: np.ndarray  # 1D detector coordinate array (meters)
    Xd: np.ndarray  # 2D detector X coordinates (meters)
    Yd: np.ndarray  # 2D detector Y coordinates (meters)
    Ld: float  # detector physical size (meters)
    numFrames: int  # number of diffraction patterns
    energyAtPos: (
        np.ndarray | jax.Array
    )  # integrated intensity at each scan position (numpy at load, JAX after GPU transfer)
    maxProbePower: float  # maximum probe power across all positions
    # --- Engine-set computed arrays ---
    filename: Path  # resolved path to loaded HDF5 file
    W: jax.Array | None  # weighting mask (set by aPIE engine)
    ptychogramDownsampled: jax.Array | None  # downsampled ptychogram (CPSC engine)
    spectralPower: jax.Array | None  # spectral power (polychromatic engines)
    PSD: jax.Array | None  # power spectral density

    def __init__(self, filename: str | Path | None = None, operationMode: str = "CPM") -> None:
        self.logger = logging.getLogger("ExperimentalData")
        self.logger.debug("Initializing ExperimentalData object")

        self.operationMode = operationMode  # operationMode: 'CPM' or 'FPM', default is CPM is not given
        self._setFields()
        if filename is not None:
            self.loadData(filename)

    def _setFields(self) -> None:
        """
        Set the required and optional fields for ptyLab to work.
        ALL VALUES MUST BE IN METERS.
        """
        # These are the fields required for ptyLab to work (depending on the operationMode)
        if self.operationMode == "CPM":
            self.requiredFields = [
                "ptychogram",  # 3D image stack
                "wavelength",  # illumination lambda
                "encoder",  # diffracted field positions
                "dxd",  # pixel size
                "zo",  # sample to detector distance
            ]
            self.optionalFields = [
                "entrancePupilDiameter",  # used in CPM as the probe diameter
                "spectralDensity",  # CPM parameters: different wavelengths required for polychromatic ptychography
                "theta",  # CPM parameters: reflection tilt angle, required for
                "emptyBeam",  # image of the probe
            ]

        elif self.operationMode == "FPM":
            self.requiredFields = [
                "ptychogram",  # 3D image stack
                "wavelength",  # illumination lambda
                "encoder",  # diffracted field positions
                "dxd",  # detector pixel size
                "zled",  # LED to sample distance
                "magnification",  # magnification, used for FPM computations of dxp
            ]
            self.optionalFields = [
                # entrance pupil diameter, defined in lens-based microscopes as the aperture diameter, reqquired for FPM
                # 'entrancePupilDiameter'
                "NA",  # numerical aperture of the microscope
            ]
        else:
            raise ValueError('operationMode is not properly set, choose "CPM" or "FPM"')
        # Engine-set computed arrays (initialized to None; set during reconstruction)
        self.W = None
        self.ptychogramDownsampled = None
        self.spectralPower = None
        self.PSD = None

    def loadData(self, filename: str | Path) -> None:
        """
        Load data specified in filename.
        :type filename: str or Path
            Filename of dataset. There are three additional options:
                - example:simulation_cpm will load an example cmp dataset.
                - example:simulation_fpm will load an example fpm dataset.
                - test:nodata will load an essentially empty object
        :param python_order: bool
                Weather to change the input order of the files to match python convention.
                 Only in very special cases should this be false.
        :return:
        """
        import os

        if not os.path.exists(filename) and str(filename).startswith("example:"):
            from PtyLabX.io.readExample import examplePath

            self.filename = Path(examplePath(str(filename)))
        else:
            self.filename = Path(filename)

        # 1. check if the dataset contains what we need before loading
        readHdf5.checkDataFields(self.filename, self.requiredFields)
        # 2. load dictionary. Only the values specified by 'requiredFields'
        # in readHdf.py file were loaded
        measurementDict = readHdf5.loadInputData(self.filename, self.requiredFields, self.optionalFields)
        # 3. 'requiredFields' will be the attributes that must be set
        attributesToSet = measurementDict.keys()
        # 4. set object attributes as the essential data fields
        for a in attributesToSet:
            # make sure that property is not an attribtue
            attribute = str(a)
            if not isinstance(getattr(type(self), attribute, None), property):
                setattr(self, attribute, measurementDict[a])
            self.logger.debug("Setting %s", a)

        self._setData()
        # last step, just to be sure that it's the last thing we do: set orientation
        # this has to be last as it can actually change the data in self.ptychogram
        # depending on the orientation
        self.setOrientation(readHdf5.getOrientation(self.filename))

    def reduce_positions(self, start: int, end: int) -> None:
        """
        Reduce the number of positions for the reconstruction
        """
        self.ptychogram = self.ptychogram[start:end]
        self.encoder = self.encoder[start:end]

    def cropCenter(self, size: int) -> None:
        """
        The parameter size corresponds to the finale size of the diffraction patterns
        """
        if not isinstance(size, int):
            raise TypeError("Crop value is not valid. Int expected")

        x = self.ptychogram.shape[-1]
        startx = x // 2 - (size // 2)

        startx += 1

        self.ptychogram = self.ptychogram[..., startx : startx + size, startx : startx + size]

    def binData(self, binning: int) -> None:
        """
        :param binning: Binning parameter (int, e.g. 2)
        :return:
        """
        Ndp = self.ptychogram.shape[0]
        Ny = self.ptychogram.shape[1]
        Nx = self.ptychogram.shape[2]

        ptychogram_temp = np.copy(self.ptychogram)
        self.ptychogram = np.zeros((Ndp, Ny // binning, Nx // binning))

        # Loop through all dp
        for i in range(Ndp):
            temp = ptychogram_temp[i]
            reshaped_temp = temp.reshape(Ny // binning, binning, Nx // binning, binning)
            temp_binning = reshaped_temp.mean(axis=(1, 3))
            self.ptychogram[i] = np.copy(temp_binning)

    def setOrientation(self, orientation: int | None, force_contiguous: bool = True) -> None:
        """
        Sets the correct orientation. This function follows the ptypy convention.

        If orientation is None, it won't change the current orientation.
        """
        if orientation is None:  # do not update.
            return
        if not isinstance(orientation, int):
            raise TypeError("Orientation value is not valid.")
        if orientation == 0:  # don't change anything
            return
        if orientation == 1:
            # Invert column
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
        elif orientation == 2:
            # Invert rows
            self.ptychogram = np.flip(self.ptychogram, axis=-2)
        elif orientation == 3:
            # invert columns and rows
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
            self.ptychogram = np.flip(self.ptychogram, axis=-2)
        elif orientation == 4:
            # Transpose
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
        elif orientation == 5:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
        elif orientation == 6:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
            self.ptychogram = np.flip(self.ptychogram, axis=-2)
        elif orientation == 7:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
            self.ptychogram = np.flip(self.ptychogram, axis=-2)

        else:
            raise ValueError(f"Orientation {orientation} is not implemented")
        if force_contiguous:
            self.ptychogram = np.ascontiguousarray(self.ptychogram)

    def _setData(self) -> None:

        # Set the detector coordinates
        self.Nd = self.ptychogram.shape[-1]
        # Detector coordinates 1D
        self.xd = np.linspace(-self.Nd / 2, self.Nd / 2, int(self.Nd)) * self.dxd
        # Detector coordinates (ogrid: 1D views that broadcast to 2D)
        self.Xd, self.Yd = self.xd.reshape(1, -1), self.xd.reshape(-1, 1)
        # Detector size in SI units
        self.Ld = self.Nd * self.dxd

        # number of Frames
        self.numFrames = self.ptychogram.shape[0]
        # probe energy at each position
        self.energyAtPos = np.sum(abs(self.ptychogram), (-1, -2))
        # maximum probe power
        self.maxProbePower = np.sqrt(np.max(np.sum(self.ptychogram, (-1, -2))))

    def showPtychogram(self) -> None:
        """
        show ptychogram.
        """
        print(f"Min max ptychogram: {np.min(self.ptychogram)}, {self.ptychogram.max()}")
        ptychogram_np = np.asarray(self.ptychogram)
        log_ptychogram = np.log10(np.swapaxes(np.clip(ptychogram_np.astype(np.float32), 0, None), 1, 2) + 1)
        print(f"Min max ptychogram: {np.min(log_ptychogram)}, {log_ptychogram.max()}")
        show3Dslider(log_ptychogram)

    def relative_intensity(self, index: int) -> float:
        """
        Return the relative intensity of the ptychogram at index compared to the brightest one
        """
        if not hasattr(self, "_relative_intensity"):
            self._relative_intensity = self.ptychogram.mean((-2, -1))
            self._relative_intensity /= self._relative_intensity.mean() + 2 * self._relative_intensity.std()
        return self._relative_intensity[index]
