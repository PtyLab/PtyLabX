# HDF5 Data Format

PtyLabX loads experimental data from HDF5 files. This page documents the required and optional fields for CPM and FPM.

## CPM required fields

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `ptychogram` | 3D array `(N, Nd, Nd)` | counts | Stack of N diffraction patterns, each Nd x Nd pixels |
| `wavelength` | scalar | meters | Illumination wavelength (e.g. `632.8e-9`) |
| `encoder` | 2D array `(N, 2)` | meters | Scan positions as (row, col) physical coordinates |
| `dxd` | scalar | meters | Detector pixel pitch |
| `zo` | scalar | meters | Sample-to-detector distance |

## CPM optional fields

| Field | Type | Units | Default | Description |
|-------|------|-------|---------|-------------|
| `entrancePupilDiameter` | scalar | meters | `Lp / 3` | Initial estimate of illumination beam diameter |
| `spectralDensity` | 1D array | meters | `None` | List of wavelengths for polychromatic reconstruction |
| `theta` | scalar | radians | `None` | Tilt angle for reflection-geometry ptychography |
| `emptyBeam` | 2D array `(Nd, Nd)` | counts | `None` | Reference probe image (for modulus-enforced probe) |

## Creating an HDF5 file

```python
import h5py
import numpy as np

with h5py.File("my_data.hdf5", "w") as f:
    f.create_dataset("ptychogram", data=diffraction_patterns)  # shape (N, Nd, Nd)
    f.create_dataset("wavelength", data=632.8e-9)
    f.create_dataset("encoder", data=scan_positions)            # shape (N, 2), in meters
    f.create_dataset("dxd", data=55e-6)
    f.create_dataset("zo", data=0.05)
    # optional
    f.create_dataset("entrancePupilDiameter", data=150e-6)
```

## Using example data

PtyLabX ships with built-in example datasets:

```python
from PtyLabX import ExperimentalData

# Load the built-in CPM simulation
data = ExperimentalData("example:simulation_cpm", operationMode="CPM")
```

You can also access the example data folder directly:

```python
from PtyLabX.io import getExampleDataFolder
print(getExampleDataFolder())  # path to example_data/
```

## Constructing ExperimentalData manually

Instead of loading from a file, you can set fields directly:

```python
from PtyLabX import ExperimentalData

data = ExperimentalData(operationMode="CPM")
data.ptychogram = my_diffraction_patterns
data.wavelength = 632.8e-9
data.encoder = my_scan_positions
data.dxd = 55e-6
data.zo = 0.05
data.entrancePupilDiameter = 150e-6
data._setData()  # computes derived quantities (Nd, numFrames, coordinates, etc.)
```

## Auto-computed fields

After loading, `ExperimentalData` automatically computes:

- `Nd` — detector array size (pixels)
- `numFrames` — number of diffraction patterns
- `Ld` — physical detector size (`Nd * dxd`)
- `energyAtPos` — integrated intensity at each scan position
- `maxProbePower` — maximum probe power estimate

## Data orientation

If your data appears flipped or rotated, use `setOrientation()`:

```python
data.setOrientation(3)  # try values 0-7 for different flip/rotation combos
```

## FPM required fields

For Fourier Ptychographic Microscopy, the required fields differ:

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `ptychogram` | 3D array `(N, Nd, Nd)` | counts | Diffraction patterns |
| `wavelength` | scalar | meters | Illumination wavelength |
| `encoder` | 2D array `(N, 2)` | meters | LED positions |
| `dxd` | scalar | meters | Detector pixel pitch |
| `zled` | scalar | meters | LED-to-sample distance |
| `magnification` | scalar | — | Objective magnification |

Optional FPM field: `NA` (numerical aperture of the objective).
