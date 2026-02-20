"""Demo: mPIE reconstruction with different intensity constraints.

Runs the same simulated dataset through mPIE with various intensity constraint
types to verify the fused JIT kernel works for each. Prints timing for comparison.

Usage:
    uv run python example_scripts/demo_constraints.py
"""

import time

import matplotlib.pylab as plt
import numpy as np
from scipy.signal import convolve2d

from PtyLabX import Engines, ExperimentalData, Monitor, Params, Reconstruction
from PtyLabX.Operators.Operators import aspw
from PtyLabX.utils.scanGrids import GenerateNonUniformFermat
from PtyLabX.utils.utils import cart2pol, circ, gaussian2D

# matplotlib.use("Agg")  # non-interactive backend for headless runs

# ============================================================
# 1. Simulate data (identical to demo_timeit.py)
# ============================================================
wavelength = 632.8e-9
zo = 5e-2
Nd = 2**7
dxd = 16 * 4.5e-6

Ld = Nd * dxd
dxp = wavelength * zo / Ld
Np = Nd
Lp = dxp * Np
xp = np.arange(-Np // 2, Np // 2) * dxp
Xp, Yp = np.meshgrid(xp, xp)

No = 2**9
dxo = dxp
Lo = dxo * No
xo = np.arange(-No // 2, No // 2) * dxo
Xo, Yo = np.meshgrid(xo, xo)

# Generate probe
f = 5e-3
pinhole = circ(Xp, Yp, Lp / 2)
pinhole = convolve2d(pinhole, gaussian2D(5, 1).astype(np.float32), mode="same")
probe = aspw(pinhole, 2 * f, wavelength, Lp, is_FT=False)[0]
aperture = circ(Xp, Yp, 3 * Lp / 4)
aperture = convolve2d(aperture, gaussian2D(5, 3).astype(np.float32), mode="same")
probe = probe * np.exp(-1.0j * 2 * np.pi / wavelength * (Xp**2 + Yp**2) / (2 * f)) * aperture
probe = aspw(probe, 2 * f, wavelength, Lp, is_FT=False)[0]

# Generate object
d = 1e-3
b = 33
theta, rho = cart2pol(Xo, Yo)
t = (1 + np.sign(np.sin(b * theta + 2 * np.pi * (rho / d) ** 2))) / 2
t = t * circ(Xo, Yo, Lo) * (1 - circ(Xo, Yo, 200 * dxo)) + circ(Xo, Yo, 130 * dxo)
obj = convolve2d(t, gaussian2D(5, 3), mode="same")

# Generate scan grid
numPoints = 100
radius = 100
y_coord, x_coord = GenerateNonUniformFermat(numPoints, radius=radius, power=1)

encoder = np.vstack((y_coord * dxo, x_coord * dxo)).T
positions = np.round(encoder / dxo)
offset = 50
positions = (positions + No // 2 - Np // 2 + offset).astype(int)
numFrames = len(x_coord)

# Generate ptychogram
ptychogram = np.zeros((numFrames, Nd, Nd))
for loop in np.arange(numFrames):
    row, col = positions[loop]
    sy = slice(row, row + Np)
    sx = slice(col, col + Np)
    objectPatch = obj[..., sy, sx].copy()
    esw = objectPatch * probe
    ESW = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(esw), norm="ortho"))
    ptychogram[loop] = abs(ESW) ** 2

# ============================================================
# 2. Run mPIE with different intensity constraints
# ============================================================
constraints = [
    "standard",
    "fluctuation",
    "exponential",
    "poisson",
    "proxpoisson",
]

NUM_ITERATIONS = 20

for constraint_name in constraints:
    print(f"\n{'=' * 60}")
    print(f"Intensity Constraint: {constraint_name}")
    print(f"{'=' * 60}")

    exampleData = ExperimentalData(operationMode="CPM")
    exampleData.ptychogram = ptychogram
    exampleData.wavelength = wavelength
    exampleData.encoder = encoder
    exampleData.dxd = dxd
    exampleData.zo = zo
    exampleData.entrancePupilDiameter = 150e-6
    exampleData.spectralDensity = None
    exampleData.theta = None
    exampleData._setData()

    monitor = Monitor()
    monitor.figureUpdateFrequency = 1000  # disable visualization
    monitor.verboseLevel = "low"

    params = Params()
    params.positionOrder = "random"
    params.propagatorType = "Fraunhofer"
    params.intensityConstraint = constraint_name
    params.probePowerCorrectionSwitch = True
    params.comStabilizationSwitch = True

    reconstruction = Reconstruction(exampleData, params)
    reconstruction.npsm = 1
    reconstruction.nosm = 1
    reconstruction.nlambda = 1
    reconstruction.nslice = 1
    reconstruction.initialProbe = "circ"
    reconstruction.initialObject = "ones"
    reconstruction.initializeObjectProbe()
    reconstruction.probe = reconstruction.probe * np.exp(
        1.0j * 2 * np.pi / reconstruction.wavelength * (reconstruction.Xp**2 + reconstruction.Yp**2) / (2 * 6e-3)
    )

    engine = Engines.mPIE(reconstruction, exampleData, params, monitor)
    engine.numIterations = NUM_ITERATIONS

    t0 = time.perf_counter()
    engine.reconstruct()
    elapsed = time.perf_counter() - t0

    final_error = reconstruction.error[-1] if len(reconstruction.error) > 0 else float("nan")
    print(f"\n  Time: {elapsed:.2f}s for {NUM_ITERATIONS} iterations")
    print(f"  Final error: {final_error:.6f}")
    print(f"  Fused kernel used: {engine._can_use_fused_kernel()}")

plt.close("all")
print(f"\n{'=' * 60}")
print("All constraint tests completed!")
print(f"{'=' * 60}")
