"""Shared type aliases for PtyLabX.

Array shape conventions (6D throughout):
    Probe:       (nlambda, 1,    npsm, nslice, Np, Np)   — complex64
    Object:      (nlambda, nosm, 1,    nslice, No, No)   — complex64
    ObjectPatch: (nlambda, nosm, npsm, nslice, Np, Np)   — complex64 (probe-sized region)
    ExitWave:    (nlambda, nosm, npsm, nslice, Np, Np)   — complex64
    Ptychogram:  (numFrames, Nd, Nd)                      — float32
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import numpy as np
from jaxtyping import Complex, Float

# Array that may be either a JAX or NumPy array (e.g., before/after backend conversion)
Array: TypeAlias = jax.Array | np.ndarray

# ---- 6D field types (annotation-only, no runtime checking) ----

# Probe field: complex, shape (nlambda, 1, npsm, nslice, Np, Np)
Probe: TypeAlias = Complex[jax.Array, "nlambda 1 npsm nslice Np Np"]

# Object field: complex, shape (nlambda, nosm, 1, nslice, No, No)
ObjectArray: TypeAlias = Complex[jax.Array, "nlambda nosm 1 nslice No No"]

# Object patch (extracted probe-sized region): same 6D shape as exit wave
ObjectPatch: TypeAlias = Complex[jax.Array, "nlambda nosm npsm nslice Np Np"]

# Exit surface wave / detector-plane field
ExitWave: TypeAlias = Complex[jax.Array, "nlambda nosm npsm nslice Np Np"]

# Full ptychogram stack (intensity measurements)
PtychogramStack: TypeAlias = Float[jax.Array, "numFrames Nd Nd"]

# Single detector frame
DetectorFrame: TypeAlias = Float[jax.Array, "Nd Nd"]
