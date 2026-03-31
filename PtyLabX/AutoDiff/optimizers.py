"""Per-parameter optimizer builder for AD-based ptychographic reconstruction.

Uses ``optax.multi_transform`` to assign independent learning rates to each
field in ``PtychographyState``.  Fields set to ``None`` (frozen) or with
``lr=0`` are handled by ``optax.set_to_zero()``.

Complex-valued parameters (object, probe) are wrapped with
``optax.contrib.split_real_and_imaginary`` so that Adam tracks momentum and
variance for the real and imaginary parts independently — the mathematically
correct treatment for Wirtinger-gradient-based optimisation.
"""

from __future__ import annotations

from dataclasses import fields

import optax

from PtyLabX.AutoDiff._state import PtychographyState


def build_optimizer(
    object_lr: float = 1e-3,
    probe_lr: float = 0.0,
) -> optax.GradientTransformation:
    """Build a per-parameter optimizer for ``PtychographyState``.

    Parameters
    ----------
    object_lr : float
        Learning rate for the object. Set to ``0`` to freeze.
    probe_lr : float
        Learning rate for the probe. Set to ``0`` to freeze.

    Returns
    -------
    optax.GradientTransformation
        Optimizer that can be used with ``optax.apply_updates``.
    """
    transforms: dict[str, optax.GradientTransformation] = {}
    label_map: dict[str, str] = {}

    # Object
    if object_lr > 0:
        transforms["object"] = optax.contrib.split_real_and_imaginary(optax.adam(object_lr))
    else:
        transforms["object"] = optax.set_to_zero()
    label_map["object"] = "object"

    # Probe
    if probe_lr > 0:
        transforms["probe"] = optax.contrib.split_real_and_imaginary(optax.adam(probe_lr))
    else:
        transforms["probe"] = optax.set_to_zero()
    label_map["probe"] = "probe"

    def _label_fn(state: PtychographyState) -> PtychographyState:
        """Map each PtychographyState field to its optimizer label."""
        return PtychographyState(**{f.name: label_map[f.name] for f in fields(state)})

    return optax.multi_transform(transforms, _label_fn)
