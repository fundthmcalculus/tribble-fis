"""One-slot-at-a-time firing bounds for IT2/GT2 refinement (issue #146).

`it2_refine`/`gt2_refine` move one membership per sub-problem. Before this
module every candidate rebuilt the upper/lower ``GaussianMixtureModel`` (per
alpha-plane, for GT2) and ran full forward passes, ``O(n * F * K * L)`` each.
Type-1 coordinate refinement stopped doing that with
:class:`tribblefis.kernel.IncrementalFIS`, which caches per-cell folds and
recomputes only the moved membership's cell and label column,
``O(n * (K + F))``, with output identical to a full pass.

The IT2 footprint is two Type-1 models -- upper and lower -- with the same
features, labels and membership counts, so they compile to *one* layout, and
an IT2 slot is the same compiled slot index in both. GT2 is ``K`` such pairs,
one per alpha-plane. So:

* :class:`IT2IncrementalFiring` is an upper/lower pair of ``IncrementalFIS``.
* :class:`GT2IncrementalFiring` is ``K`` of those, one per alpha-plane; a GT2
  slot candidate is pushed into every plane through
  ``GT2*Membership.alpha_cut``.

Both raise :class:`tribblefis.kernel.NotCompilable` for a model the flat layout
cannot hold (trapezoid/triangular memberships, ragged label sets, a missing
column); callers then keep the rebuild path, so nothing needs a flag.

Both follow ``IncrementalFIS``'s propose-then-commit protocol: ``evaluate``
returns candidate firing bounds without changing anything, ``commit`` accepts
the last evaluated candidate. Returned arrays are ``IncrementalFIS``'s reused
buffers, valid until the next ``evaluate`` on the same object.

Slot keys are ``(feature_name, label, mf_index)``, the triple
``it2_refine._iter_it2_slots``/``gt2_refine._iter_gt2_slots`` yield. Their
order (features, then labels in insertion order, then memberships) is the
order ``kernel.compile_model`` assigns slot indices in, which is what lets the
mapping below be a plain enumeration.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .gauss_data import GaussianMembership, NormPair
from .gt2_kernel import default_alpha_levels, extract_alpha_plane_model
from .it2_kernel import _extract_lower_model, _extract_upper_model
from .kernel import IncrementalFIS, NotCompilable, compile_model


def _slot_keys(model) -> list[tuple]:
    return [
        (fname, label, idx)
        for fname, fmodel in model.feature_models.items()
        for label, lmodel in fmodel.label_models.items()
        for idx in range(len(lmodel.memberships))
    ]


class IT2IncrementalFiring:
    """Upper/lower ``IncrementalFIS`` pair for one IT2 model on one dataset."""

    def __init__(self, it2_model, feature_arrays: Mapping[str, np.ndarray], norms: NormPair):
        upper = _extract_upper_model(it2_model)
        lower = _extract_lower_model(it2_model)
        for t1 in (upper, lower):
            for fmodel in t1.feature_models.values():
                for lmodel in fmodel.label_models.values():
                    if not all(isinstance(mf, GaussianMembership) for mf in lmodel.memberships):
                        raise NotCompilable("non-Gaussian IT2 membership")
        available = list(feature_arrays)
        up_c = compile_model(upper, available)
        lo_c = compile_model(lower, available)
        if (up_c.feature_names != lo_c.feature_names or up_c.labels != lo_c.labels
                or up_c.mu.shape != lo_c.mu.shape
                or not np.array_equal(up_c.slot_index, lo_c.slot_index)):
            raise NotCompilable("upper and lower models compile to different layouts")
        keys = _slot_keys(it2_model)
        if len(keys) != up_c.n_slots:
            raise NotCompilable("slot count mismatch")
        fm = up_c.feature_matrix(dict(feature_arrays))
        self.upper = IncrementalFIS(up_c, fm, norms)
        self.lower = IncrementalFIS(lo_c, fm, norms)
        self.labels = list(up_c.labels)
        self._slot = {key: i for i, key in enumerate(keys)}

    def base(self) -> tuple[np.ndarray, np.ndarray]:
        """``(firing_upper, firing_lower)`` for the committed model."""
        return self.upper.base, self.lower.base

    def evaluate(self, key: tuple, it2_mf) -> tuple[np.ndarray, np.ndarray]:
        """Candidate ``(firing_upper, firing_lower)`` with slot `key` set to `it2_mf`."""
        s = self._slot[key]
        fu = self.upper.evaluate_slot(s, it2_mf.upper_mf.mu, it2_mf.upper_mf.sigma)
        fl = self.lower.evaluate_slot(s, it2_mf.lower_mf.mu, it2_mf.lower_mf.sigma)
        return fu, fl

    def commit(self) -> None:
        self.upper.commit()
        self.lower.commit()


class GT2IncrementalFiring:
    """One :class:`IT2IncrementalFiring` per alpha-plane of a GT2 model."""

    def __init__(self, gt2_model, feature_arrays: Mapping[str, np.ndarray], norms: NormPair,
                 n_alpha_planes: int):
        self.alphas = default_alpha_levels(n_alpha_planes)
        self.planes = [
            IT2IncrementalFiring(extract_alpha_plane_model(gt2_model, float(a)), feature_arrays, norms)
            for a in self.alphas
        ]
        self.labels = self.planes[0].labels

    def base(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per-plane ``(firing_uppers, firing_lowers)`` for the committed model."""
        pairs = [p.base() for p in self.planes]
        return [fu for fu, _ in pairs], [fl for _, fl in pairs]

    def evaluate(self, key: tuple, gt2_mf) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Candidate per-plane bounds with slot `key` set to `gt2_mf`."""
        fus, fls = [], []
        for alpha, plane in zip(self.alphas, self.planes):
            fu, fl = plane.evaluate(key, gt2_mf.alpha_cut(float(alpha)))
            fus.append(fu)
            fls.append(fl)
        return fus, fls

    def commit(self) -> None:
        for plane in self.planes:
            plane.commit()
