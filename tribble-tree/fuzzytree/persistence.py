"""Persistence for fitted fuzzytree estimators (trees and HME models).

A fitted model is a ``FuzzyTreeNode`` tree of plain NamedTuples/numpy arrays
plus, for the HME, ordinary scikit-learn-style sub-estimators -- all standard
pickle-safe objects. This module is a thin, explicit wrapper over pickle
rather than a bespoke serialization format; the value it adds over a bare
``pickle.dump``/``pickle.load`` is a stamped format version and the fitted
class's name, so loading a stale or mismatched file fails with a clear error
instead of a confusing downstream ``AttributeError``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_FORMAT_VERSION = 1


def save_model(model: Any, path: str | Path) -> None:
    """Pickle a fitted fuzzytree estimator (tree, MIMO wrapper, or HME) to ``path``.

    Raises ``ValueError`` if the model has not been fit (nothing useful to
    reload) rather than silently pickling an incomplete estimator.
    """
    if not getattr(model, "is_fitted_", False):
        raise ValueError("Cannot save an unfitted estimator (call .fit() first).")
    payload = {
        "format_version": _FORMAT_VERSION,
        "class_module": type(model).__module__,
        "class_name": type(model).__qualname__,
        "model": model,
    }
    with open(Path(path), "wb") as f:
        pickle.dump(payload, f)


def load_model(path: str | Path) -> Any:
    """Load a model saved with ``save_model``.

    Raises ``ValueError`` if ``path`` was not written by ``save_model`` (e.g.
    a bare pickle of something else, or an unrelated file).
    """
    with open(Path(path), "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "model" not in payload or "format_version" not in payload:
        raise ValueError(f"{path} was not written by fuzzytree.persistence.save_model")
    return payload["model"]
