"""Explicit, opt-in feature scaling for fuzzy inference systems.

FIS membership functions expect bounded inputs, and features with content at
multiple scales (e.g. spanning several decades) are usually better fit after a
log transform. Neither transformation is applied automatically by any
estimator in this package -- callers who want it compose one of the scalers
below into an ``sklearn.pipeline.Pipeline`` in front of the estimator::

    from sklearn.pipeline import make_pipeline
    from tribblefis.scaling import StandardScalar
    from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier

    pipe = make_pipeline(StandardScalar(), MixtureOfGaussiansFuzzyClassifier())
    pipe.fit(X_train, y_train)

Two scalers are offered, differing only in the final normalization:

- :class:`StandardScalar` -- z-score standardization (``mu=0``, ``sigma=1``).
- :class:`UnitScalar` -- min-max bounding to ``[0, 1]`` (or a custom range).

Both auto-detect and log1p-transform wide-dynamic-range features first (see
``log_dynamic_range``), and both mirror sklearn scalers' ``fit``/``transform``/
``fit_transform``/``inverse_transform``/``get_feature_names_out`` surface.
Which one is the better default for a given FIS is an open, empirical
question -- see the ``grad-school`` proposal-defense evaluation.

Which features get logged
-------------------------

Both scalers log1p-transform a subset of features before normalizing. There
are two ways to choose that subset:

- ``log_dynamic_range`` (default ``3.0``) -- *automatic* detection: any
  feature whose values span at least that many decades is logged.
- ``log_features`` -- an *explicit* list of the features to log. When given,
  it is authoritative and automatic detection is skipped entirely.

Prefer ``log_features`` when you know which features are multi-scale, because
a single dynamic-range threshold cannot express every subset. Concretely, on
UCI Concrete the set ``['Slag', 'FlyAsh', 'Age']`` is not reachable by *any*
threshold: the features ordered by dynamic range are ``Slag`` (4.25 decades),
``Age`` (2.56), ``Superplasticizer`` (1.27), ``FlyAsh`` (0.91), ``Cement``
(0.72), ..., so any threshold low enough to admit ``FlyAsh`` also admits
``Superplasticizer``. The desired subset is not a prefix of that ordering, so
no scalar threshold selects it::

    UnitScalar(log_features=["Slag", "FlyAsh", "Age"])
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class _FuzzyScalarBase(TransformerMixin, BaseEstimator):
    """Shared DataFrame bookkeeping and log-detection for the scalars below.

    Not part of the public API -- use :class:`StandardScalar` or
    :class:`UnitScalar`.
    """

    def _as_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy() if self.copy else X
        feature_names = getattr(self, "feature_names_in_", None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(np.asarray(X, dtype=float), columns=feature_names)

    def _resolve_log_features(self, X_df):
        """Chooses the features to log1p, and their per-feature shifts.

        ``log_features`` wins outright when it is not ``None``: automatic
        dynamic-range detection is skipped rather than unioned in, so an
        explicit list is exactly the set that gets logged. ``log_features=[]``
        therefore means "log nothing" and is distinct from ``None``, which
        means "auto-detect". Delegates to :meth:`_detect_log_features` in the
        ``None`` case.
        """
        if self.log_features is None:
            return self._detect_log_features(X_df)
        columns = list(X_df.columns)
        selected = self._validate_log_features(columns)
        return selected, {col: self._log_shift_for(X_df[col]) for col in selected}

    def _validate_log_features(self, columns):
        """Maps each entry of ``log_features`` onto a column of the fitted
        data, raising on anything that does not name one.

        An entry is matched against the column labels first; an integer that
        is not itself a label is then taken as a positional index, which is
        the natural way to name features of an ndarray (whose columns
        :meth:`_as_dataframe` synthesises as ``feature_0``, ``feature_1``,
        ...). So for ndarray input both ``[0, 2]`` and
        ``["feature_0", "feature_2"]`` select the same features. Duplicate
        entries collapse to one -- logging a feature twice is never intended,
        and applying log1p twice would silently corrupt it.

        Validation happens here, at ``fit`` time, rather than in
        ``__init__``, per the sklearn convention that ``__init__`` only stores
        its parameters (otherwise ``clone`` and ``get_params`` break).
        """
        if isinstance(self.log_features, str):
            raise TypeError(
                "log_features must be a list of feature names, not a bare string "
                f"{self.log_features!r}; did you mean [{self.log_features!r}]?"
            )
        selected = []
        for entry in self.log_features:
            if entry in columns:
                col = entry
            elif isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
                index = int(entry)
                if not -len(columns) <= index < len(columns):
                    raise ValueError(
                        f"log_features contains index {index}, which is out of range "
                        f"for data with {len(columns)} features."
                    )
                col = columns[index]
            else:
                raise ValueError(
                    f"log_features contains {entry!r}, which is not a feature of the "
                    f"data passed to fit. Valid features are: {columns}."
                )
            if col not in selected:
                selected.append(col)
        return selected

    @staticmethod
    def _log_shift_for(values):
        """Shift that shows log1p a non-negative argument: the feature's
        minimum. All-NaN features have no meaningful shift, so they get 0.0
        and stay NaN through the transform rather than poisoning it."""
        vals = values.dropna()
        return float(vals.min()) if len(vals) else 0.0

    def _detect_log_features(self, X_df):
        """A feature qualifies for log1p when its dynamic range (log10 of
        max/min absolute non-zero value) is at least ``log_dynamic_range``
        decades -- the same heuristic used to flag energy/content spread
        across multiple scales.

        Note that this skips all-zero and all-NaN features, which have no
        dynamic range to speak of; naming one in ``log_features`` explicitly
        does log it (harmlessly, since ``log1p(0) == 0``).
        """
        log_features = []
        log_shift = {}
        if self.log_dynamic_range is not None:
            for col in X_df.columns:
                vals = X_df[col].dropna()
                non_zero = vals[vals != 0]
                if len(non_zero) == 0:
                    continue
                abs_vals = np.abs(non_zero)
                dynamic_range = np.log10(abs_vals.max() / abs_vals.min())
                if dynamic_range >= self.log_dynamic_range:
                    log_features.append(col)
                    log_shift[col] = float(vals.min())
        return log_features, log_shift

    def _apply_log(self, X_df):
        if not self.log_features_:
            return X_df
        X_df = X_df.copy()
        for col in self.log_features_:
            shift = self.log_shift_[col]
            X_df[col] = np.log1p(np.clip(X_df[col] - shift, a_min=0, a_max=None))
        return X_df

    def _undo_log(self, X_df):
        if not self.log_features_:
            return X_df
        X_df = X_df.copy()
        for col in self.log_features_:
            X_df[col] = np.expm1(X_df[col]) + self.log_shift_[col]
        return X_df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_, dtype=object)


class UnitScalar(_FuzzyScalarBase):
    """Bounds features to ``[0, 1]`` (or ``feature_range``), log-transforming
    wide-dynamic-range ones first.

    Args:
        feature_range: Desired ``(min, max)`` of transformed data.
        log_dynamic_range: See :mod:`tribblefis.scaling`. Threshold in decades
            for *automatic* log detection. ``None`` disables automatic
            detection entirely (min-max bounding only). Ignored whenever
            ``log_features`` is given.
        log_features: Explicit list of features to log1p, by column name or --
            for ndarray input -- by positional index. **Authoritative when
            given:** automatic dynamic-range detection is skipped rather than
            unioned in, so setting both this and ``log_dynamic_range`` is not
            an error but this one wins. ``[]`` means "log nothing" and is
            distinct from the default ``None``, which means "auto-detect".
            Naming a feature that is not in the data raises at ``fit`` time.
        clip: Whether ``transform`` clips values falling outside the range
            seen during ``fit`` (e.g. test data below the training min/max).
        copy: Whether to copy input data; ``False`` mutates it in place.
    """

    def __init__(
        self,
        feature_range=(0.0, 1.0),
        log_dynamic_range=3.0,
        log_features=None,
        clip=True,
        copy=True,
    ):
        self.feature_range = feature_range
        self.log_dynamic_range = log_dynamic_range
        self.log_features = log_features
        self.clip = clip
        self.copy = copy

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]

        self.log_features_, self.log_shift_ = self._resolve_log_features(X_df)
        X_log = self._apply_log(X_df)
        self.data_min_ = X_log.min(axis=0).to_numpy()
        self.data_max_ = X_log.max(axis=0).to_numpy()
        data_range = self.data_max_ - self.data_min_
        # Constant features would divide by zero; map them to the range's low end.
        self.scale_ = np.where(data_range > 0, data_range, 1.0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._as_dataframe(X)[self.feature_names_in_]
        X_log = self._apply_log(X_df)

        lo, hi = self.feature_range
        scaled = (X_log.to_numpy() - self.data_min_) / self.scale_
        scaled = scaled * (hi - lo) + lo
        if self.clip:
            scaled = np.clip(scaled, lo, hi)
        return scaled

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        unscaled = (X - lo) / (hi - lo)
        unscaled = unscaled * self.scale_ + self.data_min_
        X_df = pd.DataFrame(unscaled, columns=self.feature_names_in_)
        return self._undo_log(X_df).to_numpy()


class StandardScalar(_FuzzyScalarBase):
    """Standardizes features to ``mu=0``, ``sigma=1``, log-transforming
    wide-dynamic-range ones first.

    Args:
        log_dynamic_range: See :mod:`tribblefis.scaling`. Threshold in decades
            for *automatic* log detection. ``None`` disables automatic
            detection entirely (z-score standardization only). Ignored
            whenever ``log_features`` is given.
        log_features: Explicit list of features to log1p, by column name or --
            for ndarray input -- by positional index. **Authoritative when
            given:** automatic dynamic-range detection is skipped rather than
            unioned in, so setting both this and ``log_dynamic_range`` is not
            an error but this one wins. ``[]`` means "log nothing" and is
            distinct from the default ``None``, which means "auto-detect".
            Naming a feature that is not in the data raises at ``fit`` time.
        copy: Whether to copy input data; ``False`` mutates it in place.
    """

    def __init__(self, log_dynamic_range=3.0, log_features=None, copy=True):
        self.log_dynamic_range = log_dynamic_range
        self.log_features = log_features
        self.copy = copy

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]

        self.log_features_, self.log_shift_ = self._resolve_log_features(X_df)
        X_log = self._apply_log(X_df)
        self.mean_ = X_log.mean(axis=0).to_numpy()
        std = X_log.std(axis=0, ddof=0).to_numpy()
        self.var_ = std**2
        # Constant features would divide by zero; leave them at their mean (0 after centering).
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._as_dataframe(X)[self.feature_names_in_]
        X_log = self._apply_log(X_df)
        return (X_log.to_numpy() - self.mean_) / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        unscaled = X * self.scale_ + self.mean_
        X_df = pd.DataFrame(unscaled, columns=self.feature_names_in_)
        return self._undo_log(X_df).to_numpy()
