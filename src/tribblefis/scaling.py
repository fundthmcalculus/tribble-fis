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

    def _detect_log_features(self, X_df):
        """A feature qualifies for log1p when its dynamic range (log10 of
        max/min absolute non-zero value) is at least ``log_dynamic_range``
        decades -- the same heuristic used to flag energy/content spread
        across multiple scales."""
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
        log_dynamic_range: See :mod:`tribblefis.scaling`. ``None`` disables
            log detection entirely (min-max bounding only).
        clip: Whether ``transform`` clips values falling outside the range
            seen during ``fit`` (e.g. test data below the training min/max).
        copy: Whether to copy input data; ``False`` mutates it in place.
    """

    def __init__(self, feature_range=(0.0, 1.0), log_dynamic_range=3.0, clip=True, copy=True):
        self.feature_range = feature_range
        self.log_dynamic_range = log_dynamic_range
        self.clip = clip
        self.copy = copy

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]

        self.log_features_, self.log_shift_ = self._detect_log_features(X_df)
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
        log_dynamic_range: See :mod:`tribblefis.scaling`. ``None`` disables
            log detection entirely (z-score standardization only).
        copy: Whether to copy input data; ``False`` mutates it in place.
    """

    def __init__(self, log_dynamic_range=3.0, copy=True):
        self.log_dynamic_range = log_dynamic_range
        self.copy = copy

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]

        self.log_features_, self.log_shift_ = self._detect_log_features(X_df)
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
