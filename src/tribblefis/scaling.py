"""Explicit, opt-in feature scaling for fuzzy inference systems.

FIS membership functions expect bounded inputs, and features with content at
multiple scales (e.g. spanning several decades) are usually better fit after a
log transform. Neither transformation is applied automatically by any
estimator in this package -- callers who want it compose ``StandardFuzzyScalar``
into an ``sklearn.pipeline.Pipeline`` in front of the estimator, the same way
they would use ``StandardScaler``::

    from sklearn.pipeline import make_pipeline
    from tribblefis.scaling import StandardFuzzyScalar
    from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier

    pipe = make_pipeline(StandardFuzzyScalar(), MixtureOfGaussiansFuzzyClassifier())
    pipe.fit(X_train, y_train)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class StandardFuzzyScalar(TransformerMixin, BaseEstimator):
    """Bounds features to ``[0, 1]``, log-transforming wide-dynamic-range ones first.

    Mirrors the ``fit``/``transform``/``fit_transform``/``inverse_transform``/
    ``get_feature_names_out`` surface of sklearn's ``StandardScaler``, but the
    scaling itself is min-max to ``feature_range`` (default ``[0, 1]``) rather
    than z-score, and a per-feature log1p transform is applied first to any
    feature whose dynamic range (across non-zero absolute values) is at least
    ``log_dynamic_range`` decades -- the same heuristic used to flag features
    with energy/content spread across multiple scales.

    Args:
        feature_range: Desired ``(min, max)`` of transformed data.
        log_dynamic_range: A feature is log-transformed when
            ``log10(max(|x|) / min(|x|))`` (over its non-zero values) is at
            least this many decades. Set to ``None`` to disable log detection
            entirely (min-max bounding only).
        clip: Whether ``transform`` clips values falling outside the range
            seen during ``fit`` (e.g. test data below the training min/max).
        copy: Whether to copy input data; ``False`` mutates it in place.
    """

    def __init__(self, feature_range=(0.0, 1.0), log_dynamic_range=3.0, clip=True, copy=True):
        self.feature_range = feature_range
        self.log_dynamic_range = log_dynamic_range
        self.clip = clip
        self.copy = copy

    def _as_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy() if self.copy else X
        feature_names = getattr(self, "feature_names_in_", None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(np.asarray(X, dtype=float), columns=feature_names)

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]

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

        self.log_features_ = log_features
        self.log_shift_ = log_shift

        X_log = self._apply_log(X_df)
        self.data_min_ = X_log.min(axis=0).to_numpy()
        self.data_max_ = X_log.max(axis=0).to_numpy()
        data_range = self.data_max_ - self.data_min_
        # Constant features would divide by zero; map them to the range's low end.
        self.scale_ = np.where(data_range > 0, data_range, 1.0)
        return self

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

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_, dtype=object)
