"""Explicit, opt-in feature scaling for fuzzy inference systems.

FIS membership functions expect bounded inputs, and features with content at
multiple scales (e.g. spanning several decades) are usually better fit after a
log transform. Neither transformation is applied automatically by any
estimator in this package -- callers who want it compose one of the scalers
below into an ``sklearn.pipeline.Pipeline`` in front of the estimator::

    from sklearn.pipeline import make_pipeline
    from tribblefis.scaling import UnitFuzzyScalar
    from tribblefis.gaussian_classifier import TribbleClassifier

    pipe = make_pipeline(UnitFuzzyScalar(), TribbleClassifier())
    pipe.fit(X_train, y_train)

Two scalers are offered, differing only in the final normalization:

- :class:`UnitFuzzyScalar` -- min-max bounding to ``[0, 1]`` (or a custom
  range). **The recommended default for FIS estimators in this package** (see
  "Choosing between them" below).
- :class:`StandardFuzzyScalar` -- genuine z-score standardization (``mu=0``,
  ``sigma=1``), for callers who specifically need centred features. Not
  recommended for FIS estimators; see the warning on the class.

Both mirror sklearn scalers' ``fit``/``transform``/``fit_transform``/
``inverse_transform``/``get_feature_names_out`` surface.

Each name says what its class computes, and only what it computes:
``Unit`` for unit-interval bounding, ``Standard`` for standardization. That
precision is deliberate. A previous helper in this codebase was *named* for
standardization while *computing* min-max, and the mislabelling propagated
into every document that cited it -- so "standard" here never means
"recommended", and the recommended scaler is never called "standard".

The shorter names ``UnitScalar`` and ``StandardScalar`` are retained as
aliases for backwards compatibility, since they shipped first and are still
imported downstream. They are the same class objects, not wrappers. New code
should prefer the canonical ``*FuzzyScalar`` names, which match
:class:`_FuzzyScalarBase`.

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

    UnitFuzzyScalar(log_features=["Slag", "FlyAsh", "Age"])

Pre-log flooring
----------------

Both scalers floor logged features to the training minimum before taking the
logarithm, to ensure the log is well-defined (``log`` is undefined for values
below −1). At transform time, any value below the fitted training minimum is
silently shifted to the minimum before the log is applied. This flooring:

- Applies to **both** :class:`UnitFuzzyScalar` and :class:`StandardFuzzyScalar`.
- Applies only to features selected for logging (by ``log_features`` or
  automatic detection via ``log_dynamic_range``).
- Preserves domain safety but discards the magnitude of out-of-range
  excursions below the training minimum (similar to ``clip=True`` on
  :class:`UnitFuzzyScalar`, but unconditional and log-feature-only).

For :class:`StandardFuzzyScalar`, this means logged features are partially
bounded below, even though the class otherwise advertises unbounded output.
Unlogged features are not floored and remain truly unbounded.

Getting a DataFrame back
------------------------

FIS consumers frequently need column names (they build membership
dictionaries keyed by feature). ``set_output`` comes free with
``TransformerMixin``, so there is no need to re-wrap the output in
``pd.DataFrame(...)`` by hand::

    scaler = UnitFuzzyScalar(log_features=["Slag", "Age"]).set_output(transform="pandas")
    X_scaled = scaler.fit_transform(X_train)   # a DataFrame, columns preserved

    # ...or, in a pipeline:
    pipe = make_pipeline(UnitFuzzyScalar(), TribbleRegressor())
    pipe.fit(X_train, y_train)

Scaling a regression target
---------------------------

Some constructions in this package need ``y`` bounded to ``[0, 1]`` (output
partitioning, and consequent bucket means pinned to the unit interval). These
scalers are feature scalers, but a target is just a one-column frame, so the
idiom is a one-liner and this package deliberately adds no separate target
API::

    y_scaler = UnitFuzzyScalar(log_dynamic_range=None)      # or log_features=[]
    y_scaled = y_scaler.fit_transform(y.to_frame()).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

If you want that wired up automatically, use the standard sklearn tool
:class:`sklearn.compose.TransformedTargetRegressor`, which already handles
the fit/inverse plumbing::

    TransformedTargetRegressor(
        regressor=TribbleRegressor(),
        transformer=UnitFuzzyScalar(log_dynamic_range=None),
    )

Choosing between them
---------------------

``UnitFuzzyScalar`` is the recommended default. This is an empirical finding, not
a theoretical one, and it comes from a single dataset and a single pipeline
-- treat it as a strong prior, not a law.

In the ``grad-school`` proposal-defense evaluation on UCI Concrete over ten
seeds, min-max (``UnitFuzzyScalar``) was best-or-tied in 8 of 9
model/hyperparameter rows. Genuine z-score (``StandardFuzzyScalar``) was actively
harmful to the fuzzy models: it took a 1st-order flat MoG-TSK model to
R^2 0.087 +/- 0.089, *below* raw untransformed features at 0.646 +/- 0.039
(RMSE 7.8 -> 15.6 MPa), and dropped a demo-tuned mixture of experts from
0.834 to 0.706. Rank-based controls (CART, Random Forest) moved by <= 0.002
between the two transforms, as they must, since both are monotone -- which
is what confirms the effect is specific to the fuzzy machinery rather than an
artifact of the split.

The working explanation is that Gaussian membership functions -- and, in that
pipeline, extreme output-bucket means pinned to ``[0, 1]`` -- assume a
**bounded, non-negative** domain. An unbounded, centred transform violates an
assumption the rule construction relies on. The degradation shows up on
*training* data too, so it is underfitting rather than overfitting.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class _FuzzyScalarBase(TransformerMixin, BaseEstimator):
    """Shared DataFrame bookkeeping and log-detection for the scalars below.

    Not part of the public API -- use :class:`StandardFuzzyScalar` or
    :class:`UnitFuzzyScalar`.
    """

    def _as_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
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


class UnitFuzzyScalar(_FuzzyScalarBase):
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

            This default is a **modelling decision, not just numerical
            hygiene**, and it is worth choosing deliberately. In a one-class
            or anomaly-detection setup where the scaler is fitted on normal
            data only (say, benign network traffic), every attack row that
            falls outside the fitted range is saturated at the bounds rather
            than extrapolated. That is often exactly what you want -- it keeps
            membership functions inside their supported domain -- but it does
            discard the magnitude of the excursion, which is sometimes the
            most informative thing about an outlier. Pass ``clip=False`` when
            you need that magnitude to survive into the model.
    """

    def __init__(
        self,
        feature_range=(0.0, 1.0),
        log_dynamic_range=3.0,
        log_features=None,
        clip=True,
    ):
        self.feature_range = feature_range
        self.log_dynamic_range = log_dynamic_range
        self.log_features = log_features
        self.clip = clip

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]

        self.log_features_, self.log_shift_ = self._resolve_log_features(X_df)
        X_log = self._apply_log(X_df)
        self.data_min_ = X_log.min(axis=0)
        self.data_max_ = X_log.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        # Constant features would divide by zero; map them to the range's low end.
        self.scale_ = pd.Series(
            np.where(data_range > 0, data_range, 1.0),
            index=self.feature_names_in_
        )
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._as_dataframe(X)[self.feature_names_in_]
        X_log = self._apply_log(X_df)

        lo, hi = self.feature_range
        scaled = (X_log - self.data_min_) / self.scale_
        scaled = scaled.to_numpy() * (hi - lo) + lo
        if self.clip:
            scaled = np.clip(scaled, lo, hi)
        return scaled

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range

        if self.clip:
            on_bounds = np.isclose(X, lo) | np.isclose(X, hi)
            if np.any(on_bounds):
                warnings.warn(
                    "inverse_transform detected values sitting exactly on the bounds "
                    f"{(lo, hi)}. Since clip=True, these may be clipped artifacts from "
                    "transform(), and the round-trip inverse_transform(transform(X)) may not "
                    "recover the original values outside the fitted range. "
                    "See https://github.com/fundthmcalculus/tribble-fis/issues/74 for details.",
                    UserWarning,
                    stacklevel=2,
                )

        unscaled = (X - lo) / (hi - lo)
        unscaled = unscaled * self.scale_.to_numpy() + self.data_min_.to_numpy()
        X_df = pd.DataFrame(unscaled, columns=self.feature_names_in_)
        return self._undo_log(X_df).to_numpy()


class StandardFuzzyScalar(_FuzzyScalarBase):
    """Standardizes features to ``mu=0``, ``sigma=1``, log-transforming
    wide-dynamic-range ones first.

    .. warning::

        **This is not the recommended default for the FIS estimators in this
        package. Use :class:`UnitFuzzyScalar` unless you specifically need
        centred features and know why.**

        Despite the name, "standard" here describes the *transform* (z-score
        standardization), not a recommendation. This class produces
        **unbounded, centred** output, which conflicts directly with the
        bounded, non-negative input domain that Gaussian membership functions
        -- and output-bucket means pinned to ``[0, 1]`` -- assume. Feeding
        centred features to a FIS violates an assumption the rule
        construction relies on, and it measurably degrades results.

        Measured on UCI Concrete over ten seeds: this scaler took a 1st-order
        flat MoG-TSK model to R^2 0.087 +/- 0.089, which is *below raw
        untransformed features* at 0.646 +/- 0.039 (RMSE 7.8 -> 15.6 MPa),
        and dropped a demo-tuned mixture of experts from 0.834 to 0.706. Over
        the same runs :class:`UnitFuzzyScalar` was best-or-tied in 8 of 9
        model/hyperparameter rows. Rank-based controls (CART, Random Forest)
        moved by <= 0.002 between the two transforms, as they must, since both
        are monotone -- which is what localises the damage to the fuzzy
        machinery rather than the split. The degradation appears on
        *training* data too, so it is underfitting, not overfitting.

        That is one dataset and one pipeline, so treat it as a strong prior
        rather than a law -- but the burden of proof is on using this class,
        not on avoiding it. Legitimate reasons to use it do exist (a
        downstream consumer that requires zero-mean input, or comparing
        against a z-score baseline); "it sounds like the standard choice" is
        not one of them.

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
    """

    def __init__(self, log_dynamic_range=3.0, log_features=None):
        self.log_dynamic_range = log_dynamic_range
        self.log_features = log_features

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]

        self.log_features_, self.log_shift_ = self._resolve_log_features(X_df)
        X_log = self._apply_log(X_df)
        self.mean_ = X_log.mean(axis=0)
        std = X_log.std(axis=0, ddof=0)
        self.var_ = std**2
        # Constant features would divide by zero; leave them at their mean (0 after centering).
        self.scale_ = pd.Series(
            np.where(std > 0, std, 1.0),
            index=self.feature_names_in_
        )
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._as_dataframe(X)[self.feature_names_in_]
        X_log = self._apply_log(X_df)
        return ((X_log - self.mean_) / self.scale_).to_numpy()

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        unscaled = X * self.scale_.to_numpy() + self.mean_.to_numpy()
        X_df = pd.DataFrame(unscaled, columns=self.feature_names_in_)
        return self._undo_log(X_df).to_numpy()


# Backwards-compatible aliases. The ``*FuzzyScalar`` names above are canonical
# -- they carry the ``Fuzzy`` infix of :class:`_FuzzyScalarBase`, and each says
# what it actually computes. These shorter names shipped first and are still
# imported across the ``grad-school`` workspace, so they remain supported and
# are deliberately *not* deprecation-warned: they are the same objects, not
# wrappers, and ``isinstance`` checks against either name behave identically.
UnitScalar = UnitFuzzyScalar
StandardScalar = StandardFuzzyScalar
