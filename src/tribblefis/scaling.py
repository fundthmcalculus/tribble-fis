"""Explicit, opt-in feature scaling for fuzzy inference systems.

FIS membership functions expect bounded inputs, and features with content at
multiple scales (e.g. spanning several decades) are usually better fit after a
log transform. Neither transformation is applied automatically by any
estimator in this package -- callers who want it compose one of the scalers
below into an ``sklearn.pipeline.Pipeline`` in front of the estimator::

    from sklearn.pipeline import make_pipeline
    from tribblefis.scaling import MinMaxScaler
    from tribblefis.gaussian_classifier import TribbleClassifier

    pipe = make_pipeline(MinMaxScaler(), TribbleClassifier())
    pipe.fit(X_train, y_train)

Five scalers are offered, in two families.

**Affine normalizers** -- a log1p pre-step on wide-dynamic-range features,
then one linear map:

- :class:`MinMaxScaler` -- min-max bounding to ``[0, 1]`` (or a custom
  range). **The recommended default for FIS estimators in this package** (see
  "Choosing between them" below). Follows ``sklearn.preprocessing.MinMaxScaler``
  naming convention.
- :class:`StandardScaler` -- genuine z-score standardization (``mu=0``,
  ``sigma=1``), for callers who specifically need centred features. Not
  recommended for FIS estimators; see the warning on the class. Follows
  ``sklearn.preprocessing.StandardScaler`` naming convention.

**Uniformity-preserving transforms** -- reshape each marginal towards uniform,
then bound to ``feature_range`` (see "Uniformity transforms" below):

- :class:`EmpiricalCDFScaler` -- each value becomes the fraction of training
  values at or below it. No hyperparameters, nothing to overfit.
- :class:`PiecewiseLinearCDFScaler` -- the same idea approximated by
  ``n_pieces`` equal-probability affine segments. Exactly invertible, and a
  strict generalization of :class:`MinMaxScaler` (``n_pieces=1`` *is* min-max).
- :class:`QuantileUniformScaler` -- wraps
  ``sklearn.preprocessing.QuantileTransformer``. The textbook answer, and
  fragile on small samples; see the warning on the class.

All five mirror sklearn scalers' ``fit``/``transform``/``fit_transform``/
``inverse_transform``/``get_feature_names_out`` surface, and compose into an
``sklearn.pipeline.Pipeline`` identically.

The older names ``UnitFuzzyScalar``, ``StandardFuzzyScalar``, ``UnitScalar``,
and ``StandardScalar`` are retained as aliases for backwards compatibility, since
they shipped first and are still imported downstream. They are the same class
objects, not wrappers. New code should use the standard scikit-learn names
:class:`MinMaxScaler` and :class:`StandardScaler`.

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

    MinMaxScaler(log_features=["Slag", "FlyAsh", "Age"])

Pre-log flooring
----------------

Both scalers floor logged features to the training minimum before taking the
logarithm, to ensure the log is well-defined (``log`` is undefined for values
below −1). At transform time, any value below the fitted training minimum is
silently shifted to the minimum before the log is applied. This flooring:

- Applies to **both** :class:`MinMaxScaler` and :class:`StandardScaler`.
- Applies only to features selected for logging (by ``log_features`` or
  automatic detection via ``log_dynamic_range``).
- Preserves domain safety but discards the magnitude of out-of-range
  excursions below the training minimum, unconditionally and for
  log-selected features only.

For :class:`StandardScaler`, this means logged features are partially
bounded below, even though the class otherwise advertises unbounded output.
Unlogged features are not floored and remain truly unbounded.

Getting a DataFrame back
------------------------

FIS consumers frequently need column names (they build membership
dictionaries keyed by feature). ``set_output`` comes free with
``TransformerMixin``, so there is no need to re-wrap the output in
``pd.DataFrame(...)`` by hand::

    scaler = MinMaxScaler(log_features=["Slag", "Age"]).set_output(transform="pandas")
    X_scaled = scaler.fit_transform(X_train)   # a DataFrame, columns preserved

    # ...or, in a pipeline:
    pipe = make_pipeline(MinMaxScaler(), TribbleRegressor())
    pipe.fit(X_train, y_train)

Scaling a regression target
---------------------------

Some constructions in this package need ``y`` bounded to ``[0, 1]`` (output
partitioning, and consequent bucket means pinned to the unit interval). These
scalers are feature scalers, but a target is just a one-column frame, so the
idiom is a one-liner and this package deliberately adds no separate target
API::

    y_scaler = MinMaxScaler(log_dynamic_range=None)      # or log_features=[]
    y_scaled = y_scaler.fit_transform(y.to_frame()).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

If you want that wired up automatically, use the standard sklearn tool
:class:`sklearn.compose.TransformedTargetRegressor`, which already handles
the fit/inverse plumbing::

    TransformedTargetRegressor(
        regressor=TribbleRegressor(),
        transformer=MinMaxScaler(log_dynamic_range=None),
    )

Choosing between them
---------------------

``MinMaxScaler`` is the recommended default. This is an empirical finding, not
a theoretical one, and it comes from a single dataset and a single pipeline
-- treat it as a strong prior, not a law.

In the ``grad-school`` proposal-defense evaluation on UCI Concrete over ten
seeds, min-max (``MinMaxScaler``) was best-or-tied in 8 of 9
model/hyperparameter rows. Genuine z-score (``StandardScaler``) was actively
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

Uniformity transforms
---------------------

``log1p`` helps a feature whose content spans decades. It does nothing for a
feature that is merely *non-uniform* in some other way -- multimodal,
heavy-tailed, zero-inflated, clustered on a few discrete levels.

The mechanism that makes this matter is specific to FIS rather than general
preprocessing folklore. Membership functions here are placed from data
statistics, so they tile the domain evenly only when the marginal is roughly
uniform. On a skewed feature they bunch where the density is and leave the
tails uncovered; on a bimodal one they straddle a gap that contains no data.
Push the marginal towards uniform first and each membership function covers
roughly equal probability mass by construction.

That is the argument for the three uniformity scalers above. Two consequences
of it are worth stating outright, because they are easy to trip over:

- **These are not affine.** Distance in the output is *probability mass*, not
  magnitude. Anything downstream reading the transformed value as a physical
  quantity is reading it wrong, and ``inverse_transform`` is the only correct
  way back.
- **They subsume the log1p step rather than composing with it.** Rank is
  invariant under any strictly increasing function, so ``log1p`` before an
  empirical CDF changes its output by exactly nothing -- which is why these
  classes take no ``log_features`` argument. See
  :class:`_UniformityScalerBase` for the partial version of that argument for
  :class:`PiecewiseLinearCDFScaler`.

**Which to reach for.** :class:`MinMaxScaler` remains the documented default.
Reach for :class:`EmpiricalCDFScaler` when `MinMaxScaler` is underperforming
*and* you suspect the input distribution is the cause -- a marginal that is
visibly bimodal, heavy-tailed, zero-inflated, or otherwise not helped by the
log1p pre-step.

**Reported accuracy** (issues #220 and #224, ten seeds each). These numbers come
from sweeps in a separate repository and are **not reproduced by this package's
test suite**, which cannot fetch the datasets; they are recorded here because
they are the best evidence available for the accuracy claim, not because this
repository can check them::

    dataset                     log+minmax     EmpiricalCDF
    UCI Concrete       (R^2)         0.801            0.821
    Body Fat           (R^2)         0.109            0.587
    Bike Sharing       (R^2)         0.589            0.620
    Glass          (accuracy)        0.533            0.528
    Shuttle        (accuracy)        0.958            0.981

The pattern is the useful part: the large lifts are on the datasets where the
log pre-step does *not* help (Body Fat), the transform roughly matches min-max
where it already does (Concrete, Glass), and it never fails catastrophically --
unlike :class:`QuantileUniformScaler`, which reached R^2 -1.86 on Body Fat.
:class:`PiecewiseLinearCDFScaler` at ``n_pieces=10`` reached 0.836 on Concrete,
above both.

**What this package does verify** is mechanical rather than empirical:
`benchmarks/uniformity_scaling.py` measures marginal uniformity and downstream
regression quality on synthetic marginals with dialled-in pathology -- and shows
the gap tracking the pathology, which a single accuracy table cannot -- while
`tests/test_uniformity_scaling.py` pins the mathematical properties.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class _FuzzyScalarBase(TransformerMixin, BaseEstimator):
    """Shared DataFrame bookkeeping and log-detection for the scalars below.

    Not part of the public API -- use :class:`StandardScaler` or
    :class:`MinMaxScaler`.
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


class MinMaxScaler(_FuzzyScalarBase):
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
    """

    def __init__(
        self,
        feature_range=(0.0, 1.0),
        log_dynamic_range=3.0,
        log_features=None,
    ):
        self.feature_range = feature_range
        self.log_dynamic_range = log_dynamic_range
        self.log_features = log_features

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
            np.where(data_range > 0, data_range, 1.0), index=self.feature_names_in_
        )
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._as_dataframe(X)[self.feature_names_in_]
        X_log = self._apply_log(X_df)

        lo, hi = self.feature_range
        scaled = (X_log - self.data_min_) / self.scale_
        scaled = scaled.to_numpy() * (hi - lo) + lo
        return scaled

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range

        unscaled = (X - lo) / (hi - lo)
        unscaled = unscaled * self.scale_.to_numpy() + self.data_min_.to_numpy()
        X_df = pd.DataFrame(unscaled, columns=self.feature_names_in_)
        return self._undo_log(X_df).to_numpy()


class StandardScaler(_FuzzyScalarBase):
    """Standardizes features to ``mu=0``, ``sigma=1``, log-transforming
    wide-dynamic-range ones first.

    .. warning::

        **This is not the recommended default for the FIS estimators in this
        package. Use :class:`MinMaxScaler` unless you specifically need
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
        the same runs :class:`MinMaxScaler` was best-or-tied in 8 of 9
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
            np.where(std > 0, std, 1.0), index=self.feature_names_in_
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


class _UniformityScalerBase(_FuzzyScalarBase):
    """Shared plumbing for the three uniformity-preserving scalers below.

    Not part of the public API. Subclasses supply three per-column hooks --
    :meth:`_fit_column`, :meth:`_map_column`, :meth:`_unmap_column` -- and
    inherit the DataFrame bookkeeping, the ``feature_range`` affine, the
    constant-feature convention and NaN propagation.

    Why these have no ``log_features``
    ----------------------------------

    :class:`MinMaxScaler` and :class:`StandardScaler` both take log arguments;
    these do not, and that is a property of the transform rather than an
    omission.

    :class:`EmpiricalCDFScaler` maps a value to *the fraction of training values
    at or below it*. Rank is invariant under any strictly increasing function,
    and ``log1p`` is one, so ``log1p`` composed with an empirical CDF is the
    empirical CDF exactly -- to the last bit, not approximately. A
    ``log_features`` argument there would be a control that provably does
    nothing, which is worse than no control at all.
    ``test_log1p_cannot_change_the_empirical_cdf`` pins this.

    For :class:`PiecewiseLinearCDFScaler` the invariance is partial: the
    breakpoints are quantiles, so *where* they sit is rank-determined and
    unchanged by ``log1p``, but the interpolation between them happens in the
    original value space and does move. The argument for leaving it out is
    weaker there -- it is consistency with its sibling, plus the fact that
    ``make_pipeline(MinMaxScaler(log_features=[...]), PiecewiseLinearCDFScaler())``
    already expresses it for anyone who wants it.
    """

    def _validated_range(self):
        """``feature_range``, checked at fit time per the sklearn convention
        that ``__init__`` only stores its arguments."""
        try:
            lo, hi = self.feature_range
        except (TypeError, ValueError):
            raise ValueError(
                f"feature_range must be a (min, max) pair, got {self.feature_range!r}"
            ) from None
        if not hi > lo:
            raise ValueError(f"feature_range must have min < max, got ({lo!r}, {hi!r})")
        return float(lo), float(hi)

    def _record_constants(self, X_df):
        """Remember one representative value per column, for the inverse.

        A constant column has no mapping to invert, and returning NaN for it
        would make ``inverse_transform(transform(X))`` lose a column that was
        perfectly well-defined -- just constant. Recorded at fit time because
        that is the only moment the value is known.
        """
        self.constants_ = {}
        for col in X_df.columns:
            finite = X_df[col].to_numpy(dtype=float)
            finite = finite[~np.isnan(finite)]
            if finite.size:
                self.constants_[col] = float(finite[0])

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]
        self._validated_range()
        self._record_constants(X_df)
        self.mappings_ = {
            col: self._fit_column(X_df[col].to_numpy(dtype=float))
            for col in self.feature_names_in_
        }
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._as_dataframe(X)[self.feature_names_in_]
        lo, hi = self._validated_range()
        unit = np.empty((len(X_df), self.n_features_in_), dtype=float)
        for j, col in enumerate(self.feature_names_in_):
            values = X_df[col].to_numpy(dtype=float)
            mapping = self.mappings_[col]
            if mapping is None:
                # Constant (or all-NaN) feature. MinMaxScaler sends these to the
                # low end of the range rather than dividing by a zero spread;
                # matching that keeps a pipeline's behaviour on a degenerate
                # column independent of which scaler it happens to use.
                unit[:, j] = np.where(np.isnan(values), np.nan, 0.0)
            else:
                unit[:, j] = self._map_column(mapping, values)
        # The affine onto feature_range is the one piece of logic that is the
        # same for every column, so it happens once on the assembled array
        # rather than n_features times inside the loop -- where a reader looks
        # for per-column behaviour and would have to check it is not.
        return lo + unit * (hi - lo)

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        lo, hi = self._validated_range()
        u = (X - lo) / (hi - lo)
        out = np.empty_like(u, dtype=float)
        for j, col in enumerate(self.feature_names_in_):
            mapping = self.mappings_[col]
            if mapping is None:
                fill = self.constants_.get(col, np.nan)
                out[:, j] = np.where(np.isnan(u[:, j]), np.nan, fill)
            else:
                out[:, j] = self._unmap_column(mapping, u[:, j])
        return out


class EmpiricalCDFScaler(_UniformityScalerBase):
    """Maps each feature through its empirical CDF, then onto ``feature_range``.

    A value becomes the fraction of training values at or below it, rescaled so
    the training minimum lands on ``feature_range[0]`` and the maximum on
    ``feature_range[1]``. On the training data the result is as close to uniform
    as the sample allows -- exactly uniform when every value is distinct.

    Why a FIS cares
    ---------------

    Gaussian membership functions are placed from data statistics, so they cover
    the domain evenly only when the marginal distribution is roughly uniform. On
    a skewed or clustered feature they bunch where the density is and leave the
    tails uncovered. Spreading the data uniformly first makes each membership
    function cover roughly equal probability mass by construction.

    This is the most robust of the three uniformity scalers here: no
    hyperparameter beyond ``feature_range``, and nothing to overfit -- the
    "model" is the sorted training column.

    What it costs
    -------------

    - **The inverse is approximate.** An empirical CDF is a step function, so
      many inputs share an output and ``inverse_transform`` can only return the
      training value at that quantile. Round-tripping the *training* data
      recovers it exactly; round-tripping anything else snaps to the nearest
      training value at or above it. If you need an exact inverse, use
      :class:`PiecewiseLinearCDFScaler`, which is a genuine bijection.
    - **Output is bounded, and out-of-range inputs are clamped.** Anything below
      the training minimum maps to ``feature_range[0]`` and anything above the
      maximum to ``feature_range[1]``. Bounded output is the point -- it is what
      FIS rule construction assumes -- but the magnitude of an excursion past
      the training range is discarded, exactly as the pre-log flooring described
      in :mod:`tribblefis.scaling` discards it.
    - **Distances stop meaning anything.** The transform is monotone but wildly
      non-affine, so a gap of 0.1 in the output is a gap of equal *probability
      mass*, not of equal magnitude. Anything downstream that reads the output
      as a physical quantity is reading it wrong.

    Args:
        feature_range: Desired ``(min, max)`` of transformed data.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

    def _fit_column(self, values):
        finite = np.sort(values[~np.isnan(values)])
        if finite.size == 0:
            return None
        n = finite.size
        # F(x) = (number of training values <= x) / n. At the minimum this is
        # the count of ties on the minimum, not 1/n, and at the maximum it is
        # exactly 1. Rescaling by that observed span -- rather than assuming
        # (1/n, 1) -- is what pins the extremes onto feature_range even when the
        # smallest value repeats, which on a zero-inflated feature is most of
        # the column.
        f_min = np.searchsorted(finite, finite[0], side="right") / n
        span = 1.0 - f_min
        if span <= 0:
            return None  # every value identical
        # The output each sorted position maps to, computed by the *same*
        # arithmetic the forward map uses. The inverse then looks a value up in
        # this table instead of recomputing a rank from it.
        #
        # The obvious inverse -- rank = ceil(f * n) - 1 -- is wrong in floating
        # point, and quietly so: round-tripping the training data returned the
        # *next* training value wherever f * n landed a few ulps above an
        # integer (measured max round-trip error 1.03 on a lognormal column,
        # against 0 expected). Comparing against the table that produced the
        # value cannot drift, because both sides are the identical expression.
        ranks = (np.arange(1, n + 1) / n - f_min) / span
        return {
            "sorted": finite,
            "n": n,
            "f_min": f_min,
            "span": span,
            "u_of_rank": np.clip(ranks, 0.0, 1.0),
        }

    def _map_column(self, mapping, values):
        nan = np.isnan(values)
        # searchsorted places NaN at the far right, which would report the
        # maximum quantile for a missing value -- a fabricated number rather
        # than a missing one. Substitute, then restore.
        safe = np.where(nan, mapping["sorted"][0], values)
        f = np.searchsorted(mapping["sorted"], safe, side="right") / mapping["n"]
        u = np.clip((f - mapping["f_min"]) / mapping["span"], 0.0, 1.0)
        return np.where(nan, np.nan, u)

    def _unmap_column(self, mapping, u):
        nan = np.isnan(u)
        clipped = np.clip(np.where(nan, 0.0, u), 0.0, 1.0)
        # The generalized inverse of a step CDF: the smallest training value
        # whose output is at or above `u`. `u_of_rank` is non-decreasing (its
        # leading entries tie at 0 when the minimum repeats), so side="left"
        # picks the first index that qualifies.
        idx = np.clip(
            np.searchsorted(mapping["u_of_rank"], clipped, side="left"),
            0,
            mapping["n"] - 1,
        )
        return np.where(nan, np.nan, mapping["sorted"][idx])


class PiecewiseLinearCDFScaler(_UniformityScalerBase):
    """Approximates each feature's CDF with ``n_pieces`` equal-probability
    affine segments.

    Breakpoints are placed at the ``n_pieces + 1`` evenly spaced quantiles, and
    each segment is mapped by a single affine function onto its share of
    ``feature_range``. The transform is therefore continuous, strictly
    increasing, and exactly invertible -- the "affine maps for
    membership-function placement" idiom, with interpretable breakpoints a
    reader can print.

    ``n_pieces`` trades fidelity for smoothness:

    - ``n_pieces=1`` degenerates to plain min-max bounding, exactly. The
      breakpoints are the minimum and the maximum, and one affine map takes the
      column to ``feature_range``. ``test_one_piece_is_exactly_min_max`` pins
      that equivalence against :class:`MinMaxScaler`, which makes this class a
      strict generalization rather than an alternative.
    - Larger values track the empirical distribution more closely, moving
      *toward* :class:`EmpiricalCDFScaler` without ever reaching it -- this
      interpolates linearly between order statistics where the empirical CDF
      steps, so a gap of up to one step's height remains at any ``n_pieces``.
      Measured on a lognormal column, KS-to-uniform is 0.0866 at
      ``n_pieces=10`` against 0.0025 for the empirical CDF: a real dial, not a
      convergent series. Larger values also start fitting sample noise in the
      tails, the way any quantile estimate does.

    Compared with :class:`EmpiricalCDFScaler` this keeps a genuine inverse and a
    continuous derivative, at the cost of one hyperparameter and a coarser
    approximation of the marginal.

    Degenerate breakpoints
    ----------------------

    On a discrete or zero-inflated feature several adjacent quantiles can land
    on the same value -- ``n_pieces=10`` on a column that is 40% zeros puts four
    breakpoints on zero. Those collapse to a single breakpoint carrying the
    *highest* of their targets, which is the right-continuous reading of a CDF
    with an atom there, and keeps the mapping strictly increasing so the inverse
    stays well-defined. The effective number of pieces is therefore at most
    ``n_pieces`` and can be smaller; :attr:`n_pieces_` records what each feature
    actually got.

    Args:
        n_pieces: Number of equal-probability segments. Must be >= 1.
        feature_range: Desired ``(min, max)`` of transformed data.
    """

    def __init__(self, n_pieces=10, feature_range=(0.0, 1.0)):
        self.n_pieces = n_pieces
        self.feature_range = feature_range

    def fit(self, X, y=None):
        # Validated here rather than in __init__, per the sklearn convention
        # that __init__ only stores its parameters (otherwise clone and
        # get_params break). bool is excluded explicitly because it is an int
        # subclass, and PiecewiseLinearCDFScaler(n_pieces=True) is a mistake
        # that would otherwise run as n_pieces=1 and look like it worked.
        if isinstance(self.n_pieces, bool) or not isinstance(
            self.n_pieces, (int, np.integer)
        ):
            raise ValueError(f"n_pieces must be an integer, got {self.n_pieces!r}")
        if self.n_pieces < 1:
            raise ValueError(f"n_pieces must be >= 1, got {self.n_pieces}")
        super().fit(X, y)
        self.n_pieces_ = {
            col: (0 if m is None else len(m["xs"]) - 1)
            for col, m in self.mappings_.items()
        }
        return self

    def _fit_column(self, values):
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            return None
        probs = np.linspace(0.0, 1.0, self.n_pieces + 1)
        breakpoints = np.quantile(finite, probs)
        # Keep the last index of each run of equal breakpoints, so a repeated
        # value carries the highest target it is entitled to. `np.interp`
        # requires a strictly increasing `xp`; a tied pair would otherwise leave
        # both the forward map and the inverse ill-defined at that value.
        keep = np.append(np.diff(breakpoints) > 0, True)
        xs, ys = breakpoints[keep], probs[keep]
        if xs.size < 2:
            return None  # every value identical
        # Rescale the surviving targets back onto the full [0, 1]. Collapsing a
        # run of tied breakpoints keeps the *highest* target of the run, so an
        # atom at the minimum leaves ys[0] well above zero -- measured, a column
        # that is 40% zeros produced ys = [0.4 ... 1.0] and mapped every zero to
        # 0.4. The bottom 40% of feature_range was then unreachable by any
        # input, which is precisely the wasted-tail problem this class exists to
        # avoid, reproduced at the other end.
        #
        # ys is strictly increasing after the collapse, so the span is positive.
        # At n_pieces=1 this is the identity (ys is already [0, 1]), which is
        # what keeps the min-max equivalence exact.
        ys = (ys - ys[0]) / (ys[-1] - ys[0])
        return {"xs": xs, "ys": ys}

    def _map_column(self, mapping, values):
        # np.interp clamps outside [xs[0], xs[-1]] and propagates NaN, which is
        # the wanted behaviour on both counts.
        return np.interp(values, mapping["xs"], mapping["ys"])

    def _unmap_column(self, mapping, u):
        return np.interp(u, mapping["ys"], mapping["xs"])


class QuantileUniformScaler(_UniformityScalerBase):
    """Marginal uniformity via :class:`sklearn.preprocessing.QuantileTransformer`.

    .. warning::

        **Fragile on small samples. Prefer :class:`EmpiricalCDFScaler` unless
        you have measured this one to be better on your data.**

        A quantile transform estimates ``n_quantiles`` order statistics per
        feature. When the sample is small relative to that, the estimate is
        mostly noise and the transform encodes the training split rather than
        the distribution. In the experiments reported on issue #220 this took
        Body Fat (N=252, 13 features) to R^2 -1.86 -- worse than predicting the
        training mean -- while :class:`EmpiricalCDFScaler` reached 0.587 on the
        same cells.

        It is included because it is the textbook answer and therefore the right
        baseline to compare against, not because it is a good default here.

    ``n_quantiles`` is capped at the number of training samples, which is what
    sklearn does internally anyway; capping it here just avoids the warning and
    makes the effective value readable as :attr:`n_quantiles_`.

    Args:
        n_quantiles: Number of quantiles to estimate. Capped at ``n_samples``.
        feature_range: Desired ``(min, max)`` of transformed data.
        subsample: Passed through to ``QuantileTransformer``.
        random_state: Passed through to ``QuantileTransformer``; only has an
            effect when ``subsample`` is smaller than the sample.
    """

    def __init__(
        self,
        n_quantiles=1000,
        feature_range=(0.0, 1.0),
        subsample=10_000,
        random_state=None,
    ):
        self.n_quantiles = n_quantiles
        self.feature_range = feature_range
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.preprocessing import QuantileTransformer

        X_df = self._as_dataframe(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.n_features_in_ = X_df.shape[1]
        self._validated_range()
        self.n_quantiles_ = max(1, min(int(self.n_quantiles), len(X_df)))
        self.transformer_ = QuantileTransformer(
            n_quantiles=self.n_quantiles_,
            output_distribution="uniform",
            subsample=self.subsample,
            random_state=self.random_state,
        )
        # Fitted on bare values: passing the DataFrame would have
        # QuantileTransformer record its own feature names and then validate
        # against them separately from this class's, giving two sources of truth
        # for one question.
        self.transformer_.fit(X_df.to_numpy(dtype=float))
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._as_dataframe(X)[self.feature_names_in_]
        lo, hi = self._validated_range()
        u = self.transformer_.transform(X_df.to_numpy(dtype=float))
        return lo + u * (hi - lo)

    def inverse_transform(self, X):
        check_is_fitted(self)
        lo, hi = self._validated_range()
        u = (np.asarray(X, dtype=float) - lo) / (hi - lo)
        return self.transformer_.inverse_transform(u)

# Backwards-compatible aliases. MinMaxScaler and StandardScaler are the canonical
# names following scikit-learn convention. The older FuzzyScalar and Scalar names
# shipped first and remain supported without deprecation: they are the same class
# objects, not wrappers, and isinstance checks against any name behave identically.
UnitFuzzyScalar = MinMaxScaler
StandardFuzzyScalar = StandardScaler
UnitScalar = MinMaxScaler
StandardScalar = StandardScaler
