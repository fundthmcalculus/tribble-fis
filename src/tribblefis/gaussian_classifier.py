from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .gauss_data import (
    AnomalyParameters,
    DefaultNormCornorm,
    DEFAULT_DEDUP_RTOL,
    DEFAULT_DEDUP_ATOL,
    SimpleGaussianClassifierModel,
)
from .gauss_math import (
    calculate_gaussian_correlation,
    take_top_features,
    create_gaussian_membership_dict,
    tsk_predict,
    tsk_firing_strengths,
)

class TribbleClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Mixture Classifier that wraps the TSK-based Gaussian Mixture model.
    It follows scikit-learn's ClassifierMixin interface.
    """

    def __init__(self, top_n=-1, top_p=0.95, correlation_threshold=0.85, n_gaussians=0, norm_conorm=DefaultNormCornorm, member_function="gaussian", trapz_method="fast", random_state=42,
                 refine=False, refine_method="coordinate", refine_l2_shrink=0.05,
                 t_norm=None, t_conorm=None, allow_mixed_norms=False, max_samples=None):
        """Initialize TribbleClassifier.

        Parameters
        ----------
        top_n : int
            Number of top features by differentiation score (-1 means use top_p).
        top_p : float
            Per-feature score threshold (0-1); ignored if top_n > 0.
        correlation_threshold : float
            When top_n > 0, drop a candidate feature whose absolute correlation
            with an already-selected feature is >= this value, pulling in the
            next-best feature instead. Set to <= 0.0 or >= 1.0 to disable.
        n_gaussians : int or dict
            Gaussians per feature per label (0=auto, dict for per-label override).
        member_function : str
            "gaussian", "trap" (EM trapezoid), or "triangular" (special case of trap).
        trapz_method : str
            "fast" (histogram-based) or "em" (EM algorithm). Ignored for triangular.
        norm_conorm : str
            Fuzzy t-norm family: "min/max", "probability" (default), "luk", "hamacher", "einstein".
        random_state : int
            RNG seed for reproducibility.
        refine : bool
            Post-fit Gaussian antecedents on discriminative objective (improves ~5% accuracy).
        refine_method : str
            "coordinate" (default, fast) or "optimizers" (global search).
        refine_l2_shrink : float
            Ridge regularization for refinement antecedents.
        max_samples : int or None
            Max rows per (feature, label) for membership fitting. None=all.
        """
        self.is_fitted_: bool = False
        self.model_ = None
        self.top_features_ = None
        self.top_n_actual_ = None
        self.feature_differentiators_: list[tuple[Any, Any]] = []
        self.classes_ = None
        self.feature_names_in_: list[str] = []
        self.top_n = top_n
        self.top_p = top_p
        self.correlation_threshold = correlation_threshold
        self.n_gaussians = n_gaussians
        self.member_function = member_function
        self.trapz_method = trapz_method
        self.random_state = random_state
        self.norm_conorm = norm_conorm
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.allow_mixed_norms = allow_mixed_norms
        self.max_samples = max_samples
        self.refine = refine
        self.refine_method = refine_method
        self.refine_l2_shrink = refine_l2_shrink
        self.refine_info_: dict | None = None

    def _anomaly_params(self) -> AnomalyParameters:
        # Resolved on demand from the current norm settings, NOT cached in
        # __init__: sklearn clone/set_params (as GridSearchCV does) rewrites
        # norm_conorm/t_norm/t_conorm without touching a cached attribute, so a
        # cached copy would make any norm grid search a silent no-op.
        return AnomalyParameters(
            include_anomaly=False,
            norm_conorm=self.norm_conorm,
            t_norm=self.t_norm,
            t_conorm=self.t_conorm,
            allow_mixed_norms=self.allow_mixed_norms,
        )

    def fit(self, X, y):
        """Fit the Gaussian Mixture model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.

        Returns
        -------
        self : TribbleClassifier
            Fitted estimator.
        """
        # If X is a DataFrame, keep track of feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Standard sklearn validation
        X_array, y_array = check_X_y(X, y)
        check_classification_targets(y_array)

        # Store classes
        self.classes_ = np.unique(y_array)

        # We need X as DataFrame for the internal functions
        X_df = pd.DataFrame(X_array, columns=self.feature_names_in_)
        y_series = pd.Series(y_array)

        # 1. Calculate feature differentiators
        # Pass top_n to avoid computing scores for features we won't use
        self.feature_differentiators_ = calculate_gaussian_correlation(
            X_df, y_series, top_n=self.top_n, correlation_threshold=self.correlation_threshold
        )

        # 2. Select top features
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n
        )

        # 3. Create membership model (Gaussian or Trapezoid)
        if self.member_function == "gaussian":
            self.model_ = create_gaussian_membership_dict(
                X_df, y_series, top_n_var_names=self.top_features_, n_gaussians=self.n_gaussians,
                max_samples=self.max_samples, random_state=self.random_state,
            )
        elif self.member_function == "trap":
            if self.trapz_method == "fast":
                from .trapz_math_fast import create_trapz_membership_dict_fast
                self.model_ = create_trapz_membership_dict_fast(
                    X_df, y_series, top_n_var_names=self.top_features_
                )
            elif self.trapz_method == "em":
                from .trapz_math import create_trapz_membership_dict
                self.model_ = create_trapz_membership_dict(
                    X_df, y_series, top_n_var_names=self.top_features_, n_trapezoids=self.n_gaussians,
                    max_samples=self.max_samples, random_state=self.random_state,
                )
            else:
                raise ValueError(f"Unknown trapz_method: {self.trapz_method}")
        elif self.member_function == "triangular":
            # Triangle is degenerate trapezoid; reuse EM code with shape="triangle".
            from .trapz_math import create_trapz_membership_dict
            self.model_ = create_trapz_membership_dict(
                X_df, y_series, top_n_var_names=self.top_features_, n_trapezoids=self.n_gaussians,
                max_samples=self.max_samples, random_state=self.random_state, shape="triangle",
            )
        else:
            raise ValueError(f"Unknown member_function: {self.member_function}")

        # Optionally refine antecedents on discriminative objective (zeroth-order TSK).
        if self.refine and self.member_function == "gaussian":
            from .refine import refine_classifier_antecedents
            self.model_, self.refine_info_ = refine_classifier_antecedents(
                self.model_,
                X_df.reset_index(drop=True),
                y_array,
                method=self.refine_method,
                l2_shrink=self.refine_l2_shrink,
                # Refine against user's chosen norms, not default.
                norms=self._anomaly_params().norms(),
                seed=self.random_state,
                verbose=False,
            )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels.
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        return tsk_predict(X_df, self.model_, self._anomaly_params())

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            Class probabilities (normalized firing strengths).
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        firing_strengths, labels = tsk_firing_strengths(X_df, self.model_, self._anomaly_params())

        row_sums = firing_strengths.sum(axis=1, keepdims=True)
        probabilities = np.zeros_like(firing_strengths)
        zero_rows = row_sums.flatten() == 0
        nonzero_rows = ~zero_rows

        probabilities[nonzero_rows] = firing_strengths[nonzero_rows] / row_sums[nonzero_rows]
        probabilities[zero_rows] = 1.0 / len(labels)  # Uniform for zero firing strength.

        label_to_idx = {label: i for i, label in enumerate(labels)}

        reordered_probs = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            if cls in label_to_idx:
                reordered_probs[:, i] = probabilities[:, label_to_idx[cls]]

        return reordered_probs

    def firing_strengths(self, X, anomaly_details: AnomalyParameters | None = None) -> tuple[np.ndarray, list]:
        """
        Compute the raw TSK firing strengths for X, optionally including an
        anomaly column.

        Args:
            X: Input data (n_samples, n_features)
            anomaly_details: If provided, an extra "anomaly" column is appended
                whose strength rises as every class membership falls.

        Returns:
            (firing_strengths, labels) where ``firing_strengths`` is a
            (n_samples, n_labels) array and ``labels`` lists the column labels
            (the anomaly label is last when ``anomaly_details`` is supplied).
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        return tsk_firing_strengths(X_df, self.model_, anomaly_details=anomaly_details)

    def augment(self, X, y):
        """
        Augment the existing model with new data (similar to the 2-pass approach).
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        y_series = pd.Series(y)

        new_model = create_gaussian_membership_dict(
            X_df, y_series, top_n_var_names=self.top_features_, n_gaussians=self.n_gaussians,
            max_samples=self.max_samples, random_state=self.random_state,
        )

        self.model_ = self.model_.augment(new_model)
        return self

    def deduplicate(self, rtol: float = DEFAULT_DEDUP_RTOL, atol: float = DEFAULT_DEDUP_ATOL) -> int:
        """Remove near-duplicate membership functions from the fitted model, in place.

        Two membership functions of the same (feature, label) only ever feed the
        same conorm fold, so collapsing near-duplicates there cannot change any
        prediction: exactly, at ``rtol=atol=0``; within measurement noise at
        looser tolerances. See issue #85 for the measurement this tradeoff is
        based on -- the defaults reproduce the tolerance this estimator always
        used before this method existed.

        This only touches ``self.model_`` in place; ``predict``/``predict_proba``
        read from it directly, so no other state needs updating.

        Args:
            rtol, atol: Passed through to
                `GaussianMixtureModel.remove_duplicate_membership_fcns`.

        Returns:
            The number of membership functions actually removed.
        """
        check_is_fitted(self)
        removed = self.model_.remove_duplicate_membership_fcns(rtol=rtol, atol=atol)
        self.n_membership_functions_ = self.model_.n_membership_functions
        self.n_deduplicated_membership_functions_ = removed
        return removed

    def to_simple_model(
        self, rtol: float = DEFAULT_DEDUP_RTOL, atol: float = DEFAULT_DEDUP_ATOL
    ) -> SimpleGaussianClassifierModel:
        """Materialize this classifier as an explicit, deduplicated rule model.

        Unlike `deduplicate`, this leaves `self.model_` untouched and returns a
        standalone `SimpleGaussianClassifierModel` (see `gauss_math.simple_gaussian_predict`)
        whose rules reference deduplicated membership-function ids -- useful for
        inspecting or deploying the model as an explicit rule set.

        Args:
            rtol, atol: Passed through to `GaussianMixtureModel.to_simple_model`;
                see `deduplicate` for what these tradeoff.
        """
        check_is_fitted(self)
        return self.model_.to_simple_model(self._anomaly_params(), rtol=rtol, atol=atol)


class TribbleSequenceClassifier(BaseEstimator, ClassifierMixin):
    """
    A base :class:`TribbleClassifier` refined by *local experts*.

    The idea, inspired by ``gaussian_mixture/iris_v2.py``, is to keep a single
    global model that spans every class and then bolt on small, focused *binary*
    experts that only fix the specific confusions the global model actually
    makes.

    **Fitting** proceeds in two stages:

    1. A *base* model (``layers_[0]``) is fit on all of the training data.
    2. Cross-validated predictions of the base model are used to build a
       confusion matrix. For every predicted class ``P`` whose region is large
       enough (at least ``min_confused`` predicted rows), the true class ``T``
       it is *most* mistaken for (at least ``min_class_samples`` rows) is found
       and a **local expert** — a fresh binary
       :class:`TribbleClassifier` trained only on the ``P`` and
       ``T`` rows — is added. Because the expert re-selects features and re-fits
       Gaussians on just that pair, it can key on evidence that separates ``P``
       from ``T`` even when that evidence is too weak to survive the global
       feature ranking. This is the expert's whole reason to exist: on a subset
       of the classes a region-``P`` sample is still argmax-``P``, so an expert
       that merely re-solved the same problem could only add noise. At most
       ``max_layers - 1`` experts are added, most-confused region first, each
       region ``P`` gets exactly one expert, and the two directions of a single
       confusion (``P``-vs-``T`` and ``T``-vs-``P``) are collapsed to one expert
       so a reverse expert can never undo another's flips.

    **Predicting** runs the base model, then consults each expert only for the
    samples whose *current* prediction equals that expert's region ``P``. Each
    such sample is refined ``P → T`` only when **both** gates agree:

    * **anomaly gate** — the sample's *anomaly level* (``1 - `` its best
      membership among the expert's classes, i.e. how far it sits outside the
      region the expert knows) is **below** that expert's anomaly threshold.
      Each threshold is fit independently by bisection during :meth:`fit`
      (:attr:`anomaly_thresholds_`); experts are independent because each only
      sees samples routed to its own region ``P``. A sample that fails this gate
      is out of the expert's competence; it is frozen on the base prediction and
      no later expert may touch it.
    * **confidence gate** — the expert prefers ``T`` over ``P`` by more than
      ``refine_margin`` in membership strength. Without this, every flip on an
      irreducibly-overlapping (Bayes-noise) region is a coin toss that erodes
      accuracy; requiring a margin means experts only overrule the base model
      when they are genuinely decisive.

    A sample relabelled ``T`` may still be picked up by a later expert keyed to
    region ``T``, so experts chain. The anomaly label is only ever an internal
    gate — it is never returned from :meth:`predict`.

    This follows scikit-learn's ``ClassifierMixin`` interface.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        member_function="gaussian",
        norm_conorm=DefaultNormCornorm,
        t_norm=None,
        t_conorm=None,
        allow_mixed_norms=False,
        random_state=42,
        max_layers=4,
        anomaly_threshold=0.99,
        anomaly_label="anomaly",
        tune_thresholds=True,
        refine_margin=0.0,
        min_confused=20,
        min_class_samples=5,
        cv=3,
        refine=False,
        refine_method="coordinate",
        refine_l2_shrink=0.05,
        max_samples=None,
    ):
        """
        Args:
            top_n, top_p, n_gaussians, member_function, norm_conorm, random_state,
            max_samples:
                Passed through to every underlying
                :class:`TribbleClassifier` (base and experts).
            max_layers: Maximum number of models in the cascade *including* the
                base model. The cascade may be shorter if fewer confused regions
                are worth an expert.
            anomaly_threshold: An expert refines a sample only when the sample's
                anomaly level (``1 - best class membership`` under that expert)
                is strictly below this value. Lower values make experts more
                conservative (they defer to the base model unless the sample sits
                squarely inside their region).
            anomaly_label: Internal label used to gate experts. It must not
                collide with a real class label; it is never returned by
                :meth:`predict`.
            tune_thresholds: When ``True`` (default) each expert's anomaly
                threshold is fit independently by bisection during
                :meth:`fit` (see :attr:`anomaly_thresholds_`). Because each
                expert only ever sees samples routed to its own prediction
                region ``P``, the experts are effectively independent and their
                thresholds can be optimized one at a time. When ``False`` every
                expert uses the shared ``anomaly_threshold``.
            refine_margin: An expert relabels ``P → T`` only when it prefers
                ``T`` over ``P`` by more than this much membership strength.
                ``0.0`` means "flip on any preference" (argmax); the tuned
                anomaly threshold is the primary safety knob, so this defaults
                to ``0.0``.
            min_confused: Minimum number of rows predicted as a given class
                required before that region is eligible for an expert.
            min_class_samples: Minimum number of confused rows in the
                ``(predicted P, true T)`` pair before an expert is trained for
                region ``P``.
            cv: Number of folds for the cross-validated confusion estimate used
                to find confused classes. Falls back to in-sample predictions
                when a class has too few samples to split.
        """
        self.is_fitted_: bool = False
        # layers_[0] is the base model; layers_[1:] mirror experts_.
        self.layers_: list[TribbleClassifier] = []
        # Each entry is (region_class P, confused true class T, expert_model,
        # tuned anomaly threshold).
        self.experts_: list[tuple] = []
        self.classes_ = None
        self.feature_names_in_: list[str] = []
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.member_function = member_function
        self.norm_conorm = norm_conorm
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.allow_mixed_norms = allow_mixed_norms
        self.random_state = random_state
        self.max_layers = max_layers
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_label = anomaly_label
        self.tune_thresholds = tune_thresholds
        self.refine_margin = refine_margin
        self.min_confused = min_confused
        self.min_class_samples = min_class_samples
        self.cv = cv
        self.refine = refine
        self.refine_method = refine_method
        self.refine_l2_shrink = refine_l2_shrink
        self.max_samples = max_samples

    def _make_layer(self) -> TribbleClassifier:
        return TribbleClassifier(
            top_n=self.top_n,
            top_p=self.top_p,
            n_gaussians=self.n_gaussians,
            random_state=self.random_state,
            refine=self.refine,
            refine_method=self.refine_method,
            refine_l2_shrink=self.refine_l2_shrink,
            max_samples=self.max_samples,
        )

    def _anomaly_params(self) -> AnomalyParameters:
        return AnomalyParameters(
            include_anomaly=True,
            threshold=self.anomaly_threshold,
            label=self.anomaly_label,
            norm_conorm=self.norm_conorm,
            member_function=self.member_function,
            t_norm=self.t_norm,
            t_conorm=self.t_conorm,
            allow_mixed_norms=self.allow_mixed_norms,
        )

    @staticmethod
    def _as_frame_series(X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
        # Align indices so boolean masking lines up cleanly.
        X_df = X_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)
        return X_df, y_series

    def _oof_predictions(self, X_df, y_series) -> np.ndarray:
        """Out-of-fold predictions of the base model, for an honest confusion
        estimate. Falls back to in-sample predictions when the data cannot be
        stratified into ``cv`` folds."""
        counts = y_series.value_counts()
        n_splits = min(int(self.cv), int(counts.min())) if len(counts) else 0
        if n_splits >= 2:
            try:
                skf = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                )
                oof = cross_val_predict(clone(self._make_layer()), X_df, y_series, cv=skf)
                return np.asarray(oof, dtype=object)
            except Exception:
                pass
        # Fallback: in-sample predictions from the already-fit base model.
        return np.asarray(self.layers_[0].predict(X_df), dtype=object)

    def _confused_pairs(self, pred, y_true) -> list[tuple]:
        """Confused ``(region P, true class T)`` pairs from a confusion matrix,
        most-confused pair first.

        For every predicted class ``P`` whose region has at least
        ``min_confused`` rows, the single true class ``T != P`` it is *most*
        mistaken for (by at least ``min_class_samples`` rows) is selected. Each
        ``P`` yields at most one pair, so region keys never repeat and a binary
        ``P``-vs-``T`` expert can be trained per pair.

        The two directions of one confusion, ``(P, T)`` and ``(T, P)``, describe
        the *same* binary boundary, so keeping both would let a reverse expert
        undo the other's flips depending on cascade order. Each unordered pair is
        therefore collapsed to a single expert — the higher-confusion direction,
        which is the larger, more reliable off-diagonal cell.
        """
        pred = np.asarray(pred, dtype=object)
        y_true = np.asarray(y_true, dtype=object)

        pairs = []
        for p in np.unique(pred):
            in_region = pred == p
            if int(in_region.sum()) < self.min_confused:
                continue
            true_here = y_true[in_region]
            best_t, best_errors = None, 0
            for t in np.unique(true_here):
                if t == p:
                    continue
                errors = int(np.sum(true_here == t))
                if errors >= self.min_class_samples and errors > best_errors:
                    best_t, best_errors = t, errors
            if best_t is not None:
                pairs.append((p, best_t, best_errors))

        # Largest confusion first, then drop the reverse of any pair already kept.
        pairs.sort(key=lambda r: r[2], reverse=True)
        result, seen = [], set()
        for p, t, _ in pairs:
            key = frozenset((p, t))
            if key in seen:
                continue
            seen.add(key)
            result.append((p, t))
        return result

    def _expert_scores(self, expert, X_sub, y_sub, p, t):
        """Out-of-fold ``(anomaly_level, prefers_t)`` for every row of the
        binary ``{p, t}`` training subset.

        Cross-validating the expert here means the threshold is bisected against
        scores the expert did *not* train on, so it is not tuned to noise it has
        memorised. Falls back to the in-sample expert when the subset is too
        small to stratify.
        """
        n = len(y_sub)
        anomaly = np.full(n, np.nan)
        prefers_t = np.full(n, np.nan)

        counts = y_sub.value_counts()
        n_splits = min(int(self.cv), int(counts.min())) if len(counts) else 0

        def fill(model, idx, X_slice):
            strengths, labels = model.firing_strengths(X_slice)
            col = {label: i for i, label in enumerate(labels)}
            anomaly[idx] = 1.0 - strengths.max(axis=1)
            prefers_t[idx] = strengths[:, col[t]] - strengths[:, col[p]]

        if n_splits >= 2:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            for train_idx, test_idx in skf.split(X_sub, y_sub):
                fold = self._make_layer()
                fold.fit(
                    X_sub.iloc[train_idx].reset_index(drop=True),
                    y_sub.iloc[train_idx].reset_index(drop=True),
                )
                fill(fold, test_idx, X_sub.iloc[test_idx])
        else:
            fill(expert, np.arange(n), X_sub)

        return anomaly, prefers_t

    def _bisect_threshold(self, anomaly, prefers_t, y_region, p, t):
        """Bisection search for the anomaly threshold that maximises this
        expert's accuracy over its own prediction region.

        Only samples the expert would flip (``prefers_t > refine_margin``)
        respond to the threshold, and ordering them by anomaly level makes
        region accuracy unimodal in the threshold: admitting the most confident
        (lowest-anomaly) flips first helps, until the flips start being wrong.
        Golden-section bisection narrows the ``[0, 1]`` interval onto that peak.
        """
        y_region = np.asarray(y_region, dtype=object)
        valid = ~np.isnan(anomaly)
        anomaly = anomaly[valid]
        prefers_t = prefers_t[valid]
        y_region = y_region[valid]
        if len(y_region) == 0:
            return self.anomaly_threshold

        would_flip = prefers_t > self.refine_margin

        def region_accuracy(threshold):
            flip = would_flip & (anomaly < threshold)
            pred = np.where(flip, t, p)
            return float(np.mean(pred == y_region))

        # Golden-section (bisection-family) search on [0, 1] for the max.
        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 0.0, 1.0 + 1e-9
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        fc, fd = region_accuracy(c), region_accuracy(d)
        for _ in range(60):
            if hi - lo < 1e-6:
                break
            if fc < fd:
                lo, c, fc = c, d, fd
                d = lo + inv_phi * (hi - lo)
                fd = region_accuracy(d)
            else:
                hi, d, fd = d, c, fc
                c = hi - inv_phi * (hi - lo)
                fc = region_accuracy(c)
        best = 0.5 * (lo + hi)
        # A threshold that refines nothing beats one that only makes things
        # worse, so never let tuning do net harm relative to leaving P alone.
        if region_accuracy(best) < region_accuracy(0.0):
            return 0.0
        return best

    def fit(self, X, y):
        """
        Fit the base model, then a binary local expert per confused pair.

        See the class docstring for the full description.
        """
        X_df, y_series = self._as_frame_series(X, y)
        self.feature_names_in_ = X_df.columns.tolist()
        self.classes_ = np.unique(y_series.values)

        if self.anomaly_label in set(self.classes_.tolist()):
            raise ValueError(
                f"anomaly_label={self.anomaly_label!r} collides with a real class label."
            )

        # Layer 0: the base model, trained on everything.
        base = self._make_layer()
        base.fit(X_df, y_series)
        self.layers_ = [base]
        self.experts_ = []

        # Find confused pairs from a cross-validated confusion estimate.
        oof = self._oof_predictions(X_df, y_series)
        pairs = self._confused_pairs(oof, y_series.values)

        for p, t in pairs[: max(0, self.max_layers - 1)]:
            # Binary expert on just the confused pair, so it re-selects the
            # features and re-fits the Gaussians that best separate P from T.
            rows = y_series.isin([p, t]).values
            X_sub = X_df[rows].reset_index(drop=True)
            y_sub = y_series[rows].reset_index(drop=True)
            if y_sub.nunique() < 2:
                continue

            expert = self._make_layer()
            expert.fit(X_sub, y_sub)

            threshold = self.anomaly_threshold
            if self.tune_thresholds:
                # Tune on the confusion this expert actually owns: the {p, t}
                # rows the base model routed into region P (base OOF pred == p).
                region = oof[rows] == p
                if region.any():
                    anomaly, prefers_t = self._expert_scores(expert, X_sub, y_sub, p, t)
                    threshold = self._bisect_threshold(
                        anomaly[region], prefers_t[region], y_sub.values[region], p, t
                    )

            self.experts_.append((p, t, expert, threshold))
            self.layers_.append(expert)

        # TODO(#85): each expert re-fits Gaussians on a {P, T} subset that, for
        # the shared class, covers rows the base model already fit -- with
        # deterministic k-means/BIC selection this frequently produces
        # bit-for-bit identical Gaussians between an expert and the base.
        # Nothing currently deduplicates across `self.layers_`. A cross-layer
        # `to_simple_model()` on this cascade -- unioning `layers_` via
        # `GaussianMixtureModel.augment()` and deduplicating the result with
        # `TribbleClassifier.deduplicate()`/`to_simple_model()` -- would recover
        # that redundancy, but is a separate, larger change: it also means
        # flattening `predict()`'s anomaly/confidence-margin gating away, whose
        # own accuracy cost (measured in #85) is a mechanism question, not a
        # numeric-approximation one, and is *not* well-described by a single
        # safe number. Do not wire dedup in here without addressing that.
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for X by running the base model then its experts.

        Returns an array of class labels in ``classes_``' own dtype. The anomaly
        label is never returned; a high anomaly level only freezes the sample on
        the base prediction and stops any further expert from touching it.
        """
        check_is_fitted(self)
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        X_df = X_df.reset_index(drop=True)

        preds = np.asarray(self.layers_[0].predict(X_df), dtype=object)

        # ``frozen`` marks samples an expert declared out-of-region; no further
        # expert may touch them.
        frozen = np.zeros(len(X_df), dtype=bool)

        for region_class, true_class, expert, threshold in self.experts_:
            # Only route samples whose current prediction *is* this expert's
            # region class.
            target = (preds == region_class) & ~frozen
            if not target.any():
                continue

            strengths, labels = expert.firing_strengths(X_df)
            col = {label: i for i, label in enumerate(labels)}

            # Anomaly level = 1 - best class membership under the expert. It is
            # low when the sample sits squarely inside a class the expert knows,
            # and approaches 1 for samples far outside its trained region. The
            # threshold is this expert's own bisection-tuned value.
            anomaly_level = 1.0 - strengths.max(axis=1)
            in_region = anomaly_level < threshold

            # Relabel P -> T only where the expert both (a) trusts the sample as
            # in-region and (b) prefers T over P by more than refine_margin.
            prefers_t = strengths[:, col[true_class]] - strengths[:, col[region_class]]
            flip = target & in_region & (prefers_t > self.refine_margin)
            preds[flip] = true_class

            # Out-of-region samples are frozen on the base prediction.
            frozen = frozen | (target & ~in_region)

        # Narrow back to the label dtype before returning. `preds` has to be
        # object *during* the loop -- writing a label into a fixed-width array
        # truncates it -- but returning object is not a cosmetic dtype quibble:
        # `type_of_target` classifies an object array as "unknown", so
        # `accuracy_score`, `ClassifierMixin.score` and every other metric raise
        # "can't handle a mix of multiclass and unknown targets". This estimator
        # could not be scored against integer labels at all. Every entry came
        # either from the base layer's prediction or from an expert's
        # `true_class`, so all of them are members of `classes_` and the cast is
        # exact.
        return preds.astype(self.classes_.dtype, copy=False)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Probabilities come from the base model, which is the only layer
        guaranteed to span every class. The experts refine the hard label via
        :meth:`predict`.
        """
        check_is_fitted(self)
        return self.layers_[0].predict_proba(X)

    @property
    def confused_classes_(self) -> list:
        """The region (predicted) class each expert is keyed to, in order."""
        return [p for p, _, _, _ in self.experts_]

    @property
    def confused_pairs_(self) -> list:
        """The ``(region_class P, true_class T)`` pair each expert arbitrates."""
        return [(p, t) for p, t, _, _ in self.experts_]

    @property
    def anomaly_thresholds_(self) -> list:
        """The bisection-tuned anomaly threshold of each expert, in order."""
        return [threshold for _, _, _, threshold in self.experts_]

    @property
    def n_layers(self) -> int:
        return len(self.layers_)
