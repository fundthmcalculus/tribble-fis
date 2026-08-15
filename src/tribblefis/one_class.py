"""One-class (novelty / outlier) detection with the fuzzy anomaly rule.

`TribbleClassifier` reaches the "none of the above" anomaly rule only through a
*multi-class* fit: it needs at least two labels, its feature selection ranks
features by class separation, and its ``predict_proba`` returns a softmax that is
trivially 1.0 when there is a single class. So the setting the anomaly rule is
most naturally suited to -- abundant "normal" data, few or no anomalies, score a
new point by how far it sits from normal -- was not expressible.

``TribbleOneClassDetector`` fills that gap. It fits Gaussian memberships on a
single "normal" class using the library's own
:func:`create_gaussian_membership_dict`, and scores a point by the complement of
its best rule firing strength:

    anomaly(x) = 1 - max_rule firing_strength(x)

A normal point fires the learned rules (high firing, low anomaly); a point far
from every learned pattern fires nothing (firing -> 0, anomaly -> 1). This is the
one-class reduction of the ``1 - max class membership`` rule, and it reuses the
exact firing-strength computation the classifier uses, so the two cannot drift.

The estimator follows scikit-learn's :class:`OutlierMixin` conventions:

* ``score_samples(X)`` -- higher means *more normal* (it returns ``-anomaly``),
* ``decision_function(X)`` -- ``score_samples`` shifted by a threshold; negative
  is an outlier,
* ``predict(X)`` -- ``+1`` inlier / ``-1`` outlier,

so it composes with ``roc_auc_score`` (on ``-score_samples`` or
``anomaly_score``), pipelines, and ``fit_predict``.

Feature selection is the one piece that cannot be inherited: the classifier's
differentiation score is a *supervised* criterion (class separation) and is
undefined with one class. See ``feature_selection`` below for the unsupervised
alternatives.

Scoring past a handful of features
----------------------------------
The complement above **saturates**, and past roughly 60 features it stops being
usable. The product t-norm over per-feature Gaussians is
``exp(-0.5 * sum_j z_j^2)``, and a *typical normal* point already carries
``sum_j z_j^2 ~ n_features``. Once that exceeds ~74, ``exp(-37)`` falls below
float64's resolution next to 1.0, so ``1 - firing`` rounds to exactly 1.0 for
normal and anomalous points alike. Measured on synthetic correlated data, 32
components whitened:

===========  ==================  ==================  =================
n_features   normal at exactly   AUROC (complement   det@1%FPR
             1.0                 / surprisal)        (complement / surprisal)
===========  ==================  ==================  =================
32           0.0%                0.955 / 0.955       0.586 / 0.586
64           22.8%               0.852 / 0.918       0.000 / 0.448
128          100%                0.500 / 0.839       0.000 / 0.206
===========  ==================  ==================  =================

At 128 the complement is *chance* -- every point is tied at the top. AUROC hides
the onset of this (0.955 at 32 features looks fine) while the strict operating
point does not: det@1%FPR is already 0.000 at 64 features, where the summed
surprisal still matches Mahalanobis exactly.

``score="surprisal"`` sums ``-log(membership_j)`` instead of taking the
complement of their product. It is the same fitted memberships and the same
ordering the complement *intends*, just kept in the log domain where it does not
round away. Use it whenever more than a handful of features reach the rules --
which is essentially always under ``whiten=True``. See ``score`` below.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted

from .gauss_data import AnomalyParameters, DefaultNormCornorm
from .gauss_math import create_gaussian_membership_dict, tsk_firing_strengths
# Aliased: this class has a `t_conorm` *parameter*, and an unaliased import
# would read as that parameter at every use site inside the class body.
from .gauss_math import t_conorm as _t_conorm_fold

NORMAL_LABEL = "normal"


class TribbleOneClassDetector(OutlierMixin, BaseEstimator):
    """Fuzzy one-class novelty detector.

    Parameters
    ----------
    n_gaussians : int, default 0
        Gaussians per feature for the normal class (0 = automatic), passed
        straight to :func:`create_gaussian_membership_dict`. More components let
        the "normal" region be multi-modal.
    norm_conorm : str, default "probability"
        Fuzzy norm/conorm family, as in :class:`TribbleClassifier`.
    feature_selection : {"all", "variance"} or list, default "all"
        Which features to build rules over. The classifier's differentiation
        score needs two classes, so it cannot be used here.

        * ``"all"`` -- every feature (default; safest, and what a genuine
          one-class problem usually wants).
        * ``"variance"`` -- the ``top_n`` highest-variance features, a standard
          unsupervised filter for dropping near-constant columns.
        * an explicit list of column names -- use exactly these.

        A dispersion / tightness-based ranking is a natural future addition; it
        is intentionally not implemented yet.

        Under ``whiten=True`` selection applies to the whitened components
        (named ``pc0``, ``pc1``, ...), not the input columns: ``"all"`` and
        ``"variance"`` keep the leading components, and an explicit list must
        name components rather than original features -- naming input columns
        raises rather than silently selecting nothing.
    top_n : int, default -1
        Number of features to keep when ``feature_selection="variance"``. ``-1``
        keeps all.
    whiten : bool, default False
        If True, PCA-whiten the features (decorrelate to unit variance) on the
        normal data before fitting memberships, and apply the same stored
        transform when scoring. The per-feature product-t-norm rule assumes
        feature independence, so on correlated features (embeddings, activations)
        the raw detector misses cross-feature covariance and scores near chance;
        whitening makes the independence assumption approximately hold. Off by
        default -- no behaviour change for existing callers.
    whiten_components : int, float or None, default None
        ``n_components`` for the internal ``PCA`` when ``whiten=True`` (int for a
        component count, float for an explained-variance ratio). ``None`` keeps
        ``min(n_samples - 1, n_features)`` components. Ignored when
        ``whiten=False``.
    contamination : float, default 0.05
        Expected fraction of outliers, used only to place the
        ``decision_function`` threshold (the ``contamination`` quantile of the
        training anomaly scores), matching sklearn's outlier-detector convention.
        It does not affect ``score_samples`` or ``anomaly_score``.

        sklearn's detectors restrict this to ``(0, 0.5]``. This one takes the
        value as given -- a value past ``0.5`` still places the threshold where
        you asked -- but warns, since flagging most of the "normal" training set
        as outlier is far more often a mistake than an intent. ``<= 0`` puts the
        threshold at the training minimum, flagging nothing.
    score : {"complement", "surprisal", "trimmed"}, default "complement"
        How the per-feature memberships are aggregated into an anomaly score.

        * ``"complement"`` -- ``1 - max_rule firing_strength``, in ``[0, 1]``.
          The original formulation, kept as the default so existing callers are
          unaffected. **Saturates past ~60 features** -- see the module
          docstring; do not use it with ``whiten=True`` on wide input.
        * ``"surprisal"`` -- ``sum_j -log(membership_j)``, in ``[0, inf)``. The
          same ordering the complement intends, computed in the log domain where
          it does not round away. Non-saturating; recommended past a handful of
          features. Under the product t-norm this is exactly
          ``-log(firing_strength)``, so it is a monotone transform of the
          complement wherever the complement has not lost resolution.
        * ``"trimmed"`` -- the surprisal sum with the ``trim`` largest
          per-sample terms dropped, so one odd feature cannot by itself flag an
          otherwise normal point. Whether it beats ``"surprisal"`` is
          data-dependent: measured better on some targets and worse on others,
          so it is offered rather than recommended.

        ``score_samples``/``decision_function``/``predict`` are unchanged in
        meaning under all three -- only the ordering's resolution differs.
    trim : int, default 2
        Number of largest per-sample surprisal terms to drop when
        ``score="trimmed"``. Must be smaller than the number of features
        reaching the rules. Ignored by the other scores.
    max_samples : int or None, default None
        Cap on rows per feature when fitting memberships (see the classifier).
    random_state : int, default 42
        Seed for membership fitting.

    Attributes
    ----------
    model_ : GaussianMixtureModel
        The fitted normal-class membership model.
    top_features_ : list
        Features actually used.
    offset_ : float
        Threshold on ``score_samples`` such that ``decision_function`` is
        negative for the ``contamination`` fraction of the training data.
    """

    _SCORES = ("complement", "surprisal", "trimmed")

    def __init__(
        self,
        n_gaussians: int = 0,
        norm_conorm=DefaultNormCornorm,
        feature_selection: str | list = "all",
        top_n: int = -1,
        whiten: bool = False,
        whiten_components: int | float | None = None,
        contamination: float = 0.05,
        score: str = "complement",
        trim: int = 2,
        t_norm=None,
        t_conorm=None,
        allow_mixed_norms: bool = False,
        max_samples: int | None = None,
        random_state: int = 42,
    ):
        self.n_gaussians = n_gaussians
        self.norm_conorm = norm_conorm
        self.feature_selection = feature_selection
        self.top_n = top_n
        self.whiten = whiten
        self.whiten_components = whiten_components
        self.contamination = contamination
        self.score = score
        self.trim = trim
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.allow_mixed_norms = allow_mixed_norms
        self.max_samples = max_samples
        self.random_state = random_state

    # -- internals --------------------------------------------------------

    def _norm_params(self) -> AnomalyParameters:
        # include_anomaly stays False: the anomaly here is 1 - max firing over
        # the normal rules, computed in `anomaly_score`, not an extra rule column.
        return AnomalyParameters(
            include_anomaly=False,
            norm_conorm=self.norm_conorm,
            t_norm=self.t_norm,
            t_conorm=self.t_conorm,
            allow_mixed_norms=self.allow_mixed_norms,
        )

    def _feature_memberships(self, X_df: pd.DataFrame) -> np.ndarray:
        """``(n_samples, n_features)`` per-feature membership of the normal class.

        This is `tsk_firing_strengths`' inner fold stopped one step early: within
        a feature the label's membership functions are still combined with the
        t-conorm, but the t-norm *across* features -- the step that multiplies
        the terms together and destroys their magnitudes -- is left to the
        caller. `anomaly_score` either applies it (``"complement"``) or works in
        the log domain instead (``"surprisal"``/``"trimmed"``).

        Iterating `model_.feature_models` rather than `top_features_` mirrors
        `tsk_firing_strengths` exactly, so the columns here are the same terms
        its product is taken over.
        """
        norms = self._norm_params().norms()
        columns = []
        for name, feature_model in self.model_.feature_models.items():
            if NORMAL_LABEL not in feature_model.label_models or name not in X_df:
                continue
            data = np.asarray(X_df[name].values)
            membership = np.zeros(len(data))
            for mf in feature_model.label_models[NORMAL_LABEL].memberships:
                membership = _t_conorm_fold(
                    membership, mf.evaluate(data), norms.t_conorm
                )
            columns.append(membership)
        if not columns:
            raise ValueError("no fitted feature contributes a membership")
        return np.column_stack(columns)

    def _surprisal_terms(self, X_df: pd.DataFrame) -> np.ndarray:
        """Per-feature ``-log(membership)``, finite everywhere.

        A Gaussian membership underflows to exactly 0.0 around ``z = 38``, and
        ``-log(0)`` is ``inf`` -- which would make every sufficiently-far point
        tie at ``inf`` and reintroduce the very saturation this score exists to
        avoid, just at the other end. Flooring at the smallest positive double
        caps a single term at ~708 instead. Resolution is still lost past
        ``z ~ 38``, but that is five times further out than the complement's
        cliff and far beyond any separation that matters.
        """
        memberships = self._feature_memberships(X_df)
        return -np.log(np.clip(memberships, np.finfo(float).tiny, 1.0))

    def _to_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        missing = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing features seen during fit: {missing}")
        # Re-select by name in fit order rather than trusting the caller's
        # layout. Neither failure mode below announces itself:
        #   * the whitening PCA transform is *positional*, so a permuted frame
        #     is decorrelated against the wrong axes and scores nonsense;
        #   * `tsk_firing_strengths` silently skips features it cannot find in
        #     `X`, so a frame missing a fitted column fires every rule at 1.0 --
        #     i.e. reports every point as perfectly normal.
        return X[self.feature_names_in_].reset_index(drop=True).copy()

    def _transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored whitening transform (identity when whiten=False).

        Fitting and scoring both route through here so the memberships and the
        decorrelation can never desync -- the concern that folding whitening
        into the estimator (rather than leaving it to the caller) exists to
        remove.
        """
        if not self.whiten:
            return X_df
        Z = self._pca_.transform(X_df.to_numpy())
        return pd.DataFrame(
            Z, columns=[f"pc{i}" for i in range(Z.shape[1])], index=X_df.index
        )

    def _select_features(self, X: pd.DataFrame) -> list:
        fs = self.feature_selection
        if isinstance(fs, (list, tuple)):
            missing = [c for c in fs if c not in X.columns]
            if missing:
                raise ValueError(f"feature_selection lists unknown columns: {missing}")
            return list(fs)
        if fs == "all":
            return list(X.columns)
        if fs == "variance":
            if self.whiten:
                # Whitened components are unit-variance and already ordered by
                # explained variance, so a variance filter is moot; keep the
                # leading top_n components (or all).
                cols = list(X.columns)
                return cols if self.top_n <= 0 else cols[: self.top_n]
            var = X.var(axis=0, numeric_only=True)
            order = var.sort_values(ascending=False).index.tolist()
            return order if self.top_n <= 0 else order[: self.top_n]
        raise ValueError(
            f"feature_selection must be 'all', 'variance', or a list; got {fs!r}"
        )

    # -- sklearn API ------------------------------------------------------

    def fit(self, X, y=None):
        """Fit on normal data.

        ``y`` is ignored -- every row is treated as the normal class. It is
        accepted so the estimator drops into pipelines and ``fit(X, y)`` call
        sites unchanged.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            X = np.asarray(X)
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        X_raw = self._to_frame(X)

        # Fit the whitening transform on the normal data, if requested, before
        # anything else -- feature selection and membership fitting then operate
        # on the decorrelated components.
        if self.whiten:
            from sklearn.decomposition import PCA

            n_comp = self.whiten_components
            if n_comp is None:
                n_comp = min(len(X_raw) - 1, X_raw.shape[1])
            self._pca_ = PCA(
                n_components=n_comp, whiten=True, random_state=self.random_state
            ).fit(X_raw.to_numpy())
        X_df = self._transform(X_raw)

        if self.score not in self._SCORES:
            raise ValueError(
                f"score must be one of {self._SCORES}; got {self.score!r}"
            )

        self.top_features_ = self._select_features(X_df)
        if not self.top_features_:
            raise ValueError("no features selected")
        if self.score == "trimmed" and not 0 <= self.trim < len(self.top_features_):
            # Trimming every term leaves nothing to sum, which would score every
            # point identically -- caught here rather than at the first scoring
            # call, where the empty sum would just look like a broken detector.
            raise ValueError(
                f"trim={self.trim} must be in [0, {len(self.top_features_)}) "
                f"for {len(self.top_features_)} selected features"
            )

        y_normal = pd.Series([NORMAL_LABEL] * len(X_df))
        self.model_ = create_gaussian_membership_dict(
            X_df,
            y_normal,
            top_n_var_names=self.top_features_,
            n_gaussians=self.n_gaussians,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )

        # Place the decision threshold at the contamination quantile of the
        # training anomaly scores: the `contamination` fraction with the highest
        # anomaly become outliers, matching sklearn's outlier detectors. Score
        # the RAW input -- score_samples applies the whitening transform itself,
        # so passing the already-transformed frame would whiten twice.
        train_scores = self.score_samples(X_raw)  # higher = more normal
        q = float(self.contamination)
        if q > 0.5:
            # Honoured rather than clipped -- silently moving the threshold
            # somewhere other than where the caller put it is worse than an
            # unusual threshold. But past 0.5 the majority of the data the
            # detector was told is normal gets labelled outlier, which is
            # almost always a mistake, so it does not pass quietly.
            warnings.warn(
                f"contamination={q} is above 0.5, so more than half of the "
                "training data will be flagged as outlier. The value is used "
                "as given; sklearn's outlier detectors restrict contamination "
                "to (0, 0.5].",
                UserWarning,
                stacklevel=2,
            )
        self.offset_ = float(np.quantile(train_scores, q)) if q > 0 else float(
            train_scores.min()
        )
        self.is_fitted_ = True
        return self

    def anomaly_score(self, X) -> np.ndarray:
        """Anomaly score, higher = more anomalous, per the ``score`` parameter.

        ``"complement"`` returns ``1 - max_rule firing_strength`` in ``[0, 1]``;
        ``"surprisal"`` and ``"trimmed"`` return a summed surprisal in
        ``[0, inf)``. Only the first is bounded -- the scale carries no meaning
        under any of them, so this is the quantity to hand to ``roc_auc_score``
        as the positive-class score in every case.
        """
        check_is_fitted(self, "model_")
        X_df = self._transform(self._to_frame(X))

        if self.score == "complement":
            firing, _ = tsk_firing_strengths(X_df, self.model_, self._norm_params())
            max_firing = firing.max(axis=1) if firing.size else np.zeros(len(X_df))
            return 1.0 - np.clip(max_firing, 0.0, 1.0)

        terms = self._surprisal_terms(X_df)
        if self.score == "surprisal":
            return terms.sum(axis=1)
        if self.score == "trimmed":
            if self.trim <= 0:
                return terms.sum(axis=1)
            # Partition rather than a full sort: only the boundary between the
            # `trim` largest terms and the rest matters, not their order.
            cut = terms.shape[1] - self.trim
            return np.partition(terms, cut - 1, axis=1)[:, :cut].sum(axis=1)
        raise ValueError(f"score must be one of {self._SCORES}; got {self.score!r}")

    def score_samples(self, X) -> np.ndarray:
        """sklearn convention: higher = more normal. Returns ``-anomaly_score``."""
        return -self.anomaly_score(X)

    def decision_function(self, X) -> np.ndarray:
        """``score_samples`` shifted by the fitted threshold; negative = outlier."""
        check_is_fitted(self, "offset_")
        return self.score_samples(X) - self.offset_

    def predict(self, X) -> np.ndarray:
        """``+1`` inlier / ``-1`` outlier, thresholded at ``contamination``."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
