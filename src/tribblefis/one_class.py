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
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted

from .gauss_data import AnomalyParameters, DefaultNormCornorm
from .gauss_math import create_gaussian_membership_dict, tsk_firing_strengths

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
        is intentionally not implemented yet. Ignored when ``whiten=True`` (the
        whitened components are used directly).
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
        ``n_components`` for the internal ``PCA`` when ``whiten=True, cov="pca"``
        (int for a component count, float for an explained-variance ratio).
        ``None`` keeps ``min(n_samples - 1, n_features)`` components. Ignored
        otherwise.
    cov : {"pca", "ledoit_wolf"}, default "pca"
        How the whitening covariance is estimated when ``whiten=True``.

        * ``"pca"`` -- rank-truncated PCA whitening (keep ``whiten_components``).
          The truncation drops low-variance noise directions, which helps the
          strict operating point.
        * ``"ledoit_wolf"`` -- full-rank whitening from a Ledoit-Wolf shrunk
          covariance. A well-regularized full-covariance Mahalanobis; measured as
          a small but consistent zero-shot improvement in tail separation over
          rank-truncated PCA on the corpora tested. No rank to choose.
    score : {"complement", "surprisal", "trimmed"}, default "complement"
        How the anomaly score is formed from the rule firing strengths.

        * ``"complement"`` -- ``1 - max firing``, the original score. It equals
          ``1 - exp(-sum surprisal)`` and **saturates** once there are more than
          a handful of features: typical points pile up near 1, flattening the
          low-false-positive tail. Fine for AUROC, poor for strict operating
          points.
        * ``"surprisal"`` -- the summed per-feature surprisal
          ``sum_j -log(membership_j)``. Non-saturating (it never passes through
          the exponential), so it preserves the tail ordering a strict threshold
          needs. Recommended whenever the feature/component count is more than a
          handful (i.e. almost always with ``whiten=True``).
        * ``"trimmed"`` -- the surprisal sum with the ``trim`` largest
          per-sample terms dropped, so a single odd feature cannot flag an
          otherwise-normal point. The most robust low-FPR behaviour measured.
    trim : int, default 2
        Number of largest per-sample surprisals to drop for ``score="trimmed"``.
    few_shot : {"logistic", "none"}, default "logistic"
        What to do when ``fit`` is given a few labelled anomalies (see ``fit``).
        ``"logistic"`` fits an L2 discriminant on the whitened features and uses
        it as the anomaly score -- the one-class density supplies the *magnitude*
        of deviation, but a few labels supply the *direction*, which is what the
        strict operating point needs. ``"none"`` ignores the labels (pure
        one-class).
    few_shot_C : float, default 1.0
        Inverse-regularisation for the few-shot logistic discriminant.
    contamination : float, default 0.05
        Expected fraction of outliers, used only to place the
        ``decision_function`` threshold (the ``contamination`` quantile of the
        training anomaly scores), matching sklearn's outlier-detector convention.
        It does not affect ``score_samples`` or ``anomaly_score``.
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

    def __init__(
        self,
        n_gaussians: int = 0,
        norm_conorm=DefaultNormCornorm,
        feature_selection: str | list = "all",
        top_n: int = -1,
        whiten: bool = False,
        whiten_components: int | float | None = None,
        cov: str = "pca",
        score: str = "complement",
        trim: int = 2,
        few_shot: str = "logistic",
        few_shot_C: float = 1.0,
        contamination: float = 0.05,
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
        self.cov = cov
        self.score = score
        self.trim = trim
        self.few_shot = few_shot
        self.few_shot_C = few_shot_C
        self.contamination = contamination
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

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True).copy()
        return pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)

    def _transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored whitening transform (identity when whiten=False).

        Fitting and scoring both route through here so the memberships and the
        decorrelation can never desync -- the concern that folding whitening
        into the estimator (rather than leaving it to the caller) exists to
        remove.
        """
        if not self.whiten:
            return X_df
        if self.cov == "ledoit_wolf":
            Z = (X_df.to_numpy() - self._white_mu_) @ self._white_W_
        else:
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

    def _per_feature_surprisal(self, X_df: pd.DataFrame) -> np.ndarray:
        """(-log membership) per selected feature, shape (n, n_features).

        The feature's membership is the fuzzy OR (max) over its Gaussians, so
        this is the per-feature "surprise" under the fitted normal model. Summed
        (or trimmed-summed) it forms the non-saturating anomaly score, which does
        not pass through the exponential that makes ``1 - max firing`` saturate.
        """
        cols = self.top_features_
        out = np.empty((len(X_df), len(cols)))
        for j, fname in enumerate(cols):
            lm = self.model_.feature_models[fname].label_models[NORMAL_LABEL]
            x = X_df[fname].to_numpy()
            mem = np.zeros(len(x))
            for mf in lm.memberships:
                sig = max(float(mf.sigma), 1e-9)
                mem = np.maximum(mem, np.exp(-0.5 * ((x - mf.mu) / sig) ** 2))
            out[:, j] = -np.log(np.clip(mem, 1e-12, 1.0))
        return out

    # -- sklearn API ------------------------------------------------------

    def fit(self, X, y=None):
        """Fit on normal data, optionally with a few labelled anomalies.

        ``y=None`` (or a single class) -- pure one-class: every row is normal.

        ``y`` with two classes -- semi-supervised / few-shot: the normal class
        (the majority label, or ``0``) fits the whitening and the one-class
        density exactly as before, and the minority (anomaly) examples train an
        L2 logistic discriminant on the whitened features that becomes the
        anomaly score. A handful of anomalies (5-25) is enough. The one-class
        density supplies the *magnitude* of deviation; the labels supply the
        *direction*, which is what a strict operating point needs.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            X = np.asarray(X)
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        X_raw = self._to_frame(X)

        # Decide the normal-vs-anomaly split from y, if given.
        self._is_few_shot = False
        normal_mask = np.ones(len(X_raw), dtype=bool)
        if y is not None and self.few_shot != "none":
            y_arr = np.asarray(y)
            classes, counts = np.unique(y_arr, return_counts=True)
            if len(classes) >= 2:
                normal_label = classes[np.argmax(counts)]  # majority = normal
                self.normal_label_ = normal_label
                normal_mask = y_arr == normal_label
                self._is_few_shot = True
                self._y_fit = y_arr
        X_raw_all = X_raw
        X_raw = X_raw.iloc[normal_mask].reset_index(drop=True)  # density on normal only

        # Fit the whitening transform on the normal data, if requested, before
        # anything else -- feature selection and membership fitting then operate
        # on the decorrelated components.
        if self.whiten and self.cov == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf

            A = X_raw.to_numpy()
            self._white_mu_ = A.mean(0)
            cov = LedoitWolf().fit(A).covariance_
            w, V = np.linalg.eigh(cov)
            # whitening matrix Sigma^{-1/2} = V diag(w^-1/2) V^T (full rank)
            self._white_W_ = V @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-12, None))) @ V.T
        elif self.whiten:
            from sklearn.decomposition import PCA

            n_comp = self.whiten_components
            if n_comp is None:
                n_comp = min(len(X_raw) - 1, X_raw.shape[1])
            self._pca_ = PCA(
                n_components=n_comp, whiten=True, random_state=self.random_state
            ).fit(X_raw.to_numpy())
        X_df = self._transform(X_raw)

        self.top_features_ = self._select_features(X_df)
        if not self.top_features_:
            raise ValueError("no features selected")

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
        # Few-shot: fit the logistic discriminant on the whitened features over
        # ALL provided rows (normal + the few anomalies), using the whitening
        # learned from normal only.
        self._logit_ = None
        if self._is_few_shot:
            from sklearn.linear_model import LogisticRegression

            Zall = self._transform(X_raw_all).to_numpy()
            self._logit_ = LogisticRegression(
                class_weight="balanced", C=self.few_shot_C, max_iter=1000
            ).fit(Zall, (self._y_fit != self.normal_label_).astype(int))

        # Threshold at the contamination quantile of the normal training scores.
        # Score the RAW normal input (score_samples whitens internally).
        train_scores = self.score_samples(X_raw)  # higher = more normal
        q = float(np.clip(self.contamination, 0.0, 0.5))
        self.offset_ = float(np.quantile(train_scores, q)) if q > 0 else float(
            train_scores.min()
        )
        self.is_fitted_ = True
        return self

    def anomaly_score(self, X) -> np.ndarray:
        """Anomaly score; higher means more anomalous.

        With few-shot labels, returns the logistic discriminant (directional).
        Otherwise the one-class score selected by ``score``: ``"complement"``
        (``1 - max firing``, saturating), ``"surprisal"`` (summed per-feature
        surprisal, non-saturating), or ``"trimmed"`` (surprisal minus the
        ``trim`` largest per-sample terms, robust). The natural quantity to hand
        to ``roc_auc_score`` as the positive-class score.
        """
        check_is_fitted(self, "model_")
        X_df = self._transform(self._to_frame(X))

        if getattr(self, "_logit_", None) is not None:
            return self._logit_.decision_function(X_df.to_numpy())

        if self.score == "complement":
            firing, _ = tsk_firing_strengths(X_df, self.model_, self._norm_params())
            max_firing = firing.max(axis=1) if firing.size else np.zeros(len(X_df))
            return 1.0 - np.clip(max_firing, 0.0, 1.0)

        S = self._per_feature_surprisal(X_df)
        if self.score == "surprisal":
            return S.sum(axis=1)
        if self.score == "trimmed":
            t = min(self.trim, S.shape[1] - 1)
            return np.sort(S, axis=1)[:, : S.shape[1] - t].sum(axis=1) if t > 0 else S.sum(1)
        raise ValueError(
            f"score must be 'complement', 'surprisal', or 'trimmed'; got {self.score!r}"
        )

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
