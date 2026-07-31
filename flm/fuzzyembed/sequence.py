"""The fuzzy sequence model -- next-token prediction over membership vectors.

Given the fuzzy embeddings of the previous *k* tokens, predict the membership vector
of the next one. Concretely this is a MIMO TSK system: inputs are the windowed
level-``L`` memberships of the context, outputs are the level-``L`` memberships of
the next token, and every input and output dimension has a **name**. So a learned
rule reads

    IF prev1[noun.animal] is High AND prev1[verb.motion] is Low
    THEN next[verb.motion] ~ 0.7

which is a statement about language that a human can check -- the thing a
transformer's next-token head cannot offer.

Why level 2 (the ~45 WordNet supersenses)
-----------------------------------------
The finest level has thousands of dimensions; a flat FIS over that many inputs is
hopeless, and one over thousands of *outputs* is worse. Level 2 is the design
centre the plan identified: wide enough to say something semantic, narrow enough
for a rule base. Generation then proceeds coarse-to-fine -- pick the supersense
here, and let the decoder resolve which lexeme inside it (``decode.py``).

Scope note
----------
This is a first cut, and it is a *semantic-class* sequence model, not a language
model: it predicts what kind of thing comes next, not a well-formed word sequence.
Function words are unrepresented (they carry no hierarchy membership), so it cannot
produce grammatical text. That is a property of the representation, not a bug to be
tuned away -- see ``../FUZZY_EMBEDDING_PLAN.md`` section 8 on what an FLM still needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .corpus import Corpus
from .embedder import FuzzyEmbedder
from .syntax import SyntaxTagger


@dataclass
class SequenceInfo:
    n_windows: int
    n_features: int
    n_outputs: int
    output_names: list[str]
    train_mae: float
    test_mae: float
    baseline_mae: float
    separation: float          # mean predicted degree, positives minus negatives
    balanced_accuracy: float   # on the binarised "does S come next?" task


class FuzzySequenceModel:
    """MIMO TSK next-token predictor over named membership dimensions."""

    def __init__(self, embedder: FuzzyEmbedder, level: int = 2, window: int = 2,
                 n_outputs: int = 12, top_n: int = 8,
                 membership_threshold: float = 0.25, use_syntax: bool = True,
                 max_rules: int = 24, max_order: int = 2, random_state: int = 42):
        self.emb = embedder
        self.level = level
        self.window = window
        self.n_outputs = n_outputs
        self.top_n = top_n
        self.membership_threshold = membership_threshold
        self.use_syntax = use_syntax
        self.max_rules = max_rules
        self.max_order = max_order
        # The semantic embedding cannot see function words, which is why the
        # semantics-only model scored 0.528 balanced accuracy (chance 0.500). Named
        # syntactic categories supply the missing signal without giving up
        # interpretability -- see syntax.py.
        self.tagger = SyntaxTagger(
            getattr(embedder.senses, "lemma_synsets", None)) if use_syntax else None
        self.random_state = random_state
        self.models_: dict = {}
        self.output_index_: list[int] = []
        self.prior_: np.ndarray | None = None
        self.info_: SequenceInfo | None = None

    def _proba(self, X: np.ndarray, names: list[str]) -> np.ndarray:
        """Predicted membership degree per output dimension."""
        cols = [self.models_[name].predict(np.asarray(X)) for name in names]
        return np.clip(np.vstack(cols).T, 0.0, 1.0)

    # -- featurisation -----------------------------------------------------

    def _token_vector(self, token: str) -> np.ndarray:
        """Context features for one token: semantic memberships (+) syntax."""
        sem = self.emb.embed(token, self.level)
        if self.tagger is None:
            return sem
        return np.concatenate([sem, self.tagger.tag(token)])

    def _target_vector(self, token: str) -> np.ndarray:
        """What we predict about the *next* token: same joint space as the context.

        Predicting the syntactic categories too is the point -- it lets the model
        say "a noun is due" and lets the decoder emit function words, which a
        semantics-only target space made impossible.
        """
        return self._token_vector(token)

    def _output_names(self) -> list[str]:
        h = self.emb.h
        names = [h.name(k) for k in h.level_keys(self.level)]
        if self.tagger is not None:
            names += self.tagger.names()
        return names

    def _windows(self, corpus: Corpus, max_windows: int, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Build (context-window features, next-token targets)."""
        h = self.emb.h
        width = h.width(self.level)
        cache: dict[str, np.ndarray] = {}

        def vec(tok: str) -> np.ndarray:
            if tok not in cache:
                cache[tok] = self._token_vector(tok)
            return cache[tok]

        rng = np.random.default_rng(seed)
        sents = [s for s in corpus.sentences if len(s) > self.window]
        rng.shuffle(sents)

        X, Y = [], []
        for sent in sents:
            vecs = [vec(t) for t in sent]
            for i in range(self.window, len(sent)):
                target = vecs[i]
                # Skip positions whose target carries no membership at all -- a
                # function word. Training toward an all-zero target teaches the
                # model to predict silence, which dominates the loss and produces a
                # model that always says nothing.
                if target.sum() <= 0:
                    continue
                ctx = np.concatenate([vecs[i - j] for j in range(self.window, 0, -1)])
                X.append(ctx)
                Y.append(target)
            if len(X) >= max_windows:
                break
        if not X:
            raise RuntimeError("no usable windows; corpus too small or coverage zero")
        return np.asarray(X, dtype=np.float32)[:max_windows], \
            np.asarray(Y, dtype=np.float32)[:max_windows]

    def feature_names(self) -> list[str]:
        """Named input features, e.g. ``prev1:noun.animal``, ``prev1:DETERMINER``.

        These strings are what appear inside the learned rules, which is the whole
        point -- both the semantic and the syntactic half stay legible.
        """
        base = self._output_names()
        return [f"prev{lag}:{n}" for lag in range(self.window, 0, -1) for n in base]

    # -- fit / predict -----------------------------------------------------

    def fit(self, corpus: Corpus, max_windows: int = 4000, test_frac: float = 0.2,
            verbose: bool = True):
        """Fit one ``MembershipRuleRegressor`` per modelled output dimension.

        Neither TRIBBLE estimator family works on this data, for two separate
        reasons, both measured:

        * ``MixtureOfGaussiansFuzzyRegressor`` raises -- ``partition_output``
          quantile-bins the target and a membership coordinate is mostly zero, so
          ``pd.qcut`` hits duplicate bin edges.
        * ``MixtureOfGaussiansFuzzyClassifier`` runs but sits at chance (AUC 0.49-0.52
          where logistic regression on the same features gets 0.66-0.78). The feature
          matrix is 93% zeros, and fitting a Gaussian membership function per
          ``(feature, class)`` to a near-binary variable yields near-identical
          antecedents for both classes; the t-norm product then collapses to 0.5.

        The inputs here are *already* membership degrees, so there is no membership
        function to fit. ``rules.py`` uses the input values directly as firing
        strengths, which recovers the signal and reads better besides.
        """
        from .rules import MembershipRuleRegressor

        X, Y = self._windows(corpus, max_windows, self.random_state)
        all_names = self._output_names()

        # Predict only the most active output dimensions. The rest are filled from
        # the corpus prior at predict time: fitting a classifier for a supersense
        # that fires 3 times in the corpus buys noise, and each output costs a fit.
        activity = Y.sum(axis=0)
        chosen = np.argsort(activity)[::-1][:self.n_outputs]
        chosen = np.array([i for i in chosen if activity[i] > 0])
        self.output_index_ = list(chosen)
        self.prior_ = Y.mean(axis=0)

        self._out_names_ = None
        n_test = max(int(len(X) * test_frac), 1)
        Xtr, Xte = X[:-n_test], X[-n_test:]
        Ytr, Yte = Y[:-n_test, chosen], Y[-n_test:, chosen]

        cols = self.feature_names()
        out_names = [all_names[i] for i in chosen]
        if verbose:
            print(f"  windows={len(X)} features={X.shape[1]} "
                  f"outputs={len(chosen)}/{len(all_names)}")
            print(f"  predicting: {', '.join(out_names)}")

        self.models_ = {}
        kept: list[int] = []
        for j, name in enumerate(out_names):
            label = (Ytr[:, j] >= self.membership_threshold).astype(float)
            if len(np.unique(label)) < 2:
                if verbose:
                    print(f"    skipping '{name}': single class at threshold "
                          f"{self.membership_threshold}")
                continue
            self.models_[name] = MembershipRuleRegressor(
                max_rules=self.max_rules, max_order=self.max_order,
            ).fit(Xtr, label, cols)
            kept.append(j)
        if not self.models_:
            raise RuntimeError("no output dimension had both classes; lower "
                               "membership_threshold or raise max_windows")

        self.output_index_ = [self.output_index_[j] for j in kept]
        out_names = [out_names[j] for j in kept]
        Ytr, Yte = Ytr[:, kept], Yte[:, kept]

        tr = self._proba(Xtr, out_names)
        te = self._proba(Xte, out_names)
        self._out_names_ = out_names
        # Baseline: always predict the training mean of each output. A sequence model
        # that cannot beat this has learned nothing about context.
        base = np.tile(Ytr.mean(axis=0), (len(Yte), 1))
        # Evaluate on the task the classifiers actually optimise -- the binarised
        # "does supersense S come next?" -- as well as on MAE against the continuous
        # membership. The two disagree sharply and both belong in the report: a
        # calibrated probability centred near 0.5 is *guaranteed* to lose on MAE to a
        # mean baseline when the targets are sparse, so MAE alone would understate
        # the model, while separation alone would hide that predict_proba is not a
        # drop-in membership degree.
        lab = (Yte >= self.membership_threshold).astype(int)
        pos, neg = te[lab == 1], te[lab == 0]
        sep = float(pos.mean() - neg.mean()) if pos.size and neg.size else 0.0
        tpr = float((pos >= 0.5).mean()) if pos.size else 0.0
        tnr = float((neg < 0.5).mean()) if neg.size else 0.0
        self.info_ = SequenceInfo(
            n_windows=len(X), n_features=X.shape[1], n_outputs=len(kept),
            output_names=out_names,
            train_mae=float(np.mean(np.abs(tr - Ytr))),
            test_mae=float(np.mean(np.abs(te - Yte))),
            baseline_mae=float(np.mean(np.abs(base - Yte))),
            separation=sep, balanced_accuracy=0.5 * (tpr + tnr),
        )
        if verbose:
            i = self.info_
            print(f"  binarised task : separation={i.separation:+.3f} "
                  f"balanced-acc={i.balanced_accuracy:.3f} (chance=0.500)")
            print(f"  continuous MAE : test={i.test_mae:.4f} "
                  f"mean-baseline={i.baseline_mae:.4f} "
                  f"-> {'beats' if i.test_mae < i.baseline_mae else 'LOSES TO'} baseline")
            if i.test_mae >= i.baseline_mae:
                print("    NOTE: predict_proba is calibrated near 0.5 while membership "
                      "targets are sparse,\n          so it cannot win on MAE. Judge "
                      "next-token skill by separation/balanced-acc;\n          treat "
                      "the proba as a ranking signal, not a calibrated degree.")
        return self

    def predict_next(self, context: list[str]) -> np.ndarray:
        """Full level-``L`` membership vector for the next token."""
        if not self.models_:
            raise RuntimeError("call fit() first")
        ctx = list(context)[-self.window:]
        while len(ctx) < self.window:
            ctx.insert(0, "")
        feats = np.concatenate([self._token_vector(t) for t in ctx])
        pred = self._proba(feats[None, :], self._out_names_)[0]

        out = self.prior_.copy()
        for idx, value in zip(self.output_index_, pred):
            out[idx] = float(np.clip(value, 0.0, 1.0))
        return out

    def explain_next(self, context: list[str], k: int = 8) -> str:
        names = self._output_names()
        pred = self.predict_next(context)
        modelled = set(self.output_index_)
        order = np.argsort(pred)[::-1][:k]
        lines = [f"context: {' '.join(context)!r}",
                 f"  predicted next-token dimensions (semantic L{self.level} + syntax):"]
        for i in order:
            marker = "*" if i in modelled else " "
            lines.append(f"   {marker} {names[i]:<26} {pred[i]:.3f}")
        lines.append("   (* = modelled dimension; others are the corpus prior)")
        return "\n".join(lines)
