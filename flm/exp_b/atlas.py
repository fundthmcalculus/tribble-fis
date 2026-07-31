"""Post-hoc naming of FIS-selected embedding dimensions -- the bridge to Experiment A.

A rule like ``IF dim_417 is High AND dim_22 is Low THEN positive`` is transparent
but not interpretable: ``dim_417`` has no meaning available to a human. This module
attaches a provisional name by finding the training documents that most strongly
activate a dimension (at each pole) and extracting their distinctive n-grams
against the corpus background.

This is the sparse-autoencoder auto-interp move, and it inherits every one of
that method's known weaknesses -- explanations that come out too broad,
polysemantic dimensions that resist a single label, and no way to falsify a bad
name without a separate experiment. Running it here is worthwhile precisely
because feeling those weaknesses first-hand is the argument for a hierarchy whose
axes are named *a priori*. See ``../FIS_ON_EMBEDDINGS_PLAN.md`` section 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DimensionProfile:
    dim: int
    high_terms: list[str]
    low_terms: list[str]
    high_examples: list[str]
    low_examples: list[str]

    def label(self, k: int = 4) -> str:
        hi = ", ".join(self.high_terms[:k]) or "?"
        lo = ", ".join(self.low_terms[:k]) or "?"
        return f"dim_{self.dim}[high: {hi} | low: {lo}]"


def _distinctive_terms(texts: list[str], background: list[str], top_k: int = 8,
                       ngram_range=(1, 2)) -> list[str]:
    """Terms over-represented in ``texts`` relative to ``background``.

    Log-odds with add-one smoothing rather than raw tf-idf: tf-idf on a 20-document
    subset mostly surfaces whatever is rare in the corpus, which is not the same
    question as what is *characteristic* of this subset.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    if not texts:
        return []
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english",
                          min_df=1, max_features=20000)
    try:
        vec.fit(background)
        fg = np.asarray(vec.transform(texts).sum(axis=0)).ravel()
        bg = np.asarray(vec.transform(background).sum(axis=0)).ravel()
    except ValueError:
        return []

    fg_rate = (fg + 1) / (fg.sum() + len(fg))
    bg_rate = (bg + 1) / (bg.sum() + len(bg))
    log_odds = np.log(fg_rate / bg_rate)
    # Require the term to actually occur in the foreground; otherwise a term with
    # zero foreground count can still score via the smoothing prior.
    log_odds[fg == 0] = -np.inf

    names = np.asarray(vec.get_feature_names_out())
    order = np.argsort(log_odds)[::-1][:top_k]
    return [str(names[i]) for i in order if np.isfinite(log_odds[i])]


def profile_dimension(dim: int, X: np.ndarray, texts: list[str],
                      n_examples: int = 20, top_k: int = 8) -> DimensionProfile:
    """Profile one dimension by its extreme-activating documents at both poles."""
    col = np.asarray(X)[:, dim]
    n = min(n_examples, max(1, len(col) // 4))
    hi_idx = np.argsort(col)[::-1][:n]
    lo_idx = np.argsort(col)[:n]

    hi_texts = [texts[i] for i in hi_idx]
    lo_texts = [texts[i] for i in lo_idx]
    return DimensionProfile(
        dim=dim,
        high_terms=_distinctive_terms(hi_texts, texts, top_k),
        low_terms=_distinctive_terms(lo_texts, texts, top_k),
        high_examples=hi_texts[:3],
        low_examples=lo_texts[:3],
    )


def build_atlas(feature_names: list[str], X: np.ndarray, texts: list[str],
                **kwargs) -> dict[str, DimensionProfile]:
    """Profile every dimension a fitted head selected.

    ``feature_names`` are the ``dim_<i>`` strings from ``top_features_`` /
    ``gate_features_``, so only the handful of dimensions a rule base actually
    mentions get profiled -- which is the whole point of the feature selection.
    """
    out = {}
    for name in feature_names:
        if not name.startswith("dim_"):
            continue
        try:
            dim = int(name.split("_", 1)[1])
        except ValueError:
            continue
        if dim < np.asarray(X).shape[1]:
            out[name] = profile_dimension(dim, X, texts, **kwargs)
    return out


def render_atlas(atlas: dict[str, DimensionProfile]) -> str:
    if not atlas:
        return "(no dimensions profiled)"
    lines = ["Dimension atlas -- provisional, post-hoc names", "=" * 60]
    for name, prof in sorted(atlas.items(), key=lambda kv: kv[1].dim):
        lines.append(f"\n{name}")
        lines.append(f"  high: {', '.join(prof.high_terms) or '?'}")
        lines.append(f"  low : {', '.join(prof.low_terms) or '?'}")
        for ex in prof.high_examples[:2]:
            lines.append(f"    (+) {ex[:110]}")
        for ex in prof.low_examples[:2]:
            lines.append(f"    (-) {ex[:110]}")
    lines.append(
        "\nNOTE: these names are post-hoc guesses with no guarantee of "
        "monosemanticity.\nA dimension whose two poles are both incoherent is "
        "evidence for Experiment A,\nnot a bug in this script."
    )
    return "\n".join(lines)
