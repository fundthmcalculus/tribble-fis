"""Dataset loading for Experiment B.

Three targets per corpus, because the fuzzy framing needs the third one:

``sst2``        binary label
``sst5``        5-class label
``sst_cont``    the *continuous* [0, 1] sentiment score

The continuous target is the interesting one. SST was annotated with a slider
admitting up to 25 levels, and SST-2/SST-5 are discretizations of those graded
values -- collapsing to buckets throws away the single property of the dataset
that suits a fuzzy system. ``MixtureOfGaussiansFuzzyRegressor`` consumes the
graded score directly. See ``../FIS_ON_EMBEDDINGS_PLAN.md`` section 3.

Requires network on first call (Hugging Face ``datasets``). ``synthetic_dataset``
needs no network and exists so the harness plumbing can be exercised offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DATASETS = ("sst2", "sst5", "sst_cont", "imdb")


@dataclass
class TextDataset:
    """Texts plus a target, with the task kind attached so heads can dispatch."""

    name: str
    train_texts: list[str]
    train_y: np.ndarray
    test_texts: list[str]
    test_y: np.ndarray
    task: str  # "classification" | "regression"

    def __post_init__(self):
        if len(self.train_texts) != len(self.train_y):
            raise ValueError("train_texts and train_y length mismatch")
        if len(self.test_texts) != len(self.test_y):
            raise ValueError("test_texts and test_y length mismatch")

    def summary(self) -> str:
        if self.task == "classification":
            tgt = f"{len(np.unique(self.train_y))} classes"
        else:
            tgt = f"continuous [{self.train_y.min():.2f}, {self.train_y.max():.2f}]"
        return (
            f"{self.name}: train={len(self.train_texts)} test={len(self.test_texts)} "
            f"task={self.task} target={tgt}"
        )


def load_dataset(name: str, max_train: int | None = None, max_test: int | None = None,
                 seed: int = 42) -> TextDataset:
    """Load one of ``DATASETS``. Requires ``datasets`` and network on first call."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pip install datasets  (or run run_sentiment.py --synthetic)"
        ) from exc

    if name == "imdb":
        ds = hf_load("imdb")
        train, test = ds["train"], ds["test"]
        out = TextDataset(
            name, list(train["text"]), np.asarray(train["label"]),
            list(test["text"]), np.asarray(test["label"]), "classification",
        )
    elif name == "sst2":
        # "stanfordnlp/sst2" ships no public test labels, so the official
        # validation split is used as the test set -- the standard convention.
        ds = hf_load("stanfordnlp/sst2")
        train, test = ds["train"], ds["validation"]
        out = TextDataset(
            name, list(train["sentence"]), np.asarray(train["label"]),
            list(test["sentence"]), np.asarray(test["label"]), "classification",
        )
    else:
        # "SetFit/sst5" carries both the 5-class label and the graded score.
        ds = hf_load("SetFit/sst5")
        train, test = ds["train"], ds["test"]
        if name == "sst5":
            out = TextDataset(
                name, list(train["text"]), np.asarray(train["label"]),
                list(test["text"]), np.asarray(test["label"]), "classification",
            )
        else:
            # Recover a graded target. If the source exposes a continuous field
            # use it; otherwise fall back to the 5-class label mapped onto
            # [0, 1], which is a coarser but still ordered stand-in.
            def _cont(split):
                for key in ("label_score", "score", "sentiment"):
                    if key in split.column_names:
                        return np.asarray(split[key], dtype=float)
                return np.asarray(split["label"], dtype=float) / 4.0
            out = TextDataset(
                name, list(train["text"]), _cont(train),
                list(test["text"]), _cont(test), "regression",
            )

    return _subsample(out, max_train, max_test, seed)


def _subsample(ds: TextDataset, max_train, max_test, seed) -> TextDataset:
    rng = np.random.default_rng(seed)

    def take(texts, y, n):
        if n is None or n >= len(texts):
            return texts, y
        idx = rng.choice(len(texts), size=n, replace=False)
        return [texts[i] for i in idx], y[idx]

    tr_t, tr_y = take(ds.train_texts, ds.train_y, max_train)
    te_t, te_y = take(ds.test_texts, ds.test_y, max_test)
    return TextDataset(ds.name, tr_t, tr_y, te_t, te_y, ds.task)


def synthetic_dataset(n_train=800, n_test=200, dim=64, n_classes=3, seed=42,
                      task="classification") -> tuple[TextDataset, dict]:
    """An offline stand-in with a *planted* signal, for plumbing checks only.

    Returns ``(dataset, embeddings)`` where ``embeddings`` maps split name to an
    array, so the harness can skip the encoder entirely. Only a handful of
    dimensions carry signal, which also exercises the feature-selection path
    (``take_top_features``) that real embeddings will hit.

    This validates the harness. It says nothing about whether a FIS head works
    on real embeddings -- planted axis-aligned Gaussian signal is precisely the
    structure a Gaussian-membership FIS is built to find.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    n_signal = min(6, dim)

    X = rng.normal(0.0, 1.0, size=(n, dim))
    if task == "classification":
        y = rng.integers(0, n_classes, size=n)
        centers = rng.normal(0.0, 2.5, size=(n_classes, n_signal))
        X[:, :n_signal] += centers[y]
        y_out = y
    else:
        w = rng.normal(0.0, 1.0, size=n_signal)
        raw = X[:, :n_signal] @ w + rng.normal(0.0, 0.3, size=n)
        y_out = (raw - raw.min()) / (raw.max() - raw.min())

    texts = [f"synthetic document {i}" for i in range(n)]
    ds = TextDataset(
        f"synthetic_{task}", texts[:n_train], y_out[:n_train],
        texts[n_train:], y_out[n_train:],
        "classification" if task == "classification" else "regression",
    )
    return ds, {"train": X[:n_train], "test": X[n_train:]}
