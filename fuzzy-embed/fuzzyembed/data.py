"""Training data mix and corpus token statistics.

All datasets are public and drawn from the ``sentence-transformers`` Hub
collection, i.e. the same pool used by the from-scratch static-embedding recipe
we are following (Aarsen & Nussbaum, HF blog "Train 400x faster Static Embedding
Models"). Availability and row counts were verified on 2026-07-31 and are
recorded here so a mismatch is visible rather than silent.

Each entry is (repo, config, columns, cap). Columns are renamed to
``anchor``/``positive``[/``negative``] because MultipleNegativesRankingLoss keys
off column *order*, not name, and having them consistent makes the mix debuggable.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    repo: str
    config: str | None
    columns: tuple[str, ...]
    cap: int
    note: str
    stream: bool = False
    """Pull via the streaming API and stop after ``cap`` rows.

    ``load_dataset`` downloads *every* shard before you can subset, which is
    unacceptable for the large repos -- s2orc's title-abstract config has 41.7M
    rows. Streaming reads only the leading shards. The trade-off is that a
    streamed subset is the *head* of the file rather than a random sample, so only
    use it where the source has no meaningful ordering.
    """

    @property
    def key(self) -> str:
        base = self.repo.split("/")[-1]
        return f"{base}-{self.config}" if self.config else base


# The mix. Deliberately spans retrieval (asymmetric query->doc), semantic
# similarity (symmetric), and topical pairs, because the task-type profile of
# static models is uneven -- strong on classification/STS, weak on retrieval and
# clustering (see docs/01-small-embedding-models.md §2).
MIX: tuple[Source, ...] = (
    Source("sentence-transformers/gooaq", None, ("question", "answer"), 1_500_000,
           "web QA; the single largest useful retrieval signal"),
    Source("sentence-transformers/s2orc", "title-abstract-pair", ("title", "abstract"),
           400_000, "scientific title->abstract; long-document retrieval", stream=True),
    Source("sentence-transformers/all-nli", "triplet", ("anchor", "positive", "negative"),
           557_850, "NLI triplets; the main STS/entailment signal, with hard negatives"),
    Source("sentence-transformers/msmarco-msmarco-distilbert-base-tas-b", "triplet",
           ("query", "positive", "negative"), 502_939,
           "web retrieval with mined hard negatives"),
    Source("sentence-transformers/agnews", None, ("title", "description"), 400_000,
           "news topical pairs; helps clustering/classification"),
    Source("sentence-transformers/stackexchange-duplicates", "post-post-pair",
           ("post1", "post2"), 300_000, "duplicate-question detection"),
    Source("sentence-transformers/altlex", None, ("text", "simplified"), 112_696,
           "paraphrase; wiki -> simple wiki"),
    Source("sentence-transformers/simple-wiki", None, ("text", "simplified"), 102_225,
           "paraphrase"),
    Source("sentence-transformers/quora-duplicates", "triplet",
           ("anchor", "positive", "negative"), 101_762, "duplicate questions, hard negatives"),
    Source("sentence-transformers/natural-questions", None, ("query", "answer"), 100_231,
           "real search queries"),
    Source("sentence-transformers/squad", None, ("question", "answer"), 87_599,
           "reading-comprehension QA"),
)

_CANON = ("anchor", "positive", "negative")


def load_mix(
    sources: tuple[Source, ...] = MIX,
    scale: float = 1.0,
    seed: int = 42,
    cache_dir: str | None = None,
) -> dict[str, Dataset]:
    """Load and normalise the training mix.

    Args:
        sources: which sources to pull.
        scale: fraction of each cap to keep. Use a small value (0.01) for smoke
            tests so the pipeline is exercised without a 5M-pair download.
        seed: shuffle seed applied before truncation, so a capped subset is a
            random sample rather than the head of the file (which for several of
            these repos is sorted and would bias the mix).

    Returns:
        ``{name: Dataset}`` with columns renamed to anchor/positive[/negative].
    """
    out: dict[str, Dataset] = {}
    for src in sources:
        n = max(1, int(src.cap * scale))
        try:
            if src.stream:
                stream = load_dataset(
                    src.repo, src.config, split="train", streaming=True, cache_dir=cache_dir
                )
                stream = stream.select_columns(list(src.columns))
                rows = list(islice(iter(stream), n))
                ds = Dataset.from_list(rows)
            else:
                ds = load_dataset(src.repo, src.config, split="train", cache_dir=cache_dir)
                ds = ds.select_columns(list(src.columns))
        except Exception as exc:  # noqa: BLE001 - report and continue; a partial mix is usable
            logger.warning("skipping %s: %s", src.key, exc)
            continue
        ds = ds.rename_columns(dict(zip(src.columns, _CANON[: len(src.columns)])))
        ds = ds.shuffle(seed=seed)
        if len(ds) > n:
            ds = ds.select(range(n))
        out[src.key] = ds
        logger.info("loaded %-40s %9d rows  (%s)", src.key, len(ds), src.note)
    if not out:
        raise RuntimeError("no datasets loaded; check network access to the HF Hub")
    return out


def mix_summary(mix: dict[str, Dataset]) -> str:
    lines = [f"{'dataset':<45} {'rows':>10} {'cols':>28}"]
    total = 0
    for k, ds in mix.items():
        total += len(ds)
        lines.append(f"{k:<45} {len(ds):>10,} {str(ds.column_names):>28}")
    lines.append(f"{'TOTAL':<45} {total:>10,}")
    return "\n".join(lines)


def token_frequencies(
    mix: dict[str, Dataset],
    tokenizer,
    vocab_size: int,
    max_texts: int = 400_000,
    seed: int = 42,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Corpus token counts, for SIF initialisation of the pooling weights.

    POTION's re-regularisation step weights tokens by ``a / (a + p(t))``; we use
    the same statistic to *initialise* a learned weight, so training starts from
    the classical baseline rather than from uniform pooling.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        counts = np.load(cache_path)
        if counts.shape[0] == vocab_size:
            logger.info("token frequencies loaded from %s", cache_path)
            return counts
        logger.warning("cached counts have wrong vocab size; recomputing")

    rng = np.random.default_rng(seed)
    texts: list[str] = []
    per_ds = max(1, max_texts // max(len(mix), 1))
    for ds in mix.values():
        idx = rng.choice(len(ds), size=min(per_ds, len(ds)), replace=False)
        sub = ds.select(idx)
        for col in ("anchor", "positive"):
            if col in sub.column_names:
                texts.extend(sub[col])

    counter: Counter[int] = Counter()
    step = 2048
    for i in range(0, len(texts), step):
        for enc in tokenizer.encode_batch(texts[i : i + step], add_special_tokens=False):
            counter.update(enc.ids)

    counts = np.zeros(vocab_size, dtype=np.int64)
    for tid, c in counter.items():
        if 0 <= tid < vocab_size:
            counts[tid] = c
    logger.info(
        "token frequencies over %d texts: %d/%d vocab items seen",
        len(texts), int((counts > 0).sum()), vocab_size,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, counts)
    return counts


def most_frequent_ids(counts: np.ndarray, top_k: int = 20_000) -> np.ndarray:
    """The ``top_k`` most frequent token ids -- where KMeans should place rule
    centres, so the scatter partition covers the region the data occupies."""
    top_k = min(top_k, int((counts > 0).sum()))
    return np.argsort(-counts)[:top_k]
