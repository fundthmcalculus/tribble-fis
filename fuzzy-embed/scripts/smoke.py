"""End-to-end pipeline smoke test on a tiny data slice.

Exercises: dataset loading, token-frequency/SIF init, KMeans centre init, the ST
trainer wiring, the two-group optimiser, the UR loss, saving, reloading, and
encoding. Fast enough to run before any real experiment.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import Tokenizer

from fuzzyembed.data import load_mix, mix_summary, most_frequent_ids, token_frequencies
from fuzzyembed.model import FuzzyEmbedding
from fuzzyembed.train import FESConfig, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tok = Tokenizer.from_pretrained("google-bert/bert-base-uncased")

    # Tiny slice: 0.2% of each cap, and only the small sources so nothing large
    # has to download for a smoke test.
    from fuzzyembed.data import MIX

    small = tuple(s for s in MIX if s.cap <= 120_000)
    mix = load_mix(small, scale=0.05)
    print(mix_summary(mix))

    counts = token_frequencies(
        mix, tok, tok.get_vocab_size(), max_texts=20_000,
        cache_path=ROOT / "artifacts" / "smoke_token_counts.npy",
    )
    top_ids = most_frequent_ids(counts, top_k=4_000)

    cfg = FESConfig(
        name="smoke", d_in=32, d_out=64, n_rules=8,
        batch_size=256, epochs=1.0, matryoshka_dims=(64, 32),
        max_seq_length=64,
    )
    out = str(ROOT / "artifacts" / "smoke-model")
    model, fuzzy, info = train(
        cfg, mix, tok, out, token_counts=counts, kmeans_ids=top_ids, steps_per_log=20
    )
    print("\n--- train info ---")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Reload from disk and confirm the saved model reproduces embeddings.
    from sentence_transformers import SentenceTransformer

    reloaded = SentenceTransformer(out)
    sents = ["a fuzzy inference system", "a neural network", "the cat sat on the mat"]
    e1 = model.encode(sents, convert_to_tensor=True, device="cpu")
    e2 = reloaded.encode(sents, convert_to_tensor=True, device="cpu")
    delta = float((e1 - e2).abs().max())
    print(f"\nreload max|delta| = {delta:.2e}")
    assert delta < 1e-4, "saved model does not reproduce embeddings"

    sim = model.similarity(e1, e1)
    print("similarity matrix:\n", torch.round(sim * 1000) / 1000)
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
