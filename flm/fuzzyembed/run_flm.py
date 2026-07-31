"""End-to-end driver: build the fuzzy embedding, sequence model, and decoder.

    uv run --with nltk python -m flm.fuzzyembed.run_flm --stage all

Stages
------
``coverage``  M0 go/no-go -- does the hierarchy name the corpus's words?
``embed``     build and demonstrate the fuzzy embedding (incl. typo robustness)
``sequence``  fit the MIMO TSK next-token model over supersenses
``generate``  decode forward -- Zadeh linguistic approximation
``all``       all of the above

First run needs the nltk data::

    uv run --with nltk python -c "import nltk; [nltk.download(p) for p in \
        ['wordnet','gutenberg','brown','opinion_lexicon']]"
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def stage_coverage(args) -> None:
    from .corpus import load_corpus
    from .coverage import measure, render

    print("=" * 72)
    print("M0 -- COVERAGE GATE")
    print("=" * 72)
    for spec in ("tiny", "brown"):
        corpus = load_corpus(spec)
        print(corpus.summary())
        print(render(measure(corpus)))
        print()


def stage_embed(args):
    from .corpus import load_corpus
    from .embedder import build_embedder
    from .similarity import (DISCRIMINATIVE_LEVEL, fuzzy_jaccard,
                             hierarchy_jaccard)

    print("=" * 72)
    print("EXPERIMENT A -- FUZZY EMBEDDING")
    print("=" * 72)
    corpus = load_corpus(args.corpus)
    print(corpus.summary())
    emb, info = build_embedder(corpus, n_levels=args.levels,
                               max_types=args.max_types)
    print()

    print("-- resolution ladder --")
    for level, width in enumerate(info["widths"]):
        note = "  <- FIS workhorse width" if level == 2 else ""
        print(f"  L{level}: {width:>5} named dimensions{note}")
    print()

    print("-- explanations --")
    for text in ("the little rabbit ran to the green house",
                 "the rabit was not very happy"):
        print(emb.explain(text))
        print()

    print("-- typo robustness (fuzzy Jaccard, clean vs perturbed) --")
    h, L = emb.h, emb.h.n_levels - 1
    pairs = [
        ("the rabbit ran home", "the rabit ran home"),
        ("the rabbit ran home", "the rabbbit ran hoem"),
        ("a happy child laughed", "a happy chidl laughed"),
    ]
    for clean, noisy in pairs:
        a, b = emb.embed(clean, L), emb.embed(noisy, L)
        print(f"  {clean!r:30s} vs {noisy!r:30s} sim={fuzzy_jaccard(a, b):.3f}")
    print()

    print(f"-- semantic similarity at L{DISCRIMINATIVE_LEVEL} "
          "(the measured discriminative level) --")
    cases = [("a happy child laughed", "a joyful boy giggled", "SIM"),
             ("the dog barked", "the wolf howled", "SIM"),
             ("the girl ate bread", "the boy ate food", "SIM"),
             ("the dog barked", "the king spoke", "DIF"),
             ("a happy child laughed", "the stone was cold", "DIF"),
             ("the girl ate bread", "the mountain was tall", "DIF")]
    sims, difs = [], []
    for x, y, tag in cases:
        a2, b2 = emb.embed(x, DISCRIMINATIVE_LEVEL), emb.embed(y, DISCRIMINATIVE_LEVEL)
        af, bf = emb.embed(x, L), emb.embed(y, L)
        j2 = fuzzy_jaccard(a2, b2)
        hj = hierarchy_jaccard(af, bf, h, L)
        (sims if tag == "SIM" else difs).append(j2)
        print(f"  {tag} {x!r:26s} vs {y!r:26s} L2={j2:.3f} hier={hj:.3f}")
    print(f"  mean SIM={np.mean(sims):.3f}  mean DIF={np.mean(difs):.3f}  "
          f"gap={np.mean(sims) - np.mean(difs):+.3f}")
    print()

    print("-- rollup exactness (the Matryoshka contrast) --")
    levels, _ = emb.embed_levels("the little rabbit ran to the green house")
    ok = True
    for lo in range(L):
        for mid in range(lo + 1, L + 1):
            if not np.allclose(levels[lo], h.rollup(levels[mid], mid, lo), atol=1e-6):
                ok = False
                print(f"  MISMATCH L{lo} != rollup(L{mid})")
    print(f"  every coarse level is an exact t-conorm rollup of every finer one: "
          f"{'PASS' if ok else 'FAIL'}")
    print()
    return emb, corpus


def stage_sequence(args, emb, corpus):
    from .sequence import FuzzySequenceModel

    print("=" * 72)
    print("FUZZY SEQUENCE MODEL")
    print("=" * 72)
    model = FuzzySequenceModel(emb, level=args.seq_level, window=args.window,
                               n_outputs=args.n_outputs, top_n=args.seq_top_n)
    t0 = time.perf_counter()
    model.fit(corpus, max_windows=args.max_windows)
    print(f"  fit in {time.perf_counter() - t0:.1f}s")
    print()
    for ctx in (["the", "little", "rabbit"], ["the", "dog"], ["she", "was", "very"]):
        print(model.explain_next(ctx))
        print()
    return model


def stage_generate(args, emb, seq_model):
    from .decode import FuzzyDecoder, LexemeAtlas, generate, render_generation

    print("=" * 72)
    print("FUZZY DECODER -- linguistic approximation")
    print("=" * 72)
    atlas = LexemeAtlas(emb, seq_model.level, verbose=True)
    print()
    for hedge in (args.hedge, 6.0):
        decoder = FuzzyDecoder(atlas, hedge=hedge, top_k=args.top_k, seed=args.seed)
        print(f"-- hedge exponent {hedge:g} "
              f"({'concentrating (decisive)' if hedge > 1 else 'dilating (diverse)'}) --")
        for prompt in (["the", "little", "rabbit"], ["the", "old", "woman"]):
            _, steps = generate(seq_model, decoder, prompt, n_tokens=args.n_tokens)
            print(render_generation(prompt, steps))
            print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all",
                   choices=["coverage", "embed", "sequence", "generate", "all"])
    p.add_argument("--corpus", default="tiny", help="tiny | brown | path to text/jsonl")
    p.add_argument("--max-types", type=int, default=3000)
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--seq-level", type=int, default=2)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--n-outputs", type=int, default=10)
    p.add_argument("--seq-top-n", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=3000)
    p.add_argument("--n-tokens", type=int, default=6)
    p.add_argument("--hedge", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.stage == "coverage":
        stage_coverage(args)
        return
    if args.stage == "all":
        stage_coverage(args)

    emb, corpus = stage_embed(args)
    if args.stage == "embed":
        return

    seq = stage_sequence(args, emb, corpus)
    if args.stage == "sequence":
        return

    stage_generate(args, emb, seq)


if __name__ == "__main__":
    main()
