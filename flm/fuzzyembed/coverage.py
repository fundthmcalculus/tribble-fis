"""Milestone 0 -- the coverage go/no-go gate.

The plan makes this the first thing to run and a hard gate, because a hierarchy
that cannot name the corpus's words cannot embed them: uncovered tokens contribute
nothing to any dimension, so coverage is an upper bound on how much of the text
the representation can see at all.

Gate (from ``../FUZZY_EMBEDDING_PLAN.md`` M0), applied to *content* tokens:

    >= 85%   proceed
    60-85%   proceed only with a graft built first
    <  60%   the scaffold is wrong; stop and reconsider

Report content-token coverage and all-token coverage separately. Function words
("the", "of", "and") are absent from WordNet by design and are not a scaffold
failure -- rolling them into one headline number would understate coverage badly,
since they are the most frequent tokens in any corpus.

    uv run python -m flm.fuzzyembed.coverage --corpus tiny brown
"""

from __future__ import annotations

import argparse

# Closed-class words WordNet does not (and should not) cover. Excluded from the
# content-token denominator.
FUNCTION_WORDS = frozenset("""
a an the this that these those which who whom whose what
i you he she it we they me him her us them my your his its our their mine yours
am is are was were be been being do does did done have has had having
will would shall should can could may might must ought
and or but nor for yet so if then than because as while when where why how
of in on at to from by with without within into onto upon about above below
under over between among through during before after since until against
not no nor none nothing very too also just only even still yet already
there here where anywhere everywhere somewhere
myself yourself himself herself itself ourselves yourselves themselves
something anything everything nothing someone anyone everyone
somebody anybody everybody nobody else other others another
s t d ll re ve m o
""".split())


def _covered(lemma: str, wn) -> bool:
    from .hierarchy import wordnet_synsets
    return bool(wordnet_synsets(lemma, wn))


def measure(corpus, wn=None) -> dict:
    """Coverage of ``corpus`` by the WordNet lexicon, by token and by type."""
    if wn is None:
        from nltk.corpus import wordnet as wn  # noqa: PLC0415

    hit_cache: dict[str, bool] = {}
    tok_all = tok_all_hit = 0
    tok_content = tok_content_hit = 0
    type_content = type_content_hit = 0
    misses: list[tuple[str, int]] = []

    for word in corpus.vocabulary:
        n = corpus.counts[word]
        hit = hit_cache.setdefault(word, _covered(word, wn))
        is_content = word not in FUNCTION_WORDS and len(word) > 1

        tok_all += n
        tok_all_hit += n if hit else 0
        if is_content:
            tok_content += n
            type_content += 1
            if hit:
                tok_content_hit += n
                type_content_hit += 1
            else:
                misses.append((word, n))

    misses.sort(key=lambda kv: -kv[1])
    return {
        "corpus": corpus.name,
        "token_coverage_all": tok_all_hit / max(tok_all, 1),
        "token_coverage_content": tok_content_hit / max(tok_content, 1),
        "type_coverage_content": type_content_hit / max(type_content, 1),
        "n_tokens": tok_all,
        "n_types": len(corpus.vocabulary),
        "top_misses": misses[:30],
    }


def verdict(content_token_coverage: float) -> str:
    if content_token_coverage >= 0.85:
        return "PASS - proceed"
    if content_token_coverage >= 0.60:
        return "MARGINAL - proceed only with a graft"
    return "FAIL - scaffold is wrong, reconsider"


def render(report: dict) -> str:
    cov = report["token_coverage_content"]
    lines = [
        f"=== M0 coverage: {report['corpus']} ===",
        f"  tokens={report['n_tokens']:,}  types={report['n_types']:,}",
        f"  content-token coverage : {cov:.1%}   <-- the gate",
        f"  all-token coverage     : {report['token_coverage_all']:.1%}",
        f"  content-type coverage  : {report['type_coverage_content']:.1%}",
        f"  VERDICT: {verdict(cov)}",
        "  top uncovered content words:",
    ]
    row = []
    for word, n in report["top_misses"][:24]:
        row.append(f"{word}({n})")
        if len(row) == 6:
            lines.append("    " + "  ".join(row))
            row = []
    if row:
        lines.append("    " + "  ".join(row))
    return "\n".join(lines)


def main() -> None:
    from .corpus import load_corpus

    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", nargs="+", default=["tiny", "brown"])
    p.add_argument("--max-types", type=int, default=None)
    args = p.parse_args()

    for spec in args.corpus:
        corpus = load_corpus(spec, max_types=args.max_types)
        print(corpus.summary())
        print(render(measure(corpus)))
        print()


if __name__ == "__main__":
    main()
