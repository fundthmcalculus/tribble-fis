# E22 measurement scripts

The four scripts behind [`../LOG.md`](../LOG.md) E22. Kept in the repo because the numbers in
the log are only checkable if the thing that produced them is readable — several earlier
results in this project turned out to be measurement bugs rather than findings (E17's
evaluation leak, E20.2's mis-weighted trigram), and both were found by re-reading the
harness rather than the model.

| script | question | LOG |
|---|---|---|
| `ab.py` | Does the linguistic parameter space beat the WordNet space at matched budget? | E22.1 |
| `mix.py` | Is it *complementary* to anything — bigram, WordNet space, or their mixture? | E22.2 |
| `size.py` | How small can the rule base get before perplexity degrades? | E22.3 |
| `small.py` | Does the smallest rule base still add to a bigram? | E22.4 |

Run from the repository root, with `nltk` available:

```
uv run --with nltk python flm/experiments/ab.py
```

Each script fixes its own budget at the top (`NPOS`, `TRAINPOS`, `K`, `W`). Absolute
perplexities move with `NPOS` because it selects the evaluation positions, so **only compare
rows produced by the same script in the same run** — the orderings replicate across budgets,
the absolute values do not. `mix.py` projects every model's distribution onto the shared
decodable vocabulary before mixing; that is load-bearing, not tidying (E22.2).
