"""If the content-selection rules exist, why is generation function-word soup? (E25.2)

E25.1 refuted the E24.3 diagnosis: the rule base already holds 386 open-class rules with high
lift (`IF prev1:DETERMINER AND cand:OPEN_NOUN`, lift +8.4), the budget was never scarce, and an
open-class quota changed nothing at all.

Competing explanation, tested here. A category-level rule says "a noun comes next" but cannot
say *which* noun, so its consequent is shared across every noun in the vocabulary. A function
word gets an identity-level rule *and* a large q(w) in the NCE inversion. So per-token, any one
function word outscores any one noun -- even where the model is right that a noun should follow.
Real text is 46% content words because thousands of individually-rare nouns have large
*aggregate* mass, and `top_k=20` out of ~2,900 candidates discards exactly that aggregate.

Two things to measure, which distinguish "the model cannot tell" from "decoding throws it away":
  1. content rate vs top_k -- if raising top_k fixes it, decoding was the culprit
  2. how much probability mass the model puts on content words in aggregate, versus how much
     survives truncation. If aggregate mass is roughly right, the ranking is fine and only the
     truncation is wrong.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.syntax import CLOSED_CLASS

TRAINPOS, W, NCAND, K = 20000, 2, 3000, 200
FUNCTION = set().union(*CLOSED_CLASS.values())
PROMPTS = [["the", "little"], ["she", "was"], ["he", "did"], ["it", "is"],
           ["there", "was"], ["the", "old"]]

c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                       lexeme_top_k=K, vocabulary=cand_vocab)
j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=2500, max_order=2,
                         beam=800, lexeme_side='ctx')
j.fit(train, cand_vocab, max_positions=TRAINPOS, verbose=False)
g = FuzzyGenerator(j, cand_vocab, counts=c.counts, seed=1)

is_content = np.array([w not in FUNCTION for w in g.words])
real = [t for s in test.sentences[:400] for t in s]
target = sum(1 for t in real if t not in FUNCTION) / len(real)
print(f"real held-out text: {100*target:.1f}% content words", flush=True)
print(f"candidates: {len(g.words)} ({is_content.sum()} content, "
      f"{(~is_content).sum()} function)\n", flush=True)

# --- 2. where does the model actually put its mass? -----------------------
print("--- aggregate content mass per step, and what truncation keeps ---", flush=True)
for prompt in PROMPTS[:3]:
    p = g.distribution(prompt)
    agg = float(p[is_content].sum())
    order = np.argsort(p)[::-1]
    line = [f"  {' '.join(prompt):<12} full={100*agg:>5.1f}%"]
    for k in (20, 100, 500):
        top = order[:k]
        kept = p[top]
        line.append(f"top{k}={100*float(kept[is_content[top]].sum()/kept.sum()):>5.1f}%")
    print("  ".join(line), flush=True)

# --- 1. content rate in free generation vs top_k --------------------------
print("\n--- content rate in free generation vs top_k ---", flush=True)
print(f"{'top_k':>7}{'content%':>10}   sample", flush=True)
for top_k in (20, 100, 500, len(g.words)):
    gen = FuzzyGenerator(j, cand_vocab, counts=c.counts, seed=1)
    toks, first = [], None
    for prompt in PROMPTS:
        out, _ = gen.generate(prompt, n_tokens=14, hedge=1.0, top_k=top_k)
        new = out[len(prompt):]
        toks.extend(new)
        if first is None:
            first = f"{' '.join(prompt)} | {' '.join(new)}"
    rate = sum(1 for t in toks if t not in FUNCTION) / len(toks)
    print(f"{top_k:>7}{100*rate:>9.1f}%   {first}", flush=True)

print("\n--- best setting, all prompts ---", flush=True)
gen = FuzzyGenerator(j, cand_vocab, counts=c.counts, seed=1)
for prompt in PROMPTS:
    out, _ = gen.generate(prompt, n_tokens=14, hedge=1.0, top_k=len(g.words))
    print(f"  {' '.join(prompt)} | {' '.join(out[len(prompt):])}", flush=True)
print("DONE", flush=True)
