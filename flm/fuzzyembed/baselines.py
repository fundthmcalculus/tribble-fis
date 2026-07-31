"""Language-model baselines, scored on exactly the same held-out positions.

On the GPT-2 comparison
-----------------------
It cannot be run in this environment: GPT-2's weights come from Hugging Face, which is
outside the egress allowlist. ``gpt2_perplexity`` implements it for a machine that can
reach HF, and takes the same restriction as everything else so the number is comparable.

It is also worth being clear about what that comparison would and would not show. GPT-2
was trained on ~40GB of text; the fuzzy model here sees ~90K tokens. GPT-2 will win on
fluency by a wide margin, and that margin measures the training-data gap, not the merit
of either method. **The controlled comparison at this scale is an n-gram LM trained on
the same corpus**, which is what this module is mainly for -- and n-grams are a genuinely
strong baseline on 90K tokens, not a straw man.

Comparability rules, applied to every model here
------------------------------------------------
1. Same train/test sentence split.
2. Same vocabulary, and the same restriction to positions whose gold token is
   *decodable* by the fuzzy model. Scoring a baseline on positions the fuzzy model
   cannot represent would flatter it or punish it arbitrarily.
3. Renormalise over the same candidate vocabulary, so all models answer the same
   question: "which of these words comes next?"
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .corpus import Corpus


class NgramLM:
    """Interpolated n-gram LM with add-k smoothing -- the fair baseline at this scale.

    Jelinek-Mercer interpolation (fixed weights over orders) rather than Kneser-Ney:
    KN's discounting is the better model, but interpolation is a handful of lines and
    the point here is a *credible* reference, not a competitive n-gram implementation.
    Stated so the baseline is not mistaken for state of the art.
    """

    def __init__(self, order: int = 3, add_k: float = 0.1,
                 weights: tuple[float, ...] | None = None):
        self.order = order
        self.add_k = add_k
        # Weight higher orders more, with mass kept on the unigram for backoff.
        self.weights = weights or tuple(
            w / sum(range(1, order + 1)) for w in range(1, order + 1))
        self.counts: list[dict] = [defaultdict(float) for _ in range(order)]
        self.totals: list[dict] = [defaultdict(float) for _ in range(order)]
        self.vocab: list[str] = []
        self.vocab_index: dict[str, int] = {}

    def fit(self, corpus: Corpus, vocab: list[str]):
        self.vocab = list(vocab)
        self.vocab_index = {w: i for i, w in enumerate(self.vocab)}
        known = set(self.vocab)
        for sent in corpus.sentences:
            toks = [t for t in sent if t in known]
            for i, tok in enumerate(toks):
                for o in range(self.order):
                    if i - o < 0:
                        break
                    ctx = tuple(toks[i - o:i])
                    self.counts[o][(ctx, tok)] += 1.0
                    self.totals[o][ctx] += 1.0
        return self

    def distribution(self, tokens: list[str], candidates: list[str]) -> np.ndarray:
        """Interpolated ``p(w | context)`` renormalised over ``candidates``."""
        probs = np.zeros(len(candidates))
        v = len(self.vocab)
        for o in range(self.order):
            ctx = tuple(tokens[len(tokens) - o:]) if o else ()
            total = self.totals[o].get(ctx, 0.0)
            denom = total + self.add_k * v
            if denom <= 0:
                continue
            w = self.weights[o]
            cnt = self.counts[o]
            for ci, cand in enumerate(candidates):
                probs[ci] += w * (cnt.get((ctx, cand), 0.0) + self.add_k) / denom
        s = probs.sum()
        return probs / s if s > 0 else np.full(len(candidates), 1.0 / len(candidates))


def ngram_perplexity(lm: NgramLM, corpus: Corpus, allowed: list[str],
                     window: int, max_positions: int = 2000,
                     seed: int = 7) -> dict:
    """Perplexity on the same positions and candidate set the fuzzy model uses."""
    allowed_set = set(allowed)
    rng = np.random.default_rng(seed)
    sents = [s for s in corpus.sentences if len(s) > window]
    order = rng.permutation(len(sents))
    index = {w: i for i, w in enumerate(allowed)}
    logs, n = [], 0
    for si in order:
        sent = sents[si]
        for i in range(window, len(sent)):
            gold = sent[i]
            if gold not in allowed_set:
                continue
            p = lm.distribution(sent[:i], allowed)
            logs.append(math.log(max(p[index[gold]], 1e-12)))
            n += 1
            if n >= max_positions:
                break
        if n >= max_positions:
            break
    nll = -float(np.mean(logs))
    return {"perplexity": float(np.exp(nll)), "nll": nll, "n": n,
            "vocab": len(allowed)}


def uniform_perplexity(allowed: list[str]) -> dict:
    """The floor: every candidate equally likely. Perplexity = |V|."""
    n = len(allowed)
    return {"perplexity": float(n), "nll": math.log(n), "n": 0, "vocab": n}


def unigram_perplexity(corpus: Corpus, allowed: list[str], test: Corpus,
                       window: int, max_positions: int = 2000,
                       seed: int = 7) -> dict:
    """Frequency-only baseline -- context-free, so it isolates what context buys."""
    lm = NgramLM(order=1).fit(corpus, allowed)
    return ngram_perplexity(lm, test, allowed, window, max_positions, seed)


def gpt2_perplexity(test: Corpus, allowed: list[str], window: int,
                    max_positions: int = 500, model_name: str = "gpt2",
                    seed: int = 7) -> dict:
    """GPT-2 on the same restricted next-word choice. Requires Hugging Face access.

    Renormalises GPT-2's next-token distribution over the *same* candidate vocabulary,
    so it answers the same question as the other models rather than a
    50257-way one. Each candidate is scored by the logprob of its first BPE token,
    which is an approximation -- exact scoring would sum over all tokenisations of each
    word -- and it is the standard cheap choice for restricted-candidate evaluation.
    Noted rather than hidden, because it slightly favours GPT-2 on words whose first
    BPE token is unambiguous.
    """
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as exc:  # pragma: no cover - optional, needs network
        raise ImportError(
            "pip install transformers torch; needs Hugging Face access "
            "(blocked in the authoring environment)"
        ) from exc

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()

    # First BPE id per candidate, with a leading space (GPT-2 is space-prefixed).
    first_ids = np.array([tok.encode(" " + w)[0] for w in allowed])
    index = {w: i for i, w in enumerate(allowed)}
    allowed_set = set(allowed)

    rng = np.random.default_rng(seed)
    sents = [s for s in test.sentences if len(s) > window]
    order = rng.permutation(len(sents))
    logs, n = [], 0
    with torch.no_grad():
        for si in order:
            sent = sents[si]
            for i in range(window, len(sent)):
                gold = sent[i]
                if gold not in allowed_set:
                    continue
                ids = tok.encode(" ".join(sent[:i]), return_tensors="pt")
                logits = model(ids).logits[0, -1]
                sub = logits[first_ids]
                p = torch.softmax(sub, dim=-1).numpy()
                logs.append(math.log(max(float(p[index[gold]]), 1e-12)))
                n += 1
                if n >= max_positions:
                    break
            if n >= max_positions:
                break
    nll = -float(np.mean(logs))
    return {"perplexity": float(np.exp(nll)), "nll": nll, "n": n,
            "vocab": len(allowed), "model": model_name}


def interpolated_perplexity(generator, lm: NgramLM, test: Corpus, allowed: list[str],
                            window: int, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
                            max_positions: int = 1500, seed: int = 7) -> list[dict]:
    """Sweep ``p = lam * p_fuzzy + (1 - lam) * p_ngram`` on shared positions.

    The point is **complementarity**, which is a different question from which model
    wins alone. If the best mixture beats the n-gram at ``lam > 0``, the fuzzy features
    carry information the n-gram does not have, and that holds even where the fuzzy
    model loses head-to-head. ``lam=0`` and ``lam=1`` recover the two endpoints on
    exactly the same positions, so the sweep is self-calibrating.
    """
    allowed_set = set(allowed)
    index = {w: i for i, w in enumerate(allowed)}
    rng = np.random.default_rng(seed)
    sents = [s for s in test.sentences if len(s) > window]
    order = rng.permutation(len(sents))

    pf, pn, gold_idx = [], [], []
    n = 0
    for si in order:
        sent = sents[si]
        for i in range(window, len(sent)):
            gold = sent[i]
            if gold not in allowed_set or gold not in generator.index:
                continue
            pf.append(generator.distribution(sent[:i]))
            pn.append(lm.distribution(sent[:i], allowed))
            gold_idx.append(index[gold])
            n += 1
            if n >= max_positions:
                break
        if n >= max_positions:
            break

    PF, PN = np.vstack(pf), np.vstack(pn)
    g = np.asarray(gold_idx)
    rows = []
    for lam in lambdas:
        mix = lam * PF + (1.0 - lam) * PN
        pick = mix[np.arange(len(g)), g]
        nll = -float(np.mean(np.log(np.maximum(pick, 1e-12))))
        rows.append({"lambda": lam, "perplexity": float(np.exp(nll)), "nll": nll,
                     "n": n})
    return rows
