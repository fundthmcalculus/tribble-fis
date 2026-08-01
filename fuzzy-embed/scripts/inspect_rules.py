"""Decode a trained FES rule base into readable rules (DESIGN.md §7).

This is the thing a lookup table cannot give you. For each rule r we report:

  * the vocabulary tokens that fire it most strongly  -> what the antecedent means
  * ||b_r|| and ||A_r||_F                             -> how much it contributes
  * its share of total firing mass over a corpus      -> how much it is used

The hypothesis under test: rules specialise into linguistically recognisable
regions (subword continuations, function words, numerals, topical content), and
low-contribution rules coincide with low-information tokens -- i.e. the model
*rediscovers* IDF/SIF weighting as emergent rule structure rather than having it
imposed.

    python scripts/inspect_rules.py artifacts/A2-fes-s
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fuzzyembed.model import FuzzyEmbedding


def load_fuzzy(path: str) -> FuzzyEmbedding:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(path, device="cpu")
    for mod in st.modules():
        if isinstance(mod, FuzzyEmbedding):
            return mod
    raise SystemExit(f"no FuzzyEmbedding module found in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--top-k", type=int, default=15, help="tokens to show per rule")
    ap.add_argument("--vocab-limit", type=int, default=30522)
    ap.add_argument("--counts", default=None, help="npy of token counts, to weight usage")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    # The bert vocabulary contains Devanagari, CJK and other non-Latin
    # wordpieces; the Windows console is cp1252 and cannot encode them.
    sys.stdout.reconfigure(errors="replace")

    m = load_fuzzy(args.model_path).eval()
    if m.n_rules == 1:
        print("model has a single rule; nothing to decode (this is a control ablation)")
        return 0

    vocab = m.tokenizer.get_vocab()
    id_to_tok = {v: k for k, v in vocab.items()}
    ids = torch.arange(min(args.vocab_limit, m.vocab_size))

    with torch.no_grad():
        v = m.norm(m.features(ids)).unsqueeze(0)  # (1, V, d_in)
        ctx = v.mean(1, keepdim=True) if m.context_conditioned else None
        f = m.firing_strengths(v, ctx)[0]  # (V, R)

    counts = None
    if args.counts and Path(args.counts).exists():
        counts = np.load(args.counts)[: len(ids)].astype(np.float64)
        counts = counts / max(counts.sum(), 1.0)

    # Per-rule usage: unweighted (share of vocabulary) and frequency-weighted
    # (share of actual corpus tokens). They differ a lot, and the second is the
    # one that matters.
    usage_vocab = f.mean(0).numpy()
    usage_corpus = (
        (f.numpy() * counts[:, None]).sum(0) if counts is not None else usage_vocab
    )

    b_norm = m.expert_b.detach().norm(dim=-1).numpy()
    a_norm = (
        m.expert_w.detach().flatten(1).norm(dim=-1).numpy()
        if m.expert_w is not None
        else np.zeros(m.n_rules)
    )

    print(f"\nmodel: {args.model_path}")
    print(f"rules: {m.n_rules}   d_in: {m.d_in}   d_out: {m.d_out}   "
          f"context_conditioned: {m.context_conditioned}")
    print(f"tau (HTSK temperature): {float(m.log_tau.detach().exp()):.4f}   D: {m.D}")
    print(f"params: {m.parameter_counts()['total']:,}")
    print(f"\nmean per-token firing entropy over vocab: "
          f"{float(-(f.clamp_min(1e-12).log() * f).sum(-1).mean() / np.log(m.n_rules)):.4f}")

    order = np.argsort(-usage_corpus)
    rows = []
    print("\n" + "=" * 100)
    for rank, r in enumerate(order):
        top = torch.topk(f[:, r], args.top_k).indices.tolist()
        toks = [id_to_tok.get(int(t), f"<{t}>") for t in top]
        print(
            f"\nRULE {int(r):>3}  (usage rank {rank + 1})   "
            f"corpus usage {usage_corpus[r] * 100:6.2f}%   vocab share {usage_vocab[r] * 100:6.2f}%"
        )
        print(f"           ||b_r|| = {b_norm[r]:.4f}   ||A_r||_F = {a_norm[r]:.4f}")
        print(f"           IF token is near prototype {int(r)}, i.e. one of:")
        print("             " + "  ".join(repr(t) for t in toks))
        rows.append({
            "rule": int(r),
            "usage_corpus": float(usage_corpus[r]),
            "usage_vocab": float(usage_vocab[r]),
            "b_norm": float(b_norm[r]),
            "a_norm": float(a_norm[r]),
            "top_tokens": toks,
        })

    # The emergent-IDF check: is contribution magnitude correlated with how
    # content-bearing the rule's tokens are? We proxy "content-bearing" by the
    # negative log frequency of the rule's top tokens.
    if counts is not None:
        print("\n" + "=" * 100)
        print("EMERGENT-IDF CHECK")
        rule_rarity, rule_contrib = [], []
        for row in rows:
            tids = [vocab.get(t) for t in row["top_tokens"] if t in vocab]
            ps = [counts[i] for i in tids if i is not None and i < len(counts)]
            ps = [p for p in ps if p > 0]
            if ps:
                rule_rarity.append(float(-np.log(np.mean(ps))))
                rule_contrib.append(row["a_norm"] + row["b_norm"])
        if len(rule_rarity) > 2:
            rho = float(np.corrcoef(rule_rarity, rule_contrib)[0, 1])
            print(f"  corr(rule token rarity, rule contribution magnitude) = {rho:+.3f}")
            print("  positive => rules over rare/content tokens contribute more,")
            print("             i.e. the model rediscovered IDF-style weighting.")

    # Pooling weights are the other interpretable surface: they are the direct
    # analogue of SIF/IDF and can be read off per token.
    with torch.no_grad():
        pw = m.pool_logit.weight.squeeze(-1)[: len(ids)]
    hi = torch.topk(pw, 20).indices.tolist()
    lo = torch.topk(-pw, 20).indices.tolist()
    print("\n" + "=" * 100)
    print("LEARNED POOLING WEIGHTS (the model's own IDF)")
    print("  highest: " + "  ".join(id_to_tok.get(int(t), "?") for t in hi))
    print("  lowest : " + "  ".join(id_to_tok.get(int(t), "?") for t in lo))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
