"""Cost curves for scaling FES: parameters, throughput, and memory vs (R, d_in, d_out, rank).

No training -- this measures the *cost* side only, so the expensive question
("does quality follow?") is asked of a short list rather than a grid.

The architectural claim being tested: because the sequence pool commutes inside the
consequent (LOG.md E001 Finding 1), the expert projection runs once per *document*,
not per token. So growing R or d_out should cost throughput far less than a
per-token formulation would, and FES should stay in the static-embedding cost tier.

    python scripts/scaling_cost.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import Tokenizer

from fuzzyembed.model import FuzzyEmbedding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scaling_cost.json"

# Realistic text lengths; short queries and long-ish documents both matter.
SAMPLE = [
    "what is the capital of france",
    "the giant panda is a bear native to south central china and eats bamboo shoots "
    "almost exclusively, though it is a carnivore by ancestry and digestion",
] * 512


def bench(m: FuzzyEmbedding, texts: list[str], device: str, batch: int, reps: int) -> dict:
    m = m.to(device).eval()
    feats = m.preprocess(texts[:batch])
    feats = {k: v.to(device) for k, v in feats.items()}
    with torch.no_grad():
        for _ in range(3):
            m(dict(feats))
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            m(dict(feats))
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return {
        "sent_per_s": batch / dt,
        "ms_per_batch": dt * 1000,
        "peak_mem_mb": (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else None,
    }


def main() -> int:
    tok = Tokenizer.from_pretrained("google-bert/bert-base-uncased")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    # (label, kwargs)
    configs = [
        ("R=1 control (d_in=81)", dict(d_in=81, d_out=256, n_rules=1)),
        ("R=32 dense (current)", dict(d_in=64, d_out=256, n_rules=32)),
        ("R=64 dense", dict(d_in=64, d_out=256, n_rules=64)),
        ("R=128 dense", dict(d_in=64, d_out=256, n_rules=128)),
        ("R=512 dense", dict(d_in=64, d_out=256, n_rules=512)),
        ("R=64 rank32", dict(d_in=64, d_out=256, n_rules=64, consequent_rank=32)),
        ("R=128 rank32", dict(d_in=64, d_out=256, n_rules=128, consequent_rank=32)),
        ("R=512 rank32", dict(d_in=64, d_out=256, n_rules=512, consequent_rank=32)),
        ("R=128 rank32 d_out=512", dict(d_in=64, d_out=512, n_rules=128, consequent_rank=32)),
        ("R=128 rank32 d_out=768", dict(d_in=64, d_out=768, n_rules=128, consequent_rank=32)),
        ("R=128 rank32 d_in=128", dict(d_in=128, d_out=256, n_rules=128, consequent_rank=32)),
    ]

    print(f"device={dev}  batch=512\n")
    hdr = f"{'config':<26} {'params':>10} {'GPU s/s':>10} {'CPU s/s':>9} {'mem MB':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, kw in configs:
        m = FuzzyEmbedding(tok, max_seq_length=256, **kw)
        p = m.parameter_counts()["total"]
        gpu = bench(m, SAMPLE, dev, batch=512, reps=20) if dev == "cuda" else None
        cpu = bench(m, SAMPLE, "cpu", batch=512, reps=3)
        rows.append({"label": label, "config": kw, "params": p,
                     "gpu": gpu, "cpu": cpu})
        print(f"{label:<26} {p:>10,} {gpu['sent_per_s'] if gpu else 0:>10,.0f} "
              f"{cpu['sent_per_s']:>9,.0f} {(gpu['peak_mem_mb'] if gpu else 0):>8.1f}")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

    base = next(r for r in rows if r["label"] == "R=32 dense (current)")
    print("\nrelative to R=32 dense:")
    for r in rows:
        pr = r["params"] / base["params"]
        tr = r["cpu"]["sent_per_s"] / base["cpu"]["sent_per_s"]
        print(f"  {r['label']:<26} params x{pr:>5.2f}   CPU throughput x{tr:>5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
