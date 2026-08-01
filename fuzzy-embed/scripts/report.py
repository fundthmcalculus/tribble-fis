"""Assemble results/records.jsonl into the comparison tables.

Writes results/table.md. Keeps FES rungs and external baselines in separate
sections, because they answer different questions: baselines fix the yardstick,
rungs attribute the difference to a specific architectural change.

    python scripts/report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzyembed.evaluate import MTEB_14, load_records

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "records.jsonl"

# Published figures for the comparison models. Per direction (2026-07-31) these are
# accepted as-is rather than re-measured. Columns: params, published MTEB avg,
# published NanoBEIR, source, in-scope-as-a-target?
QUOTED = {
    "potion-base-2M": ("~1.9M", "~45-48", "-", "MTEB 41-task", "target"),
    "potion-base-8M": ("7.56M", "51.32", "-", "MTEB 41-task", "target"),
    "potion-base-32M": ("32M", "52.83", "-", "MTEB 41-task", "target"),
    "static-retrieval-mrl-en-v1": ("~32M", "-", "0.5032", "HF blog", "target"),
    "all-MiniLM-L6-v2": ("22.7M", "~56.1", "0.5623", "MTEB 41-task / HF blog", "reference"),
    "embeddinggemma-300m": ("308M", "69.67", "-", "MTEB(eng, v2)", "NOT a target"),
    "LFM2.5-Encoder-230M": ("230M", "79.29", "-",
                            "17-task GLUE/SuperGLUE, fine-tuned", "NOT an embedding model"),
}

# Measured on four models that have both numbers (docs/03-benchmarks.md 1a).
# Stable within the static family, NOT across architecture families -- so it is
# applied only to static models, and cross-family claims use NanoBEIR instead.
MTEB14_OFFSET_STATIC = 1.3


def _fmt(v, nd=2, scale=1.0):
    try:
        f = float(v) * scale
        return f"{f:.{nd}f}" if f == f else "-"
    except (TypeError, ValueError):
        return "-"


def table(records: list[dict], types: list[str]) -> str:
    head = ["model", "params", "MTEB-14", *[t[:6] for t in types], "NanoBEIR", "H_rule", "H_fire"]
    rows = []
    for r in records:
        s = r.get("summary", {})
        bt = s.get("by_type", {})
        rows.append([
            r.get("model", "?"),
            f"{r['params'] / 1e6:.2f}M" if r.get("params") else "-",
            _fmt(s.get("mteb14_avg")),
            *[_fmt(bt.get(t), 1) for t in types],
            _fmt(r.get("nanobeir_ndcg@10"), 4),
            _fmt(r.get("rule_entropy"), 3),
            _fmt(r.get("firing_entropy"), 3),
        ])
    rows.sort(key=lambda x: float(x[2]) if x[2] != "-" else -1, reverse=True)
    if not rows:
        return "_(no records yet)_"
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(head)]
    out = ["| " + " | ".join(h.ljust(x) for h, x in zip(head, w)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(x) for c, x in zip(r, w)) + " |")
    return "\n".join(out)


def main() -> int:
    recs = [r for r in load_records(RECORDS) if "summary" in r]
    failed = [r for r in load_records(RECORDS) if r.get("partial")]
    fes = [r for r in recs if r["model"].startswith("FES-")]
    base = [r for r in recs if not r["model"].startswith("FES-")]
    types = list(MTEB_14.keys())

    parts = [
        "# Results",
        "",
        "All figures below are **self-measured** through `fuzzyembed/evaluate.py` on the",
        "frozen 14-task MTEB subset (`MTEB-14`, all 7 task types) plus NanoBEIR. They are",
        "*not* comparable to published 41-task MTEB averages; see `docs/03-benchmarks.md`.",
        "",
        "`H_rule` = rule-usage entropy (are all rules used?). `H_fire` = per-token firing",
        "entropy (is the inference actually fuzzy?). Both normalised to [0,1]. A strong",
        "score at low `H_rule` means the parameter count is a fiction.",
        "",
        "## Baselines (the yardstick)",
        "",
        table(base, types),
        "",
        "## FES ablation ladder",
        "",
        table(fes, types),
        "",
        "## Published figures for the comparison models",
        "",
        "Accepted as published rather than re-measured (directed 2026-07-31).",
        "",
        "| model | params | pub. MTEB avg | pub. NanoBEIR | source | scope |",
        "|---|---|---|---|---|---|",
    ]
    for k, (p, s, nb, m, scope) in QUOTED.items():
        parts.append(f"| {k} | {p} | {s} | {nb} | {m} | {scope} |")

    parts += [
        "",
        "### Reading these against our numbers",
        "",
        f"Our MTEB-14 runs about **+{MTEB14_OFFSET_STATIC:.1f}** higher than the published",
        "41-task average *within the static-embedding family* (measured: +1.25 on",
        "potion-32M, +1.42 on potion-8M). It is **+4.42** on all-MiniLM-L6-v2, so the",
        "offset is family-specific and MTEB-14 must not be used for cross-family",
        "comparison. **NanoBEIR needs no offset** — ours reproduces published values to",
        "four decimals — so every cross-family claim is made on NanoBEIR.",
        "See `docs/03-benchmarks.md` §1a.",
        "",
        "Equivalently, to place an FES MTEB-14 score on the published static scale,",
        f"subtract ~{MTEB14_OFFSET_STATIC:.1f}.",
    ]

    if fes:
        best = max(fes, key=lambda r: r["summary"]["mteb14_avg"])
        adj = best["summary"]["mteb14_avg"] - MTEB14_OFFSET_STATIC
        parts += [
            "",
            f"Best FES rung `{best['model']}` ({best['params'] / 1e6:.2f}M): MTEB-14 "
            f"{best['summary']['mteb14_avg']:.2f} → **~{adj:.1f} on the published static scale**, "
            f"NanoBEIR {best.get('nanobeir_ndcg@10', float('nan')):.4f}.",
        ]

    if failed:
        parts += ["", "## Failed runs", ""]
        parts += [f"- `{r.get('model')}`: {str(r.get('error', ''))[:200]}" for r in failed]

    out = "\n".join(parts) + "\n"
    (ROOT / "results" / "table.md").write_text(out, encoding="utf-8")
    # The Windows console defaults to cp1252, which cannot encode the arrows and
    # dashes used above. Write the file in full UTF-8 and degrade only the echo.
    sys.stdout.reconfigure(errors="replace")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
