"""Experiment B harness: compare fuzzy and non-fuzzy heads on frozen embeddings.

Sweeps heads x MRL widths x noise rates off a cached embedding set and reports
accuracy alongside rule count, antecedent count, and wall-clock -- because the
question is not only "is the FIS accurate?" but "is it accurate at a rule count a
human can read?"

Usage
-----
    # plumbing check, no network, no encoder
    uv run python flm/exp_b/run_sentiment.py --synthetic

    # real run against a cache built by embed.py
    uv run python flm/exp_b/run_sentiment.py --cache flm/exp_b/cache --dataset sst2

    # continuous SST target -- the framing that suits a fuzzy system
    uv run python flm/exp_b/run_sentiment.py --cache flm/exp_b/cache --dataset sst_cont
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from atlas import build_atlas, render_atlas  # noqa: E402
from heads import HeadResult, build_heads, run_head  # noqa: E402


def load_cache(cache_dir: Path, dataset: str, dim: int, noise: float,
               model_key: str | None = None) -> dict:
    """Load one cached (dataset, width, noise) slice written by ``embed.py``."""
    tag = f"noise{noise:g}" if noise > 0 else "clean"
    pattern = f"{dataset}__{model_key or '*'}__d{dim}__{tag}.npz"
    matches = sorted(cache_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no cache matching {pattern} in {cache_dir}. Run embed.py first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} caches match {pattern}; disambiguate with --model-key: "
            + ", ".join(m.name for m in matches)
        )
    z = np.load(matches[0], allow_pickle=True)
    return {
        "train_X": z["train_X"], "train_y": z["train_y"],
        "test_X": z["test_X"], "test_y": z["test_y"],
        "task": str(z["task"]),
    }


def format_table(rows: list[dict], task: str) -> str:
    """Render the sweep as a markdown table, sorted by the headline metric."""
    if not rows:
        return "(no results)"
    metric = "accuracy" if task == "classification" else "spearman"
    higher_better = True

    ok = [r for r in rows if r.get("error") is None]
    bad = [r for r in rows if r.get("error") is not None]
    ok.sort(key=lambda r: r["metrics"].get(metric, -np.inf), reverse=higher_better)

    if task == "classification":
        cols = ["accuracy", "macro_f1"]
    else:
        cols = ["spearman", "mae", "r2"]

    head = ["head", "dim", "noise", *cols, "rules", "antec", "fit_s", "pred_s"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for r in ok:
        vals = [f"{r['metrics'].get(c, float('nan')):.4f}" for c in cols]
        lines.append("| " + " | ".join([
            r["head"], str(r["dim"]), f"{r['noise']:g}", *vals,
            str(r["n_rules"] if r["n_rules"] is not None else "-"),
            f"{r['mean_antecedents']:.1f}" if r["mean_antecedents"] else "-",
            f"{r['fit_seconds']:.2f}", f"{r['predict_seconds']:.3f}",
        ]) + " |")
    for r in bad:
        lines.append(f"| {r['head']} | {r['dim']} | {r['noise']:g} | "
                     + " | ".join(["FAILED"] * len(cols))
                     + f" | - | - | - | - |  <!-- {r['error']} -->")
    if bad:
        lines.append("")
        lines.append("Failures:")
        for r in bad:
            lines.append(f"- `{r['head']}` d={r['dim']} noise={r['noise']:g}: {r['error']}")
    return "\n".join(lines)


def result_row(res: HeadResult, dim: int, noise: float) -> dict:
    return {
        "head": res.name, "dim": dim, "noise": noise, "task": res.task,
        "metrics": res.metrics, "n_rules": res.n_rules,
        "mean_antecedents": res.mean_antecedents,
        "fit_seconds": res.fit_seconds, "predict_seconds": res.predict_seconds,
        "selected_features": res.selected_features, "error": res.error,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="offline plumbing check on planted-signal random embeddings")
    p.add_argument("--synthetic-task", default="classification",
                   choices=["classification", "regression"])
    p.add_argument("--cache", type=Path, default=Path(__file__).parent / "cache")
    p.add_argument("--dataset", default="sst2")
    p.add_argument("--model-key", default=None,
                   help="disambiguate when several models are cached, e.g. gte-small")
    p.add_argument("--dims", type=int, nargs="+", default=None,
                   help="MRL widths to evaluate (default: whatever is cached)")
    p.add_argument("--noise", type=float, nargs="+", default=[0.0])
    p.add_argument("--top-n", type=int, default=20,
                   help="embedding dims the fuzzy heads may select (the key knob)")
    p.add_argument("--heads", nargs="+", default=None, help="subset of heads to run")
    p.add_argument("--atlas", action="store_true",
                   help="post-hoc name the selected dimensions (needs cached texts)")
    p.add_argument("--out", type=Path, default=None, help="write results JSON here")
    args = p.parse_args()

    rows: list[dict] = []
    texts: list[str] | None = None
    task = "classification"

    if args.synthetic:
        from data import synthetic_dataset
        task = args.synthetic_task
        ds, emb = synthetic_dataset(task=task)
        print(ds.summary())
        print("\n*** SYNTHETIC MODE: planted axis-aligned Gaussian signal. This "
              "validates the\n    harness only. It says nothing about real "
              "embeddings -- planted signal is\n    exactly the structure a "
              "Gaussian-membership FIS is built to find. ***\n")
        heads = build_heads(task, top_n=args.top_n, include=args.heads)
        for name, factory in heads.items():
            res = run_head(name, factory, emb["train"], ds.train_y,
                           emb["test"], ds.test_y, task)
            print(f"  {name:24s} {res.error or res.metrics}")
            rows.append(result_row(res, emb["train"].shape[1], 0.0))
        texts = ds.train_texts
        train_X = emb["train"]
    else:
        dims = args.dims
        if dims is None:
            found = {int(f.stem.split("__d")[1].split("__")[0])
                     for f in args.cache.glob(f"{args.dataset}__*__d*__*.npz")}
            if not found:
                raise SystemExit(f"no caches for {args.dataset} in {args.cache}; "
                                 "run embed.py first (or pass --synthetic)")
            dims = sorted(found, reverse=True)
            print(f"discovered cached widths: {dims}")

        train_X = None
        for dim in dims:
            for noise in args.noise:
                blob = load_cache(args.cache, args.dataset, dim, noise, args.model_key)
                task = blob["task"]
                if train_X is None:
                    train_X = blob["train_X"]
                print(f"\n=== dim={dim} noise={noise:g} task={task} ===")
                heads = build_heads(task, top_n=args.top_n, include=args.heads)
                for name, factory in heads.items():
                    res = run_head(name, factory, blob["train_X"], blob["train_y"],
                                   blob["test_X"], blob["test_y"], task)
                    print(f"  {name:24s} {res.error or res.metrics}")
                    rows.append(result_row(res, dim, noise))

        tp = args.cache / f"{args.dataset}__texts.npz"
        if tp.exists():
            texts = list(np.load(tp, allow_pickle=True)["train_texts"])

    print("\n" + format_table(rows, task))

    if args.atlas:
        if texts is None or train_X is None:
            print("\n(atlas skipped: no cached texts)")
        else:
            best = max(
                (r for r in rows if r["error"] is None and r["selected_features"]),
                key=lambda r: r["metrics"].get(
                    "accuracy" if task == "classification" else "spearman", -np.inf),
                default=None,
            )
            if best is None:
                print("\n(atlas skipped: no head reported selected features)")
            else:
                print(f"\nProfiling dimensions selected by `{best['head']}`:")
                print(render_atlas(build_atlas(
                    best["selected_features"], train_X, texts)))

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
