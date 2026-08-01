"""Evaluation: a fixed MTEB subset plus NanoBEIR, run identically for every model.

The protocol rule (docs/03-benchmarks.md): all comparisons use numbers produced by
*this* code path. Leaderboard figures differ by more than the effects we are
trying to measure, because of mteb version, prompt handling, and truncation.

The task list is frozen before any FES result exists, and must not be edited to
chase a number.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 14 tasks covering all 7 MTEB task types. See docs/03-benchmarks.md §1.
MTEB_14: dict[str, tuple[str, ...]] = {
    "Classification": (
        "AmazonCounterfactualClassification",
        "Banking77Classification",
        "EmotionClassification",
    ),
    "Clustering": (
        "TwentyNewsgroupsClustering.v2",
        "StackExchangeClustering.v2",
    ),
    "PairClassification": (
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
    ),
    "Reranking": (
        "AskUbuntuDupQuestions",
        "SciDocsRR",
    ),
    "Retrieval": (
        "SciFact",
        "NFCorpus",
        "ArguAna",
    ),
    "STS": (
        "STS12",
        "STSBenchmark",
        "SICK-R",
    ),
    "Summarization": ("SummEval",),
}

ALL_TASKS: tuple[str, ...] = tuple(t for ts in MTEB_14.values() for t in ts)

# Fast subset for in-training / iteration use: one cheap task per type.
MTEB_FAST: tuple[str, ...] = (
    "Banking77Classification",
    "TwentyNewsgroupsClustering.v2",
    "SprintDuplicateQuestions",
    "AskUbuntuDupQuestions",
    "SciFact",
    "STSBenchmark",
    "SummEval",
)


def _task_type(name: str) -> str:
    for ttype, tasks in MTEB_14.items():
        if name in tasks:
            return ttype
    return "Unknown"


def run_mteb(
    model,
    model_name: str,
    tasks: tuple[str, ...] = ALL_TASKS,
    output_folder: str | Path = "results/mteb",
    encode_batch_size: int = 256,
    overwrite: bool = True,
) -> dict:
    """Run the fixed task subset and return a flat ``{task: main_score}`` dict.

    ``overwrite`` defaults to **True**, deliberately. ``mteb`` caches results per
    output folder, which is keyed on ``model_name`` here — so re-running a *changed*
    model under the same tag silently returns the previous model's scores. That is
    exactly what happened in E013: the corrected dense S2 reported MTEB-14 47.79,
    identical to the broken rank-32 run, because all 16 task JSONs were reused;
    only NanoBEIR (computed live, never cached) revealed the change. Recomputing
    costs ~8 minutes per model and removes a whole class of silent wrong answers.
    """
    import mteb

    output_folder = Path(output_folder) / model_name
    selected = mteb.get_tasks(tasks=list(tasks), languages=["eng"])
    evaluation = mteb.MTEB(tasks=selected)

    t0 = time.perf_counter()
    results = evaluation.run(
        model,
        output_folder=str(output_folder),
        overwrite_results=overwrite,
        encode_kwargs={"batch_size": encode_batch_size},
        verbosity=1,
    )
    elapsed = time.perf_counter() - t0

    scores: dict[str, float] = {}
    for res in results:
        name = getattr(res, "task_name", None) or res.task_name
        try:
            scores[name] = float(res.get_score())
        except Exception:  # noqa: BLE001 - fall back to scraping the split scores
            vals = [
                s["main_score"]
                for split in getattr(res, "scores", {}).values()
                for s in split
                if "main_score" in s
            ]
            if vals:
                scores[name] = float(sum(vals) / len(vals))
    return {"model": model_name, "elapsed_s": elapsed, "tasks": scores}


def summarise(scores: dict[str, float]) -> dict:
    """Per-task-type means and the MTEB-14 average.

    The overall figure is the **mean of task-type means**, matching MTEB's own
    convention, so a type with three tasks does not outweigh one with a single
    task.
    """
    by_type: dict[str, list[float]] = {}
    for task, score in scores.items():
        by_type.setdefault(_task_type(task), []).append(score * 100.0)
    type_means = {k: sum(v) / len(v) for k, v in sorted(by_type.items())}
    known = {k: v for k, v in type_means.items() if k != "Unknown"}
    return {
        "by_type": type_means,
        "mteb14_avg": sum(known.values()) / len(known) if known else float("nan"),
        "n_tasks": len(scores),
    }


def run_nanobeir(model, batch_size: int = 256, dataset_names: list[str] | None = None) -> dict:
    """NanoBEIR nDCG@10 -- the fast retrieval signal, comparable to the published
    static-embedding numbers (MiniLM-L6 0.5623, static-retrieval-mrl 0.5032)."""
    from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator

    evaluator = NanoBEIREvaluator(
        dataset_names=dataset_names, batch_size=batch_size, show_progress_bar=False
    )
    res = evaluator(model)
    key = next((k for k in res if k.endswith("cosine_ndcg@10") and "Nano" not in k), None)
    if key is None:  # aggregated key naming varies by version
        ndcgs = [v for k, v in res.items() if k.endswith("cosine_ndcg@10")]
        return {"nanobeir_ndcg@10": sum(ndcgs) / len(ndcgs) if ndcgs else float("nan"), "raw": res}
    return {"nanobeir_ndcg@10": float(res[key]), "raw": res}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_throughput(
    model, sentences: list[str], device: str = "cpu", batch_size: int = 256, repeats: int = 1
) -> float:
    """Sentences per second. The whole justification for this model tier."""
    model.to(device)
    model.encode(sentences[:64], batch_size=batch_size, device=device)  # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.encode(sentences, batch_size=batch_size, device=device, show_progress_bar=False)
    return len(sentences) * repeats / (time.perf_counter() - t0)


def save_record(record: dict, path: str | Path) -> None:
    """Append one JSON record per line, so results accumulate across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def results_table(records: list[dict]) -> str:
    """Markdown table over accumulated records, sorted by MTEB-14 average."""
    types = list(MTEB_14.keys())
    head = ["model", "params", "MTEB-14", *[t[:7] for t in types], "NanoBEIR"]
    rows = []
    for r in records:
        s = r.get("summary", {})
        bt = s.get("by_type", {})
        rows.append([
            r.get("model", "?"),
            f"{r.get('params', 0) / 1e6:.2f}M" if r.get("params") else "-",
            f"{s.get('mteb14_avg', float('nan')):.2f}",
            *[f"{bt[t]:.1f}" if t in bt else "-" for t in types],
            f"{r['nanobeir_ndcg@10']:.4f}" if r.get("nanobeir_ndcg@10") == r.get("nanobeir_ndcg@10")
            and r.get("nanobeir_ndcg@10") is not None else "-",
        ])
    rows.sort(key=lambda x: float(x[2]) if x[2] not in ("nan", "-") else -1, reverse=True)
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(head)]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)
