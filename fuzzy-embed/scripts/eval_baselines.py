"""Measure every baseline through our own evaluation path.

Run this before any FES result exists, so the yardstick is fixed and cannot be
retrofitted. Results append to results/records.jsonl.

    python scripts/eval_baselines.py                 # the small-model targets
    python scripts/eval_baselines.py --group large   # the caveated 200M+ references
    python scripts/eval_baselines.py --fast          # 7-task subset, for iteration
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzyembed.evaluate import (
    ALL_TASKS,
    MTEB_FAST,
    count_params,
    load_records,
    results_table,
    run_mteb,
    run_nanobeir,
    save_record,
    summarise,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "records.jsonl"

# (key, loader spec, note). Loader spec is ("st", name) for a plain
# SentenceTransformer or ("m2v", name) for a model2vec static model.
SMALL = [
    ("potion-base-2M", ("m2v", "minishlab/potion-base-2M"), "target: smallest MTEB model"),
    ("potion-base-8M", ("m2v", "minishlab/potion-base-8M"), "target: primary"),
    ("potion-base-32M", ("m2v", "minishlab/potion-base-32M"), "target: best static"),
    ("static-retrieval-mrl-en-v1", ("st", "sentence-transformers/static-retrieval-mrl-en-v1"),
     "target: from-scratch static SOTA"),
    ("all-MiniLM-L6-v2", ("st", "sentence-transformers/all-MiniLM-L6-v2"),
     "reference denominator"),
]

LARGE = [
    ("embeddinggemma-300m", ("st", "google/embeddinggemma-300m"),
     "ceiling reference; a real embedding model, NOT a target"),
    ("LFM2.5-Encoder-230M-meanpool", ("meanpool", "LiquidAI/LFM2.5-Encoder-230M"),
     "CAVEAT: not an embedding model; no contrastive training. See docs/03-benchmarks.md 4"),
]


def load_model(spec: tuple[str, str]):
    kind, name = spec
    if kind == "m2v":
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import StaticEmbedding

        return SentenceTransformer(modules=[StaticEmbedding.from_model2vec(name)])
    if kind == "meanpool":
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

        tr = Transformer(name, model_args={"trust_remote_code": True}, max_seq_length=512)
        pool = Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[tr, pool])
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, trust_remote_code=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["small", "large", "both"], default="small")
    ap.add_argument("--fast", action="store_true", help="7-task subset instead of 14")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-nanobeir", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated model keys")
    args = ap.parse_args()

    targets = {"small": SMALL, "large": LARGE, "both": SMALL + LARGE}[args.group]
    if args.only:
        keep = {k.strip() for k in args.only.split(",")}
        targets = [t for t in targets if t[0] in keep]

    tasks = MTEB_FAST if args.fast else ALL_TASKS
    done = {r["model"] for r in load_records(RECORDS) if not r.get("partial")}

    for key, spec, note in targets:
        tag = f"{key}{'-fast' if args.fast else ''}"
        if tag in done:
            logging.info("skipping %s (already recorded)", tag)
            continue
        logging.info("=== %s === (%s)", tag, note)
        try:
            model = load_model(spec)
            params = count_params(model)
            logging.info("%s: %.2fM params", key, params / 1e6)

            rec = run_mteb(model, tag, tasks=tasks, output_folder=ROOT / "results" / "mteb",
                           encode_batch_size=args.batch_size)
            rec["params"] = params
            rec["note"] = note
            rec["summary"] = summarise(rec["tasks"])
            rec["task_set"] = "MTEB_FAST" if args.fast else "MTEB_14"
            if not args.skip_nanobeir:
                rec.update(run_nanobeir(model, batch_size=args.batch_size))
                rec.pop("raw", None)
            save_record(rec, RECORDS)
            logging.info(
                "%s -> MTEB-14 %.2f | NanoBEIR %s", tag,
                rec["summary"]["mteb14_avg"], rec.get("nanobeir_ndcg@10"),
            )
            del model
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - record the failure, keep going
            logging.error("FAILED %s\n%s", tag, traceback.format_exc())
            save_record({"model": tag, "note": note, "error": traceback.format_exc()[-2000:],
                         "partial": True}, RECORDS)

    print("\n" + results_table([r for r in load_records(RECORDS) if "summary" in r]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
