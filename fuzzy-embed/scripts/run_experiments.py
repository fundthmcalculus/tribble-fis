"""Run the FES ablation ladder (DESIGN.md §5) and evaluate each rung.

Every rung shares data, step count, loss, and seed, so differences are
attributable to the architecture change alone.

    python scripts/run_experiments.py --only A1-lowrank-ctrl,A2-fes-s
    python scripts/run_experiments.py --scale 0.05 --fast     # quick pass
    python scripts/run_experiments.py                          # the full ladder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer

from fuzzyembed.data import MIX, load_mix, mix_summary, most_frequent_ids, token_frequencies
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
from fuzzyembed.train import FESConfig, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("httpx", "datasets", "filelock", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "records.jsonl"
TRAINLOG = ROOT / "results" / "train_runs.jsonl"

D_IN, D_OUT, R = 64, 256, 32

# The ladder. Order matters: the control (A1) comes before the fuzzy models so a
# null result cannot be explained away afterwards.
LADDER: dict[str, FESConfig] = {
    # Reference: the standard static-embedding architecture at potion-base-8M
    # size, trained on our data with our recipe. Answers "what does the
    # conventional design get here?", NOT parameter-matched.
    "A0-static-256": FESConfig(
        name="A0-static-256", d_in=D_OUT, d_out=D_OUT, n_rules=1, consequent_order=1,
        pool="mean", feature_norm=False, kmeans_init=False,
    ),
    # THE CONTROL. Narrow table + learned SIF pool + one linear expert. Isolates
    # everything except the rule base.
    "A1-lowrank-ctrl": FESConfig(
        name="A1-lowrank-ctrl", d_in=D_IN, d_out=D_OUT, n_rules=1, consequent_order=1,
        kmeans_init=False,
    ),
    # PARAMETER-MATCHED CONTROL. A1 has 2.00M params and A2 has 2.52M, so A2
    # beating A1 could just be the extra 26% of capacity. This rung widens the
    # table to d_in=81 -> 2,524,121 params, within 0.14% of A2's 2,520,635, with
    # still only one rule. If A2 does not beat *this*, the rule base is not
    # earning its parameters and the compression claim fails.
    "A1b-ctrl-matched": FESConfig(
        name="A1b-ctrl-matched", d_in=81, d_out=D_OUT, n_rules=1, consequent_order=1,
        kmeans_init=False,
    ),
    # The compression claim: does a rule base beat one rule at ~matched params?
    "A2-fes-s": FESConfig(name="A2-fes-s", d_in=D_IN, d_out=D_OUT, n_rules=R),
    # Confirms the curse-of-dimensionality failure mode (arXiv:2102.04271).
    "A3-product-tnorm": FESConfig(
        name="A3-product-tnorm", d_in=D_IN, d_out=D_OUT, n_rules=R,
        defuzzify="product", learn_temperature=False,
    ),
    # Measures rule collapse without the load-balancing term.
    "A4-no-ur": FESConfig(name="A4-no-ur", d_in=D_IN, d_out=D_OUT, n_rules=R, ur_weight=0.0),
    # The expressivity claim: break the bag-of-words ceiling.
    "A5c-fes-c": FESConfig(
        name="A5c-fes-c", d_in=D_IN, d_out=D_OUT, n_rules=R, context_conditioned=True,
    ),
    # E003: A5c collapsed to H_fire = 0.000 (hard one-hot routing) and lost 0.149
    # NanoBEIR. Hypothesis: hard *context* gating breaks query-document symmetry --
    # a query and its answer document have different context vectors, so they route
    # to different experts and their embeddings stop being comparable. If that is
    # right, holding the routing soft should recover most of the loss.
    "A5c-anchored": FESConfig(
        name="A5c-anchored", d_in=D_IN, d_out=D_OUT, n_rules=R,
        context_conditioned=True, entropy_weight=10.0,
    ),
    "A7-logtsk": FESConfig(
        name="A7-logtsk", d_in=D_IN, d_out=D_OUT, n_rules=R,
        defuzzify="logtsk", learn_temperature=False,
    ),
    # Keep the system genuinely fuzzy instead of letting it sharpen to a hard router.
    "A8-fuzzy-anchor": FESConfig(
        name="A8-fuzzy-anchor", d_in=D_IN, d_out=D_OUT, n_rules=R, entropy_weight=10.0,
    ),
    # Rule-count sweep. d_in shrinks as R grows to hold the budget roughly level.
    "A6-R4": FESConfig(name="A6-R4", d_in=D_IN, d_out=D_OUT, n_rules=4),
    "A6-R16": FESConfig(name="A6-R16", d_in=D_IN, d_out=D_OUT, n_rules=16),
    "A6-R64": FESConfig(name="A6-R64", d_in=D_IN, d_out=D_OUT, n_rules=64),
    # The "as small as possible" point.
    "T-tiny": FESConfig(name="T-tiny", d_in=32, d_out=D_OUT, n_rules=16),

    # ---- scaling ladder (E007) -------------------------------------------------
    # Cost measurements (results/scaling_cost.json) say: d_out is nearly free
    # (+3.5% params, -8% throughput for 256->768 at rank 32), the vocabulary table
    # is the dominant parameter cost, and R is cheap in *parameters* under low rank
    # but expensive in *compute* because firing strengths are O(L*R*d_in) per token
    # and do not factor out of the sequence pool. So scale d_in and d_out first,
    # and keep R moderate.
    #
    # All use ur_weight=0 because A4 (no UR) was the best rung at R=32.
    "S1-table-128": FESConfig(  # ~4.5M: spend on the table, potion-8M-ish budget
        name="S1-table-128", d_in=128, d_out=D_OUT, n_rules=32, ur_weight=0.0,
    ),
    "S2-wide-out-512": FESConfig(  # 3.05M: d_out doubled, DENSE consequent
        name="S2-wide-out-512", d_in=64, d_out=512, n_rules=32,
        ur_weight=0.0, matryoshka_dims=(512, 256, 128, 64, 32),
    ),
    "S3-balanced": FESConfig(  # 8.18M: table + output + more rules together
        name="S3-balanced", d_in=128, d_out=512, n_rules=64,
        ur_weight=0.0, matryoshka_dims=(512, 256, 128, 64, 32),
    ),
    "S4-potion8M-matched": FESConfig(  # 7.41M vs potion-base-8M's 7.56M
        name="S4-potion8M-matched", d_in=190, d_out=D_OUT, n_rules=32, ur_weight=0.0,
    ),
    "S5-manyrules-rank": FESConfig(  # 6.28M: does R=256 help? dense, so no rank cap
        name="S5-manyrules-rank", d_in=64, d_out=D_OUT, n_rules=256, ur_weight=0.0,
    ),
    # E012: the low-rank consequent shares V across rules, which caps the model's
    # entire embedding rank at k + min(R, d_out). S2 at rank 32 produced 512-d
    # vectors living in a 65-d subspace and lost 3.5 MTEB-14 / 0.095 NanoBEIR.
    # Kept as an explicit ablation of that failure, not as a scaling config.
    "S6-rank-bottleneck-demo": FESConfig(
        name="S6-rank-bottleneck-demo", d_in=64, d_out=512, n_rules=32,
        consequent_rank=256, ur_weight=0.0, matryoshka_dims=(512, 256, 128, 64, 32),
    ),

    # ---- optimisation: resolve the Finding 6 confound --------------------------
    # A2's *training* loss was worse than A1's. Either the rule base regularises,
    # or 32 experts each see a fraction of the gradient and lr_dense=2e-3 is simply
    # too low. These separate the two.
    "O1-lr-dense-8e-3": FESConfig(
        name="O1-lr-dense-8e-3", d_in=D_IN, d_out=D_OUT, n_rules=R,
        ur_weight=0.0, lr_dense=8e-3,
    ),
    "O2-lr-dense-3e-2": FESConfig(
        name="O2-lr-dense-3e-2", d_in=D_IN, d_out=D_OUT, n_rules=R,
        ur_weight=0.0, lr_dense=3e-2,
    ),
    "O3-2epoch": FESConfig(  # is 1 epoch simply not enough for the rule base?
        name="O3-2epoch", d_in=D_IN, d_out=D_OUT, n_rules=R,
        ur_weight=0.0, epochs=2.0,
    ),
}

DEFAULT_ORDER = [
    "A1-lowrank-ctrl", "A2-fes-s", "A5c-fes-c", "A0-static-256",
    "A3-product-tnorm", "A4-no-ur", "A8-fuzzy-anchor", "A7-logtsk",
    "A6-R4", "A6-R16", "A6-R64", "T-tiny",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated rung names")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of each dataset cap")
    ap.add_argument("--fast", action="store_true", help="7-task MTEB subset")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=None,
                    help="override every rung's epoch count. Left unset so rungs "
                         "that specify their own (O3-2epoch) keep it.")
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--skip-nanobeir", action="store_true")
    ap.add_argument("--redo", action="store_true", help="re-run even if recorded")
    ap.add_argument("--records", default=None,
                    help="alternate records file. Use a separate file when running two "
                         "instances concurrently so appends cannot interleave.")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated seeds. >1 seed replicates each rung, which is "
                         "how a sub-threshold difference gets tested rather than asserted.")
    args = ap.parse_args()
    global RECORDS
    if args.records:
        RECORDS = Path(args.records)

    names = [n.strip() for n in args.only.split(",")] if args.only else DEFAULT_ORDER
    unknown = [n for n in names if n not in LADDER]
    if unknown:
        raise SystemExit(f"unknown rungs: {unknown}\navailable: {list(LADDER)}")

    tok = Tokenizer.from_pretrained("google-bert/bert-base-uncased")
    logging.info("loading training mix (scale=%.3f)", args.scale)
    mix = load_mix(MIX, scale=args.scale)
    print(mix_summary(mix))

    counts = token_frequencies(
        mix, tok, tok.get_vocab_size(),
        cache_path=ROOT / "artifacts" / f"token_counts_s{args.scale}.npy",
    )
    top_ids = most_frequent_ids(counts, top_k=20_000)

    tasks = MTEB_FAST if args.fast else ALL_TASKS
    suffix = "-fast" if args.fast else ""
    done = {r["model"] for r in load_records(RECORDS) if "summary" in r}

    seeds = [int(x) for x in args.seeds.split(",")]
    for name, seed in [(n, s) for n in names for s in seeds]:
        tag = f"FES-{name}{suffix}" + (f"-s{seed}" if seed != 42 else "")
        if tag in done and not args.redo:
            logging.info("skipping %s (already recorded)", tag)
            continue

        cfg = LADDER[name]
        cfg.batch_size = args.batch_size
        if args.epochs is not None:
            cfg.epochs = args.epochs
        cfg.seed = seed
        logging.info("=" * 70)
        logging.info("RUNG %s", name)
        logging.info("=" * 70)
        try:
            outdir = str(ROOT / "artifacts" / (name if seed == 42 else f"{name}-s{seed}"))
            model, fuzzy, info = train(
                cfg, mix, tok, outdir,
                token_counts=counts, kmeans_ids=top_ids, steps_per_log=100,
            )
            info["rung"] = name
            info["seed"] = seed
            save_record(info, TRAINLOG)

            rec = run_mteb(model, tag, tasks=tasks,
                           output_folder=ROOT / "results" / "mteb",
                           encode_batch_size=args.eval_batch_size)
            rec["params"] = count_params(model)
            rec["summary"] = summarise(rec["tasks"])
            rec["task_set"] = "MTEB_FAST" if args.fast else "MTEB_14"
            rec["rung"] = name
            rec["seed"] = seed
            rec["config"] = cfg.as_dict()
            rec["param_breakdown"] = fuzzy.parameter_counts()
            rec["rule_entropy"] = info["final_rule_entropy"]
            rec["firing_entropy"] = info["final_firing_entropy"]
            rec["train_runtime_s"] = info["train_runtime_s"]
            if not args.skip_nanobeir:
                rec.update(run_nanobeir(model, batch_size=args.eval_batch_size))
                rec.pop("raw", None)
            save_record(rec, RECORDS)
            logging.info(
                "%s -> params %.2fM | MTEB-14 %.2f | NanoBEIR %s | H_rule %.3f | H_fire %.3f",
                tag, rec["params"] / 1e6, rec["summary"]["mteb14_avg"],
                rec.get("nanobeir_ndcg@10"), rec["rule_entropy"], rec["firing_entropy"],
            )
            del model, fuzzy
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            logging.error("FAILED %s\n%s", tag, traceback.format_exc())
            save_record({"model": tag, "rung": name, "partial": True,
                         "error": traceback.format_exc()[-3000:]}, RECORDS)

    recs = [r for r in load_records(RECORDS) if "summary" in r]
    print("\n" + results_table(recs))
    (ROOT / "results" / "table.md").write_text(results_table(recs), encoding="utf-8")
    print(f"\n{len(recs)} records in {RECORDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
