"""Benchmark runner: ``python -m benchmarks.bench [options]``.

Records wall-clock per workload to JSON and prints a Markdown table. Pass
``--compare <baseline.json>`` to print the speedup against an earlier run and to
flag any workload whose checksum moved -- a changed checksum means the
optimization changed the answer, which disqualifies the timing next to it.

Examples::

    python -m benchmarks.bench -o benchmarks/results/baseline.json
    python -m benchmarks.bench --compare benchmarks/results/baseline.json
    python -m benchmarks.bench -k forward --profile
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .workloads import Workload, all_workloads, workloads_by_name


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _environment() -> dict[str, Any]:
    import numpy as np

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_rev": _git_rev(),
    }


def run_workload(w: Workload) -> dict[str, Any]:
    """Time one workload.

    ``min`` is the headline number: it is the run least polluted by scheduler
    noise and is the most reproducible statistic on a laptop. The median is kept
    alongside it so a result that is fast only once is visible as such.
    """
    state = w.setup()
    for _ in range(w.warmups):
        w.run(state)

    samples: list[float] = []
    checksum: float | None = None
    for _ in range(w.repeats):
        t0 = time.perf_counter()
        result = w.run(state)
        samples.append(time.perf_counter() - t0)
        checksum = w.checksum(result)

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    return {
        "name": w.name,
        "description": w.description,
        "tags": list(w.tags),
        "repeats": w.repeats,
        "min_s": samples[0],
        "median_s": median,
        "max_s": samples[-1],
        "checksum": checksum,
    }


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def render_table(results: list[dict[str, Any]], baseline: dict[str, Any] | None) -> str:
    base_index = (
        {r["name"]: r for r in baseline["results"]} if baseline else {}
    )
    has_base = bool(base_index)

    header = ["workload", "min", "median"]
    if has_base:
        header += ["baseline min", "speedup", "checksum"]
    else:
        header += ["checksum"]

    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    for r in results:
        if "skipped" in r:
            lines.append("| " + " | ".join(
                [r["name"], f"skipped: {r['skipped']}"] + ["-"] * (len(header) - 2)
            ) + " |")
            continue
        row = [r["name"], _fmt_time(r["min_s"]), _fmt_time(r["median_s"])]
        if has_base:
            b = base_index.get(r["name"])
            if b is not None and "skipped" in b:
                b = None
            if b is None:
                row += ["-", "-"]
            else:
                speed = b["min_s"] / r["min_s"] if r["min_s"] > 0 else float("inf")
                row += [_fmt_time(b["min_s"]), f"{speed:.2f}x"]
            # A moved checksum invalidates the speedup on the same row, so say
            # so inline rather than in a footnote.
            if b is not None and not _checksums_match(b["checksum"], r["checksum"]):
                row.append(f"CHANGED ({b['checksum']!r} -> {r['checksum']!r})")
            else:
                row.append(_fmt_checksum(r["checksum"]))
        else:
            row.append(_fmt_checksum(r["checksum"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_checksum(c: Any) -> str:
    return f"{c:.10g}" if isinstance(c, float) else str(c)


def _checksums_match(a: Any, b: Any, rtol: float = 1e-9) -> bool:
    """Compare checksums with a relative tolerance.

    Exact equality is the goal for a pure refactor, but a kernel that reassociates
    a floating-point reduction (vectorizing a fold, say) legitimately shifts the
    last bits. A relative tolerance accepts that and still catches any change
    large enough to be a behavioural difference.
    """
    if isinstance(a, float) and isinstance(b, float):
        scale = max(abs(a), abs(b), 1e-300)
        return abs(a - b) <= rtol * scale
    return a == b


def profile_workload(w: Workload, top: int = 25) -> str:
    """cProfile one timed call of `w`, returning the cumulative-time table."""
    import cProfile
    import io
    import pstats

    state = w.setup()
    w.run(state)  # warm caches so the profile is of steady-state work
    pr = cProfile.Profile()
    pr.enable()
    w.run(state)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-k", "--select", action="append", default=None,
                   help="substring filter on workload name (repeatable)")
    p.add_argument("-n", "--name", action="append", default=None,
                   help="exact workload name (repeatable)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="write results JSON here")
    p.add_argument("--compare", type=Path, default=None,
                   help="baseline results JSON to compare against")
    p.add_argument("--profile", action="store_true",
                   help="cProfile each selected workload instead of timing it")
    p.add_argument("--label", default="", help="free-text label stored in the JSON")
    args = p.parse_args(argv)

    workloads = workloads_by_name(args.name)
    if args.select:
        workloads = [w for w in workloads
                     if any(s in w.name for s in args.select)]
    if not workloads:
        raise SystemExit("no workloads selected")

    if args.profile:
        for w in workloads:
            print(f"\n===== profile: {w.name} =====")
            print(w.description)
            print(profile_workload(w))
        return 0

    results = []
    for w in workloads:
        ok, reason = w.available()
        if not ok:
            # Recorded, not dropped: a missing GPU row should say "no CUDA
            # device", not look like a workload nobody bothered to run.
            print(f"skipping {w.name}: {reason}", flush=True)
            results.append({
                "name": w.name, "description": w.description, "tags": list(w.tags),
                "skipped": reason,
            })
            continue
        print(f"running {w.name} ...", flush=True)
        results.append(run_workload(w))

    payload = {
        "label": args.label,
        "environment": _environment(),
        "results": results,
    }

    baseline = None
    if args.compare:
        baseline = json.loads(args.compare.read_text())

    print()
    print(render_table(results, baseline))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output}")

    if baseline:
        base_index = {r["name"]: r for r in baseline["results"]}
        moved = [r["name"] for r in results
                 if "skipped" not in r and r["name"] in base_index
                 and "skipped" not in base_index[r["name"]]
                 and not _checksums_match(base_index[r["name"]]["checksum"], r["checksum"])]
        if moved:
            print(f"\nCHECKSUM MISMATCH on: {', '.join(moved)}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
