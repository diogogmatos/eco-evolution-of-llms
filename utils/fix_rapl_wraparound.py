"""
Recover the CPU and DRAM energy values affected by the RAPL wrap-around bug.

Until it was fixed, evaluate_from_local.py computed RAPL deltas as `(end - start) % 2**32`.
The powercap counters, however, wrap around at `max_energy_range_uj` (R), not at 2**32 µJ (M).
For a stored value s, the true energy of the inference is therefore one of:

    true = s + k * M              (counter did not wrap during the inference)
    true = s + (R mod M) + k * M  (counter wrapped once during the inference)

with k >= 0 lost multiples of M (~4,295 J) and k = -1 possible only in the wrapped case.

Within a run, CPU and DRAM power are stable, so the expected energy of each inference
(reference power * inference time) identifies k reliably, as candidates with different k are
~4,295 J apart. The wrapped case cannot be identified in general: its candidates are only
R mod M (~150 J) apart from the non-wrapped ones, which is within the natural variation of a
single inference. It is therefore only assumed when the non-wrapped reading is impossible
(a short inference reading ~4,000+ J). Undetected wraps leave the stored value ~150 J too low,
at most once every R joules (~50 min at 90 W), and are reported as a bound on the residual error.

Usage:
    python utils/fix_rapl_wraparound.py               # dry run, prints a report
    python utils/fix_rapl_wraparound.py --apply       # rewrites the results in place

The original values are kept in git history and in each corrected record's `rapl_correction`
field, and every decision is logged to `<results>/rapl_corrections.csv`.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

M = 2**32  # Modulo wrongly used by the script (µJ)
DOMAINS = {"cpu": "inference_cpu_joules", "dram": "inference_dram_joules"}

# A k is considered uncertain when the reading is this far (as a fraction of M) from the expected value
UNCERTAIN_FRACTION = 0.25


def load_run(run_dir):
    files = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "CoT", "all", "*.json"))):
        with open(path, "r") as f:
            files[path] = json.load(f)
    return files


def reference_power(records, field):
    """
    Robust estimate of a run's steady power (W).

    A simple median of energy / time fails for runs where most readings were truncated, so
    the power is chosen as the one that explains the most readings modulo M: a reading s at
    time t agrees with power P when P * t is close to s + k * M for some k.
    """
    timed = [(r[field] * 1e6, r["inference_time_seconds"]) for r in records if r["inference_time_seconds"] >= 1]

    def residual(s, t, p):
        d = (p * t * 1e6 - s) % M
        return min(d, M - d)

    def inliers(p):
        return [(s, t) for s, t in timed if residual(s, t, p) < 0.05 * p * t * 1e6 + 20e6]

    candidates = {round(s / t / 1e6 * 2) / 2 for s, t in timed if s > 0}
    best = max(candidates, key=lambda p: len(inliers(p)))

    # Refine with the median power of the agreeing readings, after restoring their lost multiples of M
    return statistics.median((s + round((best * t * 1e6 - s) / M) * M) / t / 1e6 for s, t in inliers(best))


def correct_value(stored_j, time_s, p_ref, r_mod_m):
    s = round(stored_j * 1e6)
    expected = p_ref * time_s * 1e6

    # Not wrapped: restore the k lost multiples of M closest to the expected energy
    k = max(0, round((expected - s) / M))
    value, wrapped = s + k * M, 0
    error = abs(value - expected)

    # Wrapped, with the true energy below R mod M: the reading is ~M too high for the inference
    wrapped_value = s + r_mod_m - M
    if error > 0.5 * M and wrapped_value >= 0 and abs(wrapped_value - expected) < 0.1 * error:
        value, wrapped, k = wrapped_value, 1, -1
        error = abs(value - expected)

    return {
        "value_j": value / 1e6,
        "wrapped": wrapped,
        "k": k,
        "expected_j": expected / 1e6,
        "error_fraction": error / M,
        "uncertain": error / M > UNCERTAIN_FRACTION,
    }


def aggregate(records, run_data):
    """Recompute the energy-dependent fields of run_data.json, as evaluate_from_local.py does."""
    n = len(records)
    cpu = sum(r["inference_cpu_joules"] for r in records)
    dram = sum(r["inference_dram_joules"] for r in records)
    gpu = sum(r["inference_gpu_joules"] for r in records)
    total = sum(r["total_inference_joules"] for r in records)
    tokens = sum(r["total_tokens"] for r in records)
    time_s = sum(r["inference_time_seconds"] for r in records)
    avg_j = total / n
    return {
        "accuracy_per_joule": run_data["avg_accuracy"] / avg_j,
        "tokens_per_joule": tokens / total,
        "joules_per_token": total / tokens,
        "avg_joules_per_inference": avg_j,
        "total_inference_joules": total,
        "total_inference_cpu_joules": cpu,
        "total_inference_dram_joules": dram,
        "total_inference_gpu_joules": gpu,
        "avg_total_power_watts": total / time_s,
        "avg_cpu_power_watts": cpu / time_s,
        "avg_dram_power_watts": dram / time_s,
        "avg_gpu_power_watts": gpu / time_s,
    }


def close(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default=str(ROOT_DIR / "results/main"))
    parser.add_argument("--max_energy_range_uj", type=int, default=262143328850)
    parser.add_argument("--apply", action="store_true", help="Rewrite the result files in place.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    log_path = results_dir / "rapl_corrections.csv"
    if args.apply and log_path.exists():
        raise SystemExit(f"{log_path} already exists: the corrections seem to have been applied already.")

    r_mod_m = args.max_energy_range_uj % M
    log_rows = []
    new_run_data = {}
    totals = {"records": 0, "changed": 0, "uncertain": 0, "runs_changed": 0}
    run_report = []

    run_dirs = sorted(
        (d for d in results_dir.glob("run_*") if d.is_dir()),
        key=lambda d: tuple(int(x) for x in d.name.split("_")[1::2]),
    )
    for run_dir in run_dirs:
        files = load_run(str(run_dir))
        records = [r for recs in files.values() for r in recs]
        if not records:
            print(f"[skip] {run_dir.name}: no per-prompt records")
            continue
        with open(run_dir / "run_data.json", "r") as f:
            run_data = json.load(f)

        # Sanity check: the aggregation must reproduce the stored run_data.json before any change
        before = aggregate(records, run_data)
        mismatched = [k for k, v in before.items() if not close(v, run_data[k])]
        if mismatched:
            print(f"[skip] {run_dir.name}: cannot reproduce run_data.json fields {mismatched}")
            continue

        refs = {d: reference_power(records, field) for d, field in DOMAINS.items()}
        changed_in_run = 0
        wraps_in_run = {d: 0 for d in DOMAINS}

        for path, recs in files.items():
            for r in recs:
                totals["records"] += 1
                correction = {}
                for domain, field in DOMAINS.items():
                    res = correct_value(r[field], r["inference_time_seconds"], refs[domain], r_mod_m)
                    wraps_in_run[domain] += res["wrapped"]
                    is_change = res["wrapped"] or res["k"] != 0
                    if res["uncertain"]:
                        totals["uncertain"] += 1
                    if is_change or res["uncertain"]:
                        log_rows.append({
                            "run": run_dir.name,
                            "file": os.path.relpath(path, run_dir),
                            "question_id": r["question_id"],
                            "domain": domain,
                            "inference_time_seconds": r["inference_time_seconds"],
                            "stored_joules": r[field],
                            "corrected_joules": res["value_j"],
                            "expected_joules": round(res["expected_j"], 3),
                            "wrapped": res["wrapped"],
                            "k": res["k"],
                            "error_fraction": round(res["error_fraction"], 3),
                            "uncertain": res["uncertain"],
                        })
                    if is_change:
                        correction[domain] = {"stored_joules": r[field], "wrapped": res["wrapped"], "k": res["k"]}
                        r[field] = res["value_j"]
                if correction:
                    r["total_inference_joules"] = (
                        r["inference_cpu_joules"] + r["inference_dram_joules"] + r["inference_gpu_joules"]
                    )
                    r["rapl_correction"] = correction
                    changed_in_run += 1

        totals["changed"] += changed_in_run
        totals["runs_changed"] += bool(changed_in_run)
        after = aggregate(records, run_data)
        # Only rewrite runs that changed, so untouched runs keep their exact original values
        if changed_in_run:
            new_run_data[run_dir.name] = (files, run_data, after)

        # Wraps the counters can have done during inference, each leaving at most R mod M undetected
        max_wraps = {d: math.ceil(after[f"total_inference_{d}_joules"] * 1e6 / args.max_energy_range_uj) for d in DOMAINS}
        residual_bound = sum(max(0, max_wraps[d] - wraps_in_run[d]) * r_mod_m / 1e6 for d in DOMAINS)
        run_report.append((
            run_dir.name, changed_in_run,
            before["total_inference_joules"], after["total_inference_joules"],
            wraps_in_run["cpu"], wraps_in_run["dram"], refs["cpu"],
            residual_bound, residual_bound / after["total_inference_joules"],
        ))

    # ---------------------------------------------------------------- report
    print(f"{'run':24} {'changed':>7} {'total J before':>14} {'total J after':>14} {'change':>7} {'cpu wraps':>9} {'dram wraps':>10} {'P_cpu':>6}")
    for name, ch, b, a, wc, wd, p, _, _ in run_report:
        if ch:
            print(f"{name:24} {ch:7d} {b:14.0f} {a:14.0f} {100 * (a / b - 1):6.1f}% {wc:9d} {wd:10d} {p:6.1f}")
    print()
    print(f"records: {totals['records']}  changed: {totals['changed']}  runs changed: {totals['runs_changed']}/{len(run_report)}")
    print(f"uncertain corrections (review in the log): {totals['uncertain']}")
    total_before = sum(r[2] for r in run_report)
    total_after = sum(r[3] for r in run_report)
    print(f"total energy (all runs): {total_before:.0f} J -> {total_after:.0f} J (+{100 * (total_after / total_before - 1):.2f}%)")
    worst = max(run_report, key=lambda r: r[8])
    print(f"undetectable wrap error: at most {max(r[7] for r in run_report):.0f} J per run, worst relative {100 * worst[8]:.3f}% ({worst[0]})")

    if not args.apply:
        print("\nDry run: nothing was written. Re-run with --apply to rewrite the results.")
        return

    # ---------------------------------------------------------------- write
    for run_name, (files, run_data, after) in new_run_data.items():
        for path, recs in files.items():
            if any("rapl_correction" in r for r in recs):
                with open(path, "w") as f:
                    json.dump(recs, f)
        run_data.update(after)
        with open(results_dir / run_name / "run_data.json", "w") as f:
            json.dump(run_data, f, indent=4)

    run_table = results_dir / "run_table.csv"
    with open(run_table, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row["__run_id"] in new_run_data:
            after = new_run_data[row["__run_id"]][2]
            for key, value in after.items():
                if key in row:
                    row[key] = value
    with open(run_table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nWrote corrected results and {log_path}")


if __name__ == "__main__":
    main()
