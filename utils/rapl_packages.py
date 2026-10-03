"""
Reports the power drawn by every Intel RAPL domain (all CPU packages and their sub-domains).

evaluate_from_local.py only reads package 0 (intel-rapl:0) and its DRAM sub-domain, but the
server has two CPU packages. This script shows what the second package (intel-rapl:1) and its
DRAM consume, both when the system is idle and while an evaluation is running.

Usage:
    python utils/rapl_packages.py                 # 30 seconds, one line every 5 seconds
    python utils/rapl_packages.py --duration 600  # e.g. while an evaluation runs in another shell
"""

import argparse
import glob
import os
import time

RAPL_DIR = "/sys/class/powercap"


def read_int(path):
    with open(path, "r") as f:
        return int(f.read().strip())


def read_text(path):
    with open(path, "r") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=30, help="Total time to measure, in seconds.")
    parser.add_argument("--interval", type=float, default=5, help="Time between reports, in seconds.")
    args = parser.parse_args()

    domains = []
    for path in sorted(glob.glob(os.path.join(RAPL_DIR, "intel-rapl:*"))):
        domains.append({
            "id": os.path.basename(path),
            "name": read_text(os.path.join(path, "name")),
            "energy": os.path.join(path, "energy_uj"),
            "max_range": read_int(os.path.join(path, "max_energy_range_uj")),
        })

    labels = [f"{d['id']} ({d['name']})" for d in domains]
    print(f"{'time (s)':>8}  " + "  ".join(f"{label:>26}" for label in labels))

    start_time = time.time()
    first = previous = [read_int(d["energy"]) for d in domains]
    previous_time = start_time
    totals = [0] * len(domains)

    while time.time() - start_time < args.duration:
        time.sleep(args.interval)
        now = time.time()
        current = [read_int(d["energy"]) for d in domains]

        watts = []
        for i, d in enumerate(domains):
            delta = current[i] - previous[i]
            if delta < 0:
                delta += d["max_range"]
            totals[i] += delta
            watts.append(delta / 1e6 / (now - previous_time))

        print(f"{now - start_time:8.1f}  " + "  ".join(f"{w:24.2f} W" for w in watts))
        previous, previous_time = current, now

    elapsed = previous_time - start_time
    print(f"{'average':>8}  " + "  ".join(f"{t / 1e6 / elapsed:24.2f} W" for t in totals))


if __name__ == "__main__":
    main()
