#!/usr/bin/env python3
"""Per-case compute time of an FEM campaign, from its HTCondor job logs.

Each case runs as one single-process job, so the CPU time HTCondor records when the job ends is
the cost of that case. Writes a CSV beside the case files; for the grid of the article this is
results/fem/runs/2026-08-26_ning-grid32/timing_M12.csv.

    python scripts/fem_campaign_timing.py LOG_DIR [--mesh 12] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import statistics as st

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "fem" / "runs" / "2026-08-26_ning-grid32"
USAGE = re.compile(r"Usr (\d+) (\d+):(\d+):(\d+), Sys (\d+) (\d+):(\d+):(\d+)\s+-\s+Run Remote Usage")
NAME = re.compile(r"M(\d+)_D(\d+(?:\.\d+)?)_P(\d+(?:\.\d+)?)\.log$")


def cpu_hours(log: pathlib.Path) -> tuple[float, int] | None:
    """User + system CPU time of the job's last termination event, in hours."""
    text = log.read_text(errors="ignore")
    if "Job terminated" not in text:
        return None
    use = USAGE.findall(text)
    if not use:
        return None
    ud, uh, um, us, sd, sh, sm, ss = map(int, use[-1])
    secs = (ud * 86400 + uh * 3600 + um * 60 + us) + (sd * 86400 + sh * 3600 + sm * 60 + ss)
    cluster = re.search(r"^\d{3} \((\d+)\.", text, re.M)
    return secs / 3600.0, int(cluster.group(1)) if cluster else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log_dir", type=pathlib.Path)
    ap.add_argument("--mesh", type=int, default=12)
    ap.add_argument("--out", type=pathlib.Path, default=RUN / "timing_M12.csv")
    a = ap.parse_args()

    rows = []
    for log in sorted(a.log_dir.glob(f"M{a.mesh}_D*_P*.log")):
        m = NAME.search(log.name)
        t = cpu_hours(log)
        if m is None or t is None:
            raise SystemExit(f"no completed-job usage in {log}")
        rows.append({"case": log.stem, "needle_um": float(m.group(2)), "pressure_kPa": float(m.group(3)),
                     "cpu_hours": round(t[0], 4)})
    if not rows:
        raise SystemExit(f"no M{a.mesh} logs in {a.log_dir}")
    rows.sort(key=lambda r: (r["needle_um"], r["pressure_kPa"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    h = [r["cpu_hours"] for r in rows]
    print(f"  {len(h)} cases: mean {st.mean(h):.3f} h, median {st.median(h):.3f} h, "
          f"range {min(h):.2f} to {max(h):.2f} h, total {sum(h):.1f} CPU-h")
    print(f"  written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
