#!/usr/bin/env python3
"""One-command reproduction: download (optional) -> ingest -> instances -> schedulers -> report.

    python run_all.py                 # full pipeline (needs data/raw, ~2-10 min)
    python run_all.py --fetch         # also pull the three trace archives and check sha256
    python run_all.py --stage report  # only re-render tables/figures from results/metrics.csv
    python run_all.py --sweep         # + 08_sweep.py: 4 window ranks x 3 seeds x 2 pools x 3
                                      #   load tiers (72 instances, ~1-2 h with --sweep-jobs 6);
                                      #   its window_rank=1/seed=20261013 cell reproduces the
                                      #   committed six instances as a positive control
    python run_all.py --stage sweep --sweep-no-ilp   # sweep with list policies only (~10 min)

08_sweep.py is excluded from the default `all` run: it re-runs the same policies on other
windows and seeds, which is a robustness check on the six committed instances rather than a
step that produces them.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

# Official checksums published in cluster-trace-gpu-v2020/README.md (alibaba/clusterdata).
FILES = {
    "pai_job_table.tar.gz": "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
    "pai_task_table.tar.gz": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
    "pai_machine_spec.tar.gz": "cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d",
}
BASE = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/"


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, want in FILES.items():
        p = RAW / name
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == want:
            print(f"  {name} already present and checksum matches")
            continue
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, p)
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"checksum mismatch for {name}: {got}")
        print(f"  {name} sha256 OK")


def run(script: str) -> None:
    print(f"\n### python scripts/{script}")
    r = subprocess.run([sys.executable, script], cwd=str(ROOT / "scripts"))
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with exit {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the public trace archives")
    ap.add_argument("--stage", choices=["fetch", "ingest", "instance", "solve", "backlog", "priority", "sweep", "report", "notebook", "all"], default="all")
    ap.add_argument("--sweep", action="store_true",
                    help="include 08_sweep.py in --stage all (72 instances, ~1-2 h with --jobs 6)")
    ap.add_argument("--sweep-jobs", type=int, default=6, help="parallel instances for 08_sweep.py")
    ap.add_argument("--sweep-no-ilp", action="store_true",
                    help="sweep with list policies only (no MILP; ~10 min instead of ~1-2 h)")
    args = ap.parse_args()

    if args.fetch or args.stage == "fetch":
        fetch()
    stages = {
        "ingest": "01_ingest.py",
        "instance": "02_instance.py",
        "solve": "03_optimize.py",
        "backlog": "06_backlog_idle.py",
        "priority": "07_priority_baselines.py",
        "report": "04_report.py",
        "notebook": "05_notebook.py",
    }
    if args.stage == "all":
        order = ["ingest", "instance", "solve", "backlog", "priority"]
        # 08 is a robustness sweep over other windows and seeds, not part of the six
        # committed instances, so it only runs when asked for (it is also by far the
        # longest stage).
        if args.sweep:
            order.append("sweep")
        order += ["report", "notebook"]
        for name in order:
            if name == "sweep":
                run_sweep(args)
            else:
                run(stages[name])
    elif args.stage == "sweep":
        run_sweep(args)
    else:
        run(stages[args.stage])


def run_sweep(args) -> None:
    extra = ["--jobs", str(args.sweep_jobs)]
    if args.sweep_no_ilp:
        extra.append("--no-ilp")
    print(f"\n### python scripts/08_sweep.py {' '.join(extra)}")
    r = subprocess.run([sys.executable, "08_sweep.py", *extra], cwd=str(ROOT / "scripts"))
    if r.returncode != 0:
        raise SystemExit(f"08_sweep.py failed with exit {r.returncode}")


if __name__ == "__main__":
    main()
