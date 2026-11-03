#!/usr/bin/env python3
"""One-command reproduction: download (optional) -> ingest -> instances -> schedulers -> report.

    python run_all.py                 # full pipeline (needs data/raw, ~2-10 min)
    python run_all.py --fetch         # also pull the three trace archives and check sha256
    python run_all.py --stage report  # only re-render tables/figures from results/metrics.csv
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
    ap.add_argument("--stage", choices=["fetch", "ingest", "instance", "solve", "backlog", "priority", "report", "notebook", "all"], default="all")
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
        for s in stages.values():
            run(s)
    else:
        run(stages[args.stage])


if __name__ == "__main__":
    main()
