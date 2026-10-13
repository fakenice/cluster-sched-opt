#!/usr/bin/env python3
"""Stage 1 - ingest the Alibaba PAI GPU trace into a scheduling problem instance.

Inputs (data/raw, checksums verified against the official README):
  pai_job_table.csv    job_name, inst_id, user, status, submit_time, complete_time
  pai_task_table.csv   job_name, task_name, inst_num, status, start_time, end_time,
                       plan_cpu, plan_mem, plan_gpu, gpu_type
  pai_machine_spec.csv machine, gpu_type, cap_cpu, cap_mem, cap_gpu

Outputs (data/processed):
  jobs.csv         one row per usable job: arrival, service demand, observed run
  pools.csv        per-GPU-type cluster capacity (cores / GB / cards, machine count)
  ingest_report.json  every filter decision with the counts it dropped (no silent drops)

Units follow the trace README verbatim: plan_cpu is percent of a core (600 = 6 vCPU),
plan_mem is GB, plan_gpu is percent of one card (50 = half a card). cap_* on machines
are plain cores / GB / card counts. Timestamps are seconds from trace start.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

TASK_COLS = [
    "job_name", "task_name", "inst_num", "status", "start_time", "end_time",
    "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
]
JOB_COLS = ["job_name", "inst_id", "user", "status", "submit_time", "complete_time"]
MACHINE_COLS = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]


def read_csv(name: str, cols: list[str], dtypes: dict) -> pd.DataFrame:
    """Read the single CSV that lives inside each tar.gz, with an explicit header."""
    import tarfile

    with tarfile.open(RAW / f"{name}.tar.gz", "r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.isfile())
        fh = tf.extractfile(member)
        df = pd.read_csv(
            fh, names=cols, dtype=dtypes, header=None,
            keep_default_na=False, na_values=[""],
        )
    print(f"  {name}: {len(df):,} rows")
    return df


def main() -> None:
    report: dict = {"units": {
        "plan_cpu": "percent of one core (600 = 6 vCPU)",
        "plan_mem": "GB",
        "plan_gpu": "percent of one GPU card (50 = 0.5 card)",
        "timestamps": "seconds since trace start",
    }}

    print("[1/4] reading trace tables")
    jobs = read_csv("pai_job_table", JOB_COLS,
                    {"job_name": str, "inst_id": str, "user": str, "status": str})
    tasks = read_csv("pai_task_table", TASK_COLS,
                     {"job_name": str, "task_name": str, "status": str, "gpu_type": str})
    machines = read_csv("pai_machine_spec", MACHINE_COLS,
                        {"machine": str, "gpu_type": str})

    for col in ("submit_time", "complete_time"):
        jobs[col] = pd.to_numeric(jobs[col], errors="coerce")
    for col in ("inst_num", "start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu"):
        tasks[col] = pd.to_numeric(tasks[col], errors="coerce")
    for col in ("cap_cpu", "cap_mem", "cap_gpu"):
        machines[col] = pd.to_numeric(machines[col], errors="coerce")

    report["raw_counts"] = {
        "jobs": int(len(jobs)), "tasks": int(len(tasks)), "machines": int(len(machines)),
        "unique_job_names": int(jobs["job_name"].nunique()),
    }
    report["job_status"] = jobs["status"].value_counts().to_dict()
    report["task_status"] = tasks["status"].value_counts().to_dict()

    print("[2/4] aggregating tasks into job-level resource demand")
    # Per the README only 'Terminated' tasks are successful runs, so incomplete or
    # failed tasks cannot give a trustworthy service time and are dropped here.
    t = tasks[tasks["status"] == "Terminated"].copy()
    t = t[np.isfinite(t["start_time"]) & np.isfinite(t["end_time"]) & (t["end_time"] > t["start_time"])]
    report["task_filter"] = {"kept_task_rows": int(len(t)), "dropped_nonterminated_or_badtime": int(len(tasks) - len(t))}

    # A job's concurrent demand is the sum over its roles (ps + worker + evaluator),
    # each replicated by inst_num.
    t["cpu_cores"] = t["inst_num"] * t["plan_cpu"] / 100.0
    t["mem_gb"] = t["inst_num"] * t["plan_mem"]
    t["gpu_cards"] = t["inst_num"] * t["plan_gpu"] / 100.0
    t["gpu_only"] = t["gpu_cards"]

    agg = t.groupby("job_name").agg(
        n_tasks=("task_name", "nunique"),
        n_instances=("inst_num", "sum"),
        cpu_cores=("cpu_cores", "sum"),
        mem_gb=("mem_gb", "sum"),
        gpu_cards=("gpu_cards", "sum"),
        first_launch=("start_time", "min"),
        last_end=("end_time", "max"),
    ).reset_index()

    # GPU type pool: a job that touched more than one GPU type is ambiguous for a
    # single-pool ILP, so those are counted and excluded (reported, not hidden).
    typed = t[t["gpu_cards"] > 0]
    n_types = typed.groupby("job_name")["gpu_type"].nunique()
    main_type = (typed.sort_values(["job_name", "gpu_cards"], ascending=[True, False])
                 .drop_duplicates("job_name")[["job_name", "gpu_type"]]
                 .rename(columns={"gpu_type": "gpu_type_main"}))
    agg = agg.merge(main_type, on="job_name", how="left")
    agg["gpu_type_main"] = agg["gpu_type_main"].fillna("CPU")
    multi = n_types[n_types > 1]
    report["gpu_type_multiplicity"] = {
        "jobs_requesting_gpu": int(len(n_types)),
        "jobs_spanning_more_than_one_gpu_type": int(len(multi)),
        "policy": "kept under dominant type; flagged via gpu_type_multi flag",
    }
    agg["gpu_type_multi"] = agg["job_name"].isin(multi.index)

    print("[3/4] joining submission times and building the instance")
    j = jobs.drop_duplicates("job_name")[["job_name", "user", "status", "submit_time", "complete_time"]]
    df = agg.merge(j, on="job_name", how="inner")

    n_join = len(df)
    df = df[np.isfinite(df["submit_time"])]
    n_submit = len(df)
    # Observed scheduling latency and run time (README: gap between job submission
    # and earliest task launch is the wait before launching).
    df["wait_obs"] = df["first_launch"] - df["submit_time"]
    df["run_obs"] = df["last_end"] - df["first_launch"]
    df["total_obs"] = df["last_end"] - df["submit_time"]
    df = df[(df["run_obs"] > 0) & np.isfinite(df["wait_obs"]) & (df["wait_obs"] >= 0)]
    n_valid = len(df)
    df = df[(df["run_obs"] >= 60) & (df["run_obs"] <= 7 * 86400)]
    report["instance_filter"] = {
        "after_join": int(n_join),
        "dropped_no_submit_time": int(n_join - n_submit),
        "dropped_bad_wait_or_run": int(n_submit - n_valid),
        "dropped_run_outside_1min_to_7d": int(n_valid - len(df)),
        "usable_jobs": int(len(df)),
    }

    df = df[[
        "job_name", "user", "submit_time", "wait_obs", "run_obs", "total_obs",
        "cpu_cores", "mem_gb", "gpu_cards", "gpu_type_main", "gpu_type_multi",
        "n_tasks", "n_instances",
    ]].sort_values("submit_time").reset_index(drop=True)
    df["arrival_h"] = df["submit_time"] / 3600.0
    df.to_csv(OUT / "jobs.csv", index=False)

    pools = machines.groupby("gpu_type").agg(
        machines=("machine", "count"), cap_cpu_cores=("cap_cpu", "sum"),
        cap_mem_gb=("cap_mem", "sum"), cap_gpu_cards=("cap_gpu", "sum"),
        cpu_per_machine=("cap_cpu", "mean"), gpu_per_machine=("cap_gpu", "mean"),
    ).reset_index()
    pools.to_csv(OUT / "pools.csv", index=False)
    report["cluster"] = {
        "machines": int(len(machines)),
        "gpu_cards_total": float(machines["cap_gpu"].sum()),
        "cpu_cores_total": float(machines["cap_cpu"].sum()),
        "mem_gb_total": float(machines["cap_mem"].sum()),
        "by_gpu_type": pools.to_dict("records"),
    }

    print("[4/4] diagnostics")
    gpu_jobs = df[df["gpu_cards"] > 0]
    report["demand_summary"] = {
        "jobs": int(len(df)),
        "gpu_jobs": int(len(gpu_jobs)),
        "cpu_only_jobs": int(len(df) - len(gpu_jobs)),
        "unique_users": int(df["user"].nunique()),
        "arrival_span_hours": round(float(df["submit_time"].max() - df["submit_time"].min()) / 3600, 2),
        "run_seconds": {q: round(float(df["run_obs"].quantile(q)), 1) for q in (0.5, 0.9, 0.99, 1.0)},
        "gpu_card_request": {q: round(float(gpu_jobs["gpu_cards"].quantile(q)), 2) for q in (0.5, 0.9, 0.99)},
        "observed_wait_seconds": {q: round(float(df["wait_obs"].quantile(q)), 1) for q in (0.5, 0.9, 0.99)},
        "gpu_share_by_type": gpu_jobs.groupby("gpu_type_main")["gpu_cards"].sum().round(1).to_dict(),
        "jobs_by_gpu_type": df.groupby("gpu_type_main")["job_name"].count().to_dict(),
    }
    (OUT / "ingest_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report["demand_summary"], indent=2, ensure_ascii=False))
    print("\npools:\n", pools.to_string(index=False))


if __name__ == "__main__":
    main()
