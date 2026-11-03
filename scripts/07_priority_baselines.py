#!/usr/bin/env python3
"""07_priority_baselines.py -- is the ILP's gain reproducible by a one-line priority rule?

The study compares FIFO and EASY backfill with two offline solvers and one rolling solver.
That leaves a gap: if a request-only rule such as "smallest GPU gang first" already captures
most of the reordering gain, the optimiser is not doing the work -- the priority order is.
This script replays four extra online list schedules through the same slot occupancy,
capacity and placement primitives (`sched_core.run_priority_online`), so slot granularity and
tie-breaking match the rest of the study. A job that has not arrived cannot jump the queue;
each slot starts the arrived, unstarted jobs in ascending key order whenever all three
resources fit.

  priority-gpu   ascending requested GPU cards        (online: known at submit)
  priority-cpu   ascending requested CPU cores        (online: known at submit)
  priority-mem   ascending requested memory GB        (online: known at submit)
  priority-gpu-age{1,2,4}h  as priority-gpu, but a job waiting past the threshold goes
                            first, oldest-first (online; bounds the starvation tail)
  priority-short ascending actual service time        (ORACLE: not known at submit)

All but `priority-short` are deployable. The last is the strongest possible size rule and is
marked as an oracle so it cannot be quoted as an online policy.

Outputs results/metrics_priority.csv and results/priority_baselines.json; the "captured"
column is (FIFO - policy) / (FIFO - ILP-flow) on mean wait, i.e. the share of the offline
gain the rule reproduces.

Usage
  python scripts/07_priority_baselines.py
"""
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sched_core as sc   # noqa: E402

# 06 owns the backlog-idle definition; load it as a module rather than copying the formula.
_spec = importlib.util.spec_from_file_location(
    "backlog_idle", ROOT / "scripts" / "06_backlog_idle.py")
_backlog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backlog)
backlog_metrics = _backlog.backlog_metrics

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

POLICIES = {
    "priority-gpu": ("gpu_cards", False, None),
    "priority-cpu": ("cpu_cores", False, None),
    "priority-mem": ("mem_gb", False, None),
    "priority-gpu-age1h": ("gpu_cards", False, 6),
    "priority-gpu-age2h": ("gpu_cards", False, 12),
    "priority-gpu-age4h": ("gpu_cards", False, 24),
    "priority-short": ("service", True, None),
}


def main() -> int:
    baseline = pd.read_csv(RES / "metrics.csv")
    rows, summary = [], {"generated_by": "scripts/07_priority_baselines.py",
                         "note": ("All policies replay the same instance through "
                                  "sched_core.run_fifo with a substituted queue order. "
                                  "priority-short uses actual runtime and is an oracle."),
                         "instances": {}}

    for meta, inst in sc.load_instances(PROC):
        tag = f"{meta['pool']}_rho{meta['achieved_rho']:.2f}"
        b = baseline[baseline.instance == tag].set_index("policy")
        fifo_wait = float(b.loc["FIFO", "mean_wait_s"])
        fifo_p90 = float(b.loc["FIFO", "p90_wait_s"])
        fifo_max = float(b.loc["FIFO", "max_wait_s"])
        ilp_wait = float(b.loc["ILP-flow(offline)", "mean_wait_s"])
        entry = {"fifo_wait_s": fifo_wait, "ilp_flow_wait_s": ilp_wait,
                 "policies": {}}

        for name, (col, oracle, aging) in POLICIES.items():
            starts = sc.run_priority_online(inst, key=col, aging_slots=aging)
            met = sc.metrics(inst, starts)
            met.update(sc.add_deadlines(inst, starts))
            met.update(sc.packing_check(inst, starts))
            met.update(backlog_metrics(inst, starts))
            captured = ((fifo_wait - met["mean_wait_s"])
                        / (fifo_wait - ilp_wait)) if fifo_wait > ilp_wait else None
            assert met["capacity_violation_slots"] == 0, (
                f"{tag}/{name} overcommitted capacity")
            rows.append({
                "instance": tag, "pool": meta["pool"], "rho": meta["achieved_rho"],
                "policy": name, "oracle": oracle, "aging_slots": aging,
                **met,
                "wait_change_vs_fifo": round(
                    (met["mean_wait_s"] - fifo_wait) / fifo_wait, 4),
                "p90_change_vs_fifo": round(
                    (met["p90_wait_s"] - fifo_p90) / fifo_p90, 4),
                "max_change_vs_fifo": round(
                    (met["max_wait_s"] - fifo_max) / fifo_max, 4),
                "captured_ilp_wait_gain": (None if captured is None
                                            else round(float(captured), 4)),
            })
            entry["policies"][name] = {
                "oracle": oracle,
                "aging_slots": aging,
                "mean_wait_s": met["mean_wait_s"],
                "p90_wait_s": met["p90_wait_s"],
                "max_wait_s": met["max_wait_s"],
                "mean_wait_change_vs_fifo": round(
                    (met["mean_wait_s"] - fifo_wait) / fifo_wait, 4),
                "p90_wait_change_vs_fifo": round(
                    (met["p90_wait_s"] - fifo_p90) / fifo_p90, 4),
                "max_wait_change_vs_fifo": round(
                    (met["max_wait_s"] - fifo_max) / fifo_max, 4),
                "deadline_miss_rate": met["deadline_miss_rate"],
                "throughput_jobs_per_hour": met["throughput_jobs_per_hour"],
                "idle_addressable_backlog_card_hours":
                    met["idle_addressable_backlog_card_hours"],
                "capacity_violation_slots": met["capacity_violation_slots"],
                "captured_ilp_wait_gain": (None if captured is None
                                           else round(float(captured), 4)),
            }
        summary["instances"][tag] = entry

    df = pd.DataFrame(rows)
    df.to_csv(RES / "metrics_priority.csv", index=False)

    deployable = df[~df.oracle]
    plain = deployable[~deployable.policy.str.contains("age")]
    summary["aggregate"] = {
        "instances": int(df.instance.nunique()),
        "deployable_policies": sorted(deployable.policy.unique()),
        "best_deployable_per_instance": {
            tag: min(
                [p for p in entry["policies"] if not entry["policies"][p]["oracle"]],
                key=lambda p: entry["policies"][p]["mean_wait_s"])
            for tag, entry in summary["instances"].items()},
        "captured_ilp_gain_range_deployable": [
            round(float(deployable.captured_ilp_wait_gain.min()), 4),
            round(float(deployable.captured_ilp_wait_gain.max()), 4)],
        "captured_ilp_gain_range_priority_only": [
            round(float(plain.captured_ilp_wait_gain.min()), 4),
            round(float(plain.captured_ilp_wait_gain.max()), 4)],
        "p90_change_range_deployable": [
            round(float(deployable.p90_change_vs_fifo.min()), 4),
            round(float(deployable.p90_change_vs_fifo.max()), 4)],
        "p90_change_range_priority_only": [
            round(float(plain.p90_change_vs_fifo.min()), 4),
            round(float(plain.p90_change_vs_fifo.max()), 4)],
        "max_wait_change_range_deployable": [
            round(float(deployable.max_change_vs_fifo.min()), 4),
            round(float(deployable.max_change_vs_fifo.max()), 4)],
        "max_wait_change_range_priority_only": [
            round(float(plain.max_change_vs_fifo.min()), 4),
            round(float(plain.max_change_vs_fifo.max()), 4)],
        "max_wait_multiplier_max_priority_only": round(
            float((plain.max_wait_s / plain.instance.map(
                baseline[baseline.policy == "FIFO"].set_index("instance").max_wait_s)).max()), 3),
        "oracle_short_captured_ilp_gain_range": [
            round(float(df[df.oracle].captured_ilp_wait_gain.min()), 4),
            round(float(df[df.oracle].captured_ilp_wait_gain.max()), 4)],
    }
    (RES / "priority_baselines.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for tag, entry in summary["instances"].items():
        print(f"=== {tag}  FIFO {entry['fifo_wait_s']/60:.1f}m  "
              f"ILP-flow {entry['ilp_flow_wait_s']/60:.1f}m")
        for name, p in entry["policies"].items():
            print(f"  {name:15s}{' (oracle)' if p['oracle'] else '         '} "
                  f"wait {p['mean_wait_s']/60:7.1f}m  "
                  f"vs FIFO {p['mean_wait_change_vs_fifo']*100:+7.1f}%  "
                  f"captured {p['captured_ilp_wait_gain']*100 if p['captured_ilp_wait_gain'] is not None else float('nan'):6.1f}%  "
                  f"miss {p['deadline_miss_rate']*100:5.1f}%")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
