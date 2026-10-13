#!/usr/bin/env python3
"""Stage 6 - add the idle metrics that actually discriminate between policies.

Over a horizon long enough for every job to finish, allocated card-hours equal demanded
card-hours for *every* policy, so "GPU idle rate over the fixed window" is constant
(= 1 - rho) and says nothing about scheduling quality. The measures that do:

    idle-with-backlog        = GPU-card-hours that stood unused while at least one job was
                               already submitted and still waiting.
    addressable idle-with-    = the same, but only counted in slots where at least one of
      backlog                   those waiting jobs could actually have started (its GPU,
                               CPU *and* memory request fits the spare capacity of the slot).

The second one is the number to read. Raw idleness with a queue behind it is not always the
scheduler's fault: on the MISC pool most spare GPU cards sit in slots where CPU or memory is
already exhausted, so no queued job can use them. Head-of-line blocking creates *addressable*
waste; reordering removes it; deferring work out of the window creates it again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sched_core as sc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

POLICY_BY_FILE = {
    "FIFO": "FIFO", "EASY_backfill": "EASY-backfill", "ILP_rolling": "ILP-rolling",
    "ILP_flowoffline": "ILP-flow(offline)", "ILP_makespanoffline": "ILP-makespan(offline)",
    "LP_bound": "LP-bound",
}


def backlog_metrics(inst: sc.Instance, starts: np.ndarray) -> dict:
    starts = sc.clamp_starts(inst, starts)
    d = inst.slot_seconds
    cap = np.array([inst.capacity[r] for r in sc.RESOURCES], dtype=float)
    occ = sc.Occupancy(inst)
    for j in range(len(starts)):
        occ.reserve(j, int(starts[j]))
    arrivals_slot = np.ceil(inst.jobs["arrival_rel"].to_numpy() / d).astype(int)
    need = inst.demand
    idle_bh = 0.0
    idle_addr = 0.0
    slots_with_backlog = 0
    depths = []
    for s in range(inst.n_slots):
        waiting = np.where((arrivals_slot <= s) & (starts > s))[0]
        free = cap - occ.use[s]
        depths.append(int(waiting.size))
        if waiting.size == 0:
            continue
        slots_with_backlog += 1
        free_gpu = max(float(free[0]), 0.0)
        if free_gpu > 1e-9:
            idle_bh += free_gpu * d / 3600.0
            # could *any* queued job have used this slot, on all three dimensions?
            if bool(np.any(np.all(need[waiting] <= free + 1e-9, axis=1))):
                idle_addr += free_gpu * d / 3600.0
    window_card_hours = cap[0] * inst.n_slots * d / 3600.0
    return {
        "idle_with_backlog_card_hours": round(idle_bh, 2),
        "idle_with_backlog_share": round(idle_bh / max(window_card_hours, 1e-9), 4),
        "idle_addressable_backlog_card_hours": round(idle_addr, 2),
        "idle_addressable_backlog_share": round(idle_addr / max(window_card_hours, 1e-9), 4),
        "addressable_share_of_backlog_idle": round(idle_addr / max(idle_bh, 1e-9), 3),
        "gpu_peak_allocated_share": round(float(occ.use[:, 0].max() / cap[0]), 4),
        "slots_with_backlog": slots_with_backlog,
        "mean_queue_depth": round(float(np.mean(depths)), 3),
    }


def main() -> None:
    metas = json.loads((PROC / "instance_meta.json").read_text(encoding="utf-8"))
    by_tag = {f"{m['pool']}_rho{m['achieved_rho']:.2f}": m for m in metas}
    rows = []
    for path in sorted(RES.glob("schedule_*.csv")):
        stem = path.stem[len("schedule_"):]
        # tags look like "P100_rho0.41" and policy names may contain underscores
        # ("ILP_flowoffline"), so split on the rho marker rather than on an underscore
        hit = re.match(r"^(?P<tag>.+?_rho\d+\.\d+)_(?P<pol>.+)$", stem)
        if hit is None:
            print(f"  skip {path.name}: cannot parse '{stem}'")
            continue
        tag, pol_file = hit.group("tag"), hit.group("pol")
        policy = POLICY_BY_FILE.get(pol_file, pol_file)
        m = by_tag.get(tag)
        if m is None:
            print(f"  skip {path.name}: no instance for tag {tag}")
            continue
        inst = sc.load(PROC / m["instance_file"], m)
        sched = pd.read_csv(path)
        starts = sched["start_slot"].to_numpy()
        if len(starts) != len(inst.jobs):
            print(f"  skip {path.name}: {len(starts)} rows vs {len(inst.jobs)} jobs")
            continue
        rows.append({"instance": tag, "policy": policy, **backlog_metrics(inst, starts)})
        r = rows[-1]
        print(f"  {tag:16s} {policy:24s} backlog-idle={r['idle_with_backlog_card_hours']:8.1f} card-h "
              f"addressable={r['idle_addressable_backlog_card_hours']:8.1f} "
              f"({100*r['addressable_share_of_backlog_idle']:5.1f}% of it) "
              f"peak={r['gpu_peak_allocated_share']:.2f}")

    add = pd.DataFrame(rows)
    df = pd.read_csv(RES / "metrics.csv")
    df = df.drop(columns=[c for c in add.columns if c != "instance" and c != "policy" and c in df.columns])
    df = df.merge(add, on=["instance", "policy"], how="left")
    df.to_csv(RES / "metrics.csv", index=False)
    df.to_json(RES / "metrics.json", orient="records", indent=2)
    print(f"\nupdated metrics.csv ({len(df)} rows) with backlog/idle columns")


if __name__ == "__main__":
    main()
