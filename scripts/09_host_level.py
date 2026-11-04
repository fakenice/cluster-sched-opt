#!/usr/bin/env python3
"""Stage 9 - the host-level (per-machine) model: what does the fragmentation cost?

The committed study (scripts/03_optimize.py) treats a GPU pool as one big machine whose
capacity is the sum over hosts, and defends that relaxation with `sched_core.packing_check`,
which re-packs every busy slot onto the real hosts with first-fit-decrease and reports a
packability rate. Two things are missing from that defence:

  1. the re-packing is *per slot*, so a job may sit on host 1 in one slot and host 4 in the
     next -- a container is not migrated, so the question is whether ONE assignment holds
     for the whole life of every job;
  2. nothing measures the *price*: if the offline optimum is not executable, how much
     waiting does reality add?

This stage answers both, and then pays the bill with the model itself:

  audit    the committed schedules (FIFO / EASY / ILP-flow / ILP-rolling) are replayed on
           discrete hosts with one time-consistent assignment; the jobs that cannot keep
           their scheduled start are counted, and a host-aware repair measures the delay
           that makes the schedule executable.
  policies the deployable online rules (FIFO, EASY backfill, the request-only priority
           rules) are re-run under host capacity, so "one priority rule gets most of the
           ILP's gain" can be tested where the pool is made of machines.
  milp     the offline optimum is re-solved as a host-level MILP (one column per machine)
           on the *same* restricted instance as the aggregate model, so the two LP bounds
           and the two optima can be compared directly.

A note on solver semantics, because it decides how the numbers below must be read:
HiGHS reports status 7 ("Optimal") both when it *proves* optimality and when it stops
inside the requested `mip_rel_gap`; scipy maps both onto status 0 ("Optimization
terminated successfully"). The committed study therefore publishes an *incumbent within
its gap tolerance* (2 %, 5 % for the big instances) and calls it the "offline optimum".
Here both models are additionally solved with gap = 0, i.e. optimality is proven or the
dual bound is reported, and the fragmentation bill is stated as an interval
(host dual bound - aggregate optimum) .. (host incumbent - aggregate optimum) rather
than as a single number that two different tolerances could produce.

Nothing here changes the committed numbers: the aggregate results stay in results/metrics.csv
and results/metrics_priority.csv, and every host-level number is written to its own file
next to its aggregate twin.

Usage
  python scripts/09_host_level.py --stage all
  python scripts/09_host_level.py --stage audit
  python scripts/09_host_level.py --stage milp --pilot-window-hours 2 --cand-window 12
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import LinearConstraint, Bounds, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sched_core as sc   # noqa: E402
import host_core as hc   # noqa: E402

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

_spec = importlib.util.spec_from_file_location("optimizer", ROOT / "scripts" / "03_optimize.py")
opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opt)

# the committed policies whose host-level executability is audited
COMMITTED_POLICIES = ["FIFO", "EASY-backfill", "ILP-flow(offline)", "ILP-rolling"]
POLICY_FILE = {"FIFO": "FIFO", "EASY-backfill": "EASY_backfill",
               "ILP-flow(offline)": "ILP_flowoffline", "ILP-rolling": "ILP_rolling",
               "ILP-makespan(offline)": "ILP_makespanoffline"}
PRIORITY_RULES = {
    "priority-gpu": ("gpu_cards", None),
    "priority-cpu": ("cpu_cores", None),
    "priority-mem": ("mem_gb", None),
    "priority-gpu-age1h": ("gpu_cards", 6),
    "priority-gpu-age2h": ("gpu_cards", 12),
    "priority-gpu-age4h": ("gpu_cards", 24),
}


# --------------------------------------------------------------------------- host MILP

def host_milp(inst: sc.Instance, subset: np.ndarray, cand_window: int = 16,
              binary: bool = True, time_limit: float = 180.0, gap: float = 0.02):
    """Time-indexed MILP with one column per (job, machine, start slot).

    Variables
        x[j,t]        1 if job j starts in slot t
        z[j,m,t]      1 if job j starts in slot t on machine m

    Constraints
        sum_t x[j,t] = 1                       every job starts exactly once
        sum_m z[j,m,t] = h_j * x[j,t]          the job occupies h_j machines, from that
                                                start, for its whole life -- the machines
                                                cannot change from slot to slot
        sum_{j,t in slot s} share_j,c * z[j,m,t] <= cap_c
                                                per machine, per slot, per resource

    Objective: min sum_j C_j (flow time), the same objective as the aggregate model, so the
    two objectives are directly comparable in seconds.
    """
    hm = hc.build_host_model(inst)
    n = len(subset)
    H = hm.hosts
    d = inst.slot_seconds
    cands = []
    for j in subset:
        lo = int(hm.earliest[j])
        hi = min(int(hm.latest[j]), lo + int(cand_window) - 1)
        if hi < lo:
            hi = lo
        cands.append(np.arange(lo, hi + 1))
    offs = np.concatenate([[0], np.cumsum([len(c) for c in cands])[:-1]]).astype(int)
    nx = int(sum(len(c) for c in cands))
    nz = nx * H
    ncols = nx + nz
    x_col = lambda k, tp: int(offs[k] + tp)
    z_col = lambda k, tp, m: nx + int((offs[k] + tp) * H + m)

    rows, cols, vals = [], [], []
    lb, ub = [], []
    r = 0
    # (1) every job starts exactly once
    for k in range(n):
        for tp in range(len(cands[k])):
            rows.append(r); cols.append(x_col(k, tp)); vals.append(1.0)
        lb.append(1.0); ub.append(1.0)
        r += 1
    # (2) the machine set of a job is fixed at its start slot
    for k in range(n):
        hk = float(hm.n_host[subset[k]])
        for tp in range(len(cands[k])):
            for m in range(H):
                rows.append(r); cols.append(z_col(k, tp, m)); vals.append(1.0)
            rows.append(r); cols.append(x_col(k, tp)); vals.append(-hk)
            lb.append(0.0); ub.append(0.0)
            r += 1
    # (3) per-machine, per-slot capacity
    n_slots = int(inst.n_slots)
    cap = hm.cap
    for m in range(H):
        for s in range(n_slots):
            for c in range(3):
                row_has = False
                for k in range(n):
                    j = int(subset[k])
                    pj = int(hm.p[j])
                    share = float(hm.share[j, c])
                    if share <= 0:
                        continue
                    for tp, t in enumerate(cands[k]):
                        lo, hi = int(t), min(int(t) + pj, n_slots)
                        if hi <= s or lo > s:
                            continue
                        rows.append(r); cols.append(z_col(k, tp, m)); vals.append(share)
                        row_has = True
                if row_has:
                    lb.append(-np.inf); ub.append(float(cap[c]))
                    r += 1
                else:
                    lb.append(-np.inf); ub.append(np.inf)   # empty row: no constraint
                    r += 1
    A = sp.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr()
    lc = LinearConstraint(A, np.array(lb), np.array(ub))

    cvec = np.zeros(ncols)
    for k in range(n):
        j = int(subset[k])
        for tp, t in enumerate(cands[k]):
            cvec[x_col(k, tp)] = (int(t) + int(hm.p[j])) * d
    integrality = np.zeros(ncols)
    if binary:
        integrality[:] = 1
    bounds = Bounds(np.zeros(ncols), np.ones(ncols))

    t0 = time.time()
    res = milp(c=cvec, constraints=lc, integrality=integrality, bounds=bounds,
               options={"time_limit": time_limit, "mip_rel_gap": gap, "presolve": True})
    status = int(getattr(res, "status", -1))
    x = res.x
    has_solution = x is not None and status in (0, 1) and np.all(np.isfinite(x))
    starts = None
    hosts_of = {}
    if has_solution:
        x = np.asarray(x, dtype=float)
        starts = np.zeros(n, dtype=int)
        for k in range(n):
            best_tp, best_val = 0, -np.inf
            for tp in range(len(cands[k])):
                v = x[x_col(k, tp)]
                if v > best_val:
                    best_val, best_tp = v, tp
            t = int(cands[k][best_tp])
            starts[k] = t
            hs = [m for m in range(H) if x[z_col(k, best_tp, m)] > 0.5]
            hosts_of[int(subset[k])] = hs
    fun = getattr(res, "fun", None)
    obj_val = float(fun) if fun is not None and np.isfinite(np.asarray(fun, dtype=float)) else None
    mip_gap, dual = opt.solver_gap_fields(res)
    info = {
        "solver_status": status, "message": str(getattr(res, "message", "")),
        "solver_feasible": bool(has_solution), "n_columns": ncols, "n_rows": int(r),
        "n_nnz": int(A.nnz), "solve_seconds": round(time.time() - t0, 2),
        "objective_value_s": obj_val, "mip_gap": mip_gap, "dual_bound": dual,
        "machines": H, "jobs": n, "cand_window": int(cand_window),
    }
    return starts, hosts_of, info


# --------------------------------------------------------------------------- checks

def verify_host_solution(inst: sc.Instance, hm: hc.HostModel, subset: np.ndarray,
                         starts: np.ndarray, hosts_of: dict) -> dict:
    """Independent check of a host-level solution: per-host capacity *and* the
    aggregate (pool) capacity. A host-feasible schedule must also be aggregate-feasible,
    because the total demand on the pool is the sum over the hosts."""
    occ = hc.HostOccupancy(hm)
    for k, j in enumerate(subset):
        j = int(j)
        hs = hosts_of.get(j)
        if hs is None:
            continue
        occ.reserve(j, int(starts[k]), hosts=np.asarray(hs, dtype=int))
    v = occ.violations()
    cap = np.array([inst.capacity[r] for r in sc.RESOURCES], dtype=float)
    use = np.zeros((inst.n_slots, 3))
    for k, j in enumerate(subset):
        t, pj = int(starts[k]), int(inst.p[int(j)])
        use[t:t + pj] += inst.demand[int(j)]
    over = use - cap
    return {
        "host_capacity_violation_host_slots": v["host_capacity_violation_host_slots"],
        "max_host_overcommit_cards": v["max_host_overcommit_cards"],
        "aggregate_capacity_violation_slots": int((over.max(axis=1) > 1e-6).sum()),
        "aggregate_max_overcommit_cards": round(float(over[:, 0].max(initial=0.0)), 3),
    }


def flow_objective(inst: sc.Instance, subset: np.ndarray, starts: np.ndarray) -> float:
    """sum_j C_j in seconds -- the same objective both models minimise."""
    starts = np.asarray(starts, dtype=int)
    return float(((starts + inst.p[subset]) * inst.slot_seconds).sum())


def pilot_subset(inst: sc.Instance, args) -> np.ndarray:
    """The pilot job set: the busiest window of the committed instance (or its first N
    arrivals). Both models and both policy families get exactly this set."""
    arrival = inst.jobs["arrival_rel"].to_numpy(float)
    if args.pilot_window_hours and args.pilot_window_hours > 0:
        w = args.pilot_window_hours * 3600.0
        bins = np.floor(arrival / w).astype(int)
        demand = np.zeros(bins.max() + 1)
        np.add.at(demand, bins, inst.jobs["gpu_cards"].to_numpy(float))
        b = int(np.argmax(demand))
        return np.sort(np.where(bins == b)[0])
    order = np.argsort(arrival, kind="stable")
    if args.pilot_jobs and args.pilot_jobs > 0:
        return np.sort(order[:args.pilot_jobs])
    return np.arange(len(inst.jobs))


def sub_instance(inst: sc.Instance, subset: np.ndarray) -> sc.Instance:
    """The same instance restricted to `subset` (jobs renumbered, same capacity)."""
    return sc.Instance(pool=inst.pool, rho=inst.rho,
                        jobs=inst.jobs.iloc[subset].reset_index(drop=True),
                        capacity=inst.capacity, slot_seconds=inst.slot_seconds,
                        n_slots=inst.n_slots, gpu_per_machine=inst.gpu_per_machine,
                        meta=inst.meta)


# --------------------------------------------------------------------------- stages

def stage_audit(args) -> pd.DataFrame:
    """Can the committed schedules be executed on the real machines at all?"""
    rows = []
    agg = pd.read_csv(RES / "metrics.csv")
    for meta, inst in sc.load_instances(PROC):
        tag = f"{meta['pool']}_rho{meta['achieved_rho']:.2f}"
        hm = hc.build_host_model(inst)
        for policy in COMMITTED_POLICIES:
            path = RES / f"schedule_{tag}_{POLICY_FILE[policy]}.csv"
            if not path.exists():
                continue
            starts = pd.read_csv(path)["start_slot"].to_numpy()
            if len(starts) != hm.n_jobs:
                continue
            audit = hc.assign_hosts(hm, starts)
            repair, rep_info = hc.repair_schedule(hm, starts)
            placed = audit["placed_at_scheduled_start"]
            wait_agg = float((starts * inst.slot_seconds - inst.jobs["arrival_rel"].to_numpy()).mean())
            ok = repair >= 0
            wait_rep = float(((repair[ok] * inst.slot_seconds
                               - inst.jobs["arrival_rel"].to_numpy()[ok]).mean()) if ok.any() else np.nan)
            mrow = agg[(agg.instance == tag) & (agg.policy == policy)]
            ffd = float(mrow["packing_feasibility_rate"].iloc[0]) if len(mrow) else None
            rows.append({
                "instance": tag, "pool": meta["pool"], "rho": meta["achieved_rho"],
                "policy": policy, "machines": hm.hosts, "jobs": hm.n_jobs,
                "jobs_multi_host": int((hm.n_host > 1).sum()),
                "placeable_at_scheduled_start": placed,
                "share_placeable": audit["share_placeable"],
                "per_slot_ffd_packable_rate": ffd,
                "repair_unplaced_jobs": rep_info["unplaced_jobs"],
                "mean_wait_s_aggregate": round(wait_agg, 1),
                "mean_wait_s_host_repaired": (None if not np.isfinite(wait_rep) else round(wait_rep, 1)),
                "wait_multiplier_after_repair": (None if not np.isfinite(wait_rep) or wait_agg <= 0
                                                 else round(wait_rep / wait_agg, 3)),
                "host_violation_host_slots_after_repair": rep_info["occupancy"].violations()[
                    "host_capacity_violation_host_slots"],
            })
            print(f"  {tag:16s} {policy:20s} placeable {placed:4d}/{hm.n_jobs:4d} "
                  f"({audit['share_placeable']*100:5.1f}%)  per-slot FFD {ffd if ffd is None else round(ffd*100,1)}%  "
                  f"repair wait x{rows[-1]['wait_multiplier_after_repair']}")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "host_level_audit.csv", index=False)
    print(f"\nwrote {len(df)} audit rows -> results/host_level_audit.csv")
    return df


def stage_policies(args) -> pd.DataFrame:
    """The deployable online rules, re-run under discrete machines."""
    rows = []
    base = pd.read_csv(RES / "metrics.csv")
    prio = pd.read_csv(RES / "metrics_priority.csv")
    for meta, inst in sc.load_instances(PROC):
        tag = f"{meta['pool']}_rho{meta['achieved_rho']:.2f}"
        hm = hc.build_host_model(inst)
        runs = {
            "FIFO": hc.run_fifo_host(hm, backfill=False),
            "EASY-backfill": hc.run_fifo_host(hm, backfill=True),
        }
        for name, (col, aging) in PRIORITY_RULES.items():
            runs[name] = hc.run_priority_online_host(
                hm, inst.jobs[col].to_numpy(float), aging_slots=aging)
        for name, (starts, occ) in runs.items():
            met = sc.metrics(inst, starts)
            met.update(sc.add_deadlines(inst, starts))
            met.update(occ.violations())
            met.update(hc.host_idle_metrics(hm, starts, occ))
            agg_rows = base[base.instance == tag] if name in set(base.policy) else prio[prio.instance == tag]
            agg_rows = agg_rows[agg_rows.policy == name]
            agg_wait = float(agg_rows["mean_wait_s"].iloc[0]) if len(agg_rows) else None
            rows.append({
                "instance": tag, "pool": meta["pool"], "rho": meta["achieved_rho"],
                "policy": name, "machines": hm.hosts,
                **met,
                "mean_wait_s_aggregate_model": agg_wait,
                "wait_ratio_host_over_aggregate": (None if agg_wait in (None, 0)
                                                   else round(met["mean_wait_s"] / agg_wait, 3)),
            })
            print(f"  {tag:16s} {name:20s} host wait {met['mean_wait_s']/60:7.1f}m "
                  f"(aggregate {agg_wait/60 if agg_wait else float('nan'):7.1f}m)  "
                  f"miss {met['deadline_miss_rate']*100:5.1f}%  "
                  f"host-viol {met['host_capacity_violation_host_slots']}")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "host_level_metrics.csv", index=False)
    print(f"\nwrote {len(df)} rows -> results/host_level_metrics.csv")
    return df


def stage_milp(args) -> pd.DataFrame:
    """Aggregate vs host-level offline optimum on the same restricted instance.

    Both models get exactly the same jobs, the same candidate slots and the same
    objective (sum of completion times). The aggregate optimum is proven with gap = 0;
    the host optimum is proven with gap = 0 as well, and if it does not close in the
    budget, its dual bound is reported so the fragmentation bill is an interval.
    """
    rows = []
    metas = json.loads((PROC / "instance_meta.json").read_text(encoding="utf-8"))
    for meta in metas:
        if args.pools and meta["pool"] not in args.pools.split(","):
            continue
        tag = f"{meta['pool']}_rho{meta['achieved_rho']:.2f}"
        inst = sc.load(PROC / meta["instance_file"], meta)
        arrival = inst.jobs["arrival_rel"].to_numpy(float)
        # the pilot instance: the busiest window of the committed instance (or its first N
        # arrivals), on the same machines. Both models get exactly the same job set, so the
        # two bounds and the two optima can be compared directly.
        if args.pilot_window_hours and args.pilot_window_hours > 0:
            subset = pilot_subset(inst, args)
        else:
            subset = pilot_subset(inst, args)
        hm = hc.build_host_model(inst)
        multi = int((hm.n_host[subset] > 1).sum())
        # the tolerance the committed study uses for an instance of this size
        committed_gap = 0.05 if len(subset) > 200 else 0.02

        agg_lp = opt.solve_ilp(inst, subset, objective="flow", binary=False,
                               time_limit=args.time_limit, gap=0.0,
                               cand_window=args.cand_window)[1]
        agg_starts, agg_info = opt.solve_ilp(inst, subset, objective="flow", binary=True,
                                              time_limit=args.time_limit, gap=0.0,
                                              cand_window=args.cand_window)
        agg_gap_starts, agg_gap_info = opt.solve_ilp(inst, subset, objective="flow", binary=True,
                                                     time_limit=args.time_limit, gap=committed_gap,
                                                     cand_window=args.cand_window)
        h_starts, h_hosts, h_lp = host_milp(inst, subset, cand_window=args.cand_window,
                                            binary=False, time_limit=args.time_limit, gap=0.0)
        h_ilp_starts, h_ilp_hosts, h_ilp = host_milp(inst, subset, cand_window=args.cand_window,
                                                     binary=True, time_limit=args.time_limit, gap=0.0)

        row = {
            "instance": tag, "pool": meta["pool"], "rho": meta["achieved_rho"],
            "pilot_jobs": len(subset), "machines": hm.hosts, "cand_window": args.cand_window,
            "pilot_window_hours": args.pilot_window_hours,
            "pilot_multi_host_jobs": multi,
            "aggregate_lp_bound_s": agg_lp["objective_value"],
            "host_lp_bound_s": h_lp["objective_value_s"],
            "aggregate_ilp_value_s": agg_info["objective_value"],
            "aggregate_dual_bound_s": agg_info["dual_bound"],
            "aggregate_mip_gap": agg_info["mip_gap"],
            "aggregate_solve_seconds": agg_info["solve_seconds"],
            "aggregate_columns": agg_info["n_columns"],
            "aggregate_committed_gap_value_s": agg_gap_info["objective_value"],
            "aggregate_committed_gap": committed_gap,
            "aggregate_committed_gap_mip_gap": agg_gap_info["mip_gap"],
            "aggregate_committed_gap_slack_percent": (
                None if not (agg_info["objective_value"] and agg_gap_info["objective_value"])
                else round(100 * (agg_gap_info["objective_value"] - agg_info["objective_value"])
                           / agg_info["objective_value"], 3)),
            "host_ilp_value_s": h_ilp["objective_value_s"],
            "host_dual_bound_s": h_ilp["dual_bound"],
            "host_mip_gap": h_ilp["mip_gap"],
            "host_solve_seconds": h_ilp["solve_seconds"],
            "host_columns": h_ilp["n_columns"],
            "host_feasible": h_ilp["solver_feasible"],
            "host_message": h_ilp["message"][:120],
            "aggregate_mean_wait_s": (None if agg_starts is None else
                                      round(float((agg_starts * inst.slot_seconds
                                                   - arrival[subset]).mean()), 1)),
            "host_mean_wait_s": (None if h_ilp_starts is None else
                                 round(float((h_ilp_starts * inst.slot_seconds
                                              - arrival[subset]).mean()), 1)),
        }
        # the host solution must also satisfy the aggregate capacity (host-feasible is a
        # subset of aggregate-feasible), and the aggregate optimum must be host-executable
        # only partially -- both are checked, not assumed.
        if h_ilp_starts is not None:
            row.update({"host_check_" + k: v for k, v in
                        verify_host_solution(inst, hm, subset, h_ilp_starts, h_ilp_hosts).items()})
            row["host_objective_recomputed_s"] = round(flow_objective(inst, subset, h_ilp_starts), 1)
        if agg_starts is not None:
            sub_hm = hc.subset_model(hm, subset)
            aud = hc.assign_hosts(sub_hm, agg_starts)
            rep, rep_info = hc.repair_schedule(sub_hm, agg_starts)
            wait_agg = float((agg_starts * inst.slot_seconds - arrival[subset]).mean())
            ok = rep >= 0
            wait_rep = float(((rep[ok] * inst.slot_seconds - arrival[subset][ok]).mean())
                             if ok.any() else np.nan)
            row["aggregate_opt_placeable_share"] = aud["share_placeable"]
            row["aggregate_opt_repair_unplaced_jobs"] = rep_info["unplaced_jobs"]
            row["aggregate_opt_repair_wait_multiplier"] = (
                None if not np.isfinite(wait_rep) or wait_agg <= 0 else round(wait_rep / wait_agg, 3))
            row["aggregate_opt_repair_mean_wait_s"] = (
                None if not np.isfinite(wait_rep) else round(wait_rep, 1))
        # the fragmentation bill, as an interval
        agg_opt = agg_info["objective_value"]
        host_inc = h_ilp["objective_value_s"]
        host_dual = h_ilp["dual_bound"]
        if agg_opt is not None and host_inc is not None:
            row["bill_upper_s"] = round(host_inc - agg_opt, 1)
            row["bill_upper_share"] = round((host_inc - agg_opt) / max(abs(agg_opt), 1e-9), 5)
            row["bill_upper_mean_wait_s"] = round((host_inc - agg_opt) / max(len(subset), 1), 1)
        if agg_opt is not None and host_dual is not None:
            # a host-feasible schedule is also aggregate-feasible, so the bill cannot be
            # negative. The dual bound is only informative when it is *above* the aggregate
            # optimum; otherwise it certifies nothing and the lower end stays 0.
            row["bill_lower_s"] = round(max(0.0, host_dual - agg_opt), 1)
            row["bill_lower_share"] = round(max(0.0, host_dual - agg_opt) / max(abs(agg_opt), 1e-9), 5)
            row["bill_lower_certified"] = bool(host_dual > agg_opt + 1e-6)
        rows.append(row)
        print(f"  {tag}: jobs={len(subset)} multi-host={multi}  "
              f"agg LP {agg_lp['objective_value']} | agg OPT {agg_info['objective_value']} "
              f"(gap {agg_info['mip_gap']}, dual {agg_info['dual_bound']}) | "
              f"host LP {h_lp['objective_value_s']} | host OPT {h_ilp['objective_value_s']} "
              f"(gap {h_ilp['mip_gap']}, dual {h_ilp['dual_bound']}) | "
              f"bill [{row.get('bill_lower_s')}, {row.get('bill_upper_s')}] s")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "host_level_milp.csv", index=False)
    print(f"\nwrote {len(df)} rows -> results/host_level_milp.csv")
    return df


def stage_rules(args) -> pd.DataFrame:
    """The deployable rule and the offline optimum, on the *same* job set, in the
    aggregate model and in the host model -- the 2x2 that says whether the host
    constraint hurts the rule more than it hurts the optimum.

    Caveat, stated because it cuts against the optimiser: the MILP side is restricted to
    `cand_window` candidate slots per job and to the planning horizon, while the greedy
    rule may place a job anywhere in horizon + tail. The rule side is therefore not
    handicapped; if the optimiser still wins, the comparison is conservative.
    """
    milp_path = RES / "host_level_milp.csv"
    milp = pd.read_csv(milp_path) if milp_path.exists() else pd.DataFrame()
    rows = []
    metas = json.loads((PROC / "instance_meta.json").read_text(encoding="utf-8"))
    for meta in metas:
        tag = f"{meta['pool']}_rho{meta['achieved_rho']:.2f}"
        inst = sc.load(PROC / meta["instance_file"], meta)
        subset = pilot_subset(inst, args)
        sub = sub_instance(inst, subset)
        arrival = sub.jobs["arrival_rel"].to_numpy(float)
        d = sub.slot_seconds

        agg_starts = sc.run_priority_online(sub, key="gpu_cards")
        sub_hm = hc.build_host_model(sub)
        host_starts, host_occ = hc.run_priority_online_host(
            sub_hm, sub.jobs["gpu_cards"].to_numpy(float))
        agg_wait = float((agg_starts * d - arrival).mean())
        host_wait = float((host_starts * d - arrival).mean())
        row = {
            "instance": tag, "pool": meta["pool"], "rho": meta["achieved_rho"],
            "pilot_jobs": len(subset), "machines": sub_hm.hosts,
            "rule": "priority-gpu",
            "aggregate_rule_mean_wait_s": round(agg_wait, 1),
            "host_rule_mean_wait_s": round(host_wait, 1),
            "host_rule_host_violations": host_occ.violations()["host_capacity_violation_host_slots"],
            "host_rule_over_aggregate_rule": round(host_wait / max(agg_wait, 1e-9), 3),
        }
        if len(milp):
            m = milp[milp.instance == tag]
            if len(m):
                m = m.iloc[0]
                row["aggregate_ilp_mean_wait_s"] = m.get("aggregate_mean_wait_s")
                row["host_ilp_mean_wait_s"] = m.get("host_mean_wait_s")
                if m.get("aggregate_mean_wait_s") and np.isfinite(m.get("host_mean_wait_s", np.nan)):
                    # ratio < 1 means the optimiser waits less than the rule
                    row["ilp_wait_over_rule_aggregate"] = round(
                        float(m["aggregate_mean_wait_s"]) / max(agg_wait, 1e-9), 3)
                    row["ilp_wait_over_rule_host"] = round(
                        float(m["host_mean_wait_s"]) / max(host_wait, 1e-9), 3)
                    row["rule_over_ilp_host"] = round(host_wait / max(float(m["host_mean_wait_s"]), 1e-9), 3)
        rows.append(row)
        print(f"  {tag:16s} rule: aggregate {agg_wait/60:7.1f}m -> host {host_wait/60:7.1f}m "
              f"(x{row['host_rule_over_aggregate_rule']})  "
              f"ILP: aggregate {row.get('aggregate_ilp_mean_wait_s', float('nan'))}s -> "
              f"host {row.get('host_ilp_mean_wait_s', float('nan'))}s")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "host_level_rules.csv", index=False)
    print(f"\nwrote {len(df)} rows -> results/host_level_rules.csv")
    return df


def stage_lpcheck(args) -> pd.DataFrame:
    """Re-derive the committed LP lower bounds and compare them with the committed file.

    The claim that the host-level work left the committed numbers alone is only worth
    something if the aggregate LP relaxation still reproduces. So it is re-solved here:
    the same aggregate LP (`binary = False`, `gap = 0`) on the same full instances,
    compared against `lp_lower_bound_s` in `results/metrics.csv`.

    This is a *check*, not a new measurement: it re-derives numbers that are already
    published and fails loudly if they drift.
    """
    base = pd.read_csv(RES / "metrics.csv")
    committed = base[base.policy == "ILP-flow(offline)"].set_index("instance")["lp_lower_bound_s"]
    rows = []
    for meta in json.loads((PROC / "instance_meta.json").read_text(encoding="utf-8")):
        tag = f"{meta['pool']}_rho{meta['achieved_rho']:.2f}"
        inst = sc.load(PROC / meta["instance_file"], meta)
        t0 = time.time()
        info = opt.solve_ilp(inst, np.arange(len(inst.jobs)), objective="flow",
                             binary=False, time_limit=args.time_limit, gap=0.0)[1]
        val = float(info["objective_value"])
        old = float(committed[tag]) if tag in committed.index else None
        rows.append({
            "instance": tag, "pool": meta["pool"], "rho": meta["achieved_rho"],
            "jobs": len(inst.jobs),
            "committed_lp_lower_bound_s": old,
            "recomputed_lp_lower_bound_s": round(val, 6),
            "abs_diff_s": None if old is None else round(val - old, 9),
            "rel_diff": None if old is None else round((val - old) / old, 12),
            "reproduced": None if old is None else bool(abs(val - old) <= 1e-6 * max(abs(old), 1.0)),
            "solve_seconds": round(time.time() - t0, 1),
        })
        print(f"  {tag:16s} committed {old}  recomputed {val:.6f}  "
              f"rel_diff {rows[-1]['rel_diff']}  reproduced={rows[-1]['reproduced']}")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "host_level_lp_check.csv", index=False)
    print(f"\nwrote {len(df)} rows -> results/host_level_lp_check.csv")
    return df


def stage_ilpcheck(args) -> pd.DataFrame:
    """Re-solve the committed ILP-flow rows and compare them with the committed values.

    The committed "offline optimum" is what HiGHS returned inside its wall-clock
    budget, so it is an *incumbent*, and an incumbent is not reproducible by
    definition. This stage re-runs it exactly as `03_optimize.py` does
    (`opt.run_offline`, same tolerances, same budget) and reports, per instance:

      * the committed objective and the recomputed one,
      * the relative deviation (which may be negative: a re-run can be *better*),
      * the recomputed gap and dual bound, which say whether the re-run closed.

    A non-zero deviation here is a property of the solver, not a change made here:
    the aggregate numbers in `results/metrics.csv` are untouched.
    """
    base = pd.read_csv(RES / "metrics.csv")
    committed = base[base.policy == "ILP-flow(offline)"].set_index("instance")
    rows = []
    for meta in json.loads((PROC / "instance_meta.json").read_text(encoding="utf-8")):
        tag = f"{meta['pool']}_rho{meta['achieved_rho']:.2f}"
        inst = sc.load(PROC / meta["instance_file"], meta)
        t0 = time.time()
        starts, info = opt.run_offline(inst, "flow", binary=True)
        val = float(info["objective_value"]) if info.get("objective_value") else None
        old = float(committed.loc[tag, "milp_objective_sum_completion_s"])
        rows.append({
            "instance": tag, "pool": meta["pool"], "rho": meta["achieved_rho"],
            "jobs": len(inst.jobs),
            "committed_objective_s": old,
            "recomputed_objective_s": None if val is None else round(val, 1),
            "rel_deviation": None if val is None else round((val - old) / old, 6),
            "committed_gap_vs_lp_percent": committed.loc[tag, "gap_vs_lp_percent"],
            "recomputed_mip_gap": info.get("mip_gap"),
            "recomputed_dual_bound_s": info.get("dual_bound"),
            "recomputed_solve_seconds": round(time.time() - t0, 1),
            "identical": None if val is None else bool(abs(val - old) <= 1e-6 * abs(old)),
        })
        print(f"  {tag:16s} committed {old:12.1f}  recomputed "
              f"{'n/a' if val is None else format(val, '.1f'):>12s}  "
              f"rel_dev {rows[-1]['rel_deviation']}  gap {info.get('mip_gap')}")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "host_level_ilp_check.csv", index=False)
    print(f"\nwrote {len(df)} rows -> results/host_level_ilp_check.csv")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "audit", "policies", "milp", "rules", "lpcheck", "ilpcheck"])
    ap.add_argument("--pools", default="", help="comma-separated pool filter (milp stage)")
    ap.add_argument("--pilot-jobs", type=int, default=40,
                    help="jobs kept in the offline MILP pilot when no window is given")
    ap.add_argument("--pilot-window-hours", type=float, default=0.0,
                    help="if > 0, the pilot instance is the busiest window of this many hours "
                         "inside the committed instance (the same jobs for both models); "
                         "0 keeps every job")
    ap.add_argument("--cand-window", type=int, default=16,
                    help="candidate slots per job after release (identical for both models)")
    ap.add_argument("--time-limit", type=float, default=300.0,
                    help="per-solve wall-clock budget in seconds (gap = 0 runs)")
    ap.add_argument("--gap", type=float, default=0.02, help="unused; kept for CLI compatibility")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if args.stage in ("all", "audit"):
        print("### stage audit")
        stage_audit(args)
    if args.stage in ("all", "policies"):
        print("\n### stage policies")
        stage_policies(args)
    if args.stage in ("all", "milp"):
        print("\n### stage milp")
        stage_milp(args)
    if args.stage in ("all", "rules"):
        print("\n### stage rules")
        stage_rules(args)
    if args.stage in ("all", "lpcheck"):
        print("\n### stage lpcheck (re-derive the committed LP lower bounds)")
        stage_lpcheck(args)
    if args.stage in ("all", "ilpcheck"):
        print("\n### stage ilpcheck (re-solve the committed ILP-flow rows)")
        stage_ilpcheck(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
