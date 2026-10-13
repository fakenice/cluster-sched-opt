#!/usr/bin/env python3
"""Stage 3 - integer-programming scheduler (time-indexed formulation) + online replay.

Model (per GPU pool, non-preemptive, three coupled resources):
    x[j,t] = 1 if job j starts in slot t          binary
    sum_t x[j,t] = 1                               every job runs exactly once
    sum_j sum_{t <= s < t+p_j} r[j,c] x[j,t] <= cap[c]   for every slot s, c in {GPU, CPU, mem}
    t >= earliest[j], t + p_j <= n_slots           release time and horizon / 时限

    objectives:  flow-time  min sum_j C_j                (queue delay)
                 makespan   min max_j C_j               (throughput / tail idleness)
                 tardiness  min sum_j max(0, C_j - due_j) with due_j = arrival + p_j + slack

Solved with scipy's HiGHS MILP backend; the LP relaxation of the flow objective gives a
provable lower bound, so the reported improvement carries an optimality gap instead of
just "it looks better".

`run_rolling` is the honest version of the experiment: an online re-optimiser with bounded
lookahead L that re-solves every C minutes and commits only decisions inside the commitment
window. It does not get to see the future, so its comparison against FIFO / EASY backfill is
a like-for-like policy comparison, while the offline optimum stays an upper bound.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import LinearConstraint, Bounds, milp

import sched_core as sc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LOOKAHEAD_S = 2 * 3600      # rolling-horizon: jobs that far ahead are visible
COMMIT_S = 60 * 60          # decisions this far ahead are frozen (also the re-plan period)


def _candidate_slots(inst: sc.Instance, subset: np.ndarray, min_start: np.ndarray | None):
    cands = []
    for j in subset:
        lo = int(inst.earliest[j])
        if min_start is not None:
            lo = max(lo, int(min_start[j]))
        hi = int(inst.latest[j])
        if hi < lo:                     # cannot finish inside the horizon: pin at lo
            hi = min(lo, inst.n_slots - 1)
        cands.append(np.arange(lo, hi + 1))
    return cands


def solve_ilp(inst: sc.Instance, subset: np.ndarray, objective: str = "flow",
              binary: bool = True, time_limit: float = 120.0, gap: float = 0.02,
              occ_base: np.ndarray | None = None, min_start: np.ndarray | None = None,
              weights: np.ndarray | None = None, slack_s: float = 1800.0):
    """Return (starts for subset, info dict). occ_base carries committed occupancy."""
    n_jobs = len(subset)
    cands = _candidate_slots(inst, subset, min_start)
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in cands])[:-1]])
    nx = int(sum(len(c) for c in cands))
    extra = 1 if objective == "makespan" else (n_jobs if objective == "tardiness" else 0)
    ncols = nx + extra

    cap = np.array([inst.capacity[r] for r in sc.RESOURCES], dtype=float)
    if occ_base is not None:
        cap_remaining = cap[None, :] - occ_base
    else:
        cap_remaining = np.tile(cap, (inst.n_slots, 1))

    rows, cols, vals = [], [], []
    r_idx = 0
    # (1) assignment: sum_t x[j,t] = 1
    for k in range(n_jobs):
        for t_pos in range(len(cands[k])):
            rows.append(r_idx); cols.append(int(offsets[k] + t_pos)); vals.append(1.0)
        r_idx += 1
    lb_assign = np.full(n_jobs, 1.0)
    ub_assign = np.full(n_jobs, 1.0)
    n_cap_rows = inst.n_slots * 3
    A_rows = r_idx + n_cap_rows

    # (2) capacity rows, one per (slot, resource)
    p_all = inst.p
    for k, j in enumerate(subset):
        pj = int(p_all[j])
        for t_pos, t in enumerate(cands[k]):
            col = int(offsets[k] + t_pos)
            for c in range(3):
                r = inst.demand[j, c]
                if r <= 0:
                    continue
                lo = int(t)
                hi = min(int(t) + pj, inst.n_slots)
                idx = np.arange(r_idx + lo * 3, r_idx + lo * 3 + (hi - lo) * 3, 3)
                rows.extend(idx + c)
                cols.extend([col] * len(idx))
                vals.extend([r] * len(idx))
    lb_cap = np.full(n_cap_rows, -np.inf)
    ub_cap = cap_remaining.reshape(-1).astype(float)

    # (3) objective-dependent rows
    obj_rows = []
    if objective == "makespan":
        m_col = nx
        for k in range(n_jobs):
            for t_pos, t in enumerate(cands[k]):
                rows.append(A_rows + k); cols.append(m_col); vals.append(-1.0)
                rows.append(A_rows + k); cols.append(int(offsets[k] + t_pos))
                vals.append((int(t) + int(p_all[subset[k]])) * inst.slot_seconds)
        obj_rows = [(A_rows + k, -np.inf, 0.0) for k in range(n_jobs)]
        A_rows += n_jobs
    elif objective == "tardiness":
        due = inst.jobs["arrival_rel"].to_numpy()[subset] + p_all[subset] * inst.slot_seconds + slack_s
        for k in range(n_jobs):
            t_col = nx + k
            for t_pos, t in enumerate(cands[k]):
                rows.append(A_rows + k); cols.append(t_col); vals.append(-1.0)
                rows.append(A_rows + k); cols.append(int(offsets[k] + t_pos))
                vals.append((int(t) + int(p_all[subset[k]])) * inst.slot_seconds)
        obj_rows = [(A_rows + k, -due[k], np.inf) for k in range(n_jobs)]
        A_rows += n_jobs

    A = sp.coo_matrix((vals, (rows, cols)), shape=(A_rows, ncols)).tocsr()
    lb = np.concatenate([lb_assign, lb_cap] + [[o[1] for o in obj_rows] if obj_rows else []])
    ub = np.concatenate([ub_assign, ub_cap] + [[o[2] for o in obj_rows] if obj_rows else []])
    lc = LinearConstraint(A, lb, ub)

    c = np.zeros(ncols)
    w = np.ones(n_jobs) if weights is None else np.asarray(weights, dtype=float)
    if objective == "flow":
        for k in range(n_jobs):
            for t_pos, t in enumerate(cands[k]):
                c[int(offsets[k] + t_pos)] = w[k] * (int(t) + int(p_all[subset[k]])) * inst.slot_seconds
    elif objective == "makespan":
        c[nx] = 1.0
    elif objective == "tardiness":
        c[nx:nx + n_jobs] = 1.0

    integrality = np.zeros(ncols)
    if binary:
        integrality[:nx] = 1
    lo_b = np.zeros(ncols)
    hi_b = np.ones(ncols) * np.inf
    hi_b[:nx] = 1.0
    if objective == "makespan":
        hi_b[nx] = np.inf
        lo_b[nx] = 0.0

    t0 = time.time()
    res = milp(c=c, constraints=lc, integrality=integrality, bounds=Bounds(lo_b, hi_b),
               options={"time_limit": time_limit, "mip_rel_gap": gap, "presolve": True})
    x = res.x
    status = int(getattr(res, "status", -1))
    # scipy returns an incumbent only when it actually found one; a failed solve must never
    # be turned into "every job starts at its release slot", which would overcommit capacity.
    has_solution = x is not None and status in (0, 1) and np.all(np.isfinite(x))
    starts = None
    if has_solution:
        x = np.asarray(x, dtype=float)
        starts = np.zeros(n_jobs, dtype=int)
        for k in range(n_jobs):
            block = x[offsets[k]:offsets[k] + len(cands[k])]
            starts[k] = int(cands[k][int(np.argmax(block))]) if len(block) else int(inst.earliest[subset[k]])
    fun = getattr(res, "fun", None)
    try:
        obj_val = float(fun) if fun is not None and np.isfinite(np.asarray(fun, dtype=float)) else None
    except (TypeError, ValueError):
        obj_val = None
    info = {
        "solver_status": status, "message": str(getattr(res, "message", "")),
        "solver_feasible": bool(has_solution),
        "objective": objective, "binary": binary,
        "n_columns": ncols, "n_rows": int(A_rows), "n_nnz": int(A.nnz),
        "solve_seconds": round(time.time() - t0, 2),
        "objective_value": obj_val,
        "mip_gap": float(np.asarray(res.mip_rel_gap).item()) if getattr(res, "mip_rel_gap", None) is not None
                     and np.isfinite(np.asarray(res.mip_rel_gap, dtype=float)) else None,
    }
    return starts, info


def run_offline(inst: sc.Instance, objective: str, **kw):
    subset = np.arange(len(inst.jobs))
    # Larger instances get a bigger but still bounded budget; the reported optimality
    # gap versus the LP relaxation is what makes a time-limited answer interpretable.
    kw.setdefault("time_limit", 180.0 if len(subset) > 200 else 120.0)
    kw.setdefault("gap", 0.05 if len(subset) > 200 else 0.02)
    starts, info = solve_ilp(inst, subset, objective=objective, **kw)
    if starts is None:
        # No incumbent inside the budget: fall back to a *feasible* first-fit schedule and say
        # so, rather than reporting a schedule the solver never produced.
        occ = sc.Occupancy(inst)
        starts, unplaceable = sc.first_fit_leftovers(inst, occ, subset)
        info["solver_fell_back_to_first_fit"] = True
        info["unplaceable_jobs"] = len(unplaceable)
        print(f"    ! ILP-{objective}: no incumbent ({info['message']}); first-fit fallback used "
              f"({info['n_columns']} cols / {info['solve_seconds']}s)")
    else:
        info["solver_fell_back_to_first_fit"] = False
        info["unplaceable_jobs"] = 0
    return starts, info


def run_rolling(inst: sc.Instance, lookahead_s: float = LOOKAHEAD_S, commit_s: float = COMMIT_S,
                objective: str = "flow", time_limit: float = 15.0, gap: float = 0.10):
    """Bounded-lookahead online re-optimisation. Never sees jobs beyond the lookahead.

    Each epoch gets a small solve budget (15 s, 10% gap): a production scheduler
    re-plans on a wall-clock deadline, not to optimality, so the comparison against
    FIFO / EASY stays fair in compute terms as well as in information terms.

    Decisions are only frozen when they start inside the commitment window; everything
    still uncommitted when the planning horizon runs out is placed by capacity-aware first
    fit in the documented tail and counted as `deferred_jobs_past_horizon` - never stacked
    onto the release slot, which would invent capacity.
    """
    n = len(inst.jobs)
    d = inst.slot_seconds
    starts = -np.ones(n, dtype=int)
    window = sc.window_slots(inst)
    occ = np.zeros((window, 3))
    arrivals = inst.jobs["arrival_rel"].to_numpy()
    horizon_s = inst.n_slots * d
    t_now = 0.0
    iters = 0
    epochs_without_incumbent = 0
    solve_secs = 0.0
    while t_now < horizon_s and (starts < 0).any():
        visible = np.where((starts < 0) & (arrivals <= t_now + lookahead_s))[0]
        if visible.size:
            t0 = time.time()
            min_start = np.full(n, int(np.ceil(t_now / d)))
            s_local, info = solve_ilp(inst, visible, objective=objective, binary=True,
                                      time_limit=time_limit, gap=gap, occ_base=occ[:inst.n_slots],
                                      min_start=min_start)
            solve_secs += time.time() - t0
            iters += 1
            if not info["solver_feasible"]:
                epochs_without_incumbent += 1
            else:
                commit_before = int(np.ceil((t_now + commit_s) / d))
                for pos, j in enumerate(visible):
                    if s_local[pos] <= commit_before:
                        starts[j] = int(s_local[pos])
                        pj = int(inst.p[j])
                        occ[int(s_local[pos]):int(s_local[pos]) + pj] += inst.demand[j]
        t_now += commit_s
    leftover = np.where(starts < 0)[0]
    occ_obj = sc.Occupancy(inst, n_slots=window)
    occ_obj.use = occ
    floor = np.full(n, int(inst.n_slots))
    extra, unplaceable = sc.first_fit_leftovers(inst, occ_obj, leftover, floor_slot=floor)
    if leftover.size:
        starts[leftover] = extra[leftover]
    still = np.array(unplaceable, dtype=int)
    if still.size:                       # saturated even with the tail: visible, not silent
        starts[still] = inst.earliest[still]
    return starts, {"rolling_horizon_iterations": iters,
                    "rolling_solve_seconds": round(solve_secs, 2),
                    "rolling_epochs_without_incumbent": epochs_without_incumbent,
                    "deferred_jobs_past_horizon": int(leftover.size),
                    "unplaceable_jobs": int(still.size),
                    "lookahead_minutes": lookahead_s / 60, "commit_minutes": commit_s / 60,
                    "objective": objective}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="", help="comma-separated pool filter (default: all)")
    ap.add_argument("--max-jobs", type=int, default=0, help="skip instances larger than this")
    args = ap.parse_args()
    metas = json.loads((PROC / "instance_meta.json").read_text(encoding="utf-8"))
    if args.pools:
        keep = args.pools.split(",")
        metas = [m for m in metas if m["pool"] in keep]
    if args.max_jobs:
        metas = [m for m in metas if m["n_jobs"] <= args.max_jobs]
    out_rows = []
    schedules = {}
    for m in metas:
        inst = sc.load(PROC / m["instance_file"], m)
        tag = f"{m['pool']}_rho{m['achieved_rho']:.2f}"
        print(f"=== {tag}  jobs={len(inst.jobs)} slots={inst.n_slots} Δ={m['slot_seconds']}s "
              f"cards={m['capacity']['gpu_cards']:.0f} machines={m['capacity']['machines']:.0f}")

        policies = {}
        policies["FIFO"] = sc.run_fifo(inst, backfill=False)
        policies["EASY-backfill"] = sc.run_fifo(inst, backfill=True)
        s, i = run_offline(inst, "flow"); policies["ILP-flow(offline)"] = s; i["policy"] = "ILP-flow(offline)"
        schedules[(tag, "ILP-flow(offline)")] = (s, i)
        s, i = run_offline(inst, "makespan"); policies["ILP-makespan(offline)"] = s; i["policy"] = "ILP-makespan(offline)"
        schedules[(tag, "ILP-makespan(offline)")] = (s, i)
        s, i = run_offline(inst, "flow", binary=False)
        lp_bound = i["objective_value"]; i["policy"] = "LP-bound"
        schedules[(tag, "LP-bound")] = (s, i)
        s, i = run_rolling(inst, objective="flow"); policies["ILP-rolling"] = s; i["policy"] = "ILP-rolling"
        schedules[(tag, "ILP-rolling")] = (s, i)

        for name, starts in policies.items():
            met = sc.metrics(inst, starts)
            met.update(sc.add_deadlines(inst, starts))
            met.update(sc.packing_check(inst, starts))
            inf = schedules.get((tag, name), [None, {}])[1] if name in schedules else {}
            extra = {}
            if name == "ILP-flow(offline)":
                milp_obj = schedules[(tag, "ILP-flow(offline)")][1]["objective_value"]
                extra["milp_objective_sum_completion_s"] = milp_obj
                extra["lp_lower_bound_s"] = lp_bound
                extra["gap_vs_lp_percent"] = (round(100 * (milp_obj - lp_bound) / max(milp_obj, 1), 2)
                                              if lp_bound and milp_obj else None)
                extra["solve_seconds"] = inf.get("solve_seconds")
                extra["columns"] = inf.get("n_columns")
                extra["rows"] = inf.get("n_rows")
            if name.startswith("ILP-"):
                extra["solver_fell_back_to_first_fit"] = inf.get("solver_fell_back_to_first_fit")
                extra["unplaceable_jobs"] = inf.get("unplaceable_jobs", 0)
            if name == "ILP-rolling":
                rr = schedules[(tag, "ILP-rolling")][1]
                extra.update({"rolling_epochs": rr["rolling_horizon_iterations"],
                              "rolling_epochs_without_incumbent": rr["rolling_epochs_without_incumbent"],
                              "deferred_jobs_past_horizon": rr["deferred_jobs_past_horizon"],
                              "rolling_solve_seconds": rr["rolling_solve_seconds"]})
            row = {"instance": tag, "pool": m["pool"], "rho": m["achieved_rho"],
                   "machines": m["capacity"]["machines"], "policy": name, **met, **extra}
            out_rows.append(row)
            pd.DataFrame({"job_id": inst.jobs["job_id"], "arrival_s": inst.jobs["arrival_rel"],
                          "service_s": inst.jobs["service"], "start_slot": starts,
                          "start_s": starts * inst.slot_seconds,
                          "gpu_cards": inst.jobs["gpu_cards"], "cpu_cores": inst.jobs["cpu_cores"],
                          "mem_gb": inst.jobs["mem_gb"]}).to_csv(
                RESULTS / f"schedule_{tag}_{name.replace('(', '').replace(')', '').replace('-', '_')}.csv", index=False)
            print(f"  {name:24s} mean_wait={met['mean_wait_s']/60:8.1f}m p90={met['p90_wait_s']/60:8.1f}m "
                  f"thr={met['throughput_jobs_per_hour']:6.2f}/h idle_fix={met['gpu_idle_rate_fixed_window']*100:5.1f}% "
                  f"idle_busy={met['gpu_idle_rate_busy_period']*100:5.1f}% "
                  f"miss={met['deadline_miss_rate']*100:5.1f}% pack={met['packing_feasibility_rate']} "
                  f"overcap_slots={met['capacity_violation_slots']} past_horizon={met['jobs_running_past_horizon']}"
                  + (f" deferred={extra.get('deferred_jobs_past_horizon')}" if extra.get('deferred_jobs_past_horizon') else "")
                  + ("  [FIRST-FIT FALLBACK]" if extra.get("solver_fell_back_to_first_fit") else ""))

    df = pd.DataFrame(out_rows)
    if "capacity_violation_slots" in df.columns:
        bad = df[df["capacity_violation_slots"] > 0]
        if len(bad):
            print("\n!! CAPACITY VIOLATIONS (schedule not physically runnable, reported not hidden):")
            for _, r in bad.iterrows():
                print(f"   {r['instance']:16s} {r['policy']:24s} slots_over={int(r['capacity_violation_slots'])} "
                      f"max_gpu_overcommit={r['max_gpu_overcommit_cards']} cards")
        else:
            print("\nfeasibility check: every reported schedule respects GPU/CPU/mem capacity in every slot")
    if (args.pools or args.max_jobs) and (RESULTS / "metrics.csv").exists():
        # partial runs must not discard results for the instances they did not cover
        done = {f"{m['pool']}@{m['achieved_rho']}" for m in metas}
        prev = pd.read_csv(RESULTS / "metrics.csv")
        key = prev["pool"].astype(str) + "@" + prev["rho"].astype(str)
        df = pd.concat([prev[~key.isin(done)], df], ignore_index=True)
    df.to_csv(RESULTS / "metrics.csv", index=False)
    df.to_json(RESULTS / "metrics.json", orient="records", indent=2)
    print(f"\nwrote {len(df)} policy rows -> results/metrics.csv")


if __name__ == "__main__":
    main()
