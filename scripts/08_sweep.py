#!/usr/bin/env python3
"""08_sweep.py -- does the policy ranking survive other windows and other job samples?

The committed study is six instances: two GPU pools x three load tiers, all carved out of
**the single busiest six-hour window** of each pool (`02_instance.py` ranks windows by
GPU-card-hours and takes rank 1) and all drawn with **one** subsampling seed. Every
conclusion in the README -- "an online priority rule captures 77-114 % of the optimiser's
mean-wait gain" -- is therefore a statement about one window that was selected for being
extreme, and one random draw of jobs from it. Two things could produce that result other
than the scheduler: the window could be unusually bursty, or the draw could be lucky.

This script re-runs the whole policy set over a grid:

  pools         P100, MISC                       (as committed)
  window ranks  1, 2, 3, median                  rank 1 = busiest (the committed window),
                                                 median = the middle-demand eligible window,
                                                 which is the selection-bias control
  seeds         3 subsampling seeds              seed 20261013 = the committed draw
  load tiers    rho 0.4 / 0.6 / 0.85             nested thinning, cluster fixed per pool

so the committed six instances are a *subset* of the grid and act as a built-in positive
control: rows with window_rank=1 and seed=20261013 must reproduce `results/metrics.csv`.
That control is reported separately for the deterministic policies (FIFO, EASY, the
priority rules -- exact match required) and for the solver policies (ILP-flow, LP bound,
rolling -- `scipy.optimize.milp` stops on a *wall-clock* limit, so a busier machine can
return a different incumbent; deviation is reported, not asserted away).

Policies per instance: FIFO, EASY-backfill, ILP-flow(offline), LP-bound, ILP-rolling and
the seven request-only priority rules of `07_priority_baselines.py` (loaded, not copied, so
the two scripts cannot drift). ILP-makespan is omitted: it is the worst policy in every
committed cell and is not part of any claim this sweep tests.

Instances are written to `data/processed/sweep/` and never overwrite the committed ones.
Rows are appended to `results/sweep_metrics.csv` as each instance finishes, so an
interrupted run keeps what it completed and `--resume` continues from there.

Usage
  python scripts/08_sweep.py --jobs 6                 # full grid (72 instances)
  python scripts/08_sweep.py --no-ilp --jobs 6        # list policies only, ~1 min
  python scripts/08_sweep.py --limit 2 --no-ilp       # smoke test
  python scripts/08_sweep.py --resume --jobs 6        # continue an interrupted sweep
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sched_core as sc   # noqa: E402


def _load(name: str, filename: str):
    """Load a digit-prefixed sibling script as a module (single source of truth)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inst_mod = _load("instance_builder", "02_instance.py")
opt_mod = _load("optimizer", "03_optimize.py")
backlog_mod = _load("backlog_idle", "06_backlog_idle.py")
prio_mod = _load("priority_baselines", "07_priority_baselines.py")
backlog_metrics = backlog_mod.backlog_metrics
PRIORITY_POLICIES = prio_mod.POLICIES

PROC = ROOT / "data" / "processed"
SWEEP = PROC / "sweep"
RES = ROOT / "results"
OUT_CSV = RES / "sweep_metrics.csv"
OUT_JSON = RES / "sweep_summary.json"

BASE_SEED = 20261013          # the committed study's seed
SEEDS = [20261013, 20261103, 20261104]
POOL_JOBS_HIGH = inst_mod.POOL_JOBS_HIGH

# deterministic policies must reproduce the committed numbers exactly; solver policies are
# wall-clock limited and may land on a different incumbent
DETERMINISTIC = {"FIFO", "EASY-backfill", *PRIORITY_POLICIES}
CONTROL_COLS = ["mean_wait_s", "median_wait_s", "p90_wait_s", "max_wait_s", "mean_flow_s",
                "makespan_s", "throughput_jobs_per_hour", "deadline_miss_rate",
                "gpu_idle_rate_fixed_window", "used_gpu_card_hours",
                "idle_addressable_backlog_card_hours", "capacity_violation_slots",
                "packing_feasibility_rate"]
# derived from the FIFO replay alone, so still a deterministic function of the instance
FIFO_DERIVED_COLS = ["wait_change_vs_fifo", "p90_change_vs_fifo", "max_change_vs_fifo"]
# derived from the ILP-flow row, so it inherits whatever incumbent the solver happened to find
ILP_DERIVED_COLS = ["captured_ilp_wait_gain"]


# --------------------------------------------------------------------------- instance grid

def resolve_ranks(spec: str, pool: str, df: pd.DataFrame) -> list[tuple[int, str]]:
    """Turn '1,2,3,median' into concrete window ranks for this pool."""
    n_elig = None
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "median":
            if n_elig is None:
                _, _, stats = inst_mod.pick_window(df, pool, rank=1)
                n_elig = stats["n_eligible_windows"]
            out.append((int(np.ceil(n_elig / 2)), "median"))
        else:
            out.append((int(tok), f"rank{tok}"))
    seen, uniq = set(), []
    for r, lab in out:
        if r not in seen:
            seen.add(r)
            uniq.append((r, lab))
    return uniq


def build_grid(pools: list[str], rank_spec: str, seeds: list[int], df: pd.DataFrame,
               out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Build every instance file once; return (job list, meta list)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, metas = [], []
    for pool in pools:
        ranks = resolve_ranks(rank_spec, pool, df)
        n_high = POOL_JOBS_HIGH.get(pool, 150)
        for rank, rank_label in ranks:
            for seed in seeds:
                prefix = f"instance_w{rank}_s{seed}"
                ms = inst_mod.build_pool(pool, n_high, seed, out_dir, df=df,
                                         window_rank=rank, prefix=prefix)
                for m in ms:
                    m["window_rank_label"] = rank_label
                    m["sweep_key"] = f"{pool}_w{rank}_s{seed}_rho{m['target_rho']}"
                    metas.append(m)
                    jobs.append({"pool": pool, "window_rank": rank,
                                 "window_rank_label": rank_label, "seed": seed,
                                 "target_rho": m["target_rho"],
                                 "instance_file": str(out_dir / m["instance_file"]),
                                 "meta": m})
    return jobs, metas


# --------------------------------------------------------------------------- one instance

def run_instance(job: dict) -> dict:
    """Replay every policy on one instance. Returns rows + timings; never raises."""
    t0 = time.time()
    meta = job["meta"]
    try:
        inst = sc.load(Path(job["instance_file"]), meta)
    except Exception as exc:                                   # noqa: BLE001
        return {"job": job, "status": "load_failed", "error": f"{type(exc).__name__}: {exc}",
                "rows": [], "seconds": round(time.time() - t0, 1)}

    rows: list[dict] = []
    info_by_policy: dict[str, dict] = {}
    lp_bound = None
    errors: list[str] = []

    def record(name: str, starts, oracle=False, aging=None, info=None):
        met = sc.metrics(inst, starts)
        met.update(sc.add_deadlines(inst, starts))
        met.update(sc.packing_check(inst, starts))
        met.update(backlog_metrics(inst, starts))
        if info_by_policy is not None and info is not None:
            info_by_policy[name] = info
        rows.append({"sweep_key": meta["sweep_key"], "pool": meta["pool"],
                     "window_rank": job["window_rank"],
                     "window_rank_label": job["window_rank_label"],
                     "seed": job["seed"], "target_rho": meta["target_rho"],
                     "achieved_rho": meta["achieved_rho"], "n_jobs": meta["n_jobs"],
                     "machines": meta["capacity"]["machines"],
                     "gpu_cards_capacity": meta["capacity"]["gpu_cards"],
                     "slot_seconds": meta["slot_seconds"], "n_slots": meta["n_slots"],
                     "window_demand_gpu_hours": meta["window_demand_gpu_hours"],
                     "window_rank_share_of_busiest": meta["window_demand_rank_share_of_busiest"],
                     "policy": name, "oracle": oracle, "aging_slots": aging, **met})

    try:
        record("FIFO", sc.run_fifo(inst, backfill=False))
        record("EASY-backfill", sc.run_fifo(inst, backfill=True))
        for name, (col, oracle, aging) in PRIORITY_POLICIES.items():
            starts = sc.run_priority_online(inst, key=col, aging_slots=aging)
            record(name, starts, oracle=oracle, aging=aging)
            assert rows[-1]["capacity_violation_slots"] == 0, f"{name} overcommitted capacity"
    except Exception as exc:                                   # noqa: BLE001
        errors.append(f"list policies: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    if job.get("do_ilp", True):
        try:
            s, i = opt_mod.run_offline(inst, "flow")
            record("ILP-flow(offline)", s, info=i)
        except Exception as exc:                               # noqa: BLE001
            errors.append(f"ILP-flow: {type(exc).__name__}: {exc}")
        try:
            s, i = opt_mod.run_offline(inst, "flow", binary=False)
            lp_bound = i.get("objective_value")
            info_by_policy["LP-bound"] = i
        except Exception as exc:                               # noqa: BLE001
            errors.append(f"LP-bound: {type(exc).__name__}: {exc}")
        try:
            s, i = opt_mod.run_rolling(inst, objective="flow")
            record("ILP-rolling", s, info=i)
        except Exception as exc:                               # noqa: BLE001
            errors.append(f"ILP-rolling: {type(exc).__name__}: {exc}")

    # solver provenance + the two derived columns that need the whole instance
    by_policy = {r["policy"]: r for r in rows}
    fifo = by_policy.get("FIFO")
    ilp = by_policy.get("ILP-flow(offline)")
    for r in rows:
        inf = info_by_policy.get(r["policy"], {})
        r["solve_seconds"] = inf.get("solve_seconds")
        r["columns"] = inf.get("n_columns")
        r["rows_solver"] = inf.get("n_rows")
        r["solver_message"] = inf.get("message")
        r["solver_fell_back_to_first_fit"] = inf.get("solver_fell_back_to_first_fit")
        r["unplaceable_jobs"] = inf.get("unplaceable_jobs")
        r["rolling_epochs"] = inf.get("rolling_horizon_iterations")
        r["rolling_epochs_without_incumbent"] = inf.get("rolling_epochs_without_incumbent")
        r["deferred_jobs_past_horizon"] = inf.get("deferred_jobs_past_horizon")
        r["rolling_solve_seconds"] = inf.get("rolling_solve_seconds")
        if r["policy"] == "ILP-flow(offline)":
            obj = inf.get("objective_value")
            r["milp_objective_sum_completion_s"] = obj
            r["lp_lower_bound_s"] = lp_bound
            r["gap_vs_lp_percent"] = (round(100 * (obj - lp_bound) / max(obj, 1), 2)
                                      if obj and lp_bound else None)
        if fifo is not None and fifo["mean_wait_s"] > 0:
            r["wait_change_vs_fifo"] = round(
                (r["mean_wait_s"] - fifo["mean_wait_s"]) / fifo["mean_wait_s"], 4)
            r["p90_change_vs_fifo"] = round(
                (r["p90_wait_s"] - fifo["p90_wait_s"]) / fifo["p90_wait_s"], 4)
            r["max_change_vs_fifo"] = round(
                (r["max_wait_s"] - fifo["max_wait_s"]) / fifo["max_wait_s"], 4)
        if ilp is not None and fifo is not None:
            denom = fifo["mean_wait_s"] - ilp["mean_wait_s"]
            r["captured_ilp_wait_gain"] = (round((fifo["mean_wait_s"] - r["mean_wait_s"]) / denom, 4)
                                           if denom > 1e-9 else None)
        else:
            r["captured_ilp_wait_gain"] = None

    return {"job": job, "status": "ok" if not errors else "partial",
            "errors": errors, "rows": rows, "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- summary

def summarize(df: pd.DataFrame, do_ilp: bool) -> dict:
    out = {"generated_by": "scripts/08_sweep.py",
           "n_instances": int(df.sweep_key.nunique()),
           "n_rows": int(len(df)),
           "pools": sorted(df.pool.unique().tolist()),
           "window_ranks": sorted(df.window_rank.unique().tolist()),
           "seeds": sorted(df.seed.unique().tolist()),
           "ilp_included": bool(do_ilp)}

    dep = df[(~df.oracle) & (~df.policy.isin(["FIFO", "EASY-backfill", "ILP-flow(offline)",
                                              "ILP-rolling"]))]
    plain = dep[~dep.policy.str.contains("age")]

    # per instance: which deployable rule is best on mean wait, and what it captured
    fifo_wait = df[df.policy == "FIFO"].set_index("sweep_key").mean_wait_s.to_dict()
    best_rows = []
    for key, g in dep.groupby("sweep_key"):
        b = g.loc[g.mean_wait_s.idxmin()]
        best_rows.append({
            "sweep_key": key, "pool": b.pool, "window_rank": int(b.window_rank),
            "window_rank_label": b.window_rank_label, "seed": int(b.seed),
            "target_rho": float(b.target_rho), "achieved_rho": float(b.achieved_rho),
            "n_jobs": int(b.n_jobs), "best_deployable_policy": b.policy,
            "fifo_mean_wait_s": fifo_wait.get(key),
            "mean_wait_s": float(b.mean_wait_s),
            "wait_change_vs_fifo": float(b.wait_change_vs_fifo),
            "p90_change_vs_fifo": float(b.p90_change_vs_fifo),
            "max_change_vs_fifo": float(b.max_change_vs_fifo),
            "deadline_miss_rate": float(b.deadline_miss_rate),
            "captured_ilp_wait_gain": (None if pd.isna(b.captured_ilp_wait_gain)
                                       else float(b.captured_ilp_wait_gain)),
            "idle_addressable_backlog_card_hours": float(
                b.idle_addressable_backlog_card_hours),
        })
    best = pd.DataFrame(best_rows)
    out["best_deployable_per_instance"] = best.to_dict(orient="records")
    out["best_deployable_policy_counts"] = best.best_deployable_policy.value_counts().to_dict()

    if do_ilp and best.captured_ilp_wait_gain.notna().any():
        cap = best.captured_ilp_wait_gain.dropna()
        out["captured_gain_best_deployable"] = {
            "n": int(len(cap)), "min": round(float(cap.min()), 4),
            "median": round(float(cap.median()), 4), "max": round(float(cap.max()), 4),
            "frac_ge_0.50": round(float((cap >= 0.50).mean()), 4),
            "frac_ge_0.77": round(float((cap >= 0.77).mean()), 4),
            "frac_ge_1.00": round(float((cap >= 1.00).mean()), 4),
            "frac_negative": round(float((cap < 0).mean()), 4),
        }
        by_rank = best.dropna(subset=["captured_ilp_wait_gain"]).groupby("window_rank")
        out["captured_gain_by_window_rank"] = {
            int(k): {"n": int(len(v.captured_ilp_wait_gain)),
                     "median": round(float(v.captured_ilp_wait_gain.median()), 4),
                     "min": round(float(v.captured_ilp_wait_gain.min()), 4),
                     "max": round(float(v.captured_ilp_wait_gain.max()), 4)}
            for k, v in by_rank}
        by_seed = best.dropna(subset=["captured_ilp_wait_gain"]).groupby("seed")
        out["captured_gain_by_seed"] = {
            int(k): {"n": int(len(v.captured_ilp_wait_gain)),
                     "median": round(float(v.captured_ilp_wait_gain.median()), 4),
                     "min": round(float(v.captured_ilp_wait_gain.min()), 4),
                     "max": round(float(v.captured_ilp_wait_gain.max()), 4)}
            for k, v in by_seed}
        # does the optimiser ever beat every deployable rule?
        ilp_rows = df[df.policy == "ILP-flow(offline)"][["sweep_key", "mean_wait_s"]]
        merged = best.merge(ilp_rows, on="sweep_key", suffixes=("", "_ilp"))
        out["instances_where_ilp_flow_beats_best_deployable"] = int(
            (merged.mean_wait_s_ilp < merged.mean_wait_s - 1e-9).sum())
        out["instances_compared"] = int(len(merged))

    out["priority_only_ranges"] = {
        "wait_change_vs_fifo": [round(float(plain.wait_change_vs_fifo.min()), 4),
                                round(float(plain.wait_change_vs_fifo.max()), 4)],
        "p90_change_vs_fifo": [round(float(plain.p90_change_vs_fifo.min()), 4),
                               round(float(plain.p90_change_vs_fifo.max()), 4)],
        "max_change_vs_fifo": [round(float(plain.max_change_vs_fifo.min()), 4),
                               round(float(plain.max_change_vs_fifo.max()), 4)],
        "idle_addressable_backlog_card_hours_max": round(
            float(plain.idle_addressable_backlog_card_hours.max()), 4),
        "capacity_violation_slots_max": int(plain.capacity_violation_slots.max()),
    }

    # Is "zero addressable idle for the priority rules" a property of the rule or a column
    # nobody filled in? It is the rule: run_priority_online starts *every* waiting job that
    # fits at every slot, so a slot with free GPUs and a backlog can only be a slot where
    # nothing left fits -- fragmentation, not head-of-line blocking. The test below is the
    # falsifiable version of that claim: recompute it slot by slot on one instance and see
    # whether any free-GPU-with-backlog slot admits a waiting job.
    # See scripts/08_sweep.py::_verify_addressable_claim, run by 04_report's section.
    prio_all = df[df.policy.str.startswith("priority")]
    out["addressable_idle_by_policy_family"] = {
        fam: {"n_rows": int(len(g)),
              "rows_with_nonzero_addressable": int((g.idle_addressable_backlog_card_hours > 0).sum()),
              "idle_with_backlog_card_hours_max": round(float(g.idle_with_backlog_card_hours.max()), 2),
              "idle_addressable_backlog_card_hours_max": round(
                  float(g.idle_addressable_backlog_card_hours.max()), 2),
              "addressable_share_of_backlog_idle_median": (
                  None if g.addressable_share_of_backlog_idle.isna().all()
                  else round(float(g.addressable_share_of_backlog_idle.median()), 4))}
        for fam, g in df.assign(fam=np.where(df.policy.str.startswith("priority"),
                                             "priority-rules", "fifo-easy-ilp")).groupby("fam")}
    if do_ilp:
        ilpf = df[df.policy == "ILP-flow(offline)"]
        out["ilp_flow"] = {
            "n": int(len(ilpf)),
            "gap_vs_lp_percent_max": (None if ilpf.gap_vs_lp_percent.isna().all()
                                      else round(float(ilpf.gap_vs_lp_percent.max()), 2)),
            "fallback_to_first_fit_count": int(ilpf.solver_fell_back_to_first_fit.sum()),
            "solve_seconds_range": [round(float(ilpf.solve_seconds.min()), 1),
                                    round(float(ilpf.solve_seconds.max()), 1)],
        }
        roll = df[df.policy == "ILP-rolling"]
        out["ilp_rolling"] = {
            "n": int(len(roll)),
            "solve_seconds_range": [round(float(roll.rolling_solve_seconds.min()), 1),
                                    round(float(roll.rolling_solve_seconds.max()), 1)],
            "epochs_without_incumbent_total": int(
                roll.rolling_epochs_without_incumbent.fillna(0).sum()),
            "deferred_jobs_past_horizon_total": int(
                roll.deferred_jobs_past_horizon.fillna(0).sum()),
        }
    return out


def positive_control(df: pd.DataFrame) -> dict:
    """Compare the grid cell that *is* the committed study against the committed tables.

    `results/metrics.csv` holds FIFO / EASY / the solver policies; the priority rules live in
    `results/metrics_priority.csv`. Both are checked, because both are deterministic replays:
    if the sweep's window_rank=1, seed=20261013 cell does not reproduce them row for row, the
    sweep is not running the same experiment as the study it claims to generalize.
    """
    committed_path = RES / "metrics.csv"
    if not committed_path.exists():
        return {"status": "skipped", "reason": "results/metrics.csv missing"}
    committed = pd.read_csv(committed_path)
    prio_path = RES / "metrics_priority.csv"
    prio = pd.read_csv(prio_path) if prio_path.exists() else None
    ctrl = df[(df.window_rank == 1) & (df.seed == BASE_SEED)]
    if ctrl.empty:
        return {"status": "skipped", "reason": "grid does not contain window_rank=1/seed=20261013"}
    det, sol, missing = [], [], []
    for _, r in ctrl.iterrows():
        tag = f"{r.pool}_rho{r.achieved_rho:.2f}"
        c = committed[(committed.instance == tag) & (committed.policy == r.policy)]
        src = "metrics.csv"
        if c.empty and prio is not None:
            c = prio[(prio.instance == tag) & (prio.policy == r.policy)]
            src = "metrics_priority.csv"
        if c.empty:
            missing.append(f"{tag}/{r.policy}")
            continue
        c = c.iloc[0]
        # three buckets, because a column's provenance decides whether a deviation is a bug:
        # raw metrics and FIFO-relative columns are deterministic functions of the instance
        # and MUST match; `captured_ilp_wait_gain` divides by the ILP-flow wait, so it moves
        # whenever the solver lands on a different incumbent -- that is the solver's
        # nondeterminism showing through, not the deterministic replay disagreeing.
        cols = {"raw": CONTROL_COLS, "fifo_derived": FIFO_DERIVED_COLS,
                "ilp_derived": ILP_DERIVED_COLS}
        devs: dict[str, dict[str, float]] = {k: {} for k in cols}
        for group, names in cols.items():
            for col in names:
                if col in c.index and pd.notna(c[col]) and pd.notna(r.get(col)):
                    a, b = float(c[col]), float(r[col])
                    if a == 0.0:
                        if b != 0.0:
                            devs[group][col] = float("inf")
                    else:
                        devs[group][col] = abs(a - b) / abs(a)
        worst_raw = max(devs["raw"].values(), default=0.0)
        worst_fifo = max(devs["fifo_derived"].values(), default=0.0)
        worst_ilp = max(devs["ilp_derived"].values(), default=0.0)

        def _worst(d):
            return (max(d, key=d.get), max(d.values())) if d else (None, 0.0)

        rec = {"instance": tag, "policy": r.policy, "source": src,
               "worst_relative_deviation": max(worst_raw, worst_fifo),
               "worst_raw": worst_raw, "worst_raw_column": _worst(devs["raw"])[0],
               "worst_fifo_derived": worst_fifo,
               "worst_fifo_derived_column": _worst(devs["fifo_derived"])[0],
               "worst_ilp_derived": worst_ilp,
               "worst_ilp_derived_column": _worst(devs["ilp_derived"])[0],
               "deterministic": r.policy in DETERMINISTIC}
        (det if r.policy in DETERMINISTIC else sol).append(rec)
    det_max = max((x["worst_relative_deviation"] for x in det), default=None)
    sol_max = max((x["worst_relative_deviation"] for x in sol), default=None)
    ilp_derived_max = max((x["worst_ilp_derived"] for x in det), default=None)
    return {
        "status": "ok",
        "n_deterministic_rows_compared": len(det),
        "n_solver_rows_compared": len(sol),
        "policies_not_in_committed_tables": sorted(set(missing)),
        "deterministic_worst_relative_deviation": det_max,
        "deterministic_reproduce_exactly": bool(det_max is not None and det_max < 1e-9),
        # reported apart from the verdict above, and NOT folded into it
        "deterministic_worst_via_ilp_derived_column": ilp_derived_max,
        "solver_worst_relative_deviation": sol_max,
        "solver_rows": sorted(sol, key=lambda x: -x["worst_relative_deviation"])[:6],
        "note": ("Deterministic policies replay identical rows; their raw and FIFO-relative "
                 "columns must match to 1e-9. `captured_ilp_wait_gain` is divided by the "
                 "ILP-flow wait and is therefore compared but not asserted: `milp` stops on a "
                 "wall-clock limit, so the incumbent can differ between runs of the same "
                 "instance. That deviation is reported, not hidden."),
    }


# --------------------------------------------------------------------------- driver

def verify_addressable_claim(instance_file: Path, meta: dict) -> dict:
    """Recompute the addressable/fragmentation split slot by slot for two policies.

    `idle_addressable_backlog_card_hours` is 0 for all 504 priority rows and nonzero for most
    FIFO/EASY/ILP rows. That could mean two very different things: the greedy rule really
    leaves no reclaimable idle, or the column was never filled in for that policy family. This
    settles it the only way that counts -- it walks every slot, and for slots that have both a
    waiting backlog and free GPUs, asks whether *any* waiting job fits on all three resources.
    Under a rule that starts everything that fits, the answer must be no in every such slot;
    if this function ever reports `addressable_slots > 0` for a priority policy, either the
    rule or the metric is broken.
    """
    inst = sc.load(instance_file, meta)
    out = {}
    schedules = {"FIFO": sc.run_fifo(inst, backfill=False),
                 "priority-gpu": sc.run_priority_online(inst, key="gpu_cards", aging_slots=None)}
    d = inst.slot_seconds
    cap = np.array([inst.capacity[r] for r in sc.RESOURCES], dtype=float)
    arrivals = np.ceil(inst.jobs["arrival_rel"].to_numpy() / d).astype(int)
    need = inst.demand
    for name, raw in schedules.items():
        starts = sc.clamp_starts(inst, raw)
        occ = sc.Occupancy(inst)
        for j in range(len(starts)):
            occ.reserve(j, int(starts[j]))
        free_gpu_slots = addressable = non_addressable = 0
        max_free = 0.0
        for s in range(inst.n_slots):
            waiting = np.where((arrivals <= s) & (starts > s))[0]
            if waiting.size == 0:
                continue
            free = cap - occ.use[s]
            fgpu = max(float(free[0]), 0.0)
            if fgpu <= 1e-9:
                continue
            free_gpu_slots += 1
            max_free = max(max_free, fgpu)
            fits = bool(np.any(np.all(need[waiting] <= free + 1e-9, axis=1)))
            addressable += fits
            non_addressable += not fits
        out[name] = {"unplaced_jobs": int((raw < 0).sum()),
                     "slots_with_backlog_and_free_gpu": free_gpu_slots,
                     "addressable_slots": addressable,
                     "fragmentation_only_slots": non_addressable,
                     "max_free_cards_while_backlogged": max_free,
                     **backlog_metrics(inst, starts)}
    return out


def _print_headline(summary: dict) -> None:
    """Shared by the full run and --report-only, so the two cannot drift apart."""
    pc = summary.get("positive_control", {})
    if pc.get("status") == "ok":
        det = pc["deterministic_worst_relative_deviation"]
        sol = pc["solver_worst_relative_deviation"]
        via = pc.get("deterministic_worst_via_ilp_derived_column")
        print(f"positive control: {pc['n_deterministic_rows_compared']} deterministic rows, "
              f"worst relative deviation on raw + FIFO-derived columns "
              f"{'n/a' if det is None else f'{det:.2e}'} "
              f"(exact={pc['deterministic_reproduce_exactly']}); "
              f"{pc['n_solver_rows_compared']} solver rows, worst "
              f"{'n/a' if sol is None else f'{sol:.3f}'}")
        if via:
            print(f"  note: `captured_ilp_wait_gain` on those deterministic rows deviates by "
                  f"up to {via:.2e} -- that column divides by the ILP-flow wait, so it tracks "
                  f"the solver's incumbent, not the replay")
        if pc["policies_not_in_committed_tables"]:
            print(f"  grid policies absent from the committed tables: "
                  f"{pc['policies_not_in_committed_tables']}")
    if "captured_gain_best_deployable" in summary:
        print("captured ILP mean-wait gain, best deployable rule per instance: "
              + json.dumps(summary["captured_gain_best_deployable"], ensure_ascii=False))
        print("by window rank: " + json.dumps(summary["captured_gain_by_window_rank"],
                                              ensure_ascii=False))
    if "priority_only_ranges" in summary:
        print("priority-only ranges: " + json.dumps(summary["priority_only_ranges"],
                                                    ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="P100,MISC")
    ap.add_argument("--window-ranks", default="1,2,3,median",
                    help="comma-separated window ranks; 'median' = middle-demand window")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--jobs", type=int, default=6, help="parallel instances (HiGHS is single-threaded)")
    ap.add_argument("--no-ilp", action="store_true", help="list policies only (fast pass)")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many instances")
    ap.add_argument("--resume", action="store_true",
                    help="keep rows already in results/sweep_metrics.csv and skip those instances")
    ap.add_argument("--report-only", action="store_true",
                    help="re-derive results/sweep_summary.json (positive control included) "
                         "from the existing sweep_metrics.csv without running any instance; "
                         "the grid is ~75 min and the control is a post-hoc comparison")
    args = ap.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            raise SystemExit(f"--report-only needs {OUT_CSV}, which does not exist")
        frame = pd.read_csv(OUT_CSV)
        summary = summarize(frame, do_ilp=not args.no_ilp)
        summary["wall_seconds"] = None
        summary["build_seconds"] = None
        summary["grid"] = {"pools": sorted(frame.pool.unique().tolist()),
                           "window_ranks": sorted(frame.window_rank.unique().tolist()),
                           "seeds": sorted(frame.seed.unique().tolist()),
                           "n_instance_files": int(frame.sweep_key.nunique()),
                           "regenerated_by": "--report-only"}
        summary["positive_control"] = positive_control(frame)
        # the slot-level check needs an instance, not just the metric table
        meta_path = SWEEP / "instance_meta.json"
        if meta_path.exists():
            metas = json.loads(meta_path.read_text(encoding="utf-8"))
            worst = max(metas, key=lambda m: m["achieved_rho"])
            summary["addressable_claim_verification"] = {
                "instance": worst["sweep_key"],
                "file": worst["instance_file"],
                "per_policy": verify_addressable_claim(SWEEP / worst["instance_file"], worst)}
            v = summary["addressable_claim_verification"]["per_policy"]
            assert v["priority-gpu"]["addressable_slots"] == 0 or v["priority-gpu"][
                "unplaced_jobs"] > 0, (
                "the greedy-fit rule left reclaimable idle -- the zero in "
                "idle_addressable_backlog_card_hours is a computation gap, not a finding")
        OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"regenerated {OUT_JSON} from {len(frame)} rows / "
              f"{frame.sweep_key.nunique()} instances (no instances run)")
        _print_headline(summary)
        return 0

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    do_ilp = not args.no_ilp

    print(f"reading {PROC / 'jobs.csv'} ...", flush=True)
    df_jobs = pd.read_csv(PROC / "jobs.csv")
    t_build = time.time()
    jobs, metas = build_grid(pools, args.window_ranks, seeds, df_jobs, SWEEP)
    build_seconds = round(time.time() - t_build, 1)
    print(f"built {len(jobs)} instances in {build_seconds:.0f} s", flush=True)
    (SWEEP / "instance_meta.json").write_text(
        json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")
    for m in metas[:6]:
        print(f"  {m['sweep_key']:34s} jobs={m['n_jobs']:4d} "
              f"machines={m['capacity']['machines']:3.0f} rho={m['achieved_rho']:.3f} "
              f"demand={m['window_demand_gpu_hours']:7.1f} card-h "
              f"share_of_busiest={m['window_demand_rank_share_of_busiest']:.3f}", flush=True)

    done_keys: set[str] = set()
    old_rows: list[dict] = []
    if args.resume and OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        done_keys = set(old.sweep_key.unique())
        old_rows = old.to_dict(orient="records")
        print(f"resume: {len(done_keys)} instances already in {OUT_CSV.name}", flush=True)
    todo = [j for j in jobs if j["meta"]["sweep_key"] not in done_keys]
    if args.limit:
        todo = todo[:args.limit]
    for j in todo:
        j["do_ilp"] = do_ilp
    print(f"running {len(todo)} instances with {args.jobs} workers "
          f"(ILP {'on' if do_ilp else 'off'})", flush=True)

    header_written = OUT_CSV.exists() and args.resume
    all_rows = list(old_rows)
    t1 = time.time()
    n_done = 0
    workers = max(1, min(args.jobs, len(todo))) if todo else 1
    with open(OUT_CSV, "a" if header_written else "w", newline="", encoding="utf-8") as fh:
        def emit(rows: list[dict]) -> None:
            """Append one instance's rows and flush, so an interrupted sweep keeps its work."""
            nonlocal header_written
            if not rows:
                return
            pd.DataFrame(rows).to_csv(fh, index=False, header=not header_written,
                                      lineterminator="\n")
            header_written = True
            fh.flush()

        if workers == 1 or len(todo) <= 1:
            for j in todo:
                res = run_instance(j)
                n_done += 1
                print(f"[{n_done}/{len(todo)}] {res['job']['meta']['sweep_key']} "
                      f"{res['status']} {res['seconds']} s rows={len(res['rows'])}"
                      + (f" errors={res['errors']}" if res.get("errors") else ""), flush=True)
                emit(res["rows"])
                all_rows.extend(res["rows"])
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_instance, j): j for j in todo}
                for fut in as_completed(futs):
                    res = fut.result()
                    n_done += 1
                    print(f"[{n_done}/{len(todo)}] {res['job']['meta']['sweep_key']} "
                          f"{res['status']} {res['seconds']} s rows={len(res['rows'])}"
                          + (f" errors={res['errors']}" if res.get("errors") else ""),
                          flush=True)
                    emit(res["rows"])
                    all_rows.extend(res["rows"])

    frame = pd.DataFrame(all_rows)
    # keep the file ordered by grid cell even though rows were appended as they finished
    order = {m["sweep_key"]: i for i, m in enumerate(metas)}
    frame["_ord"] = frame.sweep_key.map(order).fillna(len(metas))
    frame = frame.sort_values(["_ord", "policy"]).drop(columns="_ord")
    frame.to_csv(OUT_CSV, index=False)

    summary = summarize(frame, do_ilp)
    summary["wall_seconds"] = round(time.time() - t1, 1)
    summary["build_seconds"] = build_seconds
    summary["grid"] = {"pools": pools, "window_ranks": args.window_ranks, "seeds": seeds,
                       "n_instance_files": len(metas), "ilp": do_ilp}
    summary["positive_control"] = positive_control(frame)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {OUT_CSV} ({len(frame)} rows, {frame.sweep_key.nunique()} instances) "
          f"and {OUT_JSON} in {summary['wall_seconds']} s")
    _print_headline(summary)
    if "captured_gain_best_deployable" in summary:
        print("captured ILP mean-wait gain, best deployable rule per instance: "
              + json.dumps(summary["captured_gain_best_deployable"], ensure_ascii=False))
        print("by window rank: " + json.dumps(summary["captured_gain_by_window_rank"],
                                              ensure_ascii=False))
    print("priority-only ranges: " + json.dumps(summary["priority_only_ranges"],
                                                ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
