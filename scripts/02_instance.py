#!/usr/bin/env python3
"""Stage 2 - build trace-driven scheduling instances.

Design of the load sweep (this matters for the interpretation of Stage 3):
the cluster is FIXED per pool and the arrival stream is thinned, because offered load
is  rho = (lambda * mean_service * mean_size) / capacity.  Thinning a real window keeps
inter-arrival structure, the size distribution and the CPU / memory / GPU request mix
trace-faithful, and it keeps the three tiers comparable: the low- and mid-load instances
are nested subsets of the high-load one (same jobs, fewer arrivals).

Cluster sizing has two floors, both reported:
  * load floor  - enough machines for the busiest tier to reach the target rho
  * fit floor   - enough machines to physically host the largest single request
                   (a 50-card job needs 25 of the 2-card P100 hosts; a cluster smaller
                    than that would make the instance trivially infeasible)

Machine capacities come from pai_machine_spec at machine granularity, so a slice of k
machines is a genuine hardware partition, not a synthetic capacity number.

Documented exclusions (counted in the meta, nothing dropped silently):
  * tasks whose status is not 'Terminated' (Stage 1)
  * services longer than SERVICE_CAP_HOURS: the raw trace has a handful of jobs holding
    dozens of cards for days (one is 1.1k GPU-card-hours versus a 0.9 GPU-hour median),
    which dominate any window they land in and wash out queueing effects.
Both thresholds are CLI flags (`--service-cap-hours`, `--req-cap-quantile`) so the
sensitivity of the conclusions to these two choices can be re-measured, and every count
of what each filter removed is written into instance_meta.json.

Writes data/processed/instance_<pool>_<rho>_<n>.csv plus instance_meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

BLOCK_HOURS = 6.0
SERVICE_CAP_HOURS = 6.0
REQ_CAP_QUANTILE = 0.90   # exclude pool-p90-and-above gang requests (reported)
RHO_TIERS = (0.4, 0.6, 0.85)
MIN_JOBS = 60
POOL_JOBS_HIGH = {"P100": 150, "MISC": 400, "T4": 400, "V100": 120, "V100M32": 120}


def horizon_and_slot(n_jobs: int) -> tuple[float, int]:
    """Arrival window + longest possible service + 2h slack. Coarser slots for big
    instances so the time-indexed ILP stays inside a solving budget."""
    hours = BLOCK_HOURS + SERVICE_CAP_HOURS + 2.0
    return hours, (300 if n_jobs <= 80 else 600)


def pick_window(df: pd.DataFrame, pool: str, block_hours: float = BLOCK_HOURS):
    """The busiest contiguous window of one GPU pool, ranked by GPU-card-hours."""
    s_all = df[(df["gpu_type_main"] == pool) & (df["gpu_cards"] > 0)].copy()
    if s_all.empty:
        raise SystemExit(f"no GPU jobs for pool {pool}")
    # A partition that has to host a 100-card request cannot also be a small cluster,
    # so big-monolith requests (pool p90 and above) are excluded and reported: the
    # study then covers the gang-size regime that a scheduler actually reorders.
    req_cap = float(s_all["gpu_cards"].quantile(REQ_CAP_QUANTILE))
    s = s_all[s_all["gpu_cards"] <= req_cap].copy()
    n_pool = len(s_all)
    n_excluded_big = int(n_pool - len(s))
    s = s[s["run_obs"] <= SERVICE_CAP_HOURS * 3600.0]
    n_capped = len(s_all[s_all["gpu_cards"] <= req_cap]) - len(s)
    s["bin"] = np.floor(s["submit_time"] / (block_hours * 3600.0))
    s["gpu_hours"] = s["gpu_cards"] * s["run_obs"] / 3600.0
    ranks = s.groupby("bin").agg(n=("job_name", "size"), demand=("gpu_hours", "sum"))
    ranks = ranks[ranks["n"] >= MIN_JOBS]
    if ranks.empty:
        raise SystemExit(f"no window with >= {MIN_JOBS} jobs in pool {pool}")
    t0 = float(ranks["demand"].idxmax() * block_hours * 3600.0)
    win = s[(s["submit_time"] >= t0) & (s["submit_time"] < t0 + block_hours * 3600.0)].copy()
    stats = {
        "pool_gpu_jobs_total": int(n_pool),
        "gpu_request_cap_cards": round(req_cap, 2),
        "gpu_request_cap_quantile": REQ_CAP_QUANTILE,
        "service_cap_hours": SERVICE_CAP_HOURS,
        "pool_jobs_excluded_as_too_big": n_excluded_big,
        "pool_gpu_jobs_dropped_by_service_cap": int(n_capped),
        "window_jobs_after_filters": int(len(win)),
        "window_demand_gpu_hours": round(float(win["gpu_hours"].sum()), 1),
    }
    return win, t0, stats


def machine_floor(jobs: pd.DataFrame, gpu_per_m: float, cores_per_m: float, mem_per_m: float) -> int:
    return max(
        math.ceil(float(jobs["gpu_cards"].max()) / gpu_per_m),
        math.ceil(float(jobs["cpu_cores"].max()) / cores_per_m),
        math.ceil(float(jobs["mem_gb"].max()) / mem_per_m),
    )


def build_pool(pool: str, n_high: int, seed: int, out_dir: Path) -> list[dict]:
    df = pd.read_csv(PROC / "jobs.csv")
    pools = pd.read_csv(PROC / "pools.csv")
    row = pools[pools["gpu_type"] == pool]
    if row.empty or float(row["cap_gpu_cards"].iloc[0]) == 0:
        raise SystemExit(f"pool {pool} has no GPU capacity in machine spec")
    gpu_per_m = float(row["gpu_per_machine"].iloc[0])
    cores_per_m = float(row["cpu_per_machine"].iloc[0])
    mem_per_m = float(row["cap_mem_gb"].iloc[0]) / float(row["machines"].iloc[0])
    avail = int(row["machines"].iloc[0])

    win, t0, stats = pick_window(df, pool)
    rng = np.random.default_rng(seed)
    take = min(n_high, len(win))
    idx = rng.choice(win.index.to_numpy(), size=take, replace=False)
    high = win.loc[idx].sort_values("submit_time").reset_index(drop=True)

    horizon_h, slot_s = horizon_and_slot(take)
    demand_high = float((high["gpu_cards"] * high["run_obs"] / 3600.0).sum())
    rho_target_high = max(RHO_TIERS)
    m_for_load = math.ceil(demand_high / (rho_target_high * gpu_per_m * horizon_h))
    m_fit = machine_floor(high, gpu_per_m, cores_per_m, mem_per_m)
    m_base = max(1, min(max(m_for_load, m_fit), avail))
    cards = m_base * gpu_per_m
    capacity = {"machines": m_base, "gpu_cards": cards,
                "cpu_cores": m_base * cores_per_m, "mem_gb": m_base * mem_per_m}

    metas = []
    # Nested thinning: a random permutation of the high-load sample, tiers take a prefix.
    perm = rng.permutation(take)
    for rho in RHO_TIERS:
        k = max(1, int(round(take * rho / rho_target_high)))
        sel = np.sort(perm[:k])
        jobs = high.iloc[sel].copy()
        demand = float((jobs["gpu_cards"] * jobs["run_obs"] / 3600.0).sum())
        achieved = demand / (cards * horizon_h)
        jobs = jobs.assign(job_id=np.arange(len(jobs)),
                           arrival_rel=jobs["submit_time"] - t0, service=jobs["run_obs"])
        fname = out_dir / f"instance_{pool}_{str(rho).replace('.', '')}_{len(jobs)}.csv"
        jobs[["job_id", "job_name", "user", "arrival_rel", "service", "cpu_cores",
              "mem_gb", "gpu_cards", "gpu_type_main", "n_tasks", "n_instances"]].to_csv(fname, index=False)
        n_slots = int(round(horizon_h * 3600 / slot_s))
        metas.append({
            "pool": pool, "seed": seed, "target_rho": rho, "achieved_rho": round(achieved, 4),
            "instance_file": fname.name, "n_jobs": int(len(jobs)),
            "nested_subset_of_highest_tier": bool(k < take),
            "cluster_fixed_across_tiers": True,
            **stats, "window_start_seconds_rel_trace": round(t0, 1),
            "window_hours": BLOCK_HOURS, "horizon_hours": horizon_h,
            "slot_seconds": slot_s, "n_slots": n_slots,
            "ilp_binary_count_estimate": int(len(jobs) * n_slots),
            "machine_spec": {"gpu_per_machine": gpu_per_m, "cpu_cores_per_machine": cores_per_m,
                             "mem_gb_per_machine": round(mem_per_m, 1), "machines_available_in_pool": avail},
            "capacity": {kk: round(float(v), 1) for kk, v in capacity.items()},
            "cluster_sizing": {"machines_for_load_target": m_for_load, "machines_for_largest_request": m_fit,
                               "chosen": m_base, "clamped_to_pool": bool(m_base == avail and max(m_for_load, m_fit) > avail)},
            "demand": {
                "gpu_card_hours": round(demand, 1),
                "service_seconds_median": float(jobs["service"].median()),
                "service_seconds_p90": float(jobs["service"].quantile(0.9)),
                "gpu_cards_per_job_median": float(jobs["gpu_cards"].median()),
                "gpu_cards_per_job_max": float(jobs["gpu_cards"].max()),
                "cpu_cores_per_job_median": float(jobs["cpu_cores"].median()),
                "mem_gb_per_job_median": float(jobs["mem_gb"].median()),
                "unique_users": int(jobs["user"].nunique()),
            },
        })
    return metas


def main() -> None:
    global SERVICE_CAP_HOURS, REQ_CAP_QUANTILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="P100,MISC")
    ap.add_argument("--jobs", type=int, default=0, help="jobs in the busiest tier; 0 = pool default")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--service-cap-hours", type=float, default=SERVICE_CAP_HOURS)
    ap.add_argument("--req-cap-quantile", type=float, default=REQ_CAP_QUANTILE)
    args = ap.parse_args()

    SERVICE_CAP_HOURS = args.service_cap_hours
    REQ_CAP_QUANTILE = args.req_cap_quantile

    metas = []
    for pool in args.pools.split(","):
        n_high = args.jobs or POOL_JOBS_HIGH.get(pool, 150)
        ms = build_pool(pool, n_high, args.seed, PROC)
        metas.extend(ms)
        for m in ms:
            print(f"{pool:6s} rho {m['target_rho']:<5} jobs={m['n_jobs']:4d} "
                  f"machines={m['capacity']['machines']:3.0f} cards={m['capacity']['gpu_cards']:4.0f} "
                  f"achieved={m['achieved_rho']:.3f} demand={m['demand']['gpu_card_hours']:6.0f} card-h "
                  f"slots={m['n_slots']} binaries~{m['ilp_binary_count_estimate']:,} "
                  f"(load floor {m['cluster_sizing']['machines_for_load_target']}, "
                  f"fit floor {m['cluster_sizing']['machines_for_largest_request']})")
    (PROC / "instance_meta.json").write_text(json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
