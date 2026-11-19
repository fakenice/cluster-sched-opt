#!/usr/bin/env python3
"""10_aging_threshold.py -- re-derive the aging threshold on held-out windows.

The committed study (07_priority_baselines.py) reports priority-gpu with aging at
1 / 2 / 4 h, thresholds that were picked by inspection on the *tuning set*: the
busiest window of each pool at the base seed (window_rank=1, seed=20261013, the
6 committed instances). Nothing checked whether that choice transfers, and the
sparse trio says nothing about where the optimum sits between 0.5 h and 6 h.

This experiment closes exactly that gap:

  * grid      -- aging thresholds {0.5, 1, 2, 3, 4, 6} h (converted to slots at
                 each instance's own slot length), plus no-aging priority-gpu;
  * tuning    -- the 6 committed instances: the grid's argmin is recorded so the
                 holdout comparison is "a rule re-derived on the cells it was
                 derived on" vs "fixed at the committed trio";
  * holdout   -- the other 66 sweep instances (window ranks 2, 3, median and
                 seeds 20261103 / 20261104, including the rank-1 windows re-drawn
                 with other seeds): never used for any committed decision.

Per instance we replay every threshold through the same primitives as the study
(sched_core.run_priority_online + metrics/deadlines/backlog), then report:

  1. per-cell argmin of mean wait and of max wait (aging's tail motivation);
  2. regret of the fixed committed thresholds vs the per-cell oracle, on holdout;
  3. how often each threshold wins on holdout (ties -> smaller threshold);
  4. sign tests between adjacent thresholds, so a ranking claim is countable.

No ILP is solved here: the question is threshold choice among online rules, and
every comparator in the grid is deterministic. Control: FIFO and priority-gpu
rows are recomputed and must match results/sweep_metrics.csv for the cells both
files cover (checked in the summary, max relative deviation reported).

Outputs results/aging_holdout.csv (long form) and results/aging_holdout.json.

The replay is the expensive part (~40 min for 72 instances x 8 policies), so
`--reuse-csv` re-derives the summary from an existing results/aging_holdout.csv
without re-simulating. It refuses to do so unless the file covers exactly the
instance set in instance_meta.json and the full policy grid, so a stale or
partial table can never silently produce a headline number.

Usage
  python scripts/10_aging_threshold.py              # replay + summary
  python scripts/10_aging_threshold.py --reuse-csv  # summary from existing CSV
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sched_core as sc   # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "backlog_idle", ROOT / "scripts" / "06_backlog_idle.py")
_backlog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backlog)
backlog_metrics = _backlog.backlog_metrics

PROC = ROOT / "data" / "processed"
SWEEP = PROC / "sweep"
RES = ROOT / "results"

BASE_SEED = 20261013
GRID_HOURS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]   # committed trio is {1,2,4}
COMMITTED = ["priority-gpu-age1h", "priority-gpu-age2h", "priority-gpu-age4h"]


def policy_name(hours: float) -> str:
    h = f"{hours:g}"
    return f"priority-gpu-age{h}h"


def aging_slots(hours: float, slot_seconds: int) -> int:
    """Threshold in wall-clock hours -> slots at this instance's slot length.

    The committed 07 grid hard-coded 6/12/24 slots at 600 s; sweeping the same
    *hours* across instances whose slot length could differ keeps the policy the
    same one an operator would configure (a timer, not a slot count).
    """
    return max(1, math.ceil(hours * 3600.0 / slot_seconds))


def replay(meta: dict, inst, hours: float | None) -> dict:
    starts = sc.run_priority_online(inst, key="gpu_cards",
                                    aging_slots=None if hours is None
                                    else aging_slots(hours, inst.slot_seconds))
    met = sc.metrics(inst, starts)
    met.update(sc.add_deadlines(inst, starts))
    met.update(sc.packing_check(inst, starts))
    met.update(backlog_metrics(inst, starts))
    assert met["capacity_violation_slots"] == 0, (
        f"{meta['instance_file']}/{hours}h overcommitted capacity")
    return met


def load_cached_csv(metas: list[dict]) -> pd.DataFrame:
    """Return a previously written replay, but only if it covers this grid."""
    path = RES / "aging_holdout.csv"
    if not path.exists():
        raise SystemExit(f"--reuse-csv: {path} does not exist -- run without the flag first")
    df = pd.read_csv(path)
    want_instances = {m["instance_file"] for m in metas}
    have_instances = set(df["instance"].unique())
    if have_instances != want_instances:
        missing = sorted(want_instances - have_instances)
        extra = sorted(have_instances - want_instances)
        raise SystemExit(
            f"--reuse-csv: instance coverage mismatch "
            f"(missing {len(missing)}, extra {len(extra)}); "
            f"first missing: {missing[:3]} first extra: {extra[:3]}")
    want_policies = {"FIFO", "priority-gpu", *(policy_name(h) for h in GRID_HOURS)}
    have_policies = set(df["policy"].unique())
    if have_policies != want_policies:
        raise SystemExit(
            f"--reuse-csv: policy set mismatch, expected {sorted(want_policies)}, "
            f"found {sorted(have_policies)}")
    per_cell = df.groupby("instance")["policy"].nunique()
    if (per_cell != len(want_policies)).any():
        bad = per_cell[per_cell != len(want_policies)]
        raise SystemExit(
            f"--reuse-csv: {len(bad)} instances lack the full policy grid "
            f"(e.g. {bad.index[0]} has {bad.iloc[0]})")
    print(f"[reuse] {path.name}: {len(df)} rows, {len(have_instances)} instances x "
          f"{len(want_policies)} policies -- skipping replay", flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-csv", action="store_true",
                    help="rebuild the summary from an existing aging_holdout.csv")
    args = ap.parse_args()

    metas = json.loads((SWEEP / "instance_meta.json").read_text(encoding="utf-8"))
    if args.reuse_csv:
        df = load_cached_csv(metas)
    else:
        rows = []
        for m in metas:
            inst = sc.load(SWEEP / m["instance_file"], m)
            cell = m["sweep_key"]                       # e.g. P100_w2_s20261103_rho0.6
            tuning = (m.get("window_rank_label", "rank1") == "rank1"
                      and m["seed"] == BASE_SEED)
            fifo = sc.run_fifo(inst)
            fm = sc.metrics(inst, fifo)
            fm.update(sc.add_deadlines(inst, fifo))
            rows.append({"cell": cell, "pool": m["pool"], "instance": m["instance_file"],
                         "window_rank": m.get("window_rank_label"), "seed": m["seed"],
                         "target_rho": m["target_rho"], "achieved_rho": m["achieved_rho"],
                         "role": "tuning" if tuning else "holdout",
                         "policy": "FIFO", "threshold_hours": None,
                         "aging_slots": None, "slot_seconds": inst.slot_seconds,
                         **fm})
            for hours in [None, *GRID_HOURS]:
                name = "priority-gpu" if hours is None else policy_name(hours)
                rows.append({"cell": cell, "pool": m["pool"],
                             "instance": m["instance_file"],
                             "window_rank": m.get("window_rank_label"), "seed": m["seed"],
                             "target_rho": m["target_rho"],
                             "achieved_rho": m["achieved_rho"],
                             "role": "tuning" if tuning else "holdout",
                             "policy": name, "threshold_hours": hours,
                             "aging_slots": (None if hours is None
                                             else aging_slots(hours, inst.slot_seconds)),
                             "slot_seconds": inst.slot_seconds,
                             **replay(m, inst, hours)})
            print(f"done {m['instance_file']}", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(RES / "aging_holdout.csv", index=False)

    summary = summarise(df, write_json=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "per_cell_argmin"}, ensure_ascii=False, indent=2))
    return 0


def summarise(df: pd.DataFrame, res_dir: Path | None = None,
              write_json: bool = False) -> dict:
    """Turn the long replay table into the tuning-vs-holdout verdict.

    Split out of main() so the argmin merge and the transfer test are checkable on
    a hand-built frame. The committed run crashed in exactly this code (a missing
    'role' key in the merge) only *after* ~40 min of replay, which is what
    tests/test_aging_reuse.py pins.
    """
    res_dir = RES if res_dir is None else Path(res_dir)

    # ---- control vs the committed sweep table -------------------------------
    control = {"status": "skipped", "reason": "sweep_metrics.csv not found"}
    sm_path = res_dir / "sweep_metrics.csv"
    if sm_path.exists():
        sm = pd.read_csv(sm_path)
        mine = df[df.policy.isin(["FIFO", "priority-gpu"])]
        key_sm = sm.set_index(["sweep_key", "policy"])["mean_wait_s"]
        diffs = []
        for _, r in mine.iterrows():
            k = (r["cell"], r["policy"])
            if k in key_sm.index:
                ref = float(key_sm.loc[k])
                if ref > 0:
                    diffs.append(abs(r["mean_wait_s"] - ref) / ref)
        control = {"checked_rows": len(diffs),
                   "max_relative_deviation": (max(diffs) if diffs else None)}

    # ---- per-cell argmin -----------------------------------------------------
    pol = df[~df.policy.eq("FIFO")]
    idx_w = pol.groupby(["cell", "role"])["mean_wait_s"].idxmin()
    idx_max = pol.groupby(["cell", "role"])["max_wait_s"].idxmin()
    argmin = pol.loc[idx_w, ["cell", "role", "policy"]].rename(
        columns={"policy": "argmin_mean_wait"})
    argmin = argmin.merge(
        pol.loc[idx_max, ["cell", "role", "policy"]].rename(
            columns={"policy": "argmin_max_wait"}), on=["cell", "role"])

    hold = pol[pol.role.eq("holdout")]
    piv_w = hold.pivot_table(index="cell", columns="policy", values="mean_wait_s")
    piv_x = hold.pivot_table(index="cell", columns="policy", values="max_wait_s")
    oracle_w = piv_w.min(axis=1)

    # Column order used for tie-breaking: ascending threshold, no-aging last (an
    # infinite threshold), so "ties -> smaller threshold" holds by construction
    # rather than by pandas' alphabetical column sort.
    grid_order = [policy_name(h) for h in sorted(GRID_HOURS)] + ["priority-gpu"]
    piv_w = piv_w[grid_order]
    piv_x = piv_x[grid_order]

    def regret(col: str) -> dict:
        fixed = piv_w[col]
        rel = (fixed - oracle_w) / oracle_w
        return {"mean_regret_pct": round(float(rel.mean() * 100), 3),
                "median_regret_pct": round(float(rel.median() * 100), 3),
                "max_regret_pct": round(float(rel.max() * 100), 3),
                "cells_where_worse_than_noaging": int(
                    (piv_w[col] > piv_w["priority-gpu"]).sum())}

    wins = (piv_w.idxmin(axis=1).value_counts().rename_axis("policy")
            .rename("cells_won").reindex(grid_order, fill_value=0).to_dict())
    sign_tests = {}
    for a, b in [(1.0, 2.0), (2.0, 4.0), (1.0, 4.0), (2.0, 3.0), (3.0, 4.0)]:
        pa, pb = policy_name(a), policy_name(b)
        better = int((piv_w[pa] < piv_w[pb]).sum())
        sign_tests[f"{pa}_vs_{pb}_mean_wait"] = {
            "cells": int(len(piv_w)), f"{pa}_better_in": better,
            f"{pb}_better_in": int(len(piv_w)) - better}

    tail = {}
    for col in [*COMMITTED, "priority-gpu", policy_name(6.0)]:
        rel = (piv_x[col] - piv_x.min(axis=1)) / piv_x.min(axis=1)
        tail[col] = {"median_max_wait_regret_pct": round(float(rel.median() * 100), 2),
                     "cells_argmin": int((piv_x.idxmin(axis=1) == col).sum())}

    # ---- transfer test -------------------------------------------------------
    # The committed caveat is that 1 / 2 / 4 h were "chosen by inspection" on the
    # tuning cells. So: take the choice an operator reading ONLY the 6 tuning
    # cells would make, then score that fixed choice on the 66 holdout cells.
    # Relative-to-oracle waits are what makes the two cell sets comparable.
    tuning = pol[pol.role.eq("tuning")]
    piv_t = tuning.pivot_table(index="cell", columns="policy",
                               values="mean_wait_s")[grid_order]
    rel_t = piv_t.div(piv_t.min(axis=1), axis=0)      # 1.0 = per-cell oracle
    rel_h = piv_w.div(oracle_w, axis=0)
    hold_argmin = piv_w.idxmin(axis=1)
    tuning_pick = rel_t.mean(axis=0).idxmin()
    holdout_pick = rel_h.mean(axis=0).idxmin()
    transfer = {
        "tuning_cells": int(piv_t.shape[0]),
        "holdout_cells": int(piv_w.shape[0]),
        "tuning_mean_relative_wait": {c: round(float(v), 5)
                                      for c, v in rel_t.mean(axis=0).items()},
        "tuning_choice": tuning_pick,
        "tuning_per_cell_argmin": {str(k): v for k, v in piv_t.idxmin(axis=1).items()},
        "holdout_mean_relative_wait": {c: round(float(v), 5)
                                       for c, v in rel_h.mean(axis=0).items()},
        "holdout_choice": holdout_pick,
        "tuning_choice_matches_holdout_choice": bool(tuning_pick == holdout_pick),
        "committed_headline": "priority-gpu-age1h",
        "tuning_choice_holdout_win_share": round(
            float((hold_argmin == tuning_pick).mean()), 4),
        "committed_1h_holdout_win_share": round(
            float((hold_argmin == "priority-gpu-age1h").mean()), 4),
    }

    summary = {
        "generated_by": "scripts/10_aging_threshold.py",
        "note": ("Dense aging-threshold grid replayed on every sweep instance. "
                 "Tuning role = the committed rank-1/base-seed cells; holdout = "
                 "every other cell. Regret is vs the per-cell oracle over the grid."),
        "grid_hours": GRID_HOURS,
        "instances": {"tuning": int(df[df.role.eq('tuning')].cell.nunique()),
                      "holdout": int(df[df.role.eq('holdout')].cell.nunique())},
        "fifo_priority_control_vs_sweep": control,
        "per_cell_argmin": argmin.to_dict(orient="records"),
        "holdout_regret_vs_oracle_mean_wait": {c: regret(c) for c in piv_w.columns},
        "holdout_argmin_wins_mean_wait": wins,
        "holdout_sign_tests_mean_wait": sign_tests,
        "holdout_tail_max_wait": tail,
        "tuning_to_holdout_transfer": transfer,
    }
    if write_json:
        (res_dir / "aging_holdout.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
