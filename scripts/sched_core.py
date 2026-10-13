#!/usr/bin/env python3
"""Shared scheduling primitives: instance loading, slot occupancy, metrics, packing check.

Everything is slot-based (one slot = `slot_seconds`) so that the online baselines and
the offline optimiser are compared under identical granularity, identical rounding and
identical capacity assumptions. Resources tracked per slot: GPU cards, CPU cores, memory GB.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RESOURCES = ("gpu_cards", "cpu_cores", "mem_gb")


@dataclass
class Instance:
    pool: str
    rho: float
    jobs: pd.DataFrame          # job_id, arrival_rel, service, cpu_cores, mem_gb, gpu_cards
    capacity: dict              # gpu_cards, cpu_cores, mem_gb, machines
    slot_seconds: int
    n_slots: int
    gpu_per_machine: float
    meta: dict = field(default_factory=dict)

    @property
    def p(self) -> np.ndarray:
        """Service demand rounded up to whole slots (non-preemptive)."""
        return np.ceil(self.jobs["service"].to_numpy() / self.slot_seconds).astype(int)

    @property
    def earliest(self) -> np.ndarray:
        return np.ceil(self.jobs["arrival_rel"].to_numpy() / self.slot_seconds).astype(int)

    @property
    def latest(self) -> np.ndarray:
        return self.n_slots - self.p

    @property
    def demand(self) -> np.ndarray:
        return self.jobs[list(RESOURCES)].to_numpy(dtype=float)

    def feasible(self) -> tuple[np.ndarray, np.ndarray]:
        ok = (self.p <= self.n_slots) & (self.earliest <= self.latest)
        return ok, self.p


def load(path, meta: dict) -> Instance:
    jobs = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    return Instance(
        pool=meta["pool"], rho=meta["achieved_rho"], jobs=jobs,
        capacity=meta["capacity"], slot_seconds=meta["slot_seconds"], n_slots=meta["n_slots"],
        gpu_per_machine=meta["machine_spec"]["gpu_per_machine"], meta=meta,
    )


def load_instances(proc_dir) -> list[tuple[dict, Instance]]:
    from pathlib import Path
    proc_dir = Path(proc_dir)
    metas = json.loads((proc_dir / "instance_meta.json").read_text(encoding="utf-8"))
    out = []
    for m in metas:
        out.append((m, load(proc_dir / m["instance_file"], m)))
    return out


# --------------------------------------------------------------------------- occupancy
def tail_slots(inst: "Instance") -> int:
    """Extra slots after the planning horizon in which late admits may still run.

    The horizon is a *planning window*, not a cluster shutdown. Without a tail, an online
    policy that cannot place a job would have to stack it onto the last feasible slot,
    which fabricates both a short makespan and a schedule that uses more cards than the
    pool owns. Placements are therefore allowed to overflow into a documented tail, and the
    overflow shows up honestly as `jobs_running_past_horizon` plus a longer makespan.
    """
    return int(2 * inst.p.max() + 12)


def window_slots(inst: "Instance") -> int:
    """Slots modelled for occupancy/placement: the planning horizon plus the tail."""
    return int(inst.n_slots + tail_slots(inst))


class Occupancy:
    """Per-slot usage vector for the three resources."""

    def __init__(self, inst: "Instance", n_slots: int | None = None):
        self.inst = inst
        self.n = int(n_slots) if n_slots is not None else window_slots(inst)
        self.use = np.zeros((self.n, 3), dtype=float)
        self.cap = np.array([inst.capacity[r] for r in RESOURCES], dtype=float)

    def fits(self, j: int, s: int) -> bool:
        p = int(self.inst.p[j])
        if s + p > self.n:
            return False
        need = self.inst.demand[j]
        block = self.use[s:s + p]
        return bool(np.all((block + need) <= self.cap + 1e-9))

    def reserve(self, j: int, s: int) -> None:
        p = int(self.inst.p[j])
        self.use[s:s + p] += self.inst.demand[j]

    def earliest_fit(self, j: int, floor_slot: int = 0, must_end_by: int | None = None) -> int | None:
        p = int(self.inst.p[j])
        lo = max(int(self.inst.earliest[j]), int(floor_slot))
        hi = self.n - p
        if must_end_by is not None:
            hi = min(hi, int(must_end_by) - p)
        for s in range(lo, hi + 1):
            if self.fits(j, s):
                return s
        return None

    def violations(self) -> dict:
        """Slots where the reserved demand exceeds what the pool owns."""
        over = self.use - self.cap[None, :]
        gpu_over = over[:, 0]
        bad = gpu_over > 1e-6
        return {
            "capacity_violation_slots": int((over > 1e-6).any(axis=1).sum()),
            "gpu_overcommit_slots": int(bad.sum()),
            "max_gpu_overcommit_cards": round(float(max(gpu_over.max(), 0.0)), 3),
        }


def first_fit_leftovers(inst: Instance, occ: Occupancy, pending: np.ndarray,
                        floor_slot: np.ndarray | None = None) -> tuple[np.ndarray, list[int]]:
    """Capacity-aware placement for jobs a policy could not schedule inside the horizon.

    Returns the starts (−1 where even the padded window has no room) and the list of jobs
    that stayed unplaceable, so a gap can never be papered over by stacking jobs at the
    last slot. Pending jobs are placed in arrival order for determinism.
    """
    starts = -np.ones(len(inst.jobs), dtype=int)
    unplaceable: list[int] = []
    for j in sorted(int(v) for v in pending):
        fl = int(inst.earliest[j]) if floor_slot is None else max(int(inst.earliest[j]), int(floor_slot[j]))
        s = occ.earliest_fit(j, floor_slot=fl)
        if s is None:
            unplaceable.append(j)
            continue
        occ.reserve(j, s)
        starts[j] = s
    return starts, unplaceable


# --------------------------------------------------------------------------- policies
def run_fifo(inst: Instance, backfill: bool = False) -> np.ndarray:
    """Online scheduling in arrival order.

    backfill=False -> strict FIFO with head-of-line blocking: a job may never start
                      before the job ahead of it in the queue has started.
    backfill=True  -> EASY backfill: when the head-of-line job is blocked, later jobs
                      are pulled forward only if they run to completion before the head
                      could start, so the head is never delayed.
    """
    occ = Occupancy(inst)
    n = len(inst.jobs)
    starts = -np.ones(n, dtype=int)
    queue = np.argsort(inst.jobs["arrival_rel"].to_numpy(), kind="stable")
    p = inst.p
    earliest = inst.earliest

    i = 0
    while i < len(queue):
        h = int(queue[i])
        head_slot = occ.earliest_fit(h, floor_slot=int(earliest[h]))
        if head_slot is None:                      # no room even in the padded window
            head_slot = int(earliest[h])           # kept honest by capacity_violation_slots
        if not backfill:
            # head-of-line blocking: nobody may overtake the job queued ahead of it
            last_start = 0
            for k in range(i, len(queue)):
                j = int(queue[k])
                if starts[j] >= 0:
                    continue
                s = occ.earliest_fit(j, floor_slot=max(int(earliest[j]), last_start))
                if s is None:
                    s = occ.earliest_fit(j, floor_slot=int(earliest[j]))
                    if s is None:
                        s = int(earliest[j])
                occ.reserve(j, s)
                starts[j] = s
                last_start = s
            break
        if head_slot <= int(earliest[h]):          # head runs straight away
            occ.reserve(h, head_slot)
            starts[h] = head_slot
            i += 1
            continue
        # head is blocked at head_slot: let later jobs backfill strictly before it
        for k in range(i + 1, len(queue)):
            j = int(queue[k])
            if starts[j] >= 0:
                continue
            s = occ.earliest_fit(j, floor_slot=int(earliest[j]), must_end_by=int(head_slot))
            if s is not None:
                occ.reserve(j, s)
                starts[j] = s
        occ.reserve(h, head_slot)
        starts[h] = head_slot
        i += 1

    # any job the main loop never reached: place it by first fit, capacity-aware
    if (starts < 0).any():
        extra, _ = first_fit_leftovers(inst, occ, np.where(starts < 0)[0])
        miss = starts < 0
        starts[miss] = extra[miss]
    still = starts < 0
    if still.any():                    # saturated beyond even the padded window (rare)
        starts[still] = earliest[still]
    return np.minimum(starts, window_slots(inst) - 1).astype(int)


# --------------------------------------------------------------------------- metrics
def metrics(inst: Instance, starts: np.ndarray) -> dict:
    """Policy comparison metrics on a common footing.

    Two idle rates are reported because the choice of denominator changes the answer:
      * gpu_idle_rate_fixed_window — allocated card-hours over the same fixed horizon for
        every policy. This is the comparable one (a policy cannot lower it by simply
        finishing earlier), and its floor is 1 - rho, the share of capacity the workload
        itself never uses.
      * gpu_idle_rate_busy_period — over each policy's own busy period, i.e. how tightly
        that policy packs its own work. Reported as a secondary view, not as "utilisation".
    Idle here means *unallocated*: we count reserved cards, not SM utilisation, because the
    released trace's per-instance sensor table is not part of this experiment.
    """
    d = inst.slot_seconds
    starts = clamp_starts(inst, starts)
    arrivals = inst.jobs["arrival_rel"].to_numpy()
    p = inst.p
    start_t = starts * d
    finish_t = start_t + p * d
    wait = start_t - arrivals
    flow = finish_t - arrivals
    makespan = float(finish_t.max() - arrivals.min())
    cap_cards = float(inst.capacity["gpu_cards"])
    occ = Occupancy(inst)                       # horizon + documented tail
    for j in range(len(starts)):
        occ.reserve(j, int(starts[j]))
    used_card_hours = float(occ.use[:, 0].sum()) * d / 3600.0
    used_in_horizon = float(occ.use[:inst.n_slots, 0].sum()) * d / 3600.0
    horizon_usage = occ.use[:inst.n_slots]
    window = finish_t.max() - min(arrivals.min(), start_t.min())
    idle_busy = 1.0 - used_card_hours / (cap_cards * window / 3600.0) if window > 0 else float("nan")
    horizon_h = inst.n_slots * d / 3600.0
    idle_fixed = 1.0 - used_in_horizon / (cap_cards * horizon_h)
    return {
        "n_jobs": int(len(starts)),
        "mean_wait_s": float(wait.mean()),
        "median_wait_s": float(np.median(wait)),
        "p90_wait_s": float(np.quantile(wait, 0.9)),
        "max_wait_s": float(wait.max()),
        "mean_flow_s": float(flow.mean()),
        "sum_completion_s": float(finish_t.sum()),
        "makespan_s": makespan,
        "throughput_jobs_per_hour": float(len(starts) / (makespan / 3600.0)) if makespan > 0 else float("nan"),
        "used_gpu_card_hours": round(used_card_hours, 2),
        "used_gpu_card_hours_in_horizon": round(used_in_horizon, 2),
        "jobs_running_past_horizon": int(((starts + p) > inst.n_slots).sum()),
        "gpu_idle_rate_fixed_window": round(float(idle_fixed), 4),
        "gpu_idle_rate_busy_period": round(float(idle_busy), 4),
        "theoretical_min_idle_fixed_window": round(1.0 - float(inst.rho), 4),
        "cpu_peak_util": round(float(horizon_usage[:, 1].max() / inst.capacity["cpu_cores"]), 4),
        "mem_peak_util": round(float(horizon_usage[:, 2].max() / inst.capacity["mem_gb"]), 4),
        "gpu_slot_util": round(float(horizon_usage[:, 0].mean() / cap_cards), 4),
        "deadline_miss_rate": None,
        **occ.violations(),
    }


def add_deadlines(inst: Instance, starts: np.ndarray, slack_minutes: float = 30.0) -> dict:
    """A job's due date is arrival + service + slack: the interactive-latency contract a
    production scheduler is usually judged on."""
    d = inst.slot_seconds
    due = inst.jobs["arrival_rel"].to_numpy() + inst.p * d + slack_minutes * 60.0
    finish = starts * d + inst.p * d
    late = np.maximum(finish - due, 0.0)
    return {"deadline_slack_minutes": slack_minutes,
            "deadline_miss_rate": float((late > 0).mean()),
            "mean_tardiness_s": float(late.mean()),
            "max_tardiness_s": float(late.max())}


# --------------------------------------------------------------------------- packing
def clamp_starts(inst: Instance, starts: np.ndarray) -> np.ndarray:
    """Index guard only: keep every job inside the modelled window (horizon + tail) so the
    occupancy arrays never run past their rows. Deliberately *not* a way to hide late
    placements — clamping to the horizon would turn an infeasible schedule into a silent one."""
    hi = np.maximum(0, window_slots(inst) - inst.p)
    return np.clip(starts, 0, hi).astype(int)


def _job_chunks(inst: Instance, j: int, cores_pm: float, mem_pm: float) -> list[tuple[float, float, float]]:
    """Split one job into per-host pieces (gang placement): a container takes at most
    `gpu_per_machine` cards, `cores_pm` cores and `mem_pm` GB from any single host, so the
    minimum host count is the max over the three dimensions, not just the GPU one.
    Approximation: the request is spread uniformly over those hosts."""
    gpu = float(inst.demand[j, 0])
    cores = float(inst.demand[j, 1])
    mem = float(inst.demand[j, 2])
    gpm = float(inst.gpu_per_machine)
    n_host = max(1, int(math.ceil(gpu / gpm)), int(math.ceil(cores / cores_pm)),
                 int(math.ceil(mem / mem_pm)))
    return [(gpu / n_host, cores / n_host, mem / n_host)] * n_host


def packing_check(inst: Instance, starts: np.ndarray) -> dict:
    """The ILP and the baselines treat a pool as one big machine (capacity aggregated
    across machines). Real GPUs live in discrete hosts, so a slot is only executable if
    the concurrent jobs fit into the actual machines. First-fit-decrease per slot gives a
    feasibility rate and the fragmentation penalty of the relaxation."""
    d = inst.slot_seconds
    starts = clamp_starts(inst, starts)
    occ = Occupancy(inst)
    for j in range(len(starts)):
        occ.reserve(j, int(starts[j]))
    n_machines = int(round(float(inst.capacity["machines"])))
    gpm = inst.gpu_per_machine
    cores_pm = inst.capacity["cpu_cores"] / n_machines
    mem_pm = inst.capacity["mem_gb"] / n_machines

    n_slots = int((starts + inst.p).max())
    checked = feasible = 0
    worst = 0.0
    for s in range(n_slots):
        if occ.use[s, 0] <= 1e-9:
            continue
        checked += 1
        idx = np.where((starts <= s) & (starts + inst.p > s))[0]
        items: list[tuple[float, float, float]] = []
        for i in idx:
            items.extend(_job_chunks(inst, int(i), cores_pm, mem_pm))
        # FFD on GPU cards, then verify cores / memory of each bin
        items.sort(key=lambda r: -r[0])
        bins: list[list[float]] = []
        ok = True
        for gpu, core, mem in items:
            placed = False
            for b in bins:
                if b[0] + gpu <= gpm + 1e-9 and b[1] + core <= cores_pm + 1e-9 and b[2] + mem <= mem_pm + 1e-9:
                    b[0] += gpu; b[1] += core; b[2] += mem
                    placed = True
                    break
            if not placed:
                if gpu <= gpm + 1e-9 and core <= cores_pm + 1e-9 and mem <= mem_pm + 1e-9:
                    bins.append([gpu, core, mem])
                else:
                    ok = False
                    break
        if ok and len(bins) <= n_machines:
            feasible += 1
        else:
            worst = max(worst, 1.0 - n_machines / max(len(bins), 1))
    return {"slots_with_load": checked, "slots_packable_on_real_machines": feasible,
            "packing_feasibility_rate": round(feasible / checked, 4) if checked else None,
            "worst_fragmentation_loss": round(float(worst), 4)}
