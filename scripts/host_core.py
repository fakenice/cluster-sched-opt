#!/usr/bin/env python3
"""Host-level scheduling primitives: the GPU pool is a set of real machines, not one bag.

The committed study (scripts/03_optimize.py) models a pool as a *single* machine whose
capacity is the sum over hosts: `capacity.gpu_cards = machines * gpu_per_machine`. That
relaxation is what makes an offline optimum cheap, and it is also what makes it
unexecutable: a job may take cards from a host that has none, and several jobs may be
packed onto one host in a way no host could hold.

`sched_core.packing_check` is the study's existing defence: it re-packs every busy slot
with first-fit-decrease onto the real hosts and reports a packability rate. That check is
*per slot and per policy-free*: it re-decides the host set every slot, so a job can be on
host 1 in one slot and host 4 in the next. A container is not migrated, so a slot-wise
packability rate is weaker than the question "is there one assignment that holds for the
whole life of every job".

This module keeps the same instance objects, the same slot granularity and the same
rounding, but places work on discrete hosts:

  * a job that needs `h_j` hosts keeps the same hosts for its whole life (time-consistent);
  * the host set of a multi-host job is chosen greedily, by the largest slack;
  * `h_j` and the per-host share follow the same rule as `sched_core._job_chunks`:
    h_j = max(1, ceil(gpu/gpu_per_machine), ceil(cores/cores_per_machine),
    ceil(mem/mem_per_machine)), share = demand / h_j.

Everything here is a *counterpart* of sched_core, not a replacement: the committed
aggregate numbers stay exactly where they are, and every host-level number is reported
next to its aggregate twin so the difference is auditable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import sched_core as sc

RESOURCES = sc.RESOURCES


@dataclass
class HostModel:
    """Discrete hosts plus the per-job footprint on them."""
    hosts: int
    cap: np.ndarray            # (3,) per-host capacity: gpu cards, cpu cores, mem GB
    p: np.ndarray              # (n,) service length in slots
    share: np.ndarray          # (n, 3) per-host share of the job's demand
    n_host: np.ndarray         # (n,) number of hosts the job needs
    demand: np.ndarray         # (n, 3) the job's total demand (as in the aggregate model)
    earliest: np.ndarray       # (n,) first slot the job may start
    latest: np.ndarray         # (n,) last slot it may start and still finish in the horizon
    n_slots: int               # modelled slots (horizon + documented tail)
    arrival: np.ndarray        # (n,) arrival time in seconds
    slot_seconds: int          # length of one slot in seconds

    @property
    def n_jobs(self) -> int:
        return int(self.p.shape[0])


def build_host_model(inst: sc.Instance, n_slots: int | None = None) -> HostModel:
    """Per-host footprint of every job in an instance (same rule as _job_chunks)."""
    n_machines = int(round(float(inst.capacity["machines"])))
    gpm = float(inst.gpu_per_machine)
    cores_pm = float(inst.capacity["cpu_cores"]) / n_machines
    mem_pm = float(inst.capacity["mem_gb"]) / n_machines
    demand = inst.demand
    n_host = np.maximum(1, np.ceil(np.maximum(
        np.maximum(demand[:, 0] / gpm, demand[:, 1] / cores_pm),
        demand[:, 2] / mem_pm)).astype(int))
    share = demand / n_host[:, None]
    return HostModel(
        hosts=n_machines,
        cap=np.array([gpm, cores_pm, mem_pm], dtype=float),
        p=inst.p.astype(int),
        share=share,
        n_host=n_host,
        demand=demand,
        earliest=inst.earliest.astype(int),
        latest=inst.latest.astype(int),
        n_slots=int(n_slots) if n_slots is not None else sc.window_slots(inst),
        arrival=inst.jobs["arrival_rel"].to_numpy(float),
        slot_seconds=int(inst.slot_seconds),
    )


def subset_model(model: HostModel, subset: np.ndarray) -> HostModel:
    """A view of the same pool holding only `subset` of the jobs (local indices).

    Used by the MILP pilot: the aggregate and the host model are compared on the same
    job subset, and the host audit of the aggregate optimum needs a model whose job
    indices are 0..len(subset)-1.
    """
    idx = np.asarray(subset, dtype=int)
    return HostModel(
        hosts=model.hosts,
        cap=model.cap.copy(),
        p=model.p[idx],
        share=model.share[idx],
        n_host=model.n_host[idx],
        demand=model.demand[idx],
        earliest=model.earliest[idx],
        latest=model.latest[idx],
        n_slots=model.n_slots,
        arrival=model.arrival[idx],
        slot_seconds=model.slot_seconds,
    )


class HostOccupancy:
    """Per-host, per-slot usage of the three resources."""

    def __init__(self, model: HostModel):
        self.m = model
        self.use = np.zeros((model.hosts, model.n_slots, 3), dtype=float)

    # ------------------------------------------------------------------ fitting
    def slack(self, j: int, s: int) -> np.ndarray:
        """Per-host slack (over the job's whole duration) if it started at slot `s`.

        slack[h] = min over the slots it would occupy and over the three resources of
        (host capacity - current use - this job's share). A host can take the job iff
        its slack is >= 0.
        """
        p = int(self.m.p[j])
        e = min(s + p, self.m.n_slots)
        if e <= s:
            return np.full(self.m.hosts, -np.inf)
        block = self.use[:, s:e, :]                       # (H, len, 3)
        return (self.m.cap[None, None, :] - block - self.m.share[j][None, None, :]).min(axis=(1, 2))

    def hosts_for(self, j: int, s: int) -> np.ndarray | None:
        """The host set for job j at slot s, or None when it does not fit anywhere."""
        sl = self.slack(j, s)
        need = int(self.m.n_host[j])
        ok = np.where(sl >= -1e-9)[0]
        if ok.size < need:
            return None
        # the hosts with the most room first, so the tightest fits keep their slack
        order = ok[np.argsort(-sl[ok], kind="stable")]
        return order[:need]

    def fits(self, j: int, s: int) -> bool:
        return self.hosts_for(j, s) is not None

    def earliest_fit(self, j: int, floor_slot: int = 0, must_end_by: int | None = None) -> int | None:
        lo = max(int(self.m.earliest[j]), int(floor_slot))
        hi = self.m.n_slots - int(self.m.p[j])
        if must_end_by is not None:
            hi = min(hi, int(must_end_by) - int(self.m.p[j]))
        for s in range(lo, hi + 1):
            if self.fits(j, s):
                return s
        return None

    def reserve(self, j: int, s: int, hosts: np.ndarray | None = None, force: bool = False) -> None:
        """Reserve a job on `hosts` (or the best-fitting set) for its whole duration.

        `force=True` reserves even when the job does not fit, which is how the two
        fallback paths (a job that fits nowhere in the window) stay visible as an
        overcommit instead of disappearing: the violations counter reports them.
        """
        if hosts is None:
            hosts = self.hosts_for(j, s)
            if hosts is None:
                if not force:
                    raise ValueError(f"job {j} does not fit at slot {s}")
                sl = self.slack(j, s)
                hosts = np.argsort(-sl, kind="stable")[:int(self.m.n_host[j])]
        p = int(self.m.p[j])
        for h in hosts:
            self.use[int(h), s:s + p, :] += self.m.share[j]

    # ------------------------------------------------------------------ audit
    def violations(self) -> dict:
        over = self.use - self.m.cap[None, None, :]
        bad_host_slot = over.max(axis=2) > 1e-6
        return {
            "host_capacity_violation_host_slots": int(bad_host_slot.sum()),
            "max_host_overcommit_cards": round(float(over[:, :, 0].max(initial=0.0)), 3),
        }


# --------------------------------------------------------------------------- audit

def assign_hosts(model: HostModel, starts: np.ndarray, order: np.ndarray | None = None
                 ) -> dict:
    """Try to keep every job at its scheduled start with one time-consistent assignment.

    Jobs are processed in start order; a job takes the hosts with the largest slack, and
    keeps them for its whole life. A job that cannot be placed is reported, not silently
    dropped. This is a *greedy* assignment: the numbers are an upper bound on how much of
    the schedule is executable, because a different assignment might do better.

    Returns the assignment plus the counts the report quotes.
    """
    starts = np.asarray(starts, dtype=int)
    n = len(starts)
    occ = HostOccupancy(model)
    placed = np.zeros(n, dtype=bool)
    hosts_used: dict[int, list[int]] = {}
    if order is None:
        order = np.argsort(starts, kind="stable")
    for j in order:
        j = int(j)
        s = int(starts[j])
        if s < 0:
            continue
        hs = occ.hosts_for(j, s)
        if hs is None:
            continue
        occ.reserve(j, s, hs)
        placed[j] = True
        hosts_used[j] = [int(h) for h in hs]
    return {
        "placed_at_scheduled_start": int(placed.sum()),
        "not_placeable_at_scheduled_start": int((~placed).sum()),
        "share_placeable": round(float(placed.mean()), 4) if n else None,
        "assignment": hosts_used,
        "occupancy": occ,
    }


def repair_schedule(model: HostModel, starts: np.ndarray) -> tuple[np.ndarray, dict]:
    """Make an aggregate schedule executable on real hosts, keeping its ordering.

    Jobs keep their relative order; each is delayed to the first slot where a host set
    is free. Jobs that cannot be placed anywhere in the modelled window stay unplaced and
    are counted, so the price of host reality is a measured delay, not an assumption.
    """
    starts = np.asarray(starts, dtype=int)
    occ = HostOccupancy(model)
    out = np.full(len(starts), -1, dtype=int)
    unplaced: list[int] = []
    for j in np.argsort(starts, kind="stable"):
        j = int(j)
        if starts[j] < 0:
            continue
        floor = max(int(starts[j]), int(model.earliest[j]))
        s = occ.earliest_fit(j, floor_slot=floor)
        if s is None:
            unplaced.append(j)
            continue
        occ.reserve(j, s)
        out[j] = s
    return out, {"unplaced_jobs": len(unplaced),
                 "unplaced_job_ids": [int(j) for j in unplaced],
                 "occupancy": occ}


# --------------------------------------------------------------------------- policies

def run_priority_online_host(model: HostModel, key: np.ndarray,
                              aging_slots: int | None = None) -> np.ndarray:
    """`sched_core.run_priority_online` with discrete hosts.

    Same loop, same tie-breaking, same rounding; only the capacity test changes: a job
    starts when a *set of hosts* can hold it for its whole duration.
    """
    occ = HostOccupancy(model)
    n = model.n_jobs
    starts = np.full(n, -1, dtype=int)
    arrival = model.arrival
    keys = np.asarray(key, dtype=float)
    by_arrival = np.argsort(arrival, kind="stable")
    for s in range(model.n_slots):
        waiting = [int(j) for j in by_arrival
                   if model.earliest[j] <= s and starts[j] < 0]
        if not waiting:
            continue
        if aging_slots is None:
            waiting.sort(key=lambda j: (keys[j], arrival[j], j))
        else:
            waiting.sort(key=lambda j: (0 if s - model.earliest[j] >= aging_slots else 1,
                                        arrival[j] if s - model.earliest[j] >= aging_slots else 0,
                                        keys[j], arrival[j], j))
        for j in waiting:
            hs = occ.hosts_for(j, s)
            if hs is None:
                continue
            occ.reserve(j, s, hs)
            starts[j] = s
    if (starts < 0).any():
        for j in np.where(starts < 0)[0]:
            s = occ.earliest_fit(int(j), floor_slot=int(model.earliest[j]))
            if s is None:
                continue
            occ.reserve(int(j), s)
            starts[j] = s
    still = starts < 0
    if still.any():
        # no host set is free anywhere in the window: keep the job at its release slot and
        # let the violation counter show the overcommit, instead of hiding it
        for j in np.where(still)[0]:
            occ.reserve(int(j), int(model.earliest[j]), force=True)
        starts[still] = model.earliest[still]
    return np.minimum(starts, model.n_slots - 1).astype(int), occ


def run_fifo_host(model: HostModel, backfill: bool = False) -> np.ndarray:
    """`sched_core.run_fifo` with discrete hosts (strict FIFO and EASY backfill)."""
    occ = HostOccupancy(model)
    n = model.n_jobs
    starts = np.full(n, -1, dtype=int)
    queue = np.argsort(model.arrival, kind="stable")
    i = 0
    while i < len(queue):
        h = int(queue[i])
        head_slot = occ.earliest_fit(h, floor_slot=int(model.earliest[h]))
        if head_slot is None:
            head_slot = int(model.earliest[h])
        if not backfill:
            last_start = 0
            for k in range(i, len(queue)):
                j = int(queue[k])
                if starts[j] >= 0:
                    continue
                s = occ.earliest_fit(j, floor_slot=max(int(model.earliest[j]), last_start))
                if s is None:
                    s = occ.earliest_fit(j, floor_slot=int(model.earliest[j]))
                    if s is None:
                        s = int(model.earliest[j])
                occ.reserve(j, s)
                starts[j] = s
                last_start = s
            break
        if head_slot <= int(model.earliest[h]):
            occ.reserve(h, head_slot)
            starts[h] = head_slot
            i += 1
            continue
        for k in range(i + 1, len(queue)):
            j = int(queue[k])
            if starts[j] >= 0:
                continue
            s = occ.earliest_fit(j, floor_slot=int(model.earliest[j]), must_end_by=int(head_slot))
            if s is not None:
                occ.reserve(j, s)
                starts[j] = s
        occ.reserve(h, head_slot)
        starts[h] = head_slot
        i += 1
    if (starts < 0).any():
        for j in np.where(starts < 0)[0]:
            s = occ.earliest_fit(int(j), floor_slot=int(model.earliest[j]))
            if s is None:
                continue
            occ.reserve(int(j), s)
            starts[j] = s
    still = starts < 0
    if still.any():
        for j in np.where(still)[0]:
            occ.reserve(int(j), int(model.earliest[j]), force=True)
        starts[still] = model.earliest[still]
    return np.minimum(starts, model.n_slots - 1).astype(int), occ


# --------------------------------------------------------------------------- metrics

def host_idle_metrics(model: HostModel, starts: np.ndarray, occupancy: HostOccupancy | None = None
                      ) -> dict:
    """Host-level idle accounting.

    * idle_with_backlog_card_hours: GPU-card-hours standing unused on a host while at
      least one job is already submitted and still waiting.
    * idle_addressable_backlog_card_hours: the same, but only the cards on a host where
      some waiting job's *per-host share* would fit right now.

    The aggregate version of this metric (`06_backlog_idle.py`) asks whether a waiting job
    fits in the pool's spare capacity. The host-level version asks the stricter question,
    per host: "could this job start on this host, right now?". The difference between the
    two is the part of the waste that only exists because the pool was treated as one
    machine.
    """
    starts = np.asarray(starts, dtype=int)
    if occupancy is None:
        occupancy = HostOccupancy(model)
        for j in range(model.n_jobs):
            if starts[j] >= 0:
                occupancy.reserve(j, int(starts[j]))
    d = model.slot_seconds
    idle_bh = 0.0
    idle_addr = 0.0
    slots_with_backlog = 0
    depths = []
    for s in range(model.n_slots):
        waiting = np.where((model.earliest <= s) & (starts > s))[0]
        depths.append(int(waiting.size))
        if waiting.size == 0:
            continue
        slots_with_backlog += 1
        free = model.cap[None, :] - occupancy.use[:, s, :]        # (H, 3)
        free_gpu = np.clip(free[:, 0], 0.0, None)
        idle_bh += float(free_gpu.sum()) * d / 3600.0
        # could *any* queued job use the spare capacity of this particular host?
        need = model.share[waiting]                               # (W, 3)
        ok = np.all(need[:, None, :] <= free[None, :, :] + 1e-9, axis=2)   # (W, H)
        idle_addr += float(free_gpu[ok.any(axis=0)].sum()) * d / 3600.0
    window_card_hours = model.cap[0] * model.n_slots * d / 3600.0
    return {
        "idle_with_backlog_card_hours": round(idle_bh, 2),
        "idle_with_backlog_share": round(idle_bh / max(window_card_hours, 1e-9), 4),
        "idle_addressable_backlog_card_hours": round(idle_addr, 2),
        "idle_addressable_backlog_share": round(idle_addr / max(window_card_hours, 1e-9), 4),
        "addressable_share_of_backlog_idle": round(idle_addr / max(idle_bh, 1e-9), 3),
        "slots_with_backlog": slots_with_backlog,
        "mean_queue_depth": round(float(np.mean(depths)), 3),
    }
