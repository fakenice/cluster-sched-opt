# Trace-driven GPU cluster scheduling: can an optimiser beat FIFO?

Repo: <https://github.com/fakenice/cluster-sched-opt> · 中文一页结论：[`一页结论.md`](一页结论.md)

Replaying a **real** production GPU-cluster arrival stream under FIFO with head-of-line
blocking, EASY backfill, a time-indexed integer program solved with HiGHS, its rolling online
variant, and four request-only size-priority rules — and measuring what reordering work actually
buys: replay wait, throughput, wasted GPU capacity, deadline misses, and the cost of the model's
own relaxations. **The main result is that a one-line online priority rule captures 77–114 % of
the optimiser's mean-wait gain at zero solve time; the optimiser's unique contribution is
smaller than the headline "−71 to −88 %" suggests.**

Everything runs from public data, one command, no cluster access:

```bash
pip install -r requirements.txt
python run_all.py --fetch      # downloads the trace and checks its sha256, then runs every stage
```

## 1. Data

[Alibaba PAI `cluster-trace-gpu-v2020`](https://github.com/alibaba/clusterdata) — 1,055,501 job
records, 1,261,050 task records, 1,897 machines, 6,742 GPUs, Jul–Aug 2020. If you reuse it, cite
Weng et al., *MLaaS in the Wild: Workload Analysis and Scheduling at Large Scale GPU Clusters*,
**NSDI '22**.

`run_all.py --fetch` verifies the three archives against the checksums published in the trace README:

| file | sha256 |
|---|---|
| `pai_job_table.tar.gz` | `5aad7f7c…e0a6cb0` |
| `pai_task_table.tar.gz` | `cd1d6dc3…9ac40e5` |
| `pai_machine_spec.tar.gz` | `cc0d38a4…1276c2d` |

Units are read straight from the trace documentation, not guessed: `plan_cpu` is a percentage of a
core (600 = 6 vCPU), `plan_gpu` a percentage of one card (50 = 0.5 card), `plan_mem` is GB, machine
`cap_cpu / cap_mem / cap_gpu` are cores / GB / cards, timestamps are seconds since trace start, and
only the `Terminated` task status means a successful run.

Two properties of the released data bound what can be claimed:

1. **Queueing delay is not recoverable.** The README defines scheduling latency as the gap between a
   job's `submit_time` and its first task launch; in the released file that gap is 0 for 99.98 % of
   jobs (96 of 625,769 have any gap at all, max 935 s). So this is a **counterfactual replay** of a
   real arrival stream under different policies, not a reproduction of Alibaba's production waits.
2. **Demand is heavy-tailed** (job runtime p50 = 1,079 s, p90 = 14,980 s, p99 = 76,606 s,
   max = 600,961 s; GPU request p50 = 1 card, p90 = 6.25, and some jobs hold dozens of cards for days).
   1,055,501 raw job records reduce to 625,769 usable ones (610,703 GPU jobs from 1,278 distinct users)
   after keeping only `Terminated` tasks and runtimes between 1 min and 7 d. Long-running and huge gang
   requests are then excluded from an instance **and counted** in
   `data/processed/instance_meta.json`, so the exclusion is auditable rather than a quiet way to make
   the numbers look nice.

## 2. Experiment design

Per GPU pool (P100, MISC):

* one real **6-hour busiest window** by GPU-card-hours, with the exclusions above;
* a **fixed host partition** cut from `pai_machine_spec` (machine-granular, genuine specs — P100 =
  2 cards / 64 cores / 512 GB per host), sized by the *max* of a load floor (target ρ) and a fit floor
  (`ceil(largest request / per-host capacity)`, so the instance cannot be infeasible by construction);
* three **nested arrival intensities** — the low-load job set is a subset of the high-load one, so the
  sweep moves only λ while the cluster stays identical. Achieved ρ is reported, not assumed.

| instance | jobs | cards | machines | Δ | horizon | MILP binaries |
|---|---|---|---|---|---|---|
| P100 ρ=0.41 / 0.64 / 0.85 | 71 / 106 / 150 | 44 | 22 | 600 s | 84 slots (14 h) | 5,964 / 8,904 / 12,600 |
| MISC ρ=0.39 / 0.52 / 0.72 | 188 / 282 / 400 | 40 | 5 | 600 s | 84 slots (14 h) | 15,792 / 23,688 / 33,600 |

(The MISC tiers land at 0.39 / 0.52 / 0.72 rather than the requested 0.4 / 0.6 / 0.85 because host
count is an integer: the 400-job tier needs 4.26 MISC hosts to hit ρ = 0.85, so 5 hosts are the
smallest partition that satisfies the load floor, and 40 cards × 14 h = 560 card-hours is more than
0.85 would need. Achieved ρ is always reported from the actual demand.)

Policies, all under identical slot granularity, rounding, capacity assumptions and resource
dimensions (GPU cards, CPU cores, memory GB):

| policy | information | compute |
|---|---|---|
| **FIFO** | online, arrival order, strict head-of-line blocking | — |
| **EASY backfill** | online; a later job may start only if it finishes before the blocked head could | — |
| **ILP-flow (offline)** | clairvoyant: the whole window up front | HiGHS, 120–180 s, 2–5 % gap |
| **ILP-makespan (offline)** | clairvoyant, different objective | as above |
| **ILP-rolling** | online, 2 h lookahead, commits only the next 60 min | 15 s and 10 % gap per epoch |
| **size-priority** | online; smallest requested GPU gang / CPU / memory first, optional aging | — |
| **size-priority-short** | online placement but ordered by *actual* runtime (oracle control) | — |

The rolling variant is the like-for-like comparison: FIFO and EASY get neither future knowledge nor a
solver budget, and the offline optimum stays an upper bound. The LP relaxation of the flow-time model
yields a **provable lower bound**, so the headline improvement carries an optimality gap. The four
size-priority rules are the missing **strong online baseline**: they use only what a scheduler knows
at submit time (the requested GPU/CPU/memory), cost nothing to compute, and are run through the same
slot occupancy and capacity code as the other policies. The runtime-ordered variant is marked as an
oracle and is not quoted as deployable.

## 3. Results

One-page conclusion in Chinese: [`一页结论.md`](一页结论.md).
Full per-instance tables: [`results/table_comparison.md`](results/table_comparison.md).
Machine-readable: [`results/metrics.csv`](results/metrics.csv) / `metrics.json`,
`results/summary.json`. Figures: `results/fig_mean_wait.png`, `fig_gpu_idle.png`, `fig_pareto.png`,
`fig_load_sweep.png`. Walkthrough with executed outputs:
[`notebook/trace_driven_gpu_scheduling.ipynb`](notebook/trace_driven_gpu_scheduling.ipynb).

**Findings**

<!-- FINDINGS:START -->
72 policy runs over 6 instances (30 original + 42 priority variants). **Every run respects
GPU/CPU/memory capacity in every slot (`capacity_violation_slots = 0` in all 72 rows)**, so the
deltas below are not artefacts of an infeasible plan.

vs FIFO, per-policy range over the six instances:

| | mean wait under replay | deadline misses | throughput | addressable idle waste |
|---|---|---|---|---|
| **EASY backfill** | −2 % … −29 % | −2.7 … −20.8 pp | 0 … +2.8 % | −2 % … −55 % |
| **size-priority** (online, no aging) | **−59 % … −86 %** | −43.3 … −84.7 pp | −6.2 % … +11.3 % | **−100 %** (0 card·h) |
| **ILP-rolling** (online) | −71 % … −87 %, **+29 %** at P100 ρ=0.85 | −46.0 … −84.9 pp | −36.1 % … +17.9 % | −5 % … −92 %, **+486 %** at P100 ρ=0.85 |
| **ILP-flow** (offline) | −73 % … −88 % | −56.3 … −86.8 pp | −9.0 % … +17.9 % | −26 % … −91 % |

1. **Reordering is worth an order of magnitude in waiting time.** Mean replay wait drops from
   35–240 min under FIFO to 8.7–58.8 min offline and 9.3–22.5 min for the honest online variant;
   P90 waits fall even harder (e.g. P100 ρ=0.41: 69.5 → 9.7 min). The mechanism is head-of-line
   blocking: FIFO parks a large gang request in front of small ones that would fit right now.
2. **Deadline misses collapse.** 60.6–96.0 % of jobs miss their latency contract under FIFO, versus
   1.6–30.0 % offline and 4.3–50.0 % rolling.
3. **Throughput barely moves.** 0 … +2.8 % for backfill, −9 % … +18 % for the ILP, and the gain is
   concentrated in the one instance that was not capacity-bound (MISC ρ=0.72: 24.27 → 28.60 jobs/h,
   +17.9 %). On loaded windows the work is the work — a claim that reordering multiplies GPU
   throughput deserves suspicion, and this data does not support it.
4. **The idle metric that discriminates is *addressable* idle-with-backlog** — GPU card-hours that
   sit unused while a job that could have fitted in them (GPU *and* CPU *and* memory) is already
   queued. FIFO's 63.0–86.6 card·h at the two busiest tiers goes to 21.5 and 8.2 under ILP-flow,
   and to 7.1 under rolling at MISC ρ=0.72. Fixed-window idle is exactly `1 − ρ` for every policy and
   is only a load check.
5. **Offline optimality is real but not free:** the LP relaxation gives a lower bound and the MILP
   gap is 0.90–2.20 % on five instances — and **14.15 % at P100 ρ=0.85**, where the 150-job model
   stops being nearly optimal inside its 120 s budget (the budget is 180 s only for the instances
   with more than 200 jobs).
6. **Where it loses, and why.** At P100 ρ=0.85 the *online* ILP is worse than FIFO on latency
   (+29 %, throughput −36 %) because bounded 2-hour lookahead with a 15 s/10 %-gap epoch budget
   defers 75 of 150 jobs past the planning horizon and hoards capacity waiting for a better plan;
   its 369.0 addressable card·h is the worst number in this study. Rolling wins on 4 of 6 instances
   and loses at saturation — that is the finding, not a wart to delete.
7. **Makespan is the wrong objective for latency.** Minimising makespan gives near-optimal
   throughput behaviour but 119–196 min mean waits (worse than FIFO at two tiers) and the largest
   addressable idle in every instance, because it lets early jobs wait for a tighter packing.
8. **The pool-as-one-machine relaxation costs something.** Re-packing each loaded slot onto the real
   hosts with first-fit-decrease (gang placement) gives ILP-flow packing feasibility of
   16.9 %–93.5 %; at the saturated MISC tier only 16.9 % of loaded slots pack, so fragmentation
   would eat an unknown but material part of the modelled gain there. *(That "unknown" is now
   measured — see finding 12: at the optimum it is 0.02 %–0.68 %, but repairing an aggregate plan
   into an executable one costs 1.27×–3.15× more waiting.)*
9. **The gain belongs to the priority rule, not the solver.** A one-line online rule — start the
   smallest requested GPU gang (or CPU / memory request) first, no future knowledge, no MILP —
   reproduces **77–114 %** of ILP-flow's mean-wait gain across the six instances, cuts deadline
   misses by 43–85 percentage points, drives *addressable* idle to exactly **0 card·h** (by
   construction: the rule starts every arrived job that fits), and takes no solve time. At P100
   ρ=0.85 it **beats the offline ILP** on mean wait (34.1 vs 58.8 min), and against the rolling
   ILP it is within 1.0–1.5× on five instances, while at P100 ρ=0.85 the rolling ILP is 9× slower
   (310.4 vs 34.1 min). P90 wait
   also falls (46–86 %), but the tail is the catch: max wait rises to **2.9× FIFO** because large
   gangs starve. Aging (a job waiting 1–4 h jumps the queue) trades mean wait for the tail; 1 h aging
   at P100 ρ=0.85 still captures 103 % of the ILP gain but raises deadline misses to 46 %, so the
   deployable recipe is "size priority + aging + a max-wait cap", not "run an ILP".
   `results/priority_baselines.json`, `results/metrics_priority.csv`.

10. **The finding survives a 72-instance robustness grid, but the original window was the
    friendliest one.** `scripts/08_sweep.py` replays the whole policy set over 4 window ranks
    (busiest / 2nd / 3rd / median-demand) × 3 sampling seeds × 2 pools × 3 load tiers = 72
    instances; the committed six are a subset and act as the built-in positive control. Direction
    holds everywhere: pure size priority is strictly faster than FIFO on mean wait in
    **70 of 72** instances, ties on 2 and is never worse (**−96 % … 0 %**, P90 −97 % … +36 %,
    max wait −9 % … **+252 %**); it leaves
    **0 card·h** of reclaimable idle in all 504 of its rows, with 0 capacity-violating slots.
    The *magnitude* does not: the best deployable rule captures a median **92 %** of ILP-flow's
    mean-wait gain across the 71 instances where the ratio is defined (≥50 % in 90 %, ≥77 % — the
    old study's lower bound — in 75 %, >100 % in 25 %), but the busiest-window subset the original
    study used has median 96 % and **no** instance below 50 %, while other windows reach 0 %.
    So the quoted "77–114 %" was read off the most favourable window; the honest statement is
    "usually ≥90 %, occasionally far less, and offline ILP beats every deployable rule on 53 of 72
    instances". `results/sweep_metrics.csv`, `results/sweep_summary.json`.
11. **Reclaimable idle and fragmentation idle are different problems, and the priority rule only
    solves the first.** `idle-with-backlog` splits into *head-of-line* idle (cards free, a queued
    job would fit on GPU **and** CPU **and** memory, the policy just doesn't start it — reclaimable
    by scheduling) and *fragmentation* idle (no waiting job fits — a packing problem). Priority
    rules: 0 of 504 rows with any reclaimable idle, by construction. FIFO/EASY/ILP: 274 of 288
    rows nonzero. Verified slot-by-slot rather than asserted, because "column not computed" and
    "genuinely zero" look identical in a table: on the highest-achieved-ρ instance
    (`P100_w3_s20261103_rho0.85`) FIFO has 66 idle-with-backlog slots of which 63 are reclaimable,
    `priority-gpu` has 63 of which **0** are — total idle falls 82.29 → 59.54 card·h and mean queue
    depth 33.2 → 12.6, but what remains is fragmentation. That is also why the 7 low-capture
    instances lose: the offline ILP leaves *more* backlog idle than the rule on 7/7 of them, and
    40–94 % (median 67 %) of that idle is the reclaimable kind — measured, i.e. the solver had
    queued work it could have started and did not. Reading that as "it is hoarding capacity for a
    large gang it can see coming" is an interpretation the model does not output; the observable
    facts are the hoarded idle and the lower wait. **Size priority replaces rolling re-optimisation;
    it cannot replace an optimum that needs future information.**

12. **The host-level model: fragmentation is cheap at the optimum, expensive to patch around.**
    `scripts/09_host_level.py` builds the model the study listed as its next step — one column per
    machine, a job's machine set fixed for its whole life (`sum_m z[j,m,t] = h_j·x[j,t]`,
    per-machine capacity in every slot) — and measures three things on the committed six instances.

    *Executability.* With one time-consistent host assignment, **83.0 %–95.7 %** of the jobs in a
    committed schedule keep their scheduled start (ILP-flow: 90.8 %–95.7 %). The committed
    `packing_feasibility_rate` (14.5 %–93.5 %) answers a *different* question — per slot, with the
    host set re-decided every slot — so the two numbers must not be compared; a container is not
    migrated, which is why the time-consistent count is the one that matters. Making a committed
    schedule executable by delaying jobs in their original order costs **1.27×–1.52×** the mean wait
    for ILP-flow and up to **3.15×** for ILP-rolling (MISC ρ=0.52); 0 jobs stay unplaceable and
    0 host-slots are over capacity after the repair.

    *The bill at the optimum.* Both models are solved on the **same** job set (the busiest 2-hour
    window of each instance: 29–244 jobs), the same candidate slots and the same objective
    (Σ completion time). **The host LP bound equals the aggregate LP bound in all six instances**
    (385,237.5 / 528,902.5 / 738,633.5 / 2,448,762 / 3,650,490 / 5,471,069 s; five are identical
    to the last digit, the sixth differs by 1e-9 s, i.e. floating-point noise):
    fragmentation is invisible to the relaxation, so the bill is a pure integrality effect. The
    proven optima differ by **+600 s** (P100 ρ=0.41), **+1,200 s** (P100 ρ=0.64) and **+600 s**
    (MISC ρ=0.39) — 5.5–29 s per job, **0.02 %–0.23 %** of the objective — and by at most
    **+4,200 / +4,200 / +37,200 s** (≤0.57 % / ≤0.12 % / ≤0.68 %) on the three instances where the
    host MILP did not close inside 300 s (P100 ρ=0.85 / MISC ρ=0.52 / MISC ρ=0.72; for the first
    two the dual bound still certifies a bill of at least 1,800 s and 600 s, for the third the
    lower end is 0 and is therefore reported as uncertified rather than as "no bill").
    Re-solving with machine columns costs ~2 % of mean wait (MISC ρ=0.39: 312.3 → 317.8 s);
    *patching* the same aggregate plan into an executable one costs 1.55× (482.9 s). **Model the
    hosts; do not repair the plan.**

    *And the relaxation flatters the deployable rule more than it flatters the optimiser.* On the
    same job sets the size-priority rule waits 1.00–1.64× the aggregate optimum but **1.69–2.08×**
    the host-level optimum; the host constraint costs the rule ×1.29–1.72 of waiting and the
    re-optimised optimum only ×1.02–1.20. So part of "one priority rule gets most of the ILP's
    gain" is an artefact of modelling the pool as one machine: greedy placement and fragmentation
    interact badly (the rule packs greedily and strands capacity). The rule is still 3.88–9.03×
    faster than FIFO under real hosts (e.g. MISC ρ=0.52: priority-mem 17.3 min vs FIFO 156.4 min)
    with **0** host-capacity violations in all 48 host-level runs, and deadline misses still collapse
    (FIFO 73.2 %–97.3 % → priority-mem 2.7 %–31.0 %). One aggregate-model number does change:
    the priority rules' "0 card·h of addressable idle" becomes **5.3–61.6 card·h** under the
    stricter host-level definition (a job that fits the pool's spare capacity may fit on no single
    host) — the zero was a property of the relaxed capacity, not of the rule.

    *A solver caveat that applies to the committed numbers too.* HiGHS reports status 7
    ("Optimal") both when it proves optimality **and** when it stops inside the requested
    `mip_rel_gap`, and scipy maps both onto status 0 ("Optimization terminated successfully"). The
    committed "offline optimum" is therefore an *incumbent within the tolerance* (2 %, 5 % for the
    big instances): on the pilot sub-instances the committed tolerance returns an incumbent
    **0.00 %–2.98 %** above the proven optimum. (The committed `gap_vs_lp_percent` figures are
    unaffected — the six LP lower bounds reproduce exactly, re-derived and compared in
    `results/host_level_lp_check.csv` — and no committed number was changed.)
    `03_optimize.py` also read scipy's gap *result attribute* under its old name (`mip_rel_gap`,
    renamed to `mip_gap` in scipy 1.18 — the solver *option* kept its name), so the per-solve gap
    it recorded was silently always empty; it now records the gap and the dual bound.

    *And the committed numbers themselves were checked, not assumed.* Re-running the committed
    ILP-flow rows with the same tolerances reproduces **5 of 6 objectives exactly**
    (`results/host_level_ilp_check.csv`, relative deviation 0.0). The sixth, P100 ρ=0.85, does
    not: the re-run lands **2.02 % lower** (2,797,800 s vs the committed 2,855,400 s), i.e. on a
    *better* incumbent, because both runs stopped on the wall clock rather than on the gap
    (the committed row's own `gap_vs_lp_percent` is 14.15 %, the re-run's gap 11.41 %). The
    committed value is **kept as published** — it is what the study reported — but it should be
    read as "the incumbent this machine found in 120 s", not as a floor. Nothing else moved: the
    six LP lower bounds re-derive with relative difference 0.0 (`results/host_level_lp_check.csv`).

    Scope: the MILP pilot runs on the busiest-2-hour window, not on the full 6-hour instance — at
    22 hosts (P100) the full instance does not fit a solve budget. The repair that makes a schedule
    executable is a greedy, order-preserving one, so 1.27×–3.15× is an upper bound on that price.
    The host footprint uses the study's own rule (`h_j` = max over the GPU / CPU / memory
    dimensions), which the trace's `n_instances` never contradicts. `results/host_level_audit.csv`,
    `results/host_level_metrics.csv`, `results/host_level_milp.csv`, `results/host_level_rules.csv`,
    `results/host_level_lp_check.csv` (the committed LP lower bounds re-derived and compared —
    all six reproduce with relative difference 0.0), `results/host_level_ilp_check.csv` (the
    committed ILP-flow rows re-solved and compared — 5 of 6 identical, the sixth 2.02 % better).

Statistical status: the committed study is one seed and one window per pool, so its numbers are
effect sizes; the 72-instance grid in finding 10 is what turns the *direction* into a
distribution. No bootstrapped confidence interval is claimed for either — the grid gives spread
across problem instances, not sampling error within one.
<!-- FINDINGS:END -->

## 4. What could go wrong, and what catches it

This section is the part I would want to read before trusting any of the numbers above.

* **Idle rate needs its denominator, and the obvious one is meaningless.** Over a horizon long enough
  for everything to finish, allocated card-hours equal demanded card-hours for *every* policy, so
  "GPU idle rate over the fixed window" is exactly `1 − ρ` and cannot rank schedulers. Three
  denominators are reported (`gpu_idle_rate_fixed_window`, `gpu_idle_rate_busy_period`,
  `idle_with_backlog_card_hours`); **idle-with-backlog** — cards standing unused while jobs were
  already submitted and waiting — is the waste a scheduler is actually responsible for, and is what
  the figures and headline deltas use. The first version of this repo plotted the constant metric and
  produced five identical bars; `scripts/06_backlog_idle.py` exists because of that.
* **A failed solve must not become a schedule.** `scipy.optimize.milp` returns `x = None` when it has
  no incumbent. The first version defaulted to `zeros`, whose `argmax` pinned *every* job to its
  release slot — which made the high-load rolling row show a mean wait of 8 min at ρ ≈ 0.9, a makespan
  below the capacity lower bound, a *negative* idle rate, and packing feasibility of 0.39. Caught by
  arithmetic, then fixed: `solve_ilp` now returns `None` with a `solver_feasible` flag, offline falls
  back to capacity-aware first fit and labels it, and `results/metrics.csv` carries
  `capacity_violation_slots`, `gpu_overcommit_slots` and `max_gpu_overcommit_cards` for every row.
  `03_optimize.py` prints a feasibility line per policy and refuses to hide a violation.
  (`results/run_03_attempt1.log` is that broken run, kept as evidence.)
  The grid stress-tests the *other* escape hatch and reports what it finds: 6 of 792 rows have jobs
  that fit nowhere even in the padded tail, and on 1 of them (`MISC_w2_s20261103_rho0.85`,
  ILP-rolling) those jobs are placed at their arrival slot so they cannot silently disappear —
  which makes that single row a capacity-violating schedule (4 slots, peak GPU allocation 105.9 %).
  It is labelled as such by `capacity_violation_slots` / `unplaceable_jobs`, it is confined to
  ILP-rolling (ILP-flow and all nine online policies have 0 in every row), and no headline claim
  reads that row.
* **The horizon is a planning window, not a cluster shutdown.** Jobs an online policy cannot admit in
  time are placed in a documented *tail* (`sched_core.tail_slots`) instead of being clamped into the
  window; `clamp_starts` is now an index guard only. The cost is visible as
  `jobs_running_past_horizon` / `deferred_jobs_past_horizon`, a longer makespan, and a worse
  in-window idle rate.
* **The control group was too weak, and that changed the conclusion.** FIFO and EASY are both
  head-of-line-blocking policies, so "ILP beats FIFO by 71–88 %" mostly measures the cost of the
  blocking, not the value of optimisation. Adding request-only size-priority rules (script 07)
  reproduces most of that gain with no solver and no future knowledge — 77–114 % on the six committed
  windows, median 92 % across the 72-instance grid (finding 10) — so the ILP's unique contribution is
  the remaining tail/robustness, not the average wait. Any scheduling paper that
  reports only FIFO as the baseline is leaving this check out.
* **A zero in a column can mean "not computed", and that is not the same finding.** The priority
  rules report `idle_addressable_backlog_card_hours = 0` on all 504 of their rows, which is either a
  real property of a greedy-fit rule or a gap in the pipeline — and the table cannot tell you which.
  `08_sweep.py --report-only` therefore recomputes the split slot by slot on the highest-ρ instance
  and asserts it: if the rule ever left reclaimable idle, the assertion fires instead of the number
  shipping. It does not (`priority-gpu`: 63 idle-with-backlog slots, 0 reclaimable).
* **Aggregate capacity is a relaxation.** Real GPUs live in discrete hosts, so every loaded slot is
  re-checked with first-fit-decrease against the actual machines (gang placement, host count = max over
  the GPU / CPU / memory dimensions). `packing_feasibility_rate` and `worst_fragmentation_loss` report
  the price of that relaxation instead of assuming it away. The sweep quantified the other side of it:
  the best online rule removes every reclaimable card·h and still leaves 59.54 card·h of fragmentation
  idle on the busiest instance, so the relaxation and the rule fail on the *same* dimension.
* **"Aggregate capacity is a relaxation" was an excuse, so it got measured.** The per-slot
  first-fit-decrease check re-decides the host set in every slot; a container is not migrated, so it
  does not answer "is there one assignment that holds for the whole life of the job".
  `scripts/09_host_level.py` answers that question (time-consistent assignment, machine set fixed at
  the start slot) and prices it: 83.0 %–95.7 % of the jobs keep their scheduled start, and an
  order-preserving repair that makes the schedule executable costs 1.27×–3.15× the mean wait. The
  same script re-solves the offline optimum with one column per machine and finds the bill at the
  optimum is 0.02 %–0.68 % of the flow objective — the relaxation was cheap in objective terms and
  expensive in "can I run this?" terms, which is the opposite of what finding 8 assumed.
* **"Optimal" in a solver message is not the same as "proven optimal".** HiGHS reports status 7 for
  both a proven optimum and a run that stopped inside the requested `mip_rel_gap`; scipy maps both
  onto status 0 with the message "Optimization terminated successfully". The committed study's
  "offline optimum" is therefore an incumbent within its tolerance, and the per-solve gap it recorded
  was empty because scipy renamed the attribute (`mip_rel_gap` → `mip_gap` in 1.18). Both are now
  read correctly (`solver_gap_fields`), the dual bound is reported next to the incumbent, and the
  committed LP lower bounds were re-verified to reproduce exactly (six of six), so no committed number
  moved.
* **One seed, one window per pool — for the committed six.** That is why `scripts/08_sweep.py`
  exists: the grid replays everything over 4 window ranks × 3 seeds and reports the spread
  (finding 10). What it does *not* provide is sampling error within an instance — the spread is
  across 72 different problems, so the grid rules out "this only works on the window I picked" but
  is not a confidence interval, and the six committed numbers are still effect sizes.

## 5. Layout

```
run_all.py                one-command pipeline (fetch -> ingest -> instance -> solve -> backlog -> priority -> report -> notebook;
                          --sweep adds the multi-window/multi-seed robustness grid before report)
scripts/00_probe.py       inspect the raw archives (no header row, so names are assigned here)
scripts/01_ingest.py      task -> job aggregation, pools, observed runtimes
scripts/02_instance.py    busy-window choice, exclusions, host partition, nested load sweep
                          (--window-rank k = the k-th busiest window instead of the busiest)
scripts/sched_core.py     instance/occupancy/policies/metrics/packing check (shared, so nothing drifts)
scripts/03_optimize.py    time-indexed MILP + LP bound + rolling-horizon replay
scripts/04_report.py      tables, figures, summary.json
scripts/05_notebook.py    generates AND executes the notebook
scripts/06_backlog_idle.py idle-with-backlog recomputed from saved schedules (no solver rerun)
scripts/07_priority_baselines.py request-only size/aging priorities replayed online (no solver)
scripts/08_sweep.py       the same policy set over 4 window ranks x 3 seeds x 2 pools x 3 load
                          tiers; the committed six instances are a subset and act as a positive control
scripts/host_core.py      host-level primitives: per-host footprints, time-consistent assignment,
                          host-aware repair, the online rules re-run on discrete hosts
scripts/09_host_level.py  host-level pilot: audit of the committed schedules, the online policies
                          under real hosts, and the offline MILP with one column per machine
                          (results/host_level_*.csv; the lpcheck / ilpcheck stages re-derive the
                          committed LP bounds and ILP rows and compare them)
data/processed/           instances + instance_meta.json   (committed)
data/processed/sweep/     sweep instances + their meta     (committed: they define the grid)
results/                  metrics, tables, figures, schedules (committed)
notebook/                 executed analysis
一页结论.md               one-page conclusion (Chinese)
```

`data/raw/` and `data/processed/jobs.csv` are gitignored because they are downloadable; **the
artifacts that evidence every reported number are committed on purpose**, so nobody has to take a
claim in this README on trust.

## 6. Known limitations / next steps

* Preemption is not modelled (`p_j` is `ceil(service/Δ)` whole slots), so long jobs are slightly
  over-resourced; Δ = 600 s is the price of keeping the time-indexed model inside a solve budget.
* Idle means *unallocated*, not *unused SMs*: the per-instance sensor table is not part of this
  experiment, so a job holding a card idle still counts as occupying it.
* The natural extensions are a host-level assignment model (columns per machine) to pay for
  fragmentation properly, and preemptive/migrative variants. The multi-window / multi-seed grid that
  used to sit on this list is now finding 10 (`scripts/08_sweep.py`), and the host-level model is now
  finding 12 (`scripts/09_host_level.py`) — as a *pilot*: the aggregate and host optima are compared
  on the busiest 2-hour window of each committed instance (29–244 jobs), because the full instance
  at 22 hosts does not fit a solve budget. What the pilot settles: fragmentation is cheap at the
  optimum (0.02 %–0.68 % of the flow objective) and invisible to the LP bound, but patching an
  aggregate plan into an executable one costs 1.27×–3.15× the mean wait, and the aggregate
  relaxation flatters the *deployable rule* more than the optimiser (the rule waits 1.69×–2.08× the
  host-level optimum, versus 1.00×–1.64× the aggregate one). What it does not settle: the full
  instances at host level, a host-aware rule with a better placement heuristic than greedy
  largest-slack, preemption/migration, and a host-level *rolling* MILP comparable to the committed
  ILP-rolling row.
* **The solver baselines are not run-to-run reproducible.** `milp` stops on a wall-clock limit, so
  the same instance can land on a different incumbent. The 72-instance grid measured this against the
  committed tables: the 54 deterministic rows (FIFO / EASY / 7 priority rules) replay to
  `0.00e+00` maximum relative deviation, the 12 solver rows deviate by up to **22 %** — and the
  deviation is *noted*, never averaged away or asserted as agreement. Any claim here that depends on
  an ILP number carries that much fuzz; the priority-rule claims do not. The host-level pilot is
  explicit about the same issue: the aggregate optimum is re-solved with `gap = 0` so optimality is
  proven (or the dual bound is reported), and the bill is published as an interval. Re-running the
  committed ILP-flow rows confirms it at full scale — 5 of 6 reproduce exactly, and the sixth
  (P100 ρ=0.85) lands on a **2.02 % better** incumbent, because both runs stopped on the wall
  clock rather than on the gap (`results/host_level_ilp_check.csv`).
* The priority rules are evaluated on the same 72 instances as everything else, and the aging
  thresholds (1 / 2 / 4 h) were chosen by inspection rather than tuned on a held-out window. They
  establish that the mean-wait gain is not solver-specific; they do not establish the best threshold.
