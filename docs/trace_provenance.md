# Trace provenance: which snapshot of `cluster-trace-gpu-v2020` this repository replays

Written 2026-11-18. Purpose: make the input of every number in README section 3 pin-downable
without trusting the OSS bucket to hold still (the archives are pinned by content hash, not by
a version string, because the trace ships none).

## 1. Source and version pin

- Upstream: Alibaba PAI **cluster-trace-gpu-v2020**, published in
  [`alibaba/clusterdata`](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020),
  released together with Weng et al., *MLaaS in the Wild: Workload Analysis and Scheduling in
  Large-Scale Heterogeneous GPU Clusters*, NSDI '22 (see `bibliography.bib`). The trace covers
  **Jul–Aug 2020**: 1,055,501 job records, 1,261,050 task records, 1,897 machines, 6,742 GPUs
  (counts as stated in the official trace README; README §1 of this repo repeats them).
- Download endpoint used by `run_all.py --fetch`:
  `https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/<archive>`.
- The released trace has **no version string or release date beyond the README itself**, so the
  snapshot is pinned by sha256 of the three archives. The pinned hashes are the official ones
  published in the trace README (embedded in `run_all.py`, `FILES`):

  | archive | sha256 | bytes (local snapshot) |
  |---|---|---:|
  | `pai_job_table.tar.gz` | `5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0` | 62,065,432 |
  | `pai_task_table.tar.gz` | `cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5` | 35,514,117 |
  | `pai_machine_spec.tar.gz` | `cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d` | 30,449 |

- Local snapshot: the three archives in `data/raw/` carry file dates **2026-10-13** (download and
  checksum verification on this machine); `python run_all.py --fetch` re-checks them against the
  table above and re-downloads on mismatch. To verify without any network:

  ```bash
  python - <<'PY'
  import hashlib, pathlib
  for name, want in {
      "pai_job_table.tar.gz": "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
      "pai_task_table.tar.gz": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
      "pai_machine_spec.tar.gz": "cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d",
  }.items():
      p = pathlib.Path("data/raw") / name
      got = hashlib.sha256(p.read_bytes()).hexdigest()
      print(name, "OK" if got == want else f"MISMATCH {got}")
  PY
  ```

## 2. What the CSVs contain (schema and units, verbatim from the trace documentation)

The archives carry **no header row**; column names are assigned in `scripts/00_probe.py` /
`scripts/01_ingest.py`. Units are read from the trace README, not guessed:

- `pai_job_table.csv` — `job_name, inst_id, user, status, submit_time, complete_time`
- `pai_task_table.csv` — `job_name, task_name, inst_num, status, start_time, end_time, plan_cpu,
  plan_mem, plan_gpu, gpu_type`
- `pai_machine_spec.csv` — `machine, gpu_type, cap_cpu, cap_mem, cap_gpu`
- `plan_cpu` is a percentage of a core (600 = 6 vCPU); `plan_gpu` a percentage of one card
  (50 = 0.5 card); `plan_mem` is GB; machine `cap_*` are plain cores / GB / card counts;
  timestamps are **seconds from trace start**; only task status `Terminated` means a successful run.

## 3. From raw records to the replayed instances

`01_ingest.py` aggregates tasks to jobs and records **every filter with the count it dropped**
(`data/processed/ingest_report.json` — no silent drops). On the pinned snapshot this reduces
1,055,501 raw job records to 625,769 usable ones (610,703 GPU jobs from 1,278 distinct users)
after keeping only `Terminated` tasks with runtimes between 1 min and 7 d. `02_instance.py` then
excludes long-running and huge gang requests from each instance **and counts them** in
`data/processed/instance_meta.json`, so the exclusion is auditable. Two properties of the
released data bound what can be claimed (README §1): queueing delay is not recoverable from the
release (submit→launch gap is 0 for 99.98 % of jobs), so this is a counterfactual replay of a
real arrival stream, not a reproduction of production waits.

## 4. If upstream moves

If Alibaba replaces the archives, the hashes above stop matching and `--fetch` stops the run
rather than silently analysing a different trace. In that case: pin whatever new hashes match the
current README, re-run the pipeline, and regenerate every committed table — the six committed
instances also act as a positive control inside the 72-instance sweep (`--sweep`,
window_rank=1/seed=20261013 cell).
