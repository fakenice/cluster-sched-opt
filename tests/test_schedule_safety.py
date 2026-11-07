"""Regression tests for complete, resource-conserving schedules (no result writes)."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sched_core as sc

_spec = importlib.util.spec_from_file_location("optimize_safety", ROOT / "scripts" / "03_optimize.py")
opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opt)


def fixture(n=3, service=2, arrival=0, gpu=1, capacity=1, horizon=4):
    return sc.Instance(
        pool="test", rho=0.5,
        jobs=pd.DataFrame({"job_id": np.arange(n), "arrival_rel": arrival,
                           "service": service, "gpu_cards": gpu,
                           "cpu_cores": np.ones(n), "mem_gb": np.ones(n)}),
        capacity={"gpu_cards": capacity, "cpu_cores": 4.0,
                  "mem_gb": 4.0, "machines": 1},
        slot_seconds=1, n_slots=horizon, gpu_per_machine=capacity,
    )


def no_incumbent(inst, subset, objective="flow", binary=True, time_limit=120.0,
                 gap=0.02, occ_base=None, min_start=None, weights=None,
                 slack_s=1800.0, cand_window=None):
    # Only replace the expensive solver boundary, not rolling/first-fit/occupancy.
    return None, {"solver_status": 1, "message": "Time limit reached without incumbent",
                  "solver_feasible": False, "objective": objective, "binary": binary,
                  "n_columns": 0, "n_rows": 0, "n_nnz": 0, "solve_seconds": 0.0,
                  "objective_value": None, "mip_gap": None, "dual_bound": None}


class ScheduleSafetyTests(unittest.TestCase):
    def assert_conserves_resources(self, inst, starts):
        # Independent interval summation: do not use Occupancy or metrics to certify them.
        starts = np.asarray(starts)
        self.assertEqual(starts.shape, (len(inst.jobs),))
        self.assertTrue(np.issubdtype(starts.dtype, np.integer))
        releases = np.ceil(inst.jobs["arrival_rel"].to_numpy() / inst.slot_seconds)
        durations = np.ceil(inst.jobs["service"].to_numpy() / inst.slot_seconds)
        self.assertTrue(np.all(starts >= releases), starts)
        demand = inst.jobs[list(sc.RESOURCES)].to_numpy(float)
        capacity = np.array([inst.capacity[r] for r in sc.RESOURCES])
        for slot in range(int(np.max(starts + durations))):
            active = (starts <= slot) & (slot < starts + durations)
            usage = demand[active].sum(axis=0)
            self.assertTrue(np.all(usage <= capacity + 1e-9),
                            f"slot {slot}: usage {usage} exceeds capacity {capacity}")

    def test_rolling_without_incumbent_drains_thirty_long_jobs(self):
        # A short fixed tail used to leave most jobs unplaced, then stack at release.
        inst = fixture(n=30, service=10, horizon=5)
        capacity_before = inst.capacity.copy()
        with patch.object(opt, "solve_ilp", side_effect=no_incumbent):
            starts, info = opt.run_rolling(inst, lookahead_s=0, commit_s=1)
        self.assertEqual(info["unplaceable_jobs"], 0)
        np.testing.assert_array_equal(starts, np.arange(5, 305, 10))
        self.assert_conserves_resources(inst, starts)
        met = sc.metrics(inst, starts)
        self.assertEqual(met["mean_wait_s"], 150.0)
        self.assertEqual(met["max_wait_s"], 295.0)
        self.assertEqual(met["makespan_s"], 305.0)
        self.assertEqual(met["jobs_running_past_horizon"], 30)
        self.assertEqual(inst.capacity, capacity_before)

    def test_real_misc_instance_without_incumbent_has_no_unplaced_jobs(self):
        proc = ROOT / "data" / "processed" / "sweep"
        meta_path = proc / "instance_meta.json"
        if not meta_path.exists():
            self.skipTest("optional sweep metadata not present")
        metas = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = next((m for m in metas if m.get("sweep_key") ==
                     "MISC_w2_s20261103_rho0.85"), None)
        if meta is None or not (proc / meta["instance_file"]).exists():
            self.skipTest("optional 400-job MISC instance not present")
        inst = sc.load(proc / meta["instance_file"], meta)
        self.assertEqual(len(inst.jobs), 400)
        capacity_before = inst.capacity.copy()
        with patch.object(opt, "solve_ilp", side_effect=no_incumbent):
            starts, info = opt.run_rolling(inst)
        self.assertEqual(info["unplaceable_jobs"], 0)
        self.assertEqual(info["deferred_jobs_past_horizon"], 400)
        self.assert_conserves_resources(inst, starts)
        self.assertEqual(sc.metrics(inst, starts)["capacity_violation_slots"], 0)
        self.assertEqual(inst.capacity, capacity_before)

    def test_tail_includes_late_release_before_serial_drain(self):
        inst = fixture(n=3, service=10, arrival=1000, horizon=5)
        with patch.object(opt, "solve_ilp", side_effect=no_incumbent):
            starts, info = opt.run_rolling(inst, lookahead_s=0, commit_s=1)
        self.assertEqual(info["unplaceable_jobs"], 0)
        np.testing.assert_array_equal(starts, [1000, 1010, 1020])
        self.assert_conserves_resources(inst, starts)
        self.assertEqual(sc.metrics(inst, starts)["max_wait_s"], 20.0)

    def test_occupancy_rejects_invalid_or_oversized_resource_demands(self):
        for resource in sc.RESOURCES:
            for value in (-1.0, float("nan"), float("inf"), 5.0):
                with self.subTest(resource=resource, value=value):
                    inst = fixture(n=1)
                    inst.jobs[resource] = value
                    with self.assertRaises(ValueError):
                        sc.Occupancy(inst)

    def test_clamp_rejects_negative_out_of_bounds_or_early_starts(self):
        inst = fixture(n=1, service=2, arrival=3)
        for starts in ([-1], [0], [2], [sc.window_slots(inst) - 1]):
            with self.subTest(starts=starts):
                with self.assertRaises(ValueError):
                    sc.clamp_starts(inst, np.array(starts))

    def test_clamp_requires_complete_integer_vector(self):
        inst = fixture(n=2)
        for starts in ([-1, 0], [0], [0, 1, 2], [[0, 1]], [0.5, 1],
                       [0.0, 1.0], [False, True], [float("nan"), 1]):
            with self.subTest(starts=starts):
                with self.assertRaises(ValueError):
                    sc.clamp_starts(inst, np.array(starts))

    def test_clamp_preserves_valid_tail_starts(self):
        inst = fixture(n=2, service=2, horizon=4)
        starts = np.array([4, sc.window_slots(inst) - 2])
        np.testing.assert_array_equal(sc.clamp_starts(inst, starts), starts)

    def test_metrics_rejects_capacity_overcommit_in_each_resource(self):
        for resource in sc.RESOURCES:
            with self.subTest(resource=resource):
                inst = fixture(n=2, capacity=2)
                inst.jobs[resource] = inst.capacity[resource]
                with self.assertRaisesRegex(ValueError, "capacity"):
                    sc.metrics(inst, np.array([0, 0]))

    def test_metrics_rejects_invalid_starts_instead_of_repairing(self):
        inst = fixture(n=1, service=2, arrival=3)
        for start in (-1, 0, sc.window_slots(inst) - 1):
            with self.subTest(start=start):
                with self.assertRaises(ValueError):
                    sc.metrics(inst, np.array([start]))

    def test_late_or_long_jobs_have_no_out_of_horizon_candidates(self):
        cases = ((3, 2, None), (0, 5, None), (0, 2, np.array([3])))
        for arrival, service, floor in cases:
            with self.subTest(arrival=arrival, service=service, floor=floor):
                inst = fixture(n=1, arrival=arrival, service=service, horizon=4)
                candidates = opt._candidate_slots(inst, np.array([0]), floor)[0]
                self.assertEqual(candidates.size, 0, candidates)

    def test_solver_reports_no_legal_candidates_without_fabricating_schedule(self):
        inst = fixture(n=2, service=[1, 2], arrival=[0, 3], horizon=4)
        for objective in ("flow", "makespan", "tardiness"):
            with self.subTest(objective=objective):
                starts, info = opt.solve_ilp(inst, np.arange(2), objective=objective)
                self.assertFalse(info["solver_feasible"])
                self.assertIsNone(starts)
                self.assertIn("candidate", info["message"].lower())
                self.assertIsNone(info["objective_value"])

    def test_lp_bound_is_not_an_integer_schedule(self):
        # LP splits one 2-GPU job across two slots; argmax would overbook 3 GPUs.
        inst = fixture(n=2, service=1, gpu=2, capacity=3, horizon=2)
        starts, info = opt.solve_ilp(inst, np.arange(2), binary=False)
        self.assertFalse(info["solver_feasible"])
        self.assertIsNone(starts)
        self.assertAlmostEqual(info["objective_value"], 2.5)

    def test_offline_lp_returns_only_bound_without_first_fit_schedule(self):
        inst = fixture(n=2, service=1, gpu=2, capacity=3, horizon=2)
        starts, info = opt.run_offline(inst, "flow", binary=False)
        self.assertIsNone(starts)
        self.assertFalse(info["solver_feasible"])
        self.assertFalse(info["solver_fell_back_to_first_fit"])
        self.assertAlmostEqual(info["objective_value"], 2.5)

    def test_real_integer_solver_still_returns_resource_conserving_schedule(self):
        inst = fixture(n=2, service=1, gpu=2, capacity=3, horizon=2)
        starts, info = opt.solve_ilp(inst, np.arange(2))
        self.assertTrue(info["solver_feasible"])
        self.assert_conserves_resources(inst, starts)
        self.assertAlmostEqual(info["objective_value"], 3.0)

    def test_fifo_and_priority_drain_long_queue_without_overcommit(self):
        for policy in (sc.run_fifo, lambda i: sc.run_fifo(i, backfill=True),
                       lambda i: sc.run_priority_online(i, "gpu_cards")):
            with self.subTest(policy=policy):
                inst = fixture(n=30, service=10, horizon=5)
                starts = policy(inst)
                np.testing.assert_array_equal(starts, np.arange(0, 300, 10))
                self.assert_conserves_resources(inst, starts)

    def test_fifo_and_priority_preserve_resources_on_small_legal_fixture(self):
        for policy in (sc.run_fifo, lambda i: sc.run_fifo(i, backfill=True),
                       lambda i: sc.run_priority_online(i, "gpu_cards"),
                       lambda i: sc.run_priority_online(i, "gpu_cards", aging_slots=2)):
            with self.subTest(policy=policy):
                inst = fixture(n=4, service=[2, 1, 3, 1], arrival=[0, 1, 1, 3],
                               gpu=[1, 2, 1, 2], capacity=2, horizon=8)
                starts = policy(inst)
                self.assert_conserves_resources(inst, starts)
                met = sc.metrics(inst, starts)
                self.assertEqual(met["capacity_violation_slots"], 0)
                self.assertEqual(met["used_gpu_card_hours"], round(9 / 3600, 2))


if __name__ == "__main__":
    unittest.main()
