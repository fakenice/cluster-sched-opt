"""The sweep must refuse to call an incomplete or infeasible result table a success.

Each test names the break it catches: a capacity-violating row, an unplaced job, a
missing policy, a duplicate row, or a resume that treats a partial instance as done.
The integration tests drive the real --report-only path on controlled CSV fixtures.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sweep_validation_target", ROOT / "scripts" / "08_sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

FULL_POLICIES = ["FIFO", "EASY-backfill", "ILP-flow(offline)", "ILP-rolling",
                 "priority-cpu", "priority-gpu", "priority-gpu-age1h", "priority-gpu-age2h",
                 "priority-gpu-age4h", "priority-mem", "priority-short"]
LIST_POLICIES = [p for p in FULL_POLICIES if not p.startswith("ILP")]


def make_frame(keys=("K1", "K2"), policies=FULL_POLICIES):
    """Hand-controlled metric rows: every value a literal, no simulator involved."""
    rows = []
    for key in keys:
        for pol in policies:
            rows.append({
                "sweep_key": key, "pool": "P100", "window_rank": 1,
                "window_rank_label": "busiest", "seed": 20261013,
                "target_rho": 0.6, "achieved_rho": 0.6, "n_jobs": 10,
                "policy": pol, "oracle": False, "aging_slots": None,
                "mean_wait_s": 100.0, "p90_wait_s": 200.0, "max_wait_s": 300.0,
                "wait_change_vs_fifo": 0.0, "p90_change_vs_fifo": 0.0,
                "max_change_vs_fifo": 0.0, "deadline_miss_rate": 0.0,
                "captured_ilp_wait_gain": None,
                "idle_addressable_backlog_card_hours": 0.0,
                "idle_with_backlog_card_hours": 1.0,
                "addressable_share_of_backlog_idle": 0.0,
                "capacity_violation_slots": 0, "unplaceable_jobs": 0,
            })
    return pd.DataFrame(rows)


class ValidateSweepTests(unittest.TestCase):
    def test_valid_complete_grid_passes(self):
        report = sweep.validate_sweep(make_frame(), FULL_POLICIES)
        self.assertTrue(report["ok"], report)

    def test_capacity_violation_fails(self):
        df = make_frame()
        df.loc[(df.sweep_key == "K2") & (df.policy == "ILP-rolling"),
               "capacity_violation_slots"] = 4
        report = sweep.validate_sweep(df, FULL_POLICIES)
        self.assertFalse(report["ok"])
        self.assertEqual(report["capacity_violation_rows"][0]["sweep_key"], "K2")

    def test_unplaceable_jobs_fail(self):
        df = make_frame()
        df.loc[(df.sweep_key == "K1") & (df.policy == "ILP-rolling"),
               "unplaceable_jobs"] = 8
        report = sweep.validate_sweep(df, FULL_POLICIES)
        self.assertFalse(report["ok"])
        self.assertEqual(report["unplaceable_rows"][0]["policy"], "ILP-rolling")

    def test_missing_policy_fails(self):
        df = make_frame()
        df = df[~((df.sweep_key == "K1") & (df.policy == "priority-mem"))]
        report = sweep.validate_sweep(df, FULL_POLICIES)
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing_policies"], [{"sweep_key": "K1", "missing": ["priority-mem"]}])

    def test_duplicate_row_fails(self):
        df = make_frame()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = sweep.validate_sweep(df, FULL_POLICIES)
        self.assertFalse(report["ok"])
        self.assertEqual(report["duplicate_rows"], [{"sweep_key": "K1", "policy": "FIFO", "count": 2}])

    def test_non_finite_wait_fails(self):
        df = make_frame()
        df.loc[0, "mean_wait_s"] = float("nan")
        report = sweep.validate_sweep(df, FULL_POLICIES)
        self.assertFalse(report["ok"])

    def test_completed_keys_requires_full_policy_set(self):
        df = make_frame()
        df = df[~((df.sweep_key == "K1") & (df.policy == "priority-mem"))]
        self.assertEqual(sweep.completed_keys(df, FULL_POLICIES), {"K2"})


class ReportOnlyExitTests(unittest.TestCase):
    """Drive the real CLI path: an invalid table must not exit 0 or print a headline."""

    def run_report_only(self, df):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = tmp / "sweep_metrics.csv"
            json_path = tmp / "sweep_summary.json"
            df.to_csv(csv_path, index=False)
            with patch.object(sweep, "OUT_CSV", csv_path), \
                 patch.object(sweep, "OUT_JSON", json_path), \
                 patch.object(sweep, "SWEEP", tmp / "sweep"), \
                 patch.object(sweep, "RES", tmp), \
                 patch.object(sys, "argv", ["08_sweep.py", "--report-only", "--no-ilp"]):
                code = sweep.main()
            summary = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None
            return code, summary

    def test_valid_table_exits_zero_and_records_validation(self):
        code, summary = self.run_report_only(make_frame())
        self.assertEqual(code, 0)
        self.assertTrue(summary["validation"]["ok"], summary["validation"])

    def test_capacity_violating_table_exits_nonzero(self):
        df = make_frame()
        df.loc[0, "capacity_violation_slots"] = 4
        code, summary = self.run_report_only(df)
        self.assertNotEqual(code, 0)
        self.assertFalse(summary["validation"]["ok"])

    def test_incomplete_table_exits_nonzero(self):
        df = make_frame()
        df = df[~((df.sweep_key == "K1") & (df.policy == "priority-mem"))]
        code, _ = self.run_report_only(df)
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()