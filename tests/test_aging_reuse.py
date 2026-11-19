"""The aging holdout experiment must not summarise a table it did not fully replay.

Two breaks are pinned here.

1. `--reuse-csv` is a 40-minute shortcut around the replay. It is only honest if it
   refuses anything other than a complete, current grid, so a stale or half-written
   aging_holdout.csv can never produce a headline number. Each test names the way a
   table can be incomplete: a missing instance, an extra instance, a changed policy
   grid, one dropped row inside an otherwise present instance.
2. `summarise()` used to crash on a merge key that was not selected (`KeyError: 'role'`)
   *after* the whole replay had finished. The merge and the tuning-vs-holdout transfer
   test are exercised here on a hand-built frame, with no simulator involved.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "aging_threshold_target", ROOT / "scripts" / "10_aging_threshold.py")
aging = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aging)

FULL_POLICIES = ["priority-gpu", *(aging.policy_name(h) for h in aging.GRID_HOURS)]
# The shipped table carries a FIFO row per instance as the baseline every policy is
# compared against, and load_cached_csv requires it -- the fixture mirrors that.
ALL_POLICIES = ["FIFO", *FULL_POLICIES]


def make_frame(instances=("i1.csv", "i2.csv"), role="holdout",
               policies=ALL_POLICIES, drop=None):
    """Hand-controlled replay rows: every wait a literal, no simulator involved.

    Along the grid order mean wait gets worse and max wait gets better, so the two
    objectives pick opposite ends of the grid -- which is the shape the real holdout
    data has, and what makes the transfer test non-tautological.
    """
    rows = []
    for inst in instances:
        cell = inst.replace(".csv", "")
        for k, pol in enumerate(policies):
            if drop is not None and (cell, pol) == drop:
                continue
            rows.append({
                "cell": cell, "pool": "P100", "instance": inst,
                "window_rank": 2, "seed": 20261103, "target_rho": 0.6,
                "achieved_rho": 0.6, "role": role, "policy": pol,
                "threshold_hours": None, "aging_slots": None, "slot_seconds": 600,
                "mean_wait_s": 100.0 + 10.0 * k,     # best at the last policy
                "max_wait_s": 500.0 - 40.0 * k,      # best at the first policy
                "deadline_miss_rate": 0.2,
            })
    return pd.DataFrame(rows)


def metas_for(instances):
    return [{"instance_file": i} for i in instances]


class ReuseGuardTests(unittest.TestCase):
    def test_complete_grid_is_accepted(self):
        insts = ("i1.csv", "i2.csv")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "aging_holdout.csv"
            make_frame(insts).to_csv(p, index=False)
            with patch.object(aging, "RES", Path(td)):
                df = aging.load_cached_csv(list(metas_for(insts)))
        self.assertEqual(len(df), 2 * len(ALL_POLICIES))

    def test_missing_instance_is_refused(self):
        insts = ("i1.csv", "i2.csv")
        with tempfile.TemporaryDirectory() as td:
            make_frame(("i1.csv",)).to_csv(Path(td) / "aging_holdout.csv", index=False)
            with patch.object(aging, "RES", Path(td)):
                with self.assertRaises(SystemExit) as cm:
                    aging.load_cached_csv(list(metas_for(insts)))
        self.assertIn("instance coverage mismatch", str(cm.exception))

    def test_extra_instance_is_refused(self):
        """A CSV from a wider grid is a different experiment, not a superset to trust."""
        with tempfile.TemporaryDirectory() as td:
            make_frame(("i1.csv", "i2.csv")).to_csv(Path(td) / "aging_holdout.csv",
                                                    index=False)
            with patch.object(aging, "RES", Path(td)):
                with self.assertRaises(SystemExit) as cm:
                    aging.load_cached_csv(metas_for(("i1.csv",)))
        self.assertIn("instance coverage mismatch", str(cm.exception))

    def test_narrower_policy_grid_is_refused(self):
        """If GRID_HOURS changed since the CSV was written, the CSV is stale."""
        insts = ("i1.csv",)
        with tempfile.TemporaryDirectory() as td:
            make_frame(insts, policies=ALL_POLICIES[:-2]).to_csv(
                Path(td) / "aging_holdout.csv", index=False)
            with patch.object(aging, "RES", Path(td)):
                with self.assertRaises(SystemExit) as cm:
                    aging.load_cached_csv(list(metas_for(insts)))
        self.assertIn("policy set mismatch", str(cm.exception))

    def test_partial_instance_is_refused(self):
        """Same policy set overall, one row missing inside one instance."""
        insts = ("i1.csv", "i2.csv")
        with tempfile.TemporaryDirectory() as td:
            make_frame(insts, drop=("i2", FULL_POLICIES[0])).to_csv(
                Path(td) / "aging_holdout.csv", index=False)
            with patch.object(aging, "RES", Path(td)):
                with self.assertRaises(SystemExit) as cm:
                    aging.load_cached_csv(list(metas_for(insts)))
        self.assertIn("lack the full policy grid", str(cm.exception))

    def test_absent_csv_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(aging, "RES", Path(td)):
                with self.assertRaises(SystemExit) as cm:
                    aging.load_cached_csv(metas_for(("i1.csv",)))
        self.assertIn("does not exist", str(cm.exception))


class SummariseTests(unittest.TestCase):
    def frame(self):
        return pd.concat([
            make_frame(("t1.csv", "t2.csv"), role="tuning"),
            make_frame(("h1.csv", "h2.csv"), role="holdout"),
        ], ignore_index=True)

    def test_merge_keeps_role_and_yields_one_row_per_cell(self):
        """Regression: the merge dropped 'role' and raised KeyError after the replay."""
        out = aging.summarise(self.frame(), res_dir=ROOT / "results")
        self.assertEqual(len(out["per_cell_argmin"]), 4)     # 2 tuning + 2 holdout
        for rec in out["per_cell_argmin"]:
            self.assertIn(rec["role"], ("tuning", "holdout"))
            self.assertIn("argmin_mean_wait", rec)
            self.assertIn("argmin_max_wait", rec)

    def test_ties_resolve_to_smaller_threshold(self):
        """Two cells whose whole row is constant: argmin must be the smallest threshold."""
        flat = self.frame()
        flat["mean_wait_s"] = 100.0
        out = aging.summarise(flat, res_dir=ROOT / "results")
        wins = out["holdout_argmin_wins_mean_wait"]
        self.assertEqual(wins[aging.policy_name(min(aging.GRID_HOURS))], 2)
        self.assertEqual(wins["priority-gpu"], 0)

    def test_transfer_reports_disagreement_not_agreement(self):
        """mean wait prefers the largest threshold, the tail the smallest: the tuning
        choice must be reported as a choice, and its holdout win share as a share."""
        out = aging.summarise(self.frame(), res_dir=ROOT / "results")
        t = out["tuning_to_holdout_transfer"]
        self.assertEqual((t["tuning_cells"], t["holdout_cells"]), (2, 2))
        self.assertEqual(t["tuning_choice"], FULL_POLICIES[0])
        self.assertIsInstance(t["tuning_choice_matches_holdout_choice"], bool)
        self.assertIn(t["committed_headline"], FULL_POLICIES)
        self.assertLessEqual(t["tuning_choice_holdout_win_share"], 1.0)

    def test_control_skipped_without_sweep_table(self):
        """An unreadable comparator is reported as skipped, never as agreement."""
        with tempfile.TemporaryDirectory() as td:
            out = aging.summarise(self.frame(), res_dir=Path(td))
        self.assertEqual(out["fifo_priority_control_vs_sweep"]["status"], "skipped")

    def test_control_matches_when_sweep_table_agrees(self):
        """FIFO and no-aging priority are the two rows the committed sweep also
        carries; agreement must be reported as a measured zero, not assumed."""
        frame = self.frame()
        with tempfile.TemporaryDirectory() as td:
            sm = (frame[frame.policy.isin(["FIFO", "priority-gpu"])]
                  .rename(columns={"cell": "sweep_key"})
                  [["sweep_key", "policy", "mean_wait_s"]])
            sm.to_csv(Path(td) / "sweep_metrics.csv", index=False)
            out = aging.summarise(frame, res_dir=Path(td))
        self.assertEqual(out["fifo_priority_control_vs_sweep"]["checked_rows"], len(sm))
        self.assertEqual(out["fifo_priority_control_vs_sweep"]["max_relative_deviation"],
                         0.0)

    def test_control_flags_a_shifted_replay(self):
        with tempfile.TemporaryDirectory() as td:
            frame = self.frame()
            sm = (frame[frame.policy.eq("priority-gpu")]
                  .rename(columns={"cell": "sweep_key"})
                  [["sweep_key", "policy", "mean_wait_s"]])
            sm["mean_wait_s"] = sm["mean_wait_s"] * 1.5
            sm.to_csv(Path(td) / "sweep_metrics.csv", index=False)
            out = aging.summarise(frame, res_dir=Path(td))
        self.assertAlmostEqual(
            out["fifo_priority_control_vs_sweep"]["max_relative_deviation"],
            1 / 3, places=6)

    def test_write_json_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            aging.summarise(self.frame(), res_dir=Path(td))
            self.assertFalse((Path(td) / "aging_holdout.json").exists())
            aging.summarise(self.frame(), res_dir=Path(td), write_json=True)
            written = json.loads((Path(td) / "aging_holdout.json").read_text(
                encoding="utf-8"))
        self.assertIn("tuning_to_holdout_transfer", written)


class CommittedSummaryTests(unittest.TestCase):
    """The shipped results/aging_holdout.json must still be a function of the shipped
    results/aging_holdout.csv -- i.e. the published numbers can be re-derived."""

    def test_committed_json_reproduces_from_committed_csv(self):
        df = pd.read_csv(ROOT / "results" / "aging_holdout.csv")
        with tempfile.TemporaryDirectory() as td:
            out = aging.summarise(df, res_dir=ROOT / "results")
            committed = json.loads((ROOT / "results" / "aging_holdout.json").read_text(
                encoding="utf-8"))
        for key in ("tuning_to_holdout_transfer", "holdout_regret_vs_oracle_mean_wait",
                    "holdout_argmin_wins_mean_wait", "holdout_sign_tests_mean_wait",
                    "instances", "fifo_priority_control_vs_sweep"):
            self.assertEqual(out[key], committed[key], key)


if __name__ == "__main__":
    unittest.main()
