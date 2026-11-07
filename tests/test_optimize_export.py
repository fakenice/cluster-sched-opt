"""Stage 3 export must carry solver provenance, not silently drop it.

The break caught: the info lookup tested `name in schedules` while the dict is keyed
by (tag, name) tuples, so every ILP row lost solve_seconds / fallback / gap fields.
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
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("optimize_export_target", ROOT / "scripts" / "03_optimize.py")
opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opt)


class ExportProvenanceTests(unittest.TestCase):
    def test_ilp_rows_keep_solver_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, res = Path(tmp) / "proc", Path(tmp) / "res"
            proc.mkdir(); res.mkdir()
            jobs = pd.DataFrame({"job_id": [0, 1], "arrival_rel": [0.0, 0.0],
                                 "service": [1.0, 1.0], "gpu_cards": [1.0, 1.0],
                                 "cpu_cores": [1.0, 1.0], "mem_gb": [1.0, 1.0]})
            jobs.to_csv(proc / "tiny.csv", index=False)
            meta = {"pool": "TINY", "achieved_rho": 0.5, "n_jobs": 2,
                    "instance_file": "tiny.csv", "slot_seconds": 1, "n_slots": 4,
                    "capacity": {"gpu_cards": 2.0, "cpu_cores": 4.0, "mem_gb": 4.0,
                                 "machines": 1},
                    "machine_spec": {"gpu_per_machine": 2.0}}
            (proc / "instance_meta.json").write_text(json.dumps([meta]), encoding="utf-8")
            with patch.object(opt, "PROC", proc), patch.object(opt, "RESULTS", res), \
                 patch.object(sys, "argv", ["03_optimize.py"]):
                opt.main()
            df = pd.read_csv(res / "metrics.csv")
            row = df[df.policy == "ILP-flow(offline)"].iloc[0]
            self.assertFalse(row["solver_fell_back_to_first_fit"])
            self.assertTrue(pd.notna(row["solve_seconds"]), "solve_seconds was dropped")
            self.assertTrue(pd.notna(row["milp_objective_sum_completion_s"]))
            self.assertTrue(pd.notna(row["lp_lower_bound_s"]))


if __name__ == "__main__":
    unittest.main()