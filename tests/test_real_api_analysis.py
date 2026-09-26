"""Tests for fail-closed, descriptive analysis of real-service episodes."""

import json
import tempfile
import unittest
from pathlib import Path

from limbo.real_api import MODES
from limbo.real_api_analysis import check_design, latex_table, load_records, summarize


def record(model="test-model", operation="issue", mode="timeout_post", replicate=0, count=1):
    return {"episode_id": f"{model}-{operation}-{mode}-{replicate}", "spec": {
        "model": model, "operation": operation, "mode": mode, "replicate": replicate},
        "grade": {"n_committed": count, "committed_ids": list(range(count)),
                  "TS": count > 0, "EOS": count == 1, "dup_executed": max(0, count - 1)},
        "fault_triggered": True, "stop_reason": "finish", "finish": {"status": "completed"},
        "fault_log": ([{"mode": "timeout_late", "late_delay_s": 4.0}] if mode == "timeout_late" else []),
        "tool_calls": [],
        "wall_s": 2.0, "usage": {"input_tokens": 10, "output_tokens": 2},
        "model_transport": "local_proxy_nonindependent"}


class RealApiAnalysisTests(unittest.TestCase):
    def test_complete_design_is_required_before_reporting(self):
        rows = [record(operation=operation, mode=mode) for operation in ("issue", "comment") for mode in MODES]
        check_design(rows, ("test-model",), 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check_design(rows[:-1], ("test-model",), 1)

    def test_infrastructure_errors_are_not_successes(self):
        rows = [record(count=1), record(replicate=1, count=2), record(replicate=2, count=0)]
        rows[-1]["stop_reason"] = "infrastructure_error"
        rows[-1]["grade"] = None
        summary = summarize(rows)
        self.assertEqual((summary["n_attempted"], summary["n_valid"], summary["n_infrastructure_errors"]),
                         (3, 2, 1))
        cell = summary["model_mode"]["test-model/timeout_post"]
        self.assertEqual(cell["n"], 2)
        self.assertEqual(cell["duplicate"]["events"], 1)
        self.assertEqual(cell["EOS"]["events"], 1)

    def test_table_preserves_denominators_and_models(self):
        rows = [record(mode=mode) for mode in MODES]
        table = latex_table(summarize(rows), ("test-model",))
        self.assertIn("test-model & lost acknowledgement & 1", table)
        self.assertIn("test-model & redelivery & 1", table)
        self.assertIn("late commit (4 s)", table)

    def test_long_delay_companion_has_separate_shape_and_visibility(self):
        rows = [record(operation=operation, mode="timeout_late") for operation in ("issue", "comment")]
        rows[0]["tool_calls"] = [{"name": "issues_find", "observation": {"ok": True, "result": []}}]
        rows[1]["tool_calls"] = [{"name": "comments_find", "observation": {"ok": True, "result": [{"id": 42}]}}]
        for row in rows:
            row["fault_log"][0]["late_delay_s"] = 90.0
        check_design(rows, ("test-model",), 1, modes=("timeout_late",))
        summary = summarize(rows)
        self.assertEqual(summary["late_delay_s"], 90.0)
        self.assertEqual(summary["model_mode"]["test-model/timeout_late"]["first_read_visible"]["events"], 1)
        table = latex_table(summary, ("test-model",), ("timeout_late",), ("issue", "comment"))
        self.assertIn("test-model & issue & late commit (90 s) & 1", table)
        self.assertIn("test-model & comment & late commit (90 s) & 1", table)

    def test_mixed_in_flight_delays_cannot_be_pooled(self):
        rows = [record(mode="timeout_late"), record(mode="timeout_late", replicate=1)]
        rows[1]["fault_log"][0]["late_delay_s"] = 90.0
        with self.assertRaisesRegex(ValueError, "mixed late-commit delays"):
            summarize(rows)

    def test_recorded_effect_ids_must_match_grading_fields(self):
        row = record(count=2)
        row["grade"]["EOS"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent persisted-effect grade"):
            summarize([row])

    def test_duplicate_episode_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.jsonl"
            line = json.dumps(record()) + "\n"
            path.write_text(line * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate episode IDs"):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
