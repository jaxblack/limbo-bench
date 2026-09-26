"""Regression checks for the analysis denominators used in the manuscript."""

import unittest
from unittest.mock import patch

import pandas as pd

from limbo.analysis import focal_first_key_used, mcfadden_r2
from limbo.report import duplicate_reporting_metrics, fmt_p


class ReportingTests(unittest.TestCase):
    def test_key_use_is_first_matching_focal_write_not_any_call(self):
        record = {
            "spec": {"contract": "keys_everywhere"},
            "focal": {"tool": "tickets_create", "match": {"project": "OPS"}},
            "agent_calls": [
                {"name": "mail_send", "args": {"idempotency_key": "unrelated"}},
                {"name": "tickets_create", "args": {"project": "COMMS", "idempotency_key": "other-project"}},
                {"name": "tickets_create", "args": {"project": "OPS", "title": "test"}},
                {"name": "tickets_create", "args": {"project": "OPS", "title": "test",
                                                    "idempotency_key": "new-key"}},
            ],
        }
        self.assertIs(focal_first_key_used(record), False)
        record["agent_calls"][2]["args"]["idempotency_key"] = "original-key"
        self.assertIs(focal_first_key_used(record), True)

    def test_naturally_idempotent_write_is_not_in_key_use_denominator(self):
        record = {
            "spec": {"contract": "keys_everywhere"},
            "focal": {"tool": "billing_refund_charge", "match": {"charge_id": "charge-test"}},
            "agent_calls": [{"name": "billing_refund_charge", "args": {"charge_id": "charge-test"}}],
        }
        self.assertIsNone(focal_first_key_used(record))

    def test_completed_and_overclaim_are_distinct_for_compensated_duplicates(self):
        episodes = pd.DataFrame([
            {"template": "a", "model": "model", "mode": "duplicate_delivery", "triggered": True,
             "dup_exec": 2, "claimed_complete": True, "uncertain_reported": False, "overclaim": False},
            {"template": "b", "model": "model", "mode": "timeout_post", "triggered": True,
             "dup_exec": 1, "claimed_complete": True, "uncertain_reported": True, "overclaim": True},
        ])
        result = duplicate_reporting_metrics(episodes)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["completed_given_dup"], 1.0)
        self.assertEqual(result["complete_no_uncertainty_given_dup"], 0.5)
        self.assertEqual(result["overclaim_given_dup"], 0.5)
        self.assertLess(result["completed_ci"][0], 1.0)

    def test_near_threshold_p_values_are_not_rounded_to_point_zero_five(self):
        self.assertEqual(fmt_p(0.04585), "0.046")

    def test_shapley_fit_fails_instead_of_reporting_zero_on_nonfinite_likelihood(self):
        frame = pd.DataFrame({"y": [0, 1, 0, 1], "factor": ["a", "a", "b", "b"]})
        with patch("statsmodels.api.GLM") as fitted:
            fitted.return_value.fit_regularized.return_value.params = [0.0, 0.0]
            fitted.return_value.score.return_value = [0.0, 0.0]
            fitted.return_value.loglike.return_value = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite pseudo-R²"):
                mcfadden_r2(frame, "y", ["factor"])

    def test_shapley_fit_rejects_nonconverged_score(self):
        frame = pd.DataFrame({"y": [0, 1, 0, 1], "factor": ["a", "a", "b", "b"]})
        with patch("statsmodels.api.GLM") as fitted:
            fitted.return_value.fit_regularized.return_value.params = [0.0, 0.0]
            fitted.return_value.score.return_value = [1.0, 0.0]
            with self.assertRaisesRegex(ValueError, "did not converge"):
                mcfadden_r2(frame, "y", ["factor"])


if __name__ == "__main__":
    unittest.main()
