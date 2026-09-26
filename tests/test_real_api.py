"""No-network checks for the real GitHub Issues case study."""

import json
import unittest
from unittest.mock import patch

from limbo.key_contract_probe import probe
from limbo.llm import AssistantTurn, ToolCall, Usage
from limbo.real_api import GitHubIssues, MAX_TOOL_CALLS, RealCase, RealSession, github_token_for, run_case, tool_schemas


class FakeIssues:
    repo = "owner/limbo-api-validation-test"
    _token = "never-serialize-this-token"

    def __init__(self):
        self.issues = {}
        self.comments = {}
        self.next_id = 100

    def create_issue(self, title, body, *, idempotency_key=None):
        self.next_id += 1
        row = {"id": self.next_id, "number": self.next_id, "title": title, "body": body}
        self.issues[row["number"]] = row
        return row

    def create_comment(self, issue_number, body):
        self.next_id += 1
        row = {"id": self.next_id, "issue_number": issue_number, "body": body}
        self.comments[row["id"]] = row
        return row

    def get_issue(self, number):
        return self.issues[number]

    def get_comment(self, comment_id):
        return self.comments[comment_id]

    def find_issues(self, marker):
        return [row for row in self.issues.values() if marker in row["title"] or marker in row["body"]]

    def find_comments(self, issue_number, marker):
        return [row for row in self.comments.values()
                if row["issue_number"] == issue_number and marker in row["body"]]


def case(operation="issue", mode="none"):
    return RealCase(batch_id="abcd1234", model="synthetic", operation=operation, mode=mode, replicate=0)


class FakeModel:
    def __init__(self, calls):
        self.calls = iter(calls)

    def complete(self, system, conversation, schemas):
        name, args = next(self.calls)
        return AssistantTurn("", [ToolCall(str(len(conversation)), name, json.dumps(args))], Usage(), 0.0)


class RealApiTests(unittest.TestCase):
    def test_pre_and_post_commit_return_identical_observations(self):
        observations = []
        for mode, expected_first, expected_after_retry in (("timeout_pre", 0, 1), ("timeout_post", 1, 2)):
            api = FakeIssues()
            session = RealSession(api, case(mode=mode))
            args = {"title": session.title, "body": session.body}
            observations.append(session.execute("issue_create", args))
            self.assertEqual(session.grade()["n_committed"], expected_first)
            session.execute("issue_create", args)
            self.assertEqual(session.grade()["n_committed"], expected_after_retry)
        self.assertEqual(observations[0], observations[1])

    def test_proxy_500_pair_has_same_observation_but_opposite_commit_state(self):
        observations = []
        for mode, expected_count in (("http500_pre", 0), ("http500_post", 1)):
            session = RealSession(FakeIssues(), case(mode=mode))
            obs = session.execute("issue_create", {"title": session.title, "body": session.body})
            observations.append(obs)
            self.assertEqual(session.grade()["n_committed"], expected_count)
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(observations[0]["error"]["status"], 500)

    def test_redelivery_commits_two_effects_despite_success_response(self):
        session = RealSession(FakeIssues(), case(mode="duplicate_delivery"))
        obs = session.execute("issue_create", {"title": session.title, "body": session.body})
        self.assertTrue(obs["ok"])
        self.assertEqual(session.grade()["dup_executed"], 1)

    def test_late_commit_is_graded_after_settling(self):
        session = RealSession(FakeIssues(), case(mode="timeout_late"), late_delay=0.01)
        obs = session.execute("issue_create", {"title": session.title, "body": session.body})
        self.assertEqual(obs["error"]["type"], "timeout")
        session.settle()
        self.assertEqual(session.grade()["n_committed"], 1)

    def test_comment_faults_change_only_the_fixture_issue(self):
        api = FakeIssues()
        session = RealSession(api, case(operation="comment", mode="duplicate_delivery"))
        args = {"issue_number": session.parent_number, "body": session.body}
        self.assertEqual(session.grade()["n_committed"], 0)
        session.execute("comment_create", args)
        self.assertEqual(session.grade()["dup_executed"], 1)
        self.assertEqual(len(api.issues), 1)
        self.assertEqual(len(api.comments), 2)

    def test_wrong_write_arguments_cannot_change_real_service(self):
        api = FakeIssues()
        session = RealSession(api, case(mode="timeout_post"))
        obs = session.execute("issue_create", {"title": "unrelated", "body": session.body})
        self.assertEqual(obs["error"]["type"], "invalid_arguments")
        self.assertFalse(session.fault_used)
        self.assertEqual(len(api.issues), 0)

    def test_grade_detects_an_unexpected_persisted_duplicate(self):
        api = FakeIssues()
        session = RealSession(api, case())
        session.execute("issue_create", {"title": session.title, "body": session.body})
        api.create_issue(session.title, session.body)
        self.assertEqual(session.grade()["n_committed"], 2)
        self.assertEqual(session.grade()["dup_executed"], 1)

    def test_operator_tool_does_not_promise_an_unavailable_operator(self):
        descriptions = {schema["name"]: schema["description"] for schema in tool_schemas("issue")}
        self.assertIn("No operator is available", descriptions["escalate_to_human"])

    def test_runner_retains_no_token_and_grades_verified_write(self):
        spec = case(mode="timeout_post")
        title = f"LIMBO synthetic validation {spec.marker}"
        body = f"Synthetic research issue. Marker: {spec.marker}."
        model = FakeModel([("issue_create", {"title": title, "body": body}),
                           ("issues_find", {}), ("finish", {"status": "completed", "summary": "verified"})])
        with patch.dict("os.environ", {"LIMBO_BASE_URL": "http://127.0.0.1:31400/v1"}):
            record = run_case(FakeIssues(), spec, model_client=model)
        self.assertEqual(record["stop_reason"], "finish")
        self.assertTrue(record["grade"]["EOS"])
        self.assertNotIn("never-serialize-this-token", json.dumps(record))
        self.assertEqual(record["model_transport"], "local_proxy_nonindependent")

    def test_parallel_tool_calls_cannot_exceed_case_write_limit(self):
        spec = case()
        title = f"LIMBO synthetic validation {spec.marker}"
        body = f"Synthetic research issue. Marker: {spec.marker}."

        class ManyCalls:
            def complete(self, system, conversation, schemas):
                calls = [ToolCall(str(i), "issue_create", json.dumps({"title": title, "body": body}))
                         for i in range(MAX_TOOL_CALLS + 4)]
                return AssistantTurn("", calls, Usage(), 0.0)

        record = run_case(FakeIssues(), spec, model_client=ManyCalls())
        self.assertEqual(record["stop_reason"], "max_tool_calls")
        self.assertEqual(record["grade"]["n_committed"], MAX_TOOL_CALLS)

    def test_repo_attestation_blocks_non_test_resources(self):
        api = GitHubIssues("owner/limbo-api-validation-test", "secret")
        with patch.object(api, "_request", return_value={
                "full_name": api.repo, "private": False, "description": "production", "has_issues": True}):
            with self.assertRaisesRegex(ValueError, "refusing real API writes"):
                api.verify_test_repo()
        with self.assertRaisesRegex(ValueError, "isolated"):
            GitHubIssues("owner/production-repo", "secret")

    def test_gh_auth_selects_test_repo_owner_without_logging_token(self):
        with patch.dict("os.environ", {"LIMBO_GITHUB_TOKEN": ""}):
            with patch("limbo.real_api.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "secret-token\n"
                self.assertEqual(github_token_for("owner/limbo-api-validation-test"), "secret-token")
                self.assertEqual(run.call_args.args[0], ["gh", "auth", "token", "--user", "owner"])

    def test_key_probe_reports_ignored_and_honored_headers_without_assuming_either(self):
        ignored = probe(FakeIssues(), "limbo-probe-ignored")
        self.assertEqual(ignored["n_distinct_effects"], 2)
        self.assertNotEqual(ignored["first_id"], ignored["second_id"])

        class HonoringIssues(FakeIssues):
            def __init__(self):
                super().__init__()
                self.keys = {}

            def create_issue(self, title, body, *, idempotency_key=None):
                if idempotency_key in self.keys:
                    return self.keys[idempotency_key]
                issue = super().create_issue(title, body, idempotency_key=idempotency_key)
                self.keys[idempotency_key] = issue
                return issue

        honored = probe(HonoringIssues(), "limbo-probe-honored")
        self.assertEqual(honored["n_distinct_effects"], 1)
        self.assertEqual(honored["first_id"], honored["second_id"])

    def test_probe_header_is_sent_only_on_synthetic_issue_post(self):
        class Response:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"id":1,"number":1}'

        api = GitHubIssues("owner/limbo-api-validation-test", "fake-token")
        with patch("limbo.real_api.urllib.request.urlopen", return_value=Response()) as request:
            api.create_issue("synthetic", "test", idempotency_key="limbo-probe-test")
            self.assertEqual(request.call_args.args[0].get_header("Idempotency-key"), "limbo-probe-test")
        with self.assertRaisesRegex(ValueError, "only on synthetic issue creation"):
            api._request("GET", f"/repos/{api.repo}/issues", idempotency_key="limbo-probe-test")


if __name__ == "__main__":
    unittest.main()
