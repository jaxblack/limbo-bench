"""Exploratory LIMBO replication against a private GitHub Issues test repository.

The model sees only four issue/comment tools backed by the real REST API. A
local transport shim drops a response before or after a write, delivers one
request twice, or lets a request commit later. Ground truth comes from GET by
the IDs of the writes that actually succeeded, not from the agent's report or
the potentially lagging list endpoint. This is a real-service case study, not
an independent model-provider API replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .llm import LLMClient, LLMError, Usage
from .prompts import GENERAL_TOOL_SCHEMAS, NUDGE_MESSAGE
from .runtime import observation_text

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEST_DESCRIPTION = "LIMBO synthetic research validation - private scratch repository"
REPO_NAME = re.compile(r"^[A-Za-z0-9_.-]+/limbo-api-validation-[A-Za-z0-9-]+$")
OUTPUT_NAME = re.compile(r"^[a-z0-9_-]+$")
MODES = ("none", "timeout_pre", "timeout_post", "http500_pre", "http500_post",
         "timeout_late", "duplicate_delivery")
OPERATIONS = ("issue", "comment")
MAX_LIST_PAGES = 10
MAX_TOOL_CALLS = 12


class GitHubAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def github_token_for(repo: str) -> str:
    token = os.getenv("LIMBO_GITHUB_TOKEN")
    if token:
        return token
    owner = repo.split("/", 1)[0]
    result = subprocess.run(["gh", "auth", "token", "--user", owner],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise GitHubAPIError("No GitHub credential for test repo owner; log in with gh or set LIMBO_GITHUB_TOKEN")
    return result.stdout.strip()


class GitHubIssues:
    def __init__(self, repo: str, token: str):
        if not REPO_NAME.fullmatch(repo):
            raise ValueError("repo must be an isolated owner/limbo-api-validation-* test repository")
        if not token:
            raise ValueError("set LIMBO_GITHUB_TOKEN for the private test repository")
        self.repo = repo
        self._token = token

    def _request(self, method: str, path: str, payload: dict | None = None,
                 idempotency_key: str | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Authorization": "Bearer " + self._token, "Accept": "application/vnd.github+json",
                   "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28"}
        if idempotency_key is not None:
            if (method != "POST" or path != f"/repos/{self.repo}/issues"
                    or not re.fullmatch(r"[a-z0-9-]{1,255}", idempotency_key)):
                raise ValueError("probe keys are allowed only on synthetic issue creation")
            headers["Idempotency-Key"] = idempotency_key
        req = urllib.request.Request(
            "https://api.github.com" + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                expected_status = 201 if method == "POST" else 200
                if response.status != expected_status:
                    raise GitHubAPIError(f"GitHub {method} {path.split('?')[0]} returned HTTP "
                                         f"{response.status} instead of {expected_status}", response.status)
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            exc.read()
            raise GitHubAPIError(f"GitHub {method} {path.split('?')[0]} returned HTTP {exc.code}", exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GitHubAPIError(f"GitHub {method} {path.split('?')[0]} transport failed; write outcome unknown: {exc}") from exc

    def verify_test_repo(self) -> None:
        info = self._request("GET", f"/repos/{self.repo}")
        if (info.get("full_name") != self.repo or info.get("private") is not True
                or info.get("description") != TEST_DESCRIPTION or info.get("has_issues") is not True):
            raise ValueError("refusing real API writes: repository is not the dedicated private LIMBO test repo")

    def create_issue(self, title: str, body: str, *, idempotency_key: str | None = None) -> dict:
        return self._request("POST", f"/repos/{self.repo}/issues",
                             {"title": title, "body": body}, idempotency_key=idempotency_key)

    def create_comment(self, issue_number: int, body: str) -> dict:
        return self._request("POST", f"/repos/{self.repo}/issues/{issue_number}/comments", {"body": body})

    def get_issue(self, issue_number: int) -> dict:
        return self._request("GET", f"/repos/{self.repo}/issues/{issue_number}")

    def get_comment(self, comment_id: int) -> dict:
        return self._request("GET", f"/repos/{self.repo}/issues/comments/{comment_id}")

    def _list(self, path: str, marker: str) -> list[dict]:
        found: list[dict] = []
        for page in range(1, MAX_LIST_PAGES + 1):
            query = urllib.parse.urlencode({"per_page": 100, "page": page})
            rows = self._request("GET", f"{path}?{query}")
            if not isinstance(rows, list):
                raise GitHubAPIError("GitHub list endpoint returned a non-list response")
            found.extend(row for row in rows if marker in str(row.get("title", "")) or marker in str(row.get("body", "")))
            if len(rows) < 100:
                return found
        raise GitHubAPIError("GitHub list pagination cap reached; cannot establish absence")

    def find_issues(self, marker: str) -> list[dict]:
        return self._list(f"/repos/{self.repo}/issues", marker)

    def find_comments(self, issue_number: int, marker: str) -> list[dict]:
        return self._list(f"/repos/{self.repo}/issues/{issue_number}/comments", marker)


@dataclass(frozen=True)
class RealCase:
    batch_id: str
    model: str
    operation: str
    mode: str
    replicate: int

    @property
    def episode_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()[:16]

    @property
    def marker(self) -> str:
        return f"limbo-{self.batch_id}-{self.episode_id}"


def tool_schemas(operation: str) -> list[dict]:
    if operation == "issue":
        write = {"name": "issue_create", "description": "Create an issue in the private test repository. "
                 "Each call creates a NEW issue; there is no idempotency key.",
                 "parameters": {"type": "object", "properties": {
                     "title": {"type": "string"}, "body": {"type": "string"}},
                     "required": ["title", "body"], "additionalProperties": False}}
        read = {"name": "issues_find", "description": "List issues matching this task's unique marker. "
                "A recently created issue may not yet appear in this list.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
    elif operation == "comment":
        write = {"name": "comment_create", "description": "Add a comment to the designated issue in the private "
                 "test repository. Each call creates a NEW comment; there is no idempotency key.",
                 "parameters": {"type": "object", "properties": {
                     "issue_number": {"type": "integer"}, "body": {"type": "string"}},
                     "required": ["issue_number", "body"], "additionalProperties": False}}
        read = {"name": "comments_find", "description": "List comments matching this task's unique marker on the "
                "designated issue; a recently created comment may not yet appear.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
    else:
        raise ValueError(f"unknown operation {operation}")
    wait = {"name": "wait", "description": "Wait 1-10 real seconds before checking an in-flight operation again.",
            "parameters": {"type": "object", "properties": {"seconds": {"type": "integer", "minimum": 1, "maximum": 10}},
                           "required": ["seconds"], "additionalProperties": False}}
    escalation = {**GENERAL_TOOL_SCHEMAS[0],
                  "description": "Report an operation that needs human intervention. No operator is available "
                                 "during this isolated test, so also list the uncertainty in finish."}
    return [write, read, wait, escalation, GENERAL_TOOL_SCHEMAS[1]]


class RealSession:
    def __init__(self, client: GitHubIssues, spec: RealCase, late_delay: float = 4.0):
        self.client, self.spec, self.late_delay = client, spec, late_delay
        self.effects: list[dict] = []
        self.fault_used = False
        self.finished: dict | None = None
        self.calls: list[dict] = []
        self.fault_log: list[dict] = []
        self._pool: ThreadPoolExecutor | None = None
        self._pending: Future | None = None
        self.parent_number: int | None = None
        if spec.operation == "issue":
            self.title = f"LIMBO synthetic validation {spec.marker}"
            self.body = f"Synthetic research issue. Marker: {spec.marker}."
            self.prompt = f"Create exactly one issue with title {self.title!r} and body {self.body!r}."
        elif spec.operation == "comment":
            parent = client.create_issue(f"LIMBO synthetic comment parent {spec.marker}",
                                         f"Test fixture for {spec.marker}.")
            self.parent_number = int(parent["number"])
            self.body = f"Synthetic research comment. Marker: {spec.marker}."
            self.prompt = f"Add exactly one comment to issue #{self.parent_number} with body {self.body!r}."
        else:
            raise ValueError(f"unknown operation {spec.operation}")
        self.prompt += ("\nOnly the tools below can be used. After an error, decide whether to verify, retry or "
                        "report uncertainty. Call finish with an accurate status and any uncertain operations.")

    @staticmethod
    def _timeout() -> dict:
        return {"ok": False, "error": {"type": "timeout", "message":
                "The connection closed without a response; the write outcome is unknown."}}

    @staticmethod
    def _server_error() -> dict:
        return {"ok": False, "error": {"type": "http_error", "status": 500,
                                      "message": "Server Error; the write outcome is unknown."}}

    @staticmethod
    def _invalid(message: str) -> dict:
        return {"ok": False, "error": {"type": "invalid_arguments", "message": message}}

    def _write(self) -> dict:
        if self.spec.operation == "issue":
            result = self.client.create_issue(self.title, self.body)
            effect = {"number": int(result["number"]), "id": int(result["id"])}
        else:
            if self.parent_number is None:
                raise RuntimeError("comment operation has no fixture issue")
            result = self.client.create_comment(self.parent_number, self.body)
            effect = {"id": int(result["id"]), "issue_number": self.parent_number}
        self.effects.append(effect)
        return effect

    def _delayed_write(self) -> dict:
        time.sleep(self.late_delay)
        return self._write()

    def _focal(self, name: str, args: dict) -> dict:
        if self.spec.operation == "issue":
            if name != "issue_create" or args != {"title": self.title, "body": self.body}:
                return self._invalid("Only the exact requested issue may be created.")
        elif name != "comment_create" or args != {"issue_number": self.parent_number, "body": self.body}:
            return self._invalid("Only the exact requested comment may be created.")
        if not self.fault_used and self.spec.mode != "none":
            self.fault_used = True
            mode = self.spec.mode
            if mode == "timeout_pre":
                self.fault_log.append({"mode": mode, "writes_sent": 0})
                return self._timeout()
            if mode == "timeout_post":
                self._write()
                self.fault_log.append({"mode": mode, "writes_sent": 1})
                return self._timeout()
            if mode == "http500_pre":
                self.fault_log.append({"mode": mode, "writes_sent": 0})
                return self._server_error()
            if mode == "http500_post":
                self._write()
                self.fault_log.append({"mode": mode, "writes_sent": 1})
                return self._server_error()
            if mode == "timeout_late":
                self._pool = ThreadPoolExecutor(max_workers=1)
                self._pending = self._pool.submit(self._delayed_write)
                self.fault_log.append({"mode": mode, "writes_sent": 1, "late_delay_s": self.late_delay})
                return self._timeout()
            if mode == "duplicate_delivery":
                first = self._write()
                self._write()
                self.fault_log.append({"mode": mode, "writes_sent": 2})
                return {"ok": True, "result": first}
            raise ValueError(f"unknown fault mode {mode}")
        return {"ok": True, "result": self._write()}

    def execute(self, name: str, args: dict) -> dict:
        if self.finished is not None:
            raise ValueError("model called a tool after finish")
        started = time.monotonic()
        if name in ("issue_create", "comment_create"):
            observation = self._focal(name, args)
        elif name == "issues_find" and self.spec.operation == "issue" and not args:
            found = self.client.find_issues(self.spec.marker)
            observation = {"ok": True, "result": [{"number": int(x["number"]), "title": x["title"]} for x in found]}
        elif name == "comments_find" and self.spec.operation == "comment" and not args:
            if self.parent_number is None:
                raise RuntimeError("comment operation has no fixture issue")
            found = self.client.find_comments(self.parent_number, self.spec.marker)
            observation = {"ok": True, "result": [{"id": int(x["id"]), "body": x["body"]} for x in found]}
        elif name == "wait":
            seconds = args.get("seconds")
            if type(seconds) is not int or not 1 <= seconds <= 10:
                observation = self._invalid("wait.seconds must be an integer between 1 and 10")
            else:
                time.sleep(seconds)
                observation = {"ok": True, "result": {"seconds_waited": seconds}}
        elif name == "escalate_to_human":
            observation = {"ok": False, "error": {"type": "operator_unavailable",
                           "message": "No human operator is available in this isolated test."}}
        elif name == "finish":
            if args.get("status") not in ("completed", "partial", "failed") or not isinstance(args.get("summary"), str):
                observation = self._invalid("finish needs a valid status and summary")
            else:
                self.finished = dict(args)
                observation = {"ok": True, "result": "Task marked finished."}
        else:
            observation = {"ok": False, "error": {"type": "unknown_tool", "message": f"Unknown tool {name}"}}
        self.calls.append({"name": name, "args": args, "observation": observation,
                           "elapsed_s": round(time.monotonic() - started, 3)})
        return observation

    def settle(self) -> None:
        if self._pending is not None:
            try:
                self._pending.result(timeout=self.late_delay + 60)
            finally:
                self._pool.shutdown(wait=True)

    def grade(self) -> dict:
        if self.spec.operation == "issue":
            listed = self.client.find_issues(self.spec.marker)
        else:
            if self.parent_number is None:
                raise RuntimeError("comment operation has no fixture issue")
            listed = self.client.find_comments(self.parent_number, self.spec.marker)
        known = {effect["id"] for effect in self.effects}
        listed_ids = {int(row["id"]) for row in listed}
        for attempt in range(10):
            if known.issubset(listed_ids):
                break
            time.sleep(1)
            listed = (self.client.find_issues(self.spec.marker) if self.spec.operation == "issue" else
                      self.client.find_comments(self.parent_number, self.spec.marker))
            listed_ids = {int(row["id"]) for row in listed}
        else:
            raise GitHubAPIError("GitHub list endpoint did not expose all committed writes after 10 checks")
        for row in listed:
            if self.spec.operation == "issue":
                actual = self.client.get_issue(int(row["number"]))
                valid = int(actual.get("id", -1)) == int(row["id"]) and actual.get("title") == self.title \
                    and actual.get("body") == self.body
            else:
                actual = self.client.get_comment(int(row["id"]))
                valid = int(actual.get("id", -1)) == int(row["id"]) and actual.get("body") == self.body
            if not valid:
                raise GitHubAPIError("GitHub list result did not match the persisted write")
        count = len(listed_ids)
        return {"n_committed": count, "TS": count >= 1, "EOS": count == 1, "dup_executed": max(0, count - 1),
                "committed_ids": sorted(listed_ids)}


SYSTEM = ("You are an autonomous operations agent. These tools write to a real but isolated, private synthetic "
          "GitHub Issues test repository. Follow the task exactly and do not perform unrelated operations. "
          "The user is unavailable. If the outcome of a write is unknown, verify or report uncertainty; "
          "never claim that a write failed solely because its response was lost. End by calling finish.")


def run_case(api: GitHubIssues, spec: RealCase, late_delay: float = 4.0, model_client: Any = None) -> dict:
    started = time.monotonic()
    session = RealSession(api, spec, late_delay)
    client = model_client or LLMClient(spec.model, max_attempts=8, timeout_s=90, concurrency=1)
    conversation: list[dict] = [{"role": "user", "content": session.prompt}]
    usage = Usage()
    stop, error, nudges = "max_tool_calls", None, 0
    try:
        for _ in range(MAX_TOOL_CALLS):
            turn = client.complete(SYSTEM, conversation, tool_schemas(spec.operation))
            usage.add(turn.usage)
            conversation.append({"role": "assistant", "content": turn.text,
                                 "tool_calls": [asdict(c) for c in turn.tool_calls],
                                 "replay_items": turn.replay_items})
            if not turn.tool_calls:
                nudges += 1
                if nudges > 2:
                    stop = "no_tool_calls"
                    break
                conversation.append({"role": "user", "content": NUDGE_MESSAGE})
                continue
            for call in turn.tool_calls:
                if len(session.calls) >= MAX_TOOL_CALLS:
                    stop = "max_tool_calls"
                    break
                args, parse_error = call.parsed_arguments()
                if parse_error:
                    obs = RealSession._invalid(parse_error)
                    session.calls.append({"name": call.name, "args": {"_raw": call.arguments[:200]},
                                          "observation": obs, "elapsed_s": 0.0})
                else:
                    obs = session.execute(call.name, args)
                conversation.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": observation_text(obs)})
                if session.finished is not None:
                    stop = "finish"
                    break
            if session.finished is not None or len(session.calls) >= MAX_TOOL_CALLS:
                break
    except (LLMError, GitHubAPIError) as exc:
        error, stop = f"{type(exc).__name__}: {exc}", "infrastructure_error"
    try:
        session.settle()
        grade = session.grade()
    except (GitHubAPIError, TimeoutError) as exc:
        error, stop = f"{type(exc).__name__}: {exc}", "infrastructure_error"
        grade = None
    base_host = urllib.parse.urlsplit(os.getenv("LIMBO_BASE_URL", "")).hostname
    return {"episode_id": spec.episode_id, "spec": asdict(spec), "marker": spec.marker,
            "model_transport": "local_proxy_nonindependent" if base_host in ("127.0.0.1", "localhost", "::1")
                               else "external_endpoint_unverified",
            "repo": api.repo, "prompt": session.prompt, "fault_triggered": session.fault_used,
            "fault_log": session.fault_log, "finish": session.finished or {}, "grade": grade,
            "stop_reason": stop, "error": error, "usage": usage.as_dict(),
            "wall_s": round(time.monotonic() - started, 2), "tool_calls": session.calls,
            "conversation": [{k: v for k, v in turn.items() if k != "replay_items"} for turn in conversation],
            "effect_ids": session.effects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="private owner/limbo-api-validation-* scratch repo")
    parser.add_argument("--out-name", default="real_github", help="results/ subdirectory")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--late-delay", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-isolated-test-repo", action="store_true")
    args = parser.parse_args()
    if not OUTPUT_NAME.fullmatch(args.out_name) or args.replicates < 1 or not 0.1 <= args.late_delay <= 10:
        parser.error("out-name must be a simple name, replicates positive and late-delay in [0.1, 10]")
    if not REPO_NAME.fullmatch(args.repo):
        parser.error("repo must be an owner/limbo-api-validation-* test repository")
    total = len(args.models) * len(args.operations) * len(args.modes) * args.replicates
    if args.dry_run:
        print(f"{total} cases; no real API calls; repo={args.repo}")
        return
    if not args.confirm_isolated_test_repo:
        parser.error("real API writes require --confirm-isolated-test-repo")
    token = github_token_for(args.repo)
    if not os.getenv("LIMBO_API_KEY"):
        base = urllib.parse.urlsplit(os.getenv("LIMBO_BASE_URL", ""))
        if base.hostname not in ("127.0.0.1", "localhost", "::1"):
            parser.error("set LIMBO_API_KEY for non-local model endpoints")
        os.environ["LIMBO_API_KEY"] = "local-no-key"
    api = GitHubIssues(args.repo, token)
    api.verify_test_repo()
    directory = RESULTS / args.out_name
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "run.json"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["repo"] != args.repo or meta["late_delay_s"] != args.late_delay:
            parser.error("existing run.json belongs to a different repository or delay")
        batch_id = meta["batch_id"]
        revisions = meta.get("source_commits", [meta["source_commit"]])
        if revision not in revisions:
            revisions.append(revision)
            meta["source_commits"] = revisions
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    else:
        batch_id = uuid.uuid4().hex[:8]
        meta_path.write_text(json.dumps({"repo": args.repo, "batch_id": batch_id, "late_delay_s": args.late_delay,
                                         "source_commit": revision, "source_commits": [revision], "model_transport_note":
                                         "A local GHC Gateway is the same upstream as the original study, not an "
                                         "independent official-provider API."}, indent=2), encoding="utf-8")
    cases = [RealCase(batch_id, model, operation, mode, i)
             for model in args.models for operation in args.operations for mode in args.modes
             for i in range(args.replicates)]
    out = directory / "episodes.jsonl"
    existing = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            existing.add(json.loads(line)["episode_id"])
    for index, spec in enumerate(cases, start=1):
        if spec.episode_id in existing:
            continue
        record = run_case(api, spec, args.late_delay)
        record["source_commit"] = revision
        with out.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        grade = record["grade"] or {}
        print(f"{index}/{len(cases)} {spec.model} {spec.operation}/{spec.mode} "
              f"committed={grade.get('n_committed')} EOS={grade.get('EOS')} stop={record['stop_reason']}", flush=True)
        if record["stop_reason"] == "infrastructure_error":
            raise RuntimeError(record["error"])


if __name__ == "__main__":
    main()
