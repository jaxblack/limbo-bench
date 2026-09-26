"""Test whether GitHub Issues honors an identical Idempotency-Key on two POSTs.

This separate, scripted control never calls a model and never retries an
ambiguous network error. It writes only synthetic issues to a private
repository that passes the same attestation as the real-service case study.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from .real_api import GitHubAPIError, GitHubIssues, RESULTS, github_token_for


def probe(api: GitHubIssues, marker: str) -> dict:
    title = f"LIMBO synthetic idempotency header probe {marker}"
    body = f"Private research test; marker: {marker}."
    first = api.create_issue(title, body, idempotency_key=marker)
    second = None
    second_error = None
    try:
        second = api.create_issue(title, body, idempotency_key=marker)
    except GitHubAPIError as exc:
        if exc.status is None:
            raise
        second_error = {"http_status": exc.status, "message": f"GitHub returned HTTP {exc.status} on the second POST"}

    known = {int(first["id"])}
    if second is not None:
        known.add(int(second["id"]))
    for attempt in range(10):
        found = api.find_issues(marker)
        ids = {int(issue["id"]) for issue in found}
        if known.issubset(ids):
            break
        time.sleep(1)
    else:
        raise GitHubAPIError("key probe list endpoint did not show all known writes")
    for issue in found:
        persisted = api.get_issue(int(issue["number"]))
        if (int(persisted["id"]) != int(issue["id"]) or persisted["title"] != title
                or persisted["body"] != body):
            raise GitHubAPIError("key probe persisted issue does not match synthetic request")
    return {"kind": "scripted_header_control", "model_calls": 0,
            "endpoint": "POST /repos/{owner}/{repo}/issues",
            "http_header": "Idempotency-Key", "marker": marker,
            "first_id": int(first["id"]), "second_id": int(second["id"]) if second is not None else None,
            "second_error": second_error, "persisted_ids": sorted(ids), "n_distinct_effects": len(ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", type=Path, default=RESULTS / "real_github" / "key_contract_probe.json")
    parser.add_argument("--confirm-isolated-test-repo", action="store_true")
    args = parser.parse_args()
    if not args.confirm_isolated_test_repo:
        parser.error("real API writes require --confirm-isolated-test-repo")
    if args.out.exists():
        parser.error("probe artifact already exists; never replay POSTs automatically")
    api = GitHubIssues(args.repo, github_token_for(args.repo))
    api.verify_test_repo()
    marker = "limbo-probe-" + uuid.uuid4().hex[:12]
    result = probe(api, marker)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"key probe: persisted issues={result['n_distinct_effects']}, "
          f"second status={result['second_error'] and result['second_error']['http_status'] or 'HTTP 201'}")


if __name__ == "__main__":
    main()
