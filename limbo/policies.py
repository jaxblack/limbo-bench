"""Recovery conditions.

Prompt-level conditions change only what the model is told. Harness-level
conditions wrap tool execution the way an agent framework or middleware would.
Harness policies may read tool *contracts* (MCP-style hints, identity fields,
declared read-back path, documented visibility SLA) but never the ground truth,
except the two oracles (``oracle`` = current-state oracle, ``outcome_oracle``),
which are reported only as reference points.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .runtime import ToolRuntime
from .services import similar

AMBIGUOUS_TYPES = {"timeout"}
AMBIGUOUS_STATUS = {500, 502, 504}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def is_error(obs: dict[str, Any]) -> bool:
    return not obs.get("ok", False)


def error_kind(obs: dict[str, Any]) -> tuple[str | None, int | None]:
    err = obs.get("error") or {}
    return err.get("type"), err.get("status")


def is_ambiguous(obs: dict[str, Any]) -> bool:
    t, s = error_kind(obs)
    return is_error(obs) and (t in AMBIGUOUS_TYPES or s in AMBIGUOUS_STATUS)


def same_intent(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, str) and isinstance(y, str) and max(len(x), len(y)) > 24:
            if not similar(x, y, 0.8):
                return False
        elif x != y:
            return False
    return True


@dataclass
class Pending:
    tool: str
    args: dict[str, Any]
    identity: tuple
    t: float
    key: str | None
    status: str = "unknown"  # unknown | found | absent | partial | unverifiable
    detail: dict[str, Any] = field(default_factory=dict)


class Policy:
    name = "vanilla"
    prompt_variant = "vanilla"
    reflect = False

    def __init__(self, episode_id: str = "") -> None:
        self.episode_id = episode_id
        self.interventions: list[dict[str, Any]] = []

    def execute(self, rt: ToolRuntime, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return rt.execute(name, args, source="agent")

    def log(self, kind: str, **info: Any) -> None:
        self.interventions.append({"kind": kind, **info})


class Aware(Policy):
    name = "aware"
    prompt_variant = "aware"


class Reflect(Policy):
    name = "reflect"
    reflect = True


class SDKRetry(Policy):
    """Transparent client-side retries, as in common HTTP/LLM-framework defaults."""

    name = "sdk_retry3"

    def __init__(self, episode_id: str = "", retries: int = 3) -> None:
        super().__init__(episode_id)
        self.retries = retries

    def execute(self, rt, name, args):
        obs = rt.execute(name, args, source="agent")
        for attempt in range(self.retries):
            t, s = error_kind(obs)
            if not is_error(obs) or not (t == "timeout" or s in RETRYABLE_STATUS):
                break
            wait = float((obs.get("error") or {}).get("retry_after_seconds") or 2 ** attempt)
            rt.world.advance(wait)
            self.log("auto_retry", tool=name, attempt=attempt + 1)
            obs = rt.execute(name, args, source="harness", note=f"sdk retry {attempt + 1}")
        return obs


class Rules(Policy):
    """Error-code rules: back off and retry only errors that signal 'not processed'."""

    name = "rules"

    def execute(self, rt, name, args):
        obs = rt.execute(name, args, source="agent")
        for attempt in range(3):
            t, s = error_kind(obs)
            if s == 429:
                wait = float((obs.get("error") or {}).get("retry_after_seconds") or 5)
            elif s == 503:
                wait = 2.0 * (2 ** attempt)
            else:
                break
            rt.world.advance(wait)
            self.log("rule_retry", tool=name, status=s, attempt=attempt + 1)
            obs = rt.execute(name, args, source="harness", note=f"rule retry {attempt + 1}")
        return obs


class OutcomeGuard(Policy):
    """Verify-before-retry family.

    vbr:            re-verify an unknown-outcome write before letting an identical write through.
    wait{N}:        vbr + documented-lag awareness + an assumed in-flight bound of N seconds: verification is
                    deferred until N seconds (plus the read-path lag) after the original request was sent.
    guard:          vbr + automatic idempotency keys + consistency-aware verification +
                    blocking of unverifiable non-idempotent repeats + outcome annotations.
    oracle:         guard that verifies against the current ground-truth state ("state oracle"). It cannot see
                    requests that are still in flight, so it is NOT an upper bound under late commits.
    outcome_oracle: guard that also sees in-flight requests: on an identical re-issue it waits for the in-flight
                    request to commit and returns its record, and it keeps suppressing later re-issues. This is
                    the upper bound for client-side recovery (only redelivery on key-less writes remains).
    """

    def __init__(self, episode_id: str = "", *, name: str = "guard", auto_key: bool = True,
                 consistency_aware: bool = True, block_unverifiable: bool = True, annotate: bool = True,
                 oracle: str | None = None, min_wait: float = 0.0) -> None:
        super().__init__(episode_id)
        self.name = name
        self.auto_key = auto_key
        self.consistency_aware = consistency_aware
        self.block_unverifiable = block_unverifiable
        self.annotate = annotate
        self.oracle = oracle  # None | "state" | "outcome"
        self.min_wait = float(min_wait)
        self.pending: list[Pending] = []

    def _key(self, identity: tuple) -> str:
        raw = json.dumps([self.episode_id, list(map(str, identity))])
        return "limbo-" + hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _find_pending(self, identity: tuple) -> Pending | None:
        # The outcome oracle keeps suppressing identical re-issues after a write is found, so that it is a
        # true upper bound for client-side policies; the evaluated guards suppress once per unknown outcome.
        active = ("unknown", "partial", "unverifiable") + (("found",) if self.oracle == "outcome" else ())
        for p in reversed(self.pending):
            if p.status in active and same_intent(p.identity, identity):
                return p
        return None

    def _in_flight_due(self, rt: ToolRuntime, p: Pending) -> float | None:
        spec = rt.tools[p.tool]
        dues = [float(meta.get("due", 0.0)) for meta in rt.world.pending()
                if meta.get("tool") == p.tool and same_intent(spec.contract.identity(meta.get("args") or {}), p.identity)]
        return min(dues) if dues else None

    def _verify(self, rt: ToolRuntime, p: Pending) -> None:
        spec = rt.tools[p.tool]
        if self.oracle == "outcome":
            res = rt.verify(p.tool, p.args, p.t, truth=True)
            due = self._in_flight_due(rt, p) if res.status == "absent" else None
            if due is not None:
                # Perfect knowledge of the in-flight request: wait for it to commit, then return its record.
                self.log("oracle_wait_for_commit", tool=p.tool, seconds=round(max(0.0, due - rt.world.clock), 1))
                if rt.world.clock <= due:
                    rt.world.advance(due - rt.world.clock + 0.001)
                res = rt.verify(p.tool, p.args, p.t, truth=True)
            p.status, p.detail = res.status, res.detail
            self.log("verify", tool=p.tool, status=res.status, oracle="outcome")
            return
        truth = self.oracle == "state"
        sla = float(spec.contract.visibility_sla(p.args) or 0.0) if (self.consistency_aware and not truth) else 0.0
        if self.min_wait > 0:
            ready = p.t + self.min_wait + sla + 1.0
            if rt.world.clock < ready:
                self.log("assumed_inflight_wait", tool=p.tool, seconds=round(ready - rt.world.clock, 1))
                rt.world.advance(ready - rt.world.clock)
            res = rt.verify(p.tool, p.args, p.t, truth=truth)
        else:
            res = rt.verify(p.tool, p.args, p.t, truth=truth)
            ready = p.t + sla + 1.0
            if res.status == "absent" and sla > 0 and rt.world.clock < ready:
                self.log("consistency_wait", tool=p.tool, seconds=round(ready - rt.world.clock, 1))
                rt.world.advance(ready - rt.world.clock)
                res = rt.verify(p.tool, p.args, p.t, truth=False)
        p.status, p.detail = res.status, res.detail
        self.log("verify", tool=p.tool, status=res.status)

    def execute(self, rt, name, args):
        spec = rt.tools.get(name)
        if name == "escalate_to_human":
            obs = rt.execute(name, args, source="agent")
            self._absorb_escalation(obs)
            return obs
        if spec is None or not spec.contract.write or spec.contract.identity is None:
            return rt.execute(name, args, source="agent")

        args = dict(args)
        identity = spec.contract.identity(args)
        if self.auto_key and spec.contract.supports_key(args) and not args.get("idempotency_key"):
            args["idempotency_key"] = self._key(identity)
            self.log("auto_key", tool=name)

        p = self._find_pending(identity)
        if p is not None:
            if p.status in ("unknown", "unverifiable"):
                self._verify(rt, p)
            if p.status == "found":
                self.log("suppressed_duplicate", tool=name)
                return {"ok": True, "result": p.detail,
                        "reliability_guard": "An earlier identical request returned an error but has been verified as "
                                             "applied. This request was NOT sent again, to avoid a duplicate."}
            if p.status == "partial" and spec.contract.batch_field:
                missing = p.detail.get("missing_rows") or []
                if p.key is not None and spec.contract.supports_key(args):
                    # A keyed batch resumes server-side: resend the identical request under the same key.
                    args = dict(p.args)
                    self.log("resume_keyed_batch", tool=name)
                    p.status = "superseded"
                    obs = rt.execute(name, args, source="agent", note="guard: resume keyed batch")
                    return self._after(rt, name, args, identity, obs)
                if missing:
                    args[spec.contract.batch_field] = missing
                    self.log("rewrote_batch", tool=name, rows=len(missing))
                    p.status = "absent"
                    obs = rt.execute(name, args, source="agent", note="guard: only missing rows")
                    obs = dict(obs)
                    obs["reliability_guard"] = (f"An earlier identical batch was partially applied; only the "
                                                f"{len(missing)} missing row(s) were sent.")
                    return self._after(rt, name, args, identity, obs)
            if p.status == "unverifiable":
                keyed = p.key is not None and p.key == args.get("idempotency_key")
                if not keyed and self.block_unverifiable and not spec.contract.idempotent:
                    self.log("blocked_unverifiable", tool=name)
                    return {"ok": False, "error": {
                        "type": "guard_blocked",
                        "message": "Blocked by the reliability guard: an earlier identical request returned an error and "
                                   "may already have taken effect, and this service offers no way to verify it "
                                   "automatically. Ask a human with escalate_to_human, or report the operation as "
                                   "uncertain in finish."}}
            if p.status == "absent":
                p.status = "superseded"
        obs = rt.execute(name, args, source="agent")
        return self._after(rt, name, args, identity, obs)

    def _after(self, rt, name, args, identity, obs):
        if is_ambiguous(obs):
            self.pending.append(Pending(name, dict(args), identity, rt.events[-1].t_start, args.get("idempotency_key")))
            if self.annotate:
                obs = dict(obs)
                obs["reliability_guard"] = ("Outcome UNKNOWN: this request may or may not have taken effect. "
                                            "Do not assume it failed. The guard will verify before any identical retry.")
        return obs

    def _absorb_escalation(self, obs: dict[str, Any]) -> None:
        for f in ((obs.get("result") or {}).get("findings") or []):
            for p in self.pending:
                if p.tool == f.get("tool") and p.args == f.get("arguments"):
                    finding = f.get("operator_finding", "")
                    if finding.startswith("DID"):
                        p.status, p.detail = "found", f.get("record") or {}
                    elif finding.startswith("did NOT"):
                        p.status = "absent"
                    elif finding.startswith("PARTIALLY"):
                        p.status, p.detail = "partial", {"missing_rows": f.get("missing") or []}


def make_policy(name: str, episode_id: str) -> Policy:
    if name == "vanilla":
        return Policy(episode_id)
    if name == "aware":
        return Aware(episode_id)
    if name == "reflect":
        return Reflect(episode_id)
    if name == "sdk_retry3":
        return SDKRetry(episode_id)
    if name == "rules":
        return Rules(episode_id)
    if name == "vbr":
        return OutcomeGuard(episode_id, name="vbr", auto_key=False, consistency_aware=False,
                            block_unverifiable=False, annotate=False)
    if name == "guard":
        return OutcomeGuard(episode_id, name="guard")
    if name == "oracle":  # current-state oracle; see OutcomeGuard docstring
        return OutcomeGuard(episode_id, name="oracle", oracle="state")
    if name == "outcome_oracle":
        return OutcomeGuard(episode_id, name="outcome_oracle", oracle="outcome")
    m = re.fullmatch(r"wait(\d+)", name)
    if m:
        return OutcomeGuard(episode_id, name=name, auto_key=False, consistency_aware=True, block_unverifiable=False,
                            annotate=False, min_wait=float(m.group(1)))
    # Ablations: guard minus one component.
    if name.startswith("guard-no-"):
        comp = name[len("guard-no-"):]
        kw = {"auto_key": comp != "key", "consistency_aware": comp != "consistency",
              "block_unverifiable": comp != "block", "annotate": comp != "annotate"}
        return OutcomeGuard(episode_id, name=name, **kw)
    raise ValueError(f"unknown policy {name}")


POLICIES = ["vanilla", "aware", "reflect", "sdk_retry3", "rules", "vbr", "guard", "oracle"]
