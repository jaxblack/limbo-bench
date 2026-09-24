"""Tool execution runtime with deterministic fault injection.

Faults are applied at the service boundary, so the ground-truth ledger always
reflects what the service actually committed. Two pairs of fault modes are
*observation-equivalent*: ``timeout_pre``/``timeout_post`` and
``http500_pre``/``http500_post`` produce byte-identical observations and clock
advances; only the hidden commit differs.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from .services import GENERIC_KEY_TOOLS, PartialStop, ToolSpec, VerifyResult, norm_text
from .world import ToolError, World

TIMEOUT_SECONDS = 30.0
ESCALATION_MINUTES = 15.0
HUMAN_TIMEOUT_MINUTES = 30.0
RATE_LIMIT_SECONDS = 20.0

FAULT_MODES = (
    "none",
    "timeout_pre",
    "timeout_post",
    "timeout_late",
    "timeout_late_tail",
    "http500_pre",
    "http500_post",
    "http503_transient",
    "outage",
    "rate_limit",
    "schema_drift",
    "partial_timeout",
    "duplicate_delivery",
)
# Ground truth side-effect status of the triggering call, per mode.
FAULT_TRUTH = {
    "none": "confirmed",
    "timeout_pre": "absent",
    "timeout_post": "confirmed",
    "timeout_late": "late",
    "timeout_late_tail": "late",
    "http500_pre": "absent",
    "http500_post": "confirmed",
    "http503_transient": "absent",
    "outage": "absent",
    "rate_limit": "absent",
    "schema_drift": "absent",
    "partial_timeout": "partial",
    "duplicate_delivery": "duplicated",
}
AMBIGUOUS_MODES = {"timeout_pre", "timeout_post", "timeout_late", "timeout_late_tail", "http500_pre", "http500_post",
                   "partial_timeout"}
LATE_MODES = {"timeout_late", "timeout_late_tail"}
# A timed-out request that is still in flight commits this long after it was sent.
LATE_COMMIT_SECONDS = 90.0
# timeout_late_tail: in-flight delay drawn log-uniformly from this range, deterministically per world.
LATE_TAIL_RANGE = (40.0, 7200.0)


def late_tail_delay(world_seed: int) -> float:
    lo, hi = LATE_TAIL_RANGE
    u = random.Random(f"{world_seed}:late-tail").random()
    return round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))), 1)


@dataclass
class FaultSpec:
    mode: str
    tool: str = ""
    match: dict[str, Any] = field(default_factory=dict)
    nth: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "tool": self.tool, "match": self.match, "nth": self.nth}


def _arg_matches(args: dict[str, Any], match: dict[str, Any]) -> bool:
    for k, v in match.items():
        if k.endswith("__contains"):
            base = k[: -len("__contains")]
            val = args.get(base)
            hay = " ".join(map(str, val)) if isinstance(val, list) else str(val or "")
            if norm_text(v) not in norm_text(hay):
                return False
        else:
            if norm_text(args.get(k)) != norm_text(v):
                return False
    return True


def timeout_observation(domain: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": "timeout",
            "message": f"Request timed out after {TIMEOUT_SECONDS:.0f}s: no response was received from the {domain} service "
            "before the connection was closed.",
        },
    }


def http500_observation() -> dict[str, Any]:
    return {"ok": False, "error": {"type": "http_error", "status": 500, "message": "Internal Server Error"}}


def http503_observation(domain: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": "http_error",
            "status": 503,
            "message": f"Service Unavailable: the {domain} service is temporarily overloaded. Please retry later.",
        },
    }


def _fingerprint(args: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in args.items() if k != "idempotency_key"}, sort_keys=True, default=str)


def invoke(spec: ToolSpec, w: World, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool handler, adding Stripe-style key semantics where the contract
    variant grants them to a tool that has no native key support."""
    key = args.get("idempotency_key")
    if not (w.keys_everywhere and key and spec.name in GENERIC_KEY_TOOLS):
        return spec.handler(w, args)
    slot = (spec.name, str(key))
    fp = _fingerprint(args)
    entry = w.key_store.get(slot)
    if entry is not None:
        if entry["fp"] != fp:
            raise ToolError(409, "Conflict: idempotency_key was already used with different parameters")
        if entry.get("result") is not None:
            return {**entry["result"], "idempotent_replay": True}
    entry = entry or {"fp": fp, "done": 0, "result": None}
    w.key_store[slot] = entry
    n_before = len(w.ledger)
    w.batch_skip = entry["done"]
    try:
        res = spec.handler(w, args)
    except PartialStop:
        entry["done"] += len(w.ledger) - n_before
        raise
    except ToolError:
        if entry["done"] == 0 and entry["result"] is None:
            w.key_store.pop(slot, None)
        raise
    finally:
        w.batch_skip = 0
    entry["result"] = res
    return res


@dataclass
class ToolEvent:
    index: int
    source: str
    tool: str
    args: dict[str, Any]
    observation: dict[str, Any]
    t_start: float
    t_end: float
    fault: str | None
    effects: list[int]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source": self.source,
            "tool": self.tool,
            "args": self.args,
            "observation": self.observation,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "fault": self.fault,
            "effects": self.effects,
            "note": self.note,
        }


class ToolRuntime:
    def __init__(self, world: World, tools: dict[str, ToolSpec], faults: list[FaultSpec]):
        self.world = world
        self.tools = tools
        self.faults = [f for f in faults if f.mode != "none"]
        self._match_counts = [0 for _ in self.faults]
        self._persistent: dict[int, float] = {}  # fault idx -> active-since clock
        self._rate_until: dict[int, float] = {}
        self.events: list[ToolEvent] = []
        self.fault_log: list[dict[str, Any]] = []
        self.uncertain: list[dict[str, Any]] = []
        self.escalations = 0
        self.finished: dict[str, Any] | None = None
        self._next_index = 0

    # ------------------------------------------------------------ deferred
    def _late_commit(self, name: str, args: dict[str, Any]) -> None:
        """An in-flight request finishes server-side (scheduled on the world clock)."""
        w = self.world
        spec = self.tools[name]
        n_before = len(w.ledger)
        prev_source = w.call_source
        w.call_source = "late_commit"
        try:
            invoke(spec, w, dict(args))
        except (ToolError, PartialStop, TypeError, ValueError, KeyError, AttributeError):
            pass
        finally:
            w.call_source = prev_source
        eids = [e.eid for e in w.ledger[n_before:]]
        self.events.append(ToolEvent(self._next_index, "late_commit", name, copy.deepcopy(args),
                                     {"ok": True, "note": "in-flight request committed"}, w.clock, w.clock,
                                     "timeout_late", eids))
        self._next_index += 1

    def flush(self) -> None:
        """At episode end, in-flight requests still complete eventually."""
        self.world.settle()

    # ------------------------------------------------------------ helpers
    def _fault_for(self, name: str, args: dict[str, Any]) -> tuple[int, FaultSpec] | None:
        for i, f in enumerate(self.faults):
            if f.tool != name or not _arg_matches(args, f.match):
                continue
            if f.mode in ("outage", "schema_drift") and i in self._persistent:
                return i, f
            if f.mode == "rate_limit" and i in self._rate_until and self.world.clock < self._rate_until[i]:
                return i, f
            self._match_counts[i] += 1
            if self._match_counts[i] == f.nth:
                return i, f
        return None

    def _log_event(self, source, name, args, obs, t0, fault, eids, note="") -> dict[str, Any]:
        ev = ToolEvent(self._next_index, source, name, copy.deepcopy(args), obs, t0, self.world.clock, fault, eids, note)
        self._next_index += 1
        self.events.append(ev)
        return obs

    # ----------------------------------------------------------- execution
    def execute(self, name: str, args: dict[str, Any], source: str = "agent", note: str = "") -> dict[str, Any]:
        w = self.world
        t0 = w.clock
        n_before = len(w.ledger)
        w.call_index = self._next_index
        w.call_source = source
        if name == "escalate_to_human":
            return self._log_event(source, name, args, self._escalate(args), t0, None, [], note)
        if name == "finish":
            self.finished = dict(args)
            return self._log_event(source, name, args, {"ok": True, "result": "Task marked finished."}, t0, None, [], note)
        spec = self.tools.get(name)
        if spec is None:
            obs = {"ok": False, "error": {"type": "tool_error", "message": f"Unknown tool '{name}'."}}
            w.advance(0.1)
            return self._log_event(source, name, args, obs, t0, None, [], note)

        hit = self._fault_for(name, args)
        mode = hit[1].mode if hit else None
        call_args = dict(args)
        if mode == "schema_drift":
            self._persistent.setdefault(hit[0], w.clock)
            old, new = spec.contract.drift or ("", "")
            if old and old in call_args and new not in call_args:
                w.advance(0.3)
                obs = ToolError(400, f"Bad Request: unknown field '{old}'. This API version expects '{new}' instead "
                                     f"(field renamed in API v2).").observation()
                self.fault_log.append({"mode": mode, "tool": name, "index": self._next_index, "t": t0})
                return self._log_event(source, name, args, obs, t0, mode, [], note)
            if new and new in call_args:
                call_args[old] = call_args.pop(new)
            mode = None
        elif mode == "outage":
            self._persistent.setdefault(hit[0], w.clock)
            w.advance(0.5)
            self.fault_log.append({"mode": mode, "tool": name, "index": self._next_index, "t": t0})
            return self._log_event(source, name, args, http503_observation(spec.domain), t0, mode, [], note)
        elif mode == "rate_limit":
            idx = hit[0]
            if idx not in self._rate_until:
                self._rate_until[idx] = w.clock + RATE_LIMIT_SECONDS
            w.advance(0.2)
            remaining = max(1, int(round(self._rate_until[idx] - w.clock)))
            obs = {"ok": False, "error": {"type": "http_error", "status": 429, "message": "Too Many Requests",
                                          "retry_after_seconds": remaining}}
            self.fault_log.append({"mode": mode, "tool": name, "index": self._next_index, "t": t0})
            return self._log_event(source, name, args, obs, t0, mode, [], note)
        elif mode is not None:
            # A request the service would reject anyway gets its real 4xx, and the
            # trigger stays armed for the next matching call. Checking this for
            # both pre- and post-commit modes keeps the equivalent pairs symmetric.
            rejected = self._dry_run_error(spec, call_args)
            if rejected is not None:
                self._match_counts[hit[0]] -= 1
                w.advance(0.3)
                return self._log_event(source, name, args, rejected, t0, None, [], note)
            if mode in ("timeout_pre", "http500_pre", "http503_transient") or mode in LATE_MODES:
                late_delay = None
                if mode == "timeout_pre" or mode in LATE_MODES:
                    w.advance(TIMEOUT_SECONDS)
                    obs = timeout_observation(spec.domain)
                    if mode in LATE_MODES:
                        late_delay = LATE_COMMIT_SECONDS if mode == "timeout_late" else late_tail_delay(w.seed)
                        pending_args = dict(call_args)
                        w.schedule(t0 + late_delay, lambda: self._late_commit(name, pending_args),
                                   meta={"tool": name, "args": dict(call_args), "sent_at": t0, "due": t0 + late_delay})
                elif mode == "http500_pre":
                    w.advance(1.5)
                    obs = http500_observation()
                else:
                    w.advance(0.5)
                    obs = http503_observation(spec.domain)
                self._note_uncertain(name, args, t0, obs)
                entry = {"mode": mode, "tool": name, "index": self._next_index, "t": t0}
                if late_delay is not None:
                    entry["late_delay"] = late_delay
                self.fault_log.append(entry)
                return self._log_event(source, name, args, obs, t0, mode, [], note)

        if mode == "partial_timeout" and spec.contract.batch_field:
            batch = call_args.get(spec.contract.batch_field)
            w.partial_limit = max(1, len(batch) // 2) if isinstance(batch, list) and len(batch) > 1 else None
        try:
            obs = invoke(spec, w, call_args)
            w.advance(spec.latency)
        except PartialStop:
            obs = {}
        except ToolError as exc:
            w.advance(0.3)
            obs = exc.observation()
            mode = None
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            w.advance(0.3)
            obs = ToolError(400, f"Bad Request: {type(exc).__name__}: {exc}").observation()
            mode = None
        finally:
            w.partial_limit = None

        if mode == "duplicate_delivery" and obs.get("ok"):
            # At-least-once transport: the same request reaches the service twice.
            prev = w.call_source
            w.call_source = "redelivery"
            try:
                invoke(spec, w, dict(call_args))
            except (ToolError, PartialStop, TypeError, ValueError, KeyError, AttributeError):
                pass
            finally:
                w.call_source = prev
            self.fault_log.append({"mode": mode, "tool": name, "index": self._next_index, "t": t0})
        if mode in ("timeout_post", "partial_timeout"):
            w.advance(max(0.0, TIMEOUT_SECONDS - (w.clock - t0)))
            obs = timeout_observation(spec.domain)
        elif mode == "http500_post":
            w.advance(max(0.0, 1.5 - (w.clock - t0)))
            obs = http500_observation()
        if mode in AMBIGUOUS_MODES:
            self._note_uncertain(name, args, t0, obs)
            self.fault_log.append({"mode": mode, "tool": name, "index": self._next_index, "t": t0})
        eids = [e.eid for e in w.ledger[n_before:]]
        return self._log_event(source, name, args, obs, t0, mode, eids, note)

    def _dry_run_error(self, spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any] | None:
        shadow = copy.deepcopy(self.world)
        shadow.deferred = []  # never let a shadow world fire the real world's in-flight commits
        try:
            invoke(spec, shadow, dict(args))
        except PartialStop:
            return None
        except ToolError as exc:
            return exc.observation()
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            return ToolError(400, f"Bad Request: {type(exc).__name__}: {exc}").observation()
        return None

    def _note_uncertain(self, name: str, args: dict[str, Any], t0: float, obs: dict[str, Any]) -> None:
        spec = self.tools.get(name)
        if spec and spec.contract.write:
            self.uncertain.append({"index": self._next_index, "tool": name, "args": copy.deepcopy(args), "t": t0,
                                   "error": obs.get("error", {})})

    # ---------------------------------------------------------- verification
    def verify(self, name: str, args: dict[str, Any], t_call: float, *, truth: bool = False) -> VerifyResult:
        spec = self.tools.get(name)
        if spec is None or spec.contract.verifier is None:
            return VerifyResult("unverifiable", {"reason": "no read-back path declared"})
        return spec.contract.verifier(self.world, args, t_call, truth)

    def _escalate(self, args: dict[str, Any]) -> dict[str, Any]:
        w = self.world
        self.escalations += 1
        if not w.human_available:
            w.advance(HUMAN_TIMEOUT_MINUTES * 60)
            return {"ok": False, "error": {"type": "no_response",
                                           "message": f"No response: the on-call operator did not answer within {HUMAN_TIMEOUT_MINUTES:.0f} minutes."}}
        w.advance(ESCALATION_MINUTES * 60)
        w.human_minutes += ESCALATION_MINUTES
        findings = []
        for u in self.uncertain:
            v = self.verify(u["tool"], u["args"], u["t"], truth=True)
            status = {"found": "DID take effect", "absent": "did NOT take effect",
                      "partial": "PARTIALLY took effect"}.get(v.status, "could not be determined")
            item = {"tool": u["tool"], "arguments": u["args"], "operator_finding": status}
            if v.status == "partial":
                item["missing"] = v.detail.get("missing_rows")
            elif v.status == "found":
                item["record"] = v.detail
            findings.append(item)
        if not findings:
            msg = "Operator: I looked into it and found nothing wrong with the systems. Please continue with the task."
        else:
            msg = "Operator: I manually checked the systems for every request that previously returned an error."
        return {"ok": True, "result": {"message": msg, "findings": findings}}

    # ------------------------------------------------------------- summary
    def trace(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]


def observation_text(obs: dict[str, Any]) -> str:
    return json.dumps(obs, ensure_ascii=False, sort_keys=True, default=str)
