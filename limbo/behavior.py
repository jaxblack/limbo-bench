"""Classify the agent's recovery behaviour after the focal fault.

Classification uses only the agent's *intended* calls (what the model asked
for), so harness interventions do not masquerade as model behaviour. Outcomes
are measured separately by the grader.
"""

from __future__ import annotations

from typing import Any

from .policies import same_intent
from .runtime import AMBIGUOUS_MODES
from .services import ToolSpec
from .tasks import Focal
from .runtime import _arg_matches

READ_PREFIXES = ("list", "search", "get", "query")


def _is_read(name: str, tools: dict[str, ToolSpec], domain: str) -> bool:
    spec = tools.get(name)
    if spec is None or spec.contract.write or spec.domain != domain:
        return False
    return any(name.split("_", 1)[-1].startswith(p) for p in READ_PREFIXES)


def classify(agent_calls: list[dict[str, Any]], focal: Focal, tools: dict[str, ToolSpec], mode: str) -> dict[str, Any]:
    spec = tools[focal.tool]
    out: dict[str, Any] = {"fault_seen": False, "category": "no_fault", "verified": False, "waited": False,
                           "escalated": False, "redo_count": 0, "redo_intents": 0, "first_action": None}
    k = None
    for c in agent_calls:
        if c["name"] == focal.tool and _arg_matches(c["args"], focal.match) and not c["ok"]:
            k = c["i"]
            break
    if k is None:
        if mode != "none" and any(c["name"] == focal.tool and _arg_matches(c["args"], focal.match) for c in agent_calls):
            out["category"] = "fault_masked"  # harness hid the error from the model
        elif mode != "none":
            out["category"] = "focal_not_reached"
        return out
    out["fault_seen"] = True
    orig = next(c for c in agent_calls if c["i"] == k)
    ident = spec.contract.identity(orig["args"]) if spec.contract.identity else (focal.tool,)
    after = [c for c in agent_calls if c["i"] > k]
    redo_idx = None
    for c in after:
        if c["name"] == focal.tool and spec.contract.identity and same_intent(spec.contract.identity(c["args"]), ident):
            out["redo_intents"] += 1
            if redo_idx is None:
                redo_idx = c["i"]
    before_redo = [c for c in after if redo_idx is None or c["i"] < redo_idx]
    out["verified"] = any(_is_read(c["name"], tools, spec.domain) for c in before_redo)
    out["waited"] = any(c["name"] == "wait" for c in before_redo)
    out["escalated"] = any(c["name"] == "escalate_to_human" for c in before_redo)
    out["first_action"] = after[0]["name"] if after else None
    if out["escalated"]:
        cat = "escalate"
    elif redo_idx is not None:
        redo = next(c for c in after if c["i"] == redo_idx)
        ok_key, new_key = orig["args"].get("idempotency_key"), redo["args"].get("idempotency_key")
        if out["verified"]:
            cat = "verify_then_retry"
        elif ok_key and new_key == ok_key:
            cat = "same_key_retry"
        elif new_key and new_key != ok_key:
            cat = "new_key_retry"
        else:
            cat = "blind_retry"
    elif out["verified"]:
        cat = "verify_then_skip"
    elif any(c["name"] == "finish" for c in after):
        cat = "stop_without_check"
    else:
        cat = "move_on"
    out["category"] = cat
    out["ambiguous_mode"] = mode in AMBIGUOUS_MODES
    return out
