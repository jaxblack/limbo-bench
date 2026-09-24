"""Episode runner: one task, one fault plan, one model, one recovery condition."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .behavior import classify
from .grader import grade
from .llm import LLMClient, LLMError, Usage
from .policies import is_error, make_policy
from .prompts import GENERAL_TOOL_SCHEMAS, NUDGE_MESSAGE, REFLECT_MESSAGE, system_prompt
from .runtime import FAULT_TRUTH, FaultSpec, ToolRuntime, observation_text
from .services import DOMAIN_TOOLS, build_tools
from .tasks import make_task
from .world import World


@dataclass
class EpisodeSpec:
    template: str
    index: int
    focal: str
    mode: str
    model: str
    policy: str = "vanilla"
    doc_variant: str = "neutral"
    paraphrase: int = 0
    human_available: bool = True
    reasoning_effort: str | None = None
    replicate: int = 0
    max_tool_calls: int = 40
    experiment: str = "adhoc"
    harness: str = "minimal"
    contract: str = "native"
    instruction_variant: str = "default"

    @property
    def episode_id(self) -> str:
        d = asdict(self)
        # Drop new fields at their defaults so ids of earlier episodes stay stable.
        if d.get("harness") == "minimal":
            d.pop("harness")
        if d.get("contract") == "native":
            d.pop("contract")
        if d.get("instruction_variant") == "default":
            d.pop("instruction_variant")
        raw = json.dumps(d, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def world_seed(self) -> int:
        # Same world for every model/policy/replicate of one (task, focal, mode): paired design.
        raw = f"{self.template}:{self.index}:{self.focal}:{self.mode}"
        return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


class ScriptedModel:
    """Test double: a callable returning tool calls from the conversation so far."""

    def __init__(self, fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]):
        self.fn = fn
        self.model = "scripted"


def _tool_schemas(task, tools, doc_variant: str) -> list[dict[str, Any]]:
    names: list[str] = []
    for d in task.domains:
        names.extend(DOMAIN_TOOLS[d])
    names.append("wait")
    return [tools[n].schema(doc_variant) for n in names] + GENERAL_TOOL_SCHEMAS


def run_episode(spec: EpisodeSpec, client: Any = None) -> dict[str, Any]:
    t_wall = time.time()
    task = make_task(spec.template, spec.index, spec.instruction_variant)
    world = World(seed=spec.world_seed)
    task.setup(world)
    world.human_available = spec.human_available
    world.keys_everywhere = spec.contract == "keys_everywhere"
    tools = build_tools(spec.contract)
    focal = next(f for f in task.focals if f.label == spec.focal)
    faults = [] if spec.mode == "none" else [FaultSpec(spec.mode, focal.tool, focal.match, 1)]
    rt = ToolRuntime(world, tools, faults)
    policy = make_policy(spec.policy, spec.episode_id)
    schemas = _tool_schemas(task, tools, spec.doc_variant)
    system = system_prompt(policy.prompt_variant, spec.paraphrase)
    conv: list[dict[str, Any]] = [{"role": "user", "content": task.instruction}]
    agent_calls: list[dict[str, Any]] = []
    usage = Usage()
    llm_latency = 0.0
    n_turns = n_nudges = 0
    stop_reason = "max_tool_calls"
    error = None
    if client is None:
        client = LLMClient(spec.model, reasoning_effort=spec.reasoning_effort)
    try:
        while len(agent_calls) < spec.max_tool_calls:
            n_turns += 1
            if isinstance(client, ScriptedModel):
                calls = client.fn(conv)
                text, replay = "", []
                from .llm import ToolCall
                tcs = [ToolCall(c.get("id", f"s{n_turns}_{j}"), c["name"], json.dumps(c.get("args", {}))) for j, c in enumerate(calls)]
            else:
                turn = client.complete(system, conv, schemas)
                usage.add(turn.usage)
                llm_latency += turn.latency_s
                text, tcs, replay = turn.text, turn.tool_calls, turn.replay_items
            conv.append({"role": "assistant", "content": text,
                         "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tcs],
                         "replay_items": replay})
            if not tcs:
                n_nudges += 1
                if n_nudges > 2:
                    stop_reason = "no_tool_calls"
                    break
                conv.append({"role": "user", "content": NUDGE_MESSAGE})
                continue
            had_error = False
            for c in tcs:
                args, perr = c.parsed_arguments()
                if perr is not None:
                    obs = {"ok": False, "error": {"type": "invalid_arguments", "message": perr}}
                    args = {"_raw": c.arguments}
                else:
                    obs = policy.execute(rt, c.name, args)
                err = obs.get("error") or {}
                agent_calls.append({"i": len(agent_calls), "turn": n_turns, "name": c.name, "args": args,
                                    "ok": bool(obs.get("ok")), "error_type": err.get("type"),
                                    "status": err.get("status"), "guard": obs.get("reliability_guard"),
                                    "t": world.clock})
                had_error = had_error or is_error(obs)
                conv.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": observation_text(obs)})
                if rt.finished is not None or len(agent_calls) >= spec.max_tool_calls:
                    break
            if rt.finished is not None:
                stop_reason = "finish"
                break
            if policy.reflect and had_error:
                conv.append({"role": "user", "content": REFLECT_MESSAGE})
    except LLMError as exc:
        error, stop_reason = f"LLMError: {exc}", "llm_error"
    except Exception as exc:  # keep the harness alive; the record is marked invalid
        error, stop_reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}", "harness_error"

    rt.flush()
    g = grade(task, world)
    beh = classify(agent_calls, focal, tools, spec.mode)
    fin = rt.finished or {}
    claimed_complete = str(fin.get("status", "")).lower() == "completed"
    uncertain_reported = bool(fin.get("uncertain_operations"))
    return {
        "episode_id": spec.episode_id,
        "spec": asdict(spec),
        "task": task.summary(),
        "focal": asdict(focal),
        "fault_truth": FAULT_TRUTH.get(spec.mode),
        "fault_triggered": bool(rt.fault_log),
        "fault_log": rt.fault_log,
        "grade": g,
        "behavior": beh,
        "finish": fin,
        "claimed_complete": claimed_complete,
        "overclaim": claimed_complete and not (g["TS"] and g["dup_live"] == 0),
        "uncertain_reported": uncertain_reported,
        "stop_reason": stop_reason,
        "error": error,
        "n_turns": n_turns,
        "n_agent_calls": len(agent_calls),
        "n_executions": len(rt.events),
        "escalations": rt.escalations,
        "human_minutes": world.human_minutes,
        "virtual_seconds": world.clock,
        "usage": usage.as_dict(),
        "llm_latency_s": round(llm_latency, 2),
        "wall_s": round(time.time() - t_wall, 2),
        "interventions": policy.interventions,
        "agent_calls": agent_calls,
        "events": rt.trace(),
        "assistant_texts": [t.get("content") for t in conv if t["role"] == "assistant" and t.get("content")],
        "conversation": [{k: v for k, v in t.items() if k != "replay_items"} for t in conv],
        "system_prompt": system,
        "protocol": getattr(client, "protocol", "scripted"),
    }
