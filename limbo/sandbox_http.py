"""Sandbox session served over localhost HTTP, shared by every MCP proxy process.

The runner owns the world in-process; harness CLIs reach it only through the
MCP proxy (``limbo.mcp_proxy``), which exposes exactly the tools/list and
tools/call operations. No fault plan or ground truth ever leaves this process.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import EpisodeSpec
from .behavior import classify
from .grader import grade
from .policies import make_policy
from .prompts import GENERAL_TOOL_SCHEMAS
from .runtime import FAULT_TRUTH, FaultSpec, ToolRuntime
from .services import DOMAIN_TOOLS, build_tools
from .tasks import make_task
from .world import World


class SandboxSession:
    def __init__(self, spec: EpisodeSpec) -> None:
        self.spec = spec
        self.task = make_task(spec.template, spec.index, spec.instruction_variant)
        self.world = World(seed=spec.world_seed)
        self.task.setup(self.world)
        self.world.human_available = spec.human_available
        self.world.keys_everywhere = spec.contract == "keys_everywhere"
        self.tools = build_tools(spec.contract)
        self.focal = next(f for f in self.task.focals if f.label == spec.focal)
        faults = [] if spec.mode == "none" else [FaultSpec(spec.mode, self.focal.tool, self.focal.match, 1)]
        self.rt = ToolRuntime(self.world, self.tools, faults)
        self.policy = make_policy(spec.policy, spec.episode_id)
        names: list[str] = []
        for d in self.task.domains:
            names.extend(DOMAIN_TOOLS[d])
        names.append("wait")
        self.schemas = [self.tools[n].schema(spec.doc_variant) for n in names] + GENERAL_TOOL_SCHEMAS
        self.agent_calls: list[dict[str, Any]] = []
        self.started = time.time()
        self.lock = threading.Lock()
        self.list_calls = 0

    def _annotations(self, name: str) -> dict[str, Any]:
        spec = self.tools.get(name)
        if spec is None:
            return {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
        c = spec.contract
        return {"readOnlyHint": not c.write, "destructiveHint": bool(c.destructive or c.irreversible),
                "idempotentHint": bool(c.idempotent or not c.write), "openWorldHint": True}

    def list_tools(self) -> list[dict[str, Any]]:
        with self.lock:
            self.list_calls += 1
        return [{"name": s["name"], "description": s["description"], "inputSchema": s["parameters"],
                 "annotations": self._annotations(s["name"])} for s in self.schemas]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not any(s["name"] == name for s in self.schemas):
                obs = {"ok": False, "error": {"type": "tool_error", "message": f"Unknown tool '{name}'."}}
            else:
                obs = self.policy.execute(self.rt, name, dict(args or {}))
            err = obs.get("error") or {}
            self.agent_calls.append({"i": len(self.agent_calls), "turn": None, "name": name, "args": args or {},
                                     "ok": bool(obs.get("ok")), "error_type": err.get("type"),
                                     "status": err.get("status"), "guard": obs.get("reliability_guard"),
                                     "t": self.world.clock, "wall": round(time.time() - self.started, 2)})
            return obs

    def record(self, final: bool = True) -> dict[str, Any]:
        with self.lock:
            if final:
                self.rt.flush()
            g = grade(self.task, self.world)
            fin = self.rt.finished or {}
            claimed = str(fin.get("status", "")).lower() == "completed"
            return {
                "episode_id": self.spec.episode_id,
                "spec": dict(self.spec.__dict__),
                "task": self.task.summary(),
                "focal": dict(self.focal.__dict__),
                "fault_truth": FAULT_TRUTH.get(self.spec.mode),
                "fault_triggered": bool(self.rt.fault_log),
                "fault_log": self.rt.fault_log,
                "grade": g,
                "behavior": classify(self.agent_calls, self.focal, self.tools, self.spec.mode),
                "finish": fin,
                "claimed_complete": claimed,
                "overclaim": claimed and not (g["TS"] and g["dup_live"] == 0),
                "uncertain_reported": bool(fin.get("uncertain_operations")),
                "n_agent_calls": len(self.agent_calls),
                "n_executions": len(self.rt.events),
                "escalations": self.rt.escalations,
                "human_minutes": self.world.human_minutes,
                "virtual_seconds": self.world.clock,
                "interventions": self.policy.interventions,
                "agent_calls": self.agent_calls,
                "events": self.rt.trace(),
                "tools_list_calls": self.list_calls,
            }


class SandboxServer:
    """Owns one SandboxSession behind a token-protected localhost endpoint."""

    def __init__(self, session: SandboxSession) -> None:
        self.session = session
        self.token = secrets.token_hex(16)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence stderr
                pass

            def do_POST(self):
                if self.headers.get("X-Limbo-Token") != outer.token:
                    self.send_response(403)
                    self.end_headers()
                    return
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/tools/list":
                    payload = {"tools": outer.session.list_tools()}
                elif self.path == "/tools/call":
                    payload = {"observation": outer.session.call(body.get("name", ""), body.get("arguments") or {})}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(payload, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "SandboxServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
