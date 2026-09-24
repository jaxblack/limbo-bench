"""Minimal stdio MCP server that forwards tools/list and tools/call to the
localhost sandbox (``limbo.sandbox_http``). Stdlib only, so any harness can
launch it with ``python -m limbo.mcp_proxy``.

Connection details come from the environment the harness passes to the MCP
process: LIMBO_PORT and LIMBO_TOKEN.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{os.environ['LIMBO_PORT']}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Limbo-Token": os.environ["LIMBO_TOKEN"]},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _reply(msg_id, result=None, error=None) -> None:
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, msg_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if msg_id is None:
            continue
        try:
            if method == "initialize":
                req = params.get("protocolVersion")
                _reply(msg_id, {"protocolVersion": req if req in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                                "capabilities": {"tools": {"listChanged": False}},
                                "serverInfo": {"name": "acme-ops", "version": "1.0.0"}})
            elif method == "ping":
                _reply(msg_id, {})
            elif method == "tools/list":
                _reply(msg_id, _post("/tools/list", {}))
            elif method == "tools/call":
                obs = _post("/tools/call", {"name": params.get("name", ""), "arguments": params.get("arguments") or {}})["observation"]
                _reply(msg_id, {"content": [{"type": "text", "text": json.dumps(obs, sort_keys=True, default=str)}],
                                "isError": not obs.get("ok", False)})
            elif method in ("resources/list", "prompts/list", "resources/templates/list"):
                key = {"resources/list": "resources", "prompts/list": "prompts",
                       "resources/templates/list": "resourceTemplates"}[method]
                _reply(msg_id, {key: []})
            else:
                _reply(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})
        except Exception as exc:  # report, never crash the harness session
            _reply(msg_id, error={"code": -32603, "message": f"sandbox unavailable: {exc}"})


if __name__ == "__main__":
    main()
