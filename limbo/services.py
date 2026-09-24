"""Simulated services and their tool specifications.

Each tool has two faces:
- an agent-facing description + JSON schema (all an LLM agent ever sees), and
- a machine-readable ``Contract`` (MCP-style hints, identity fields, a read-back
  verifier, the documented visibility SLA, compensation). Only harness-level
  recovery policies may read the contract.

Semantics follow real API conventions: Stripe-style idempotency keys (same key
+ same parameters replays the original, different parameters is a 409),
eventually consistent list/search endpoints with documented lag, and naturally
idempotent operations that reject repeats (refunding a refunded charge).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .world import Entity, ToolError, World

READ_LATENCY = 0.4
WRITE_LATENCY = 0.9

SOCIAL_PLATFORMS = {
    # platform: (supports idempotency key, list endpoint available, list visibility lag seconds)
    "mastodon": (True, True, 0.0),
    "weibo": (False, True, 180.0),
    "linkedin": (False, True, 0.0),
    "x": (False, False, 0.0),
}
TICKET_SEARCH_LAG = 120.0
MAIL_SENT_LAG = 120.0


def norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def similar(a: Any, b: Any, threshold: float = 0.85) -> bool:
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return a == b
    if a == b or a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


_TERM = re.compile(r"[a-z0-9]+(?:[.@\-_][a-z0-9]+)*")


def search_match(query: Any, *fields: Any) -> bool:
    """Search-engine semantics: every query term must occur somewhere in the record's
    fields (case-insensitive, AND across terms, punctuation-insensitive)."""
    terms = _TERM.findall(norm_text(query))
    if not terms:
        return True
    hay = " ".join(norm_text(" ".join(f) if isinstance(f, list) else f) for f in fields)
    hay_terms = set(_TERM.findall(hay))
    return all(t in hay_terms or t in hay for t in terms)


@dataclass
class VerifyResult:
    status: str  # "found" | "absent" | "unverifiable"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Contract:
    write: bool
    idempotent: bool = False
    destructive: bool = False
    supports_key: Callable[[dict[str, Any]], bool] = lambda args: False
    identity: Callable[[dict[str, Any]], tuple] | None = None
    verifier: Callable[[World, dict[str, Any], float, bool], VerifyResult] | None = None
    visibility_sla: Callable[[dict[str, Any]], float] = lambda args: 0.0
    compensator: str | None = None
    batch_field: str | None = None
    drift: tuple[str, str] | None = None
    irreversible: bool = False


@dataclass
class ToolSpec:
    name: str
    domain: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[World, dict[str, Any]], dict[str, Any]]
    contract: Contract
    latency: float = READ_LATENCY
    explicit_note: str = ""
    consistency_note: str = ""

    def schema(self, doc_variant: str = "neutral") -> dict[str, Any]:
        """doc_variant: neutral (default docs), explicit (+ non-idempotency warning),
        no_consistency_docs (read-path consistency/lag statements removed)."""
        desc = self.description
        if doc_variant != "no_consistency_docs" and self.consistency_note:
            desc = f"{desc} {self.consistency_note}"
        if doc_variant == "explicit" and self.explicit_note:
            desc = f"{desc} {self.explicit_note}"
        return {"name": self.name, "description": desc, "parameters": self.parameters}


class PartialStop(Exception):
    """Raised by batch handlers when the fault injector cuts a batch short."""


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": True}


def _require(args: dict[str, Any], *names: str) -> None:
    missing = [n for n in names if args.get(n) in (None, "", [])]
    if missing:
        raise ToolError(400, f"Bad Request: missing required field(s): {', '.join(missing)}")


def _after(e: Entity, t_call: float) -> bool:
    return e.created_at >= t_call - 1e-6


# ============================================================== social
def _social_publish(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "platform", "text")
    platform = str(a["platform"]).lower()
    if platform not in SOCIAL_PLATFORMS:
        raise ToolError(400, f"Bad Request: unknown platform '{platform}'. Supported: {', '.join(SOCIAL_PLATFORMS)}")
    text = str(a["text"])
    if len(text) > 1000:
        raise ToolError(400, "Bad Request: text exceeds 1000 characters")
    supports_key, _, lag = SOCIAL_PLATFORMS[platform]
    supports_key = supports_key or w.keys_everywhere
    key = a.get("idempotency_key")
    if key and supports_key:
        for e in w.find("post", live_only=False, platform=platform, idem_key=str(key)):
            if norm_text(e.fields["text"]) != norm_text(text):
                raise ToolError(409, "Conflict: idempotency_key was already used with different parameters")
            return {"ok": True, "result": _post_view(e), "idempotent_replay": True}
    ent = w.add_entity(
        "post",
        {"platform": platform, "text": text, "idem_key": str(key) if (key and supports_key) else None},
        prefix={"mastodon": "md-", "weibo": "wb-", "linkedin": "li-", "x": "x-"}[platform],
        visibility_delay=lag,
    )
    w.record("social_publish", "social", "create", ent.rid, {"platform": platform, "text": text})
    return {"ok": True, "result": _post_view(ent)}


def _post_view(e: Entity) -> dict[str, Any]:
    return {
        "post_id": e.rid,
        "platform": e.fields["platform"],
        "text": e.fields["text"],
        "created_at": e.created_at,
        "url": f"https://{e.fields['platform']}.example/p/{e.rid}",
    }


def _social_list(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "platform")
    platform = str(a["platform"]).lower()
    if platform not in SOCIAL_PLATFORMS:
        raise ToolError(400, f"Bad Request: unknown platform '{platform}'")
    if not SOCIAL_PLATFORMS[platform][1]:
        raise ToolError(400, f"Bad Request: listing posts is not supported for platform '{platform}' by this API")
    limit = max(1, min(int(a.get("limit") or 10), 50))
    posts = w.find("post", visible_only=True, platform=platform)
    posts = sorted(posts, key=lambda e: (-e.created_at, e.rid))[:limit]
    return {"ok": True, "result": {"posts": [_post_view(e) for e in posts]}}


def _social_delete(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "platform", "post_id")
    e = w.get(str(a["post_id"]), "post")
    if e is None or not e.live or e.fields["platform"] != str(a["platform"]).lower():
        raise ToolError(404, "Not Found: no such post")
    e.live = False
    w.record("social_delete_post", "social", "delete", e.rid, {"platform": e.fields["platform"]}, compensates=e.rid)
    return {"ok": True, "result": {"deleted": e.rid}}


def _verify_publish(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    platform = str(a.get("platform", "")).lower()
    if platform not in SOCIAL_PLATFORMS:
        return VerifyResult("absent")
    if not truth and not SOCIAL_PLATFORMS[platform][1]:
        return VerifyResult("unverifiable", {"reason": "platform has no list endpoint"})
    for e in w.find("post", visible_only=not truth, platform=platform):
        if _after(e, t_call) and similar(e.fields["text"], a.get("text")):
            return VerifyResult("found", _post_view(e))
    return VerifyResult("absent")


# ============================================================== billing
def _charge_view(e: Entity) -> dict[str, Any]:
    f = e.fields
    return {
        "charge_id": e.rid,
        "customer_id": f["customer_id"],
        "amount_cents": f["amount_cents"],
        "currency": f["currency"],
        "description": f["description"],
        "status": "refunded" if f.get("refunded") else "succeeded",
        "created_at": e.created_at,
    }


def _create_charge(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "customer_id", "amount_cents")
    cust = str(a["customer_id"])
    if not w.find("customer", cid=cust):
        raise ToolError(404, f"Not Found: customer '{cust}' does not exist")
    try:
        amount = int(a["amount_cents"])
    except (TypeError, ValueError):
        raise ToolError(400, "Bad Request: amount_cents must be an integer")
    if amount <= 0:
        raise ToolError(400, "Bad Request: amount_cents must be positive")
    currency = str(a.get("currency") or "usd").lower()
    desc = str(a.get("description") or "")
    key = a.get("idempotency_key")
    if key:
        for e in w.find("charge", live_only=False, idem_key=str(key)):
            f = e.fields
            if (f["customer_id"], f["amount_cents"], f["currency"]) != (cust, amount, currency):
                raise ToolError(409, "Conflict: idempotency_key was already used with different parameters")
            return {"ok": True, "result": _charge_view(e), "idempotent_replay": True}
    ent = w.add_entity(
        "charge",
        {"customer_id": cust, "amount_cents": amount, "currency": currency, "description": desc,
         "idem_key": str(key) if key else None, "refunded": False},
        prefix="ch_",
    )
    w.record("billing_create_charge", "billing", "create", ent.rid,
             {"customer_id": cust, "amount_cents": amount, "description": desc})
    return {"ok": True, "result": _charge_view(ent)}


def _list_charges(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "customer_id")
    limit = max(1, min(int(a.get("limit") or 10), 50))
    rows = sorted(w.find("charge", live_only=False, customer_id=str(a["customer_id"])),
                  key=lambda e: (-e.created_at, e.rid))[:limit]
    return {"ok": True, "result": {"charges": [_charge_view(e) for e in rows]}}


def _refund(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "charge_id")
    e = w.get(str(a["charge_id"]), "charge")
    if e is None:
        raise ToolError(404, "Not Found: no such charge")
    if e.fields.get("refunded"):
        raise ToolError(409, f"Conflict: charge {e.rid} has already been refunded")
    e.fields["refunded"] = True
    w.record("billing_refund_charge", "billing", "refund", e.rid,
             {"customer_id": e.fields["customer_id"], "amount_cents": e.fields["amount_cents"]}, compensates=e.rid)
    return {"ok": True, "result": {"refund_id": w.new_id("re_"), "charge_id": e.rid, "status": "refunded"}}


def _verify_charge(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    try:
        amount = int(a.get("amount_cents"))
    except (TypeError, ValueError):
        return VerifyResult("absent")
    for e in w.find("charge", live_only=False, customer_id=str(a.get("customer_id"))):
        if _after(e, t_call) and e.fields["amount_cents"] == amount:
            return VerifyResult("found", _charge_view(e))
    return VerifyResult("absent")


def _verify_refund(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    e = w.get(str(a.get("charge_id")), "charge")
    if e is not None and e.fields.get("refunded"):
        return VerifyResult("found", _charge_view(e))
    return VerifyResult("absent")


# ============================================================== tickets
def _ticket_view(e: Entity, with_comments: bool = False) -> dict[str, Any]:
    f = e.fields
    out = {"ticket_key": e.rid, "project": f["project"], "title": f["title"], "status": f["status"],
           "created_at": e.created_at}
    if with_comments:
        out["body"] = f["body"]
        out["comments"] = [{"comment_id": c["id"], "text": c["text"], "created_at": c["t"]} for c in f["comments"]]
    return out


def _ticket_create(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "project", "title")
    project = str(a["project"]).upper()
    if project not in w.find_projects():
        raise ToolError(404, f"Not Found: project '{project}' does not exist")
    n = 1 + sum(1 for e in w.entities.values() if e.kind == "ticket" and e.fields["project"] == project)
    rid = f"{project}-{100 + n}"
    ent = w.add_entity(
        "ticket",
        {"project": project, "title": str(a["title"]), "body": str(a.get("body") or ""), "status": "open",
         "comments": []},
        prefix="", rid=rid, visibility_delay=TICKET_SEARCH_LAG,
    )
    w.record("tickets_create", "tickets", "create", rid, {"project": project, "title": str(a["title"]),
                                                            "body": str(a.get("body") or "")})
    return {"ok": True, "result": _ticket_view(ent)}


def _ticket_search(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "project", "query")
    project = str(a["project"]).upper()
    rows = [e for e in w.find("ticket", visible_only=True, project=project)
            if search_match(a["query"], e.fields["title"], e.fields["body"], e.rid)]
    return {"ok": True, "result": {"tickets": [_ticket_view(e) for e in rows[-20:]]}}


def _ticket_list_recent(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "project")
    limit = max(1, min(int(a.get("limit") or 10), 50))
    rows = sorted(w.find("ticket", project=str(a["project"]).upper()), key=lambda e: (-e.created_at, e.rid))[:limit]
    return {"ok": True, "result": {"tickets": [_ticket_view(e) for e in rows]}}


def _ticket_get(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "ticket_key")
    e = w.get(str(a["ticket_key"]).upper(), "ticket")
    if e is None:
        raise ToolError(404, "Not Found: no such ticket")
    return {"ok": True, "result": _ticket_view(e, with_comments=True)}


def _ticket_comment(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "ticket_key", "text")
    e = w.get(str(a["ticket_key"]).upper(), "ticket")
    if e is None:
        raise ToolError(404, "Not Found: no such ticket")
    cid = w.new_id("cm-")
    e.fields["comments"].append({"id": cid, "text": str(a["text"]), "t": w.clock, "live": True})
    w.record("tickets_add_comment", "tickets", "create", cid, {"ticket_key": e.rid, "text": str(a["text"])})
    return {"ok": True, "result": {"comment_id": cid, "ticket_key": e.rid}}


def _ticket_update_status(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "ticket_key", "status")
    e = w.get(str(a["ticket_key"]).upper(), "ticket")
    if e is None:
        raise ToolError(404, "Not Found: no such ticket")
    new = str(a["status"]).lower()
    if new not in ("open", "in_progress", "resolved", "closed"):
        raise ToolError(400, "Bad Request: status must be one of open, in_progress, resolved, closed")
    exp = a.get("expected_status")
    if exp and str(exp).lower() != e.fields["status"]:
        raise ToolError(409, f"Conflict: ticket status is '{e.fields['status']}', expected '{exp}'")
    old = e.fields["status"]
    e.fields["status"] = new
    if new == "closed" and a.get("resolution"):
        e.fields["resolution"] = str(a["resolution"])
    if old != new:
        comp = e.rid if (new == "closed" and str(a.get("resolution", "")).lower() == "duplicate") else None
        w.record("tickets_update_status", "tickets", "update", e.rid, {"from": old, "to": new}, compensates=comp)
        if comp:
            e.live = False
    return {"ok": True, "result": _ticket_view(e)}


def _verify_ticket(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    project = str(a.get("project", "")).upper()
    for e in w.find("ticket", visible_only=False, project=project):
        # list_recent is strongly consistent, so the read path sees everything.
        if _after(e, t_call) and similar(e.fields["title"], a.get("title")):
            return VerifyResult("found", _ticket_view(e))
    return VerifyResult("absent")


def _verify_comment(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    e = w.get(str(a.get("ticket_key", "")).upper(), "ticket")
    if e is None:
        return VerifyResult("absent")
    for c in e.fields["comments"]:
        if c["t"] >= t_call - 1e-6 and similar(c["text"], a.get("text")):
            return VerifyResult("found", {"comment_id": c["id"]})
    return VerifyResult("absent")


# ============================================================== mail
def _as_list(v: Any) -> list[str]:
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[;,]", v) if x.strip()]
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


def _mail_view(e: Entity) -> dict[str, Any]:
    f = e.fields
    return {"message_id": e.rid, "to": f["to"], "subject": f["subject"], "sent_at": e.created_at}


def _mail_send(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "to", "subject", "body")
    to = _as_list(a["to"])
    if not to or any("@" not in x for x in to):
        raise ToolError(400, "Bad Request: 'to' must contain valid email addresses")
    ent = w.add_entity("mail", {"to": to, "subject": str(a["subject"]), "body": str(a["body"])},
                       prefix="msg-", visibility_delay=MAIL_SENT_LAG)
    w.record("mail_send", "mail", "send", ent.rid, {"to": to, "subject": str(a["subject"]), "body": str(a["body"])})
    return {"ok": True, "result": {"message_id": ent.rid, "status": "sent"}}


def _mail_search(w: World, a: dict[str, Any]) -> dict[str, Any]:
    rows = [e for e in w.find("mail", visible_only=True)
            if search_match(a.get("query"), e.fields["subject"], e.fields["to"], e.fields["body"])]
    limit = max(1, min(int(a.get("limit") or 10), 50))
    rows = sorted(rows, key=lambda e: (-e.created_at, e.rid))[:limit]
    return {"ok": True, "result": {"messages": [_mail_view(e) for e in rows]}}


def _verify_mail(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    to = set(x.lower() for x in _as_list(a.get("to")))
    for e in w.find("mail", visible_only=not truth):
        if _after(e, t_call) and to & set(x.lower() for x in e.fields["to"]) and similar(e.fields["subject"], a.get("subject")):
            return VerifyResult("found", _mail_view(e))
    return VerifyResult("absent")


# ============================================================== data
def _row_view(e: Entity) -> dict[str, Any]:
    return {"row_id": e.rid, "table": e.fields["table"], **e.fields["row"]}


def _check_table(w: World, table: Any) -> str:
    t = str(table or "")
    if t not in w.find_tables():
        raise ToolError(404, f"Not Found: table '{t}' does not exist")
    return t


def _clean_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or not row:
        raise ToolError(400, "Bad Request: row must be a non-empty object")
    return {str(k): v for k, v in row.items()}


def _db_insert_one(w: World, table: str, row: dict[str, Any], tool: str) -> Entity:
    ent = w.add_entity("row", {"table": table, "row": row}, prefix="r")
    w.record(tool, "data", "create", ent.rid, {"table": table, **row})
    return ent


def _db_insert(w: World, a: dict[str, Any]) -> dict[str, Any]:
    table = _check_table(w, a.get("table"))
    ent = _db_insert_one(w, table, _clean_row(a.get("row")), "db_insert")
    return {"ok": True, "result": _row_view(ent)}


def _db_insert_many(w: World, a: dict[str, Any]) -> dict[str, Any]:
    table = _check_table(w, a.get("table"))
    rows = a.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ToolError(400, "Bad Request: rows must be a non-empty array")
    clean = [_clean_row(r) for r in rows]
    out = []
    for i, r in enumerate(clean):
        if i < w.batch_skip:
            out.append({"row": r, "status": "already_inserted"})
            continue
        if w.partial_limit is not None and i >= w.partial_limit:
            raise PartialStop()
        out.append(_row_view(_db_insert_one(w, table, r, "db_insert_many")))
    return {"ok": True, "result": {"inserted": out}}


def _db_upsert(w: World, a: dict[str, Any]) -> dict[str, Any]:
    table = _check_table(w, a.get("table"))
    _require(a, "key_field")
    row = _clean_row(a.get("row"))
    kf = str(a["key_field"])
    if kf not in row:
        raise ToolError(400, f"Bad Request: row does not contain key_field '{kf}'")
    for e in w.find("row", table=table):
        if e.fields["row"].get(kf) == row[kf]:
            if e.fields["row"] != row:
                e.fields["row"] = dict(row)
                w.record("db_upsert", "data", "update", e.rid, {"table": table, **row})
            return {"ok": True, "result": {**_row_view(e), "upserted": "updated"}}
    ent = _db_insert_one(w, table, row, "db_upsert")
    return {"ok": True, "result": {**_row_view(ent), "upserted": "inserted"}}


def _db_query(w: World, a: dict[str, Any]) -> dict[str, Any]:
    table = _check_table(w, a.get("table"))
    where = a.get("where") or {}
    if not isinstance(where, dict):
        raise ToolError(400, "Bad Request: where must be an object")
    rows = [e for e in w.find("row", table=table)
            if all(norm_text(e.fields["row"].get(k)) == norm_text(v) for k, v in where.items())]
    return {"ok": True, "result": {"rows": [_row_view(e) for e in rows[:100]], "count": len(rows)}}


def _db_delete(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "row_id")
    e = w.get(str(a["row_id"]), "row")
    if e is None or not e.live:
        raise ToolError(404, "Not Found: no such row")
    e.live = False
    w.record("db_delete", "data", "delete", e.rid, {"table": e.fields["table"]}, compensates=e.rid)
    return {"ok": True, "result": {"deleted": e.rid}}


def _row_matches(e: Entity, row: dict[str, Any]) -> bool:
    return all(norm_text(e.fields["row"].get(k)) == norm_text(v) for k, v in row.items())


def _verify_insert(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    row = a.get("row") if isinstance(a.get("row"), dict) else {}
    for e in w.find("row", table=str(a.get("table"))):
        if _after(e, t_call) and _row_matches(e, row):
            return VerifyResult("found", _row_view(e))
    return VerifyResult("absent")


def _verify_insert_many(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    rows = [r for r in (a.get("rows") or []) if isinstance(r, dict)]
    present, missing = [], []
    cands = [e for e in w.find("row", table=str(a.get("table"))) if _after(e, t_call)]
    for r in rows:
        (present if any(_row_matches(e, r) for e in cands) else missing).append(r)
    if not rows or not present:
        return VerifyResult("absent", {"missing_rows": rows})
    if missing:
        return VerifyResult("partial", {"present_rows": present, "missing_rows": missing})
    return VerifyResult("found", {"present_rows": present})


# ============================================================== deploy
def _run_status(w: World, e: Entity) -> str:
    f = e.fields
    if f.get("cancelled"):
        return "cancelled"
    age = w.clock - e.created_at
    return "queued" if age < 10 else ("running" if age < 60 else "succeeded")


def _run_view(w: World, e: Entity) -> dict[str, Any]:
    f = e.fields
    return {"run_id": e.rid, "service": f["service"], "version": f["version"], "environment": f["environment"],
            "status": _run_status(w, e), "created_at": e.created_at}


def _deploy_trigger(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "service", "version", "environment")
    svc = str(a["service"])
    if svc not in w.find_services():
        raise ToolError(404, f"Not Found: service '{svc}' does not exist")
    env = str(a["environment"]).lower()
    if env not in ("staging", "production"):
        raise ToolError(400, "Bad Request: environment must be 'staging' or 'production'")
    ent = w.add_entity("run", {"service": svc, "version": str(a["version"]), "environment": env, "cancelled": False},
                       prefix="run-")
    w.record("deploy_trigger", "deploy", "create", ent.rid, {"service": svc, "version": str(a["version"]),
                                                               "environment": env})
    return {"ok": True, "result": _run_view(w, ent)}


def _deploy_list(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "service")
    limit = max(1, min(int(a.get("limit") or 10), 50))
    rows = sorted(w.find("run", live_only=False, service=str(a["service"])), key=lambda e: (-e.created_at, e.rid))[:limit]
    return {"ok": True, "result": {"runs": [_run_view(w, e) for e in rows]}}


def _deploy_get(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "run_id")
    e = w.get(str(a["run_id"]), "run")
    if e is None:
        raise ToolError(404, "Not Found: no such run")
    return {"ok": True, "result": _run_view(w, e)}


def _deploy_cancel(w: World, a: dict[str, Any]) -> dict[str, Any]:
    _require(a, "run_id")
    e = w.get(str(a["run_id"]), "run")
    if e is None:
        raise ToolError(404, "Not Found: no such run")
    st = _run_status(w, e)
    if st in ("succeeded", "cancelled"):
        raise ToolError(409, f"Conflict: run {e.rid} is already {st}")
    e.fields["cancelled"] = True
    e.live = False
    w.record("deploy_cancel_run", "deploy", "delete", e.rid, {"service": e.fields["service"]}, compensates=e.rid)
    return {"ok": True, "result": _run_view(w, e)}


def _verify_deploy(w: World, a: dict[str, Any], t_call: float, truth: bool) -> VerifyResult:
    for e in w.find("run", service=str(a.get("service"))):
        f = e.fields
        if _after(e, t_call) and f["version"] == str(a.get("version")) and f["environment"] == str(a.get("environment", "")).lower():
            return VerifyResult("found", _run_view(w, e))
    return VerifyResult("absent")


# ============================================================== general
def _wait(w: World, a: dict[str, Any]) -> dict[str, Any]:
    try:
        s = float(a.get("seconds", 0))
    except (TypeError, ValueError):
        raise ToolError(400, "Bad Request: seconds must be a number")
    s = max(0.0, min(s, 900.0))
    w.advance(s)
    return {"ok": True, "result": {"waited_seconds": s, "now": w.clock}}


# ---------------------------------------------------------------- registry
def _str(desc: str, enum: list[str] | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "string", "description": desc}
    if enum:
        d["enum"] = enum
    return d


_KEY_DESC = "Optional idempotency key. If a request with the same key was already processed, the original result is returned instead of performing the action again."


def build_tools(contract: str = "native") -> dict[str, ToolSpec]:
    T: dict[str, ToolSpec] = {}

    def add(spec: ToolSpec) -> None:
        T[spec.name] = spec

    # social
    add(ToolSpec(
        "social_publish", "social",
        "Publish a post to a social platform (mastodon, weibo, linkedin, x). Returns the new post_id. "
        "idempotency_key is honored by mastodon only; other platforms ignore it.",
        _obj({"platform": _str("Target platform", list(SOCIAL_PLATFORMS)), "text": _str("Post text"),
              "idempotency_key": _str(_KEY_DESC + " Supported on mastodon only.")}, ["platform", "text"]),
        _social_publish,
        Contract(write=True, supports_key=lambda a: bool(SOCIAL_PLATFORMS.get(str(a.get("platform", "")).lower(), (False,))[0]),
                 identity=lambda a: ("social_publish", str(a.get("platform", "")).lower(), norm_text(a.get("text"))[:80]),
                 verifier=_verify_publish,
                 visibility_sla=lambda a: SOCIAL_PLATFORMS.get(str(a.get("platform", "")).lower(), (0, 0, 0.0))[2],
                 compensator="social_delete_post", drift=("text", "content")),
        latency=WRITE_LATENCY,
        explicit_note="Not idempotent without an honored idempotency_key: every successful call creates a new post.",
        consistency_note="On weibo, newly published posts can take up to 3 minutes to appear in social_list_posts.",
    ))
    add(ToolSpec(
        "social_list_posts", "social",
        "List the most recent posts on a platform (newest first). Not available for x.",
        _obj({"platform": _str("Platform", list(SOCIAL_PLATFORMS)), "limit": {"type": "integer", "description": "Max posts (default 10)"}}, ["platform"]),
        _social_list, Contract(write=False, idempotent=True),
        consistency_note="Weibo listings are eventually consistent (new posts may take up to 3 minutes to appear).",
    ))
    add(ToolSpec(
        "social_delete_post", "social", "Delete a post by id.",
        _obj({"platform": _str("Platform", list(SOCIAL_PLATFORMS)), "post_id": _str("Post id")}, ["platform", "post_id"]),
        _social_delete, Contract(write=True, idempotent=True, destructive=True), latency=WRITE_LATENCY,
    ))
    # billing
    add(ToolSpec(
        "billing_create_charge", "billing",
        "Charge a customer's saved payment method. amount_cents is an integer number of cents. Returns the charge.",
        _obj({"customer_id": _str("Customer id, e.g. cus_123"), "amount_cents": {"type": "integer", "description": "Amount in cents"},
              "currency": _str("ISO currency code, default usd"), "description": _str("Statement description"),
              "idempotency_key": _str(_KEY_DESC)}, ["customer_id", "amount_cents", "description"]),
        _create_charge,
        Contract(write=True, supports_key=lambda a: True,
                 identity=lambda a: ("billing_create_charge", str(a.get("customer_id")), str(a.get("amount_cents"))),
                 verifier=_verify_charge, compensator="billing_refund_charge", drift=("amount_cents", "amount")),
        latency=WRITE_LATENCY,
        explicit_note="Not idempotent without idempotency_key: every successful call creates a new charge.",
    ))
    add(ToolSpec(
        "billing_list_charges", "billing", "List a customer's charges (newest first), including refunded ones.",
        _obj({"customer_id": _str("Customer id"), "limit": {"type": "integer"}}, ["customer_id"]),
        _list_charges, Contract(write=False, idempotent=True),
    ))
    add(ToolSpec(
        "billing_refund_charge", "billing", "Fully refund a charge. Refunding an already-refunded charge returns 409.",
        _obj({"charge_id": _str("Charge id"), "reason": _str("Reason")}, ["charge_id"]),
        _refund,
        Contract(write=True, idempotent=True, destructive=True,
                 identity=lambda a: ("billing_refund_charge", str(a.get("charge_id"))), verifier=_verify_refund),
        latency=WRITE_LATENCY,
    ))
    # tickets
    add(ToolSpec(
        "tickets_create", "tickets", "Create a ticket in a project. Returns the ticket_key (e.g. OPS-123).",
        _obj({"project": _str("Project key, e.g. OPS"), "title": _str("Title"), "body": _str("Description")}, ["project", "title"]),
        _ticket_create,
        Contract(write=True, identity=lambda a: ("tickets_create", str(a.get("project", "")).upper(), norm_text(a.get("title"))[:80]),
                 verifier=_verify_ticket, compensator="tickets_update_status", drift=("title", "summary")),
        latency=WRITE_LATENCY,
        explicit_note="Not idempotent: every successful call creates a new ticket.",
    ))
    add(ToolSpec(
        "tickets_search", "tickets",
        "Full-text search over ticket titles and descriptions.",
        _obj({"project": _str("Project key"), "query": _str("Search text")}, ["project", "query"]),
        _ticket_search, Contract(write=False, idempotent=True),
        consistency_note="The search index is updated asynchronously and can lag up to 2 minutes behind writes.",
    ))
    add(ToolSpec(
        "tickets_list_recent", "tickets", "List the most recently created tickets in a project (newest first).",
        _obj({"project": _str("Project key"), "limit": {"type": "integer"}}, ["project"]),
        _ticket_list_recent, Contract(write=False, idempotent=True), consistency_note="Strongly consistent.",
    ))
    add(ToolSpec(
        "tickets_get", "tickets", "Get a ticket, including its description and comments.",
        _obj({"ticket_key": _str("Ticket key")}, ["ticket_key"]), _ticket_get, Contract(write=False, idempotent=True),
    ))
    add(ToolSpec(
        "tickets_add_comment", "tickets", "Add a comment to a ticket.",
        _obj({"ticket_key": _str("Ticket key"), "text": _str("Comment text")}, ["ticket_key", "text"]),
        _ticket_comment,
        Contract(write=True, identity=lambda a: ("tickets_add_comment", str(a.get("ticket_key", "")).upper(), norm_text(a.get("text"))[:80]),
                 verifier=_verify_comment),
        latency=WRITE_LATENCY,
        explicit_note="Not idempotent: every successful call adds another comment.",
    ))
    add(ToolSpec(
        "tickets_update_status", "tickets",
        "Change a ticket's status (open, in_progress, resolved, closed). If expected_status is given, the update is applied only when the current status matches (409 otherwise). "
        "Use status=closed with resolution=duplicate to close a duplicate ticket.",
        _obj({"ticket_key": _str("Ticket key"), "status": _str("New status", ["open", "in_progress", "resolved", "closed"]),
              "expected_status": _str("Optional precondition on the current status"), "resolution": _str("Optional resolution, e.g. duplicate")},
             ["ticket_key", "status"]),
        _ticket_update_status, Contract(write=True, idempotent=True), latency=WRITE_LATENCY,
    ))
    # mail
    add(ToolSpec(
        "mail_send", "mail", "Send an email. Sent messages cannot be recalled. Returns the message_id.",
        _obj({"to": {"type": "array", "items": {"type": "string"}, "description": "Recipient addresses"},
              "subject": _str("Subject"), "body": _str("Plain-text body")}, ["to", "subject", "body"]),
        _mail_send,
        Contract(write=True, irreversible=True,
                 identity=lambda a: ("mail_send", ",".join(sorted(x.lower() for x in _as_list(a.get("to")))), norm_text(a.get("subject"))[:80]),
                 verifier=_verify_mail, visibility_sla=lambda a: MAIL_SENT_LAG, drift=("body", "text_body")),
        latency=WRITE_LATENCY,
        explicit_note="Not idempotent: every successful call sends another email.",
    ))
    add(ToolSpec(
        "mail_search_sent", "mail",
        "Search the Sent folder by subject, recipient or body text (newest first).",
        _obj({"query": _str("Search text"), "limit": {"type": "integer"}}, []),
        _mail_search, Contract(write=False, idempotent=True),
        consistency_note="The Sent folder syncs asynchronously; a sent message can take up to 2 minutes to appear.",
    ))
    # data
    add(ToolSpec(
        "db_insert", "data", "Insert one row into a table. Each call inserts a new row with a new row_id.",
        _obj({"table": _str("Table name"), "row": {"type": "object", "description": "Column values"}}, ["table", "row"]),
        _db_insert,
        Contract(write=True, identity=lambda a: ("db_insert", str(a.get("table")), tuple(sorted((str(k), norm_text(v)) for k, v in (a.get("row") or {}).items())) if isinstance(a.get("row"), dict) else ()),
                 verifier=_verify_insert, compensator="db_delete"),
        latency=WRITE_LATENCY,
        explicit_note="Not idempotent: repeating the call inserts duplicate rows.",
    ))
    add(ToolSpec(
        "db_insert_many", "data",
        "Insert several rows into a table in order. Rows are inserted one at a time; if the request fails midway, rows inserted before the failure remain.",
        _obj({"table": _str("Table name"), "rows": {"type": "array", "items": {"type": "object"}, "description": "Rows to insert"}}, ["table", "rows"]),
        _db_insert_many,
        Contract(write=True, identity=lambda a: ("db_insert_many", str(a.get("table")), len(a.get("rows") or [])),
                 verifier=_verify_insert_many, compensator="db_delete", batch_field="rows"),
        latency=WRITE_LATENCY,
        explicit_note="Not idempotent: repeating the call inserts duplicate rows.",
    ))
    add(ToolSpec(
        "db_upsert", "data", "Insert a row, or update the existing row whose key_field value matches.",
        _obj({"table": _str("Table name"), "key_field": _str("Column used as the unique key"), "row": {"type": "object"}}, ["table", "key_field", "row"]),
        _db_upsert, Contract(write=True, idempotent=True), latency=WRITE_LATENCY,
    ))
    add(ToolSpec(
        "db_query", "data", "Query rows of a table by exact column values.",
        _obj({"table": _str("Table name"), "where": {"type": "object", "description": "Column filters"}}, ["table"]),
        _db_query, Contract(write=False, idempotent=True), consistency_note="Strongly consistent.",
    ))
    add(ToolSpec(
        "db_delete", "data", "Delete a row by row_id.",
        _obj({"row_id": _str("Row id")}, ["row_id"]), _db_delete,
        Contract(write=True, idempotent=True, destructive=True), latency=WRITE_LATENCY,
    ))
    # deploy
    add(ToolSpec(
        "deploy_trigger", "deploy",
        "Start a deployment run of a service version to an environment (staging or production). Runs go queued -> running -> succeeded (about 1 minute).",
        _obj({"service": _str("Service name"), "version": _str("Version to deploy"), "environment": _str("staging or production", ["staging", "production"])},
             ["service", "version", "environment"]),
        _deploy_trigger,
        Contract(write=True, identity=lambda a: ("deploy_trigger", str(a.get("service")), str(a.get("version")), str(a.get("environment", "")).lower()),
                 verifier=_verify_deploy, compensator="deploy_cancel_run"),
        latency=1.5,
        explicit_note="Not idempotent: every successful call starts another deployment run.",
    ))
    add(ToolSpec(
        "deploy_list_runs", "deploy", "List recent deployment runs of a service (newest first).",
        _obj({"service": _str("Service name"), "limit": {"type": "integer"}}, ["service"]),
        _deploy_list, Contract(write=False, idempotent=True), consistency_note="Strongly consistent.",
    ))
    add(ToolSpec(
        "deploy_get_run", "deploy", "Get the status of a deployment run.",
        _obj({"run_id": _str("Run id")}, ["run_id"]), _deploy_get, Contract(write=False, idempotent=True),
    ))
    add(ToolSpec(
        "deploy_cancel_run", "deploy", "Cancel a queued or running deployment run.",
        _obj({"run_id": _str("Run id")}, ["run_id"]), _deploy_cancel,
        Contract(write=True, idempotent=True, destructive=True), latency=WRITE_LATENCY,
    ))
    # general
    add(ToolSpec(
        "wait", "general", "Wait for a number of seconds (max 900) before continuing.",
        _obj({"seconds": {"type": "number", "description": "Seconds to wait"}}, ["seconds"]),
        _wait, Contract(write=False, idempotent=True), latency=0.0,
    ))
    if contract == "keys_everywhere":
        _upgrade_contracts(T)
    return T


# Non-idempotent writes whose key semantics are added by the runtime's generic key layer.
GENERIC_KEY_TOOLS = ("tickets_create", "tickets_add_comment", "mail_send", "db_insert", "db_insert_many", "deploy_trigger")
_KEY_SENTENCE = ("Supports idempotency_key: repeating a request with the same key returns the original result "
                 "instead of acting again (an interrupted batch resumes where it stopped).")


def _upgrade_contracts(T: dict[str, ToolSpec]) -> None:
    """Counterfactual contract: every non-idempotent write honors idempotency keys."""
    pub = T["social_publish"]
    pub.description = pub.description.replace("idempotency_key is honored by mastodon only; other platforms ignore it.",
                                              "idempotency_key is honored on every platform.")
    pub.parameters["properties"]["idempotency_key"] = _str(_KEY_DESC)
    pub.contract.supports_key = lambda a: True
    for name in GENERIC_KEY_TOOLS:
        spec = T[name]
        spec.description = f"{spec.description} {_KEY_SENTENCE}"
        spec.parameters["properties"]["idempotency_key"] = _str(_KEY_DESC)
        spec.contract.supports_key = lambda a: True


GENERAL_TOOLS = ("wait", "escalate_to_human", "finish")
DOMAIN_TOOLS: dict[str, list[str]] = {}
for _name, _spec in build_tools().items():
    DOMAIN_TOOLS.setdefault(_spec.domain, []).append(_name)
