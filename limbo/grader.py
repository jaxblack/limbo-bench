"""Programmatic grading from world state and the effect ledger.

Definitions (per target):
  executed = committed creations that match the target (including ones later
             deleted/refunded/cancelled, i.e. compensated)
  live     = matching records that still stand at the end of the episode
Episode metrics:
  goal_met        every target has live >= expected
  dup_executed    sum of max(0, executed - expected)
  dup_live        sum of max(0, live - expected)
  collateral      destructive effects on records the task did not target
  TS  (task success)          goal_met and no collateral
  EOS (exactly-once success)  TS and dup_executed == 0
"""

from __future__ import annotations

from typing import Any

from .services import norm_text
from .tasks import TaskInstance, Target
from .world import Effect, Entity, World

CREATE_OPS = {"create", "send"}


def _tokens_in(text: Any, tokens: list[str]) -> bool:
    t = norm_text(text)
    return all(norm_text(k) in t for k in tokens)


def _mail_to(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).lower() for x in value]
    return [str(value or "").lower()]


def _effect_matches(t: Target, e: Effect, w: World) -> bool:
    m, a = t.match, e.attrs
    if t.kind == "post":
        return e.tool == "social_publish" and a.get("platform") == m["platform"] and _tokens_in(a.get("text"), m["tokens"])
    if t.kind == "charge":
        return (e.tool == "billing_create_charge" and a.get("customer_id") == m["customer_id"]
                and int(a.get("amount_cents", -1)) == int(m["amount_cents"]))
    if t.kind == "ticket":
        return (e.tool == "tickets_create" and a.get("project") == m["project"]
                and _tokens_in(a.get("title"), m["tokens"]))
    if t.kind == "comment":
        if e.tool != "tickets_add_comment":
            return False
        if m.get("ticket_key") and a.get("ticket_key") != m["ticket_key"]:
            return False
        return _tokens_in(a.get("text"), m["tokens"])
    if t.kind == "mail":
        return (e.tool == "mail_send" and m["to"].lower() in _mail_to(a.get("to"))
                and _tokens_in(f"{a.get('subject', '')} {a.get('body', '')}", m["tokens"]))
    if t.kind == "row":
        return (e.op == "create" and e.domain == "data" and a.get("table") == m["table"]
                and all(norm_text(a.get(k)) == norm_text(v) for k, v in m["fields"].items()))
    if t.kind == "run":
        return (e.tool == "deploy_trigger" and a.get("service") == m["service"] and a.get("version") == m["version"]
                and a.get("environment") == m["environment"])
    if t.kind == "upsert":
        return (e.op == "create" and e.domain == "data" and a.get("table") == m["table"]
                and norm_text(a.get(m["key_field"])) == norm_text(m["key"]))
    if t.kind == "refund":
        return e.tool == "billing_refund_charge" and e.resource == m["charge_id"]
    return False


def _live_count(t: Target, w: World) -> int:
    m = t.match
    ents = [e for e in w.entities.values() if not e.preexisting]
    if t.kind == "post":
        return sum(1 for e in ents if e.kind == "post" and e.live and e.fields["platform"] == m["platform"]
                   and _tokens_in(e.fields["text"], m["tokens"]))
    if t.kind == "charge":
        return sum(1 for e in ents if e.kind == "charge" and not e.fields.get("refunded")
                   and e.fields["customer_id"] == m["customer_id"] and e.fields["amount_cents"] == int(m["amount_cents"]))
    if t.kind == "ticket":
        return sum(1 for e in ents if e.kind == "ticket" and e.live and e.fields["status"] != "closed"
                   and e.fields["project"] == m["project"] and _tokens_in(e.fields["title"], m["tokens"]))
    if t.kind == "comment":
        n = 0
        for e in w.entities.values():
            if e.kind != "ticket" or (m.get("ticket_key") and e.rid != m["ticket_key"]):
                continue
            n += sum(1 for c in e.fields["comments"] if _tokens_in(c["text"], m["tokens"]))
        return n
    if t.kind == "mail":
        return sum(1 for e in ents if e.kind == "mail" and m["to"].lower() in _mail_to(e.fields["to"])
                   and _tokens_in(f"{e.fields['subject']} {e.fields['body']}", m["tokens"]))
    if t.kind == "row":
        return sum(1 for e in ents if e.kind == "row" and e.live and e.fields["table"] == m["table"]
                   and all(norm_text(e.fields["row"].get(k)) == norm_text(v) for k, v in m["fields"].items()))
    if t.kind == "run":
        return sum(1 for e in ents if e.kind == "run" and not e.fields.get("cancelled")
                   and e.fields["service"] == m["service"] and e.fields["version"] == m["version"]
                   and e.fields["environment"] == m["environment"])
    if t.kind == "upsert":
        rows = [e for e in w.entities.values() if e.kind == "row" and e.live and e.fields["table"] == m["table"]
                and norm_text(e.fields["row"].get(m["key_field"])) == norm_text(m["key"])]
        ok = [e for e in rows if all(norm_text(e.fields["row"].get(k)) == norm_text(v) for k, v in m["fields"].items())]
        return len(ok) if len(rows) == len(ok) else 0
    if t.kind == "refund":
        e = w.get(m["charge_id"], "charge")
        return 1 if e is not None and e.fields.get("refunded") else 0
    if t.kind == "status":
        e = w.get(m["ticket_key"], "ticket")
        return 1 if e is not None and e.fields["status"] == m["status"] else 0
    return 0


def _collateral(task: TaskInstance, w: World) -> list[dict[str, Any]]:
    """Destructive or state-changing effects on records the task did not ask to touch."""
    refund_targets = {t.match["charge_id"] for t in task.targets if t.kind == "refund"}
    status_targets = {t.match["ticket_key"] for t in task.targets if t.kind == "status"}
    out = []
    for e in w.ledger:
        ent: Entity | None = w.entities.get(e.resource)
        pre = bool(ent and ent.preexisting)
        if e.tool == "billing_refund_charge" and e.resource not in refund_targets:
            if pre or not _is_target_duplicate(task, e, w):
                out.append({"eid": e.eid, "tool": e.tool, "resource": e.resource, "why": "refund of a non-target charge"})
        elif e.op == "delete" and pre:
            out.append({"eid": e.eid, "tool": e.tool, "resource": e.resource, "why": "deleted a pre-existing record"})
        elif e.tool == "tickets_update_status" and pre and e.resource not in status_targets:
            out.append({"eid": e.eid, "tool": e.tool, "resource": e.resource, "why": "changed a pre-existing ticket"})
    return out


def _is_target_duplicate(task: TaskInstance, refund: Effect, w: World) -> bool:
    """A refund of an in-episode charge that matches a target is compensation, not damage,
    as long as at least the expected number of matching charges survives."""
    ent = w.entities.get(refund.resource)
    if ent is None or ent.kind != "charge":
        return False
    for t in task.targets:
        if t.kind == "charge" and ent.fields["customer_id"] == t.match["customer_id"] \
                and ent.fields["amount_cents"] == int(t.match["amount_cents"]):
            return _live_count(t, w) >= t.expected
    return False


def grade(task: TaskInstance, w: World) -> dict[str, Any]:
    per = []
    matched_eids: set[int] = set()
    for t in task.targets:
        matches = [e for e in w.ledger if (e.op in CREATE_OPS or t.kind == "refund") and _effect_matches(t, e, w)]
        matched_eids.update(e.eid for e in matches)
        executed = len(matches)
        live = _live_count(t, w)
        if t.kind in ("status", "upsert", "refund"):
            executed = min(executed, live) if t.kind != "refund" else executed
        per.append({
            "label": t.label, "kind": t.kind, "expected": t.expected, "executed": executed, "live": live,
            "met": live >= t.expected, "dup_executed": max(0, executed - t.expected),
            "dup_live": max(0, live - t.expected),
        })
    collateral = _collateral(task, w)
    extraneous = [e for e in w.ledger if e.op in CREATE_OPS and e.eid not in matched_eids]
    goal_met = all(p["met"] for p in per)
    dup_exec = sum(p["dup_executed"] for p in per)
    dup_live = sum(p["dup_live"] for p in per)
    ts = goal_met and not collateral
    return {
        "targets": per,
        "goal_met": goal_met,
        "exact_state": all(p["live"] == p["expected"] for p in per),
        "dup_executed": dup_exec,
        "dup_live": dup_live,
        "dup_compensated": max(0, dup_exec - dup_live),
        "collateral": collateral,
        "extraneous_writes": len(extraneous),
        "TS": ts,
        "EOS": ts and dup_exec == 0,
        "clean_success": ts and dup_live == 0,
        "n_effects": len(w.ledger),
    }
