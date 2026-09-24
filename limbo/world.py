"""Deterministic world state for LIMBO: virtual clock, entities, and the
ground-truth effect ledger.

Every committed side effect is appended to ``World.ledger`` exactly when the
simulated service commits it, independently of what the caller is told. That
ledger is the only source of truth used by the grader.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """A service-level error returned to the caller as an observation."""

    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra

    def observation(self) -> dict[str, Any]:
        err: dict[str, Any] = {"type": "http_error", "status": self.status, "message": self.message}
        err.update(self.extra)
        return {"ok": False, "error": err}


@dataclass
class Effect:
    eid: int
    tool: str
    domain: str
    op: str
    resource: str
    attrs: dict[str, Any]
    t: float
    call_index: int
    source: str
    compensates: str | None = None  # resource id undone by this effect


@dataclass
class Entity:
    """A stored record. ``visible_at`` models eventually consistent read paths."""

    rid: str
    kind: str
    fields: dict[str, Any]
    created_at: float
    visible_at: float
    live: bool = True
    preexisting: bool = False


@dataclass
class World:
    seed: int
    clock: float = 0.0
    ledger: list[Effect] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    human_available: bool = True
    human_minutes: float = 0.0
    rng: random.Random = field(init=False)
    _next_id: int = field(init=False, default=1001)
    _next_eid: int = field(init=False, default=1)
    call_index: int = 0
    call_source: str = "agent"
    projects: set[str] = field(default_factory=set)
    tables: set[str] = field(default_factory=set)
    deploy_services: set[str] = field(default_factory=set)
    # Set by the fault injector to cut a batch write short (partial commit).
    partial_limit: int | None = None
    # Contract variant: every non-idempotent write honors idempotency keys.
    keys_everywhere: bool = False
    key_store: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    # Rows already committed under the same key by an interrupted batch (resumable batches).
    batch_skip: int = 0
    # In-flight requests: (due time, sequence, commit callback, metadata), run by advance().
    deferred: list[tuple[float, int, Any, dict]] = field(default_factory=list)
    _in_deferred: bool = field(init=False, default=False)
    _seq: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self._next_id = 1001
        self._next_eid = 1

    def find_projects(self) -> set[str]:
        return self.projects

    def find_tables(self) -> set[str]:
        return self.tables

    def find_services(self) -> set[str]:
        return self.deploy_services

    # ---------------------------------------------------------------- clock
    def advance(self, seconds: float) -> None:
        if seconds <= 0:
            return
        target = round(self.clock + seconds, 3)
        self._run_due(target)
        self.clock = target

    def schedule(self, due: float, commit, meta: dict | None = None) -> None:
        self._seq += 1
        self.deferred.append((round(due, 3), self._seq, commit, dict(meta or {})))
        self.deferred.sort(key=lambda x: (x[0], x[1]))

    def pending(self) -> list[dict]:
        """Metadata of requests that are still in flight (visible only to oracles)."""
        return [meta for _, _, _, meta in self.deferred]

    def _run_due(self, until: float) -> None:
        if self._in_deferred:
            return
        self._in_deferred = True
        try:
            while self.deferred and self.deferred[0][0] <= until:
                due, _, commit, _ = self.deferred.pop(0)
                self.clock = max(self.clock, due)
                commit()
        finally:
            self._in_deferred = False

    def settle(self) -> None:
        """Let every in-flight request complete (used when an episode ends)."""
        if self.deferred:
            self.advance(max(0.0, self.deferred[-1][0] - self.clock) + 0.001)

    # ------------------------------------------------------------- entities
    def new_id(self, prefix: str) -> str:
        rid = f"{prefix}{self._next_id}"
        self._next_id += 1
        return rid

    def add_entity(
        self,
        kind: str,
        fields: dict[str, Any],
        *,
        prefix: str,
        visibility_delay: float = 0.0,
        preexisting: bool = False,
        rid: str | None = None,
        created_at: float | None = None,
    ) -> Entity:
        rid = rid or self.new_id(prefix)
        t = self.clock if created_at is None else created_at
        ent = Entity(rid, kind, dict(fields), t, t + visibility_delay, True, preexisting)
        self.entities[rid] = ent
        return ent

    def find(self, kind: str, *, visible_only: bool = False, live_only: bool = True, **match: Any) -> list[Entity]:
        out = []
        for e in self.entities.values():
            if e.kind != kind:
                continue
            if live_only and not e.live:
                continue
            if visible_only and e.visible_at > self.clock:
                continue
            if all(e.fields.get(k) == v for k, v in match.items()):
                out.append(e)
        return sorted(out, key=lambda x: (x.created_at, x.rid))

    def get(self, rid: str, kind: str | None = None) -> Entity | None:
        e = self.entities.get(rid)
        if e is None or (kind and e.kind != kind):
            return None
        return e

    # --------------------------------------------------------------- ledger
    def record(
        self,
        tool: str,
        domain: str,
        op: str,
        resource: str,
        attrs: dict[str, Any],
        *,
        compensates: str | None = None,
    ) -> Effect:
        eff = Effect(
            self._next_eid,
            tool,
            domain,
            op,
            resource,
            dict(attrs),
            self.clock,
            self.call_index,
            self.call_source,
            compensates,
        )
        self.ledger.append(eff)
        self._next_eid += 1
        return eff
