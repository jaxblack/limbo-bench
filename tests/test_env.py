"""Deterministic tests of the LIMBO sandbox, fault injector, grader and policies.

Scripted agents stand in for LLMs so every assertion is exact.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limbo.agent import EpisodeSpec, ScriptedModel, run_episode  # noqa: E402
from limbo.runtime import FaultSpec, ToolRuntime  # noqa: E402
from limbo.services import build_tools  # noqa: E402
from limbo.tasks import TEMPLATES, make_task  # noqa: E402
from limbo.world import World  # noqa: E402


class GenAgent:
    """Adapts a generator of tool-call batches into a ScriptedModel callback."""

    def __init__(self, gen_fn):
        self.gen = gen_fn()
        self.started = False

    def __call__(self, conv):
        last = max(i for i, t in enumerate(conv) if t["role"] == "assistant") if any(t["role"] == "assistant" for t in conv) else -1
        obs = [json.loads(t["content"]) for t in conv[last + 1:] if t["role"] == "tool"]
        try:
            calls = self.gen.send(obs if self.started else None)
            self.started = True
        except StopIteration:
            return [{"name": "finish", "args": {"status": "completed", "summary": "done"}}]
        return calls


def run(spec_kwargs, gen_fn):
    spec = EpisodeSpec(model="scripted", **spec_kwargs)
    return run_episode(spec, ScriptedModel(GenAgent(gen_fn)))


def invoice_task():
    return make_task("invoice_batch", 0)


def naive_invoice_agent(task, careful=False, use_key=False):
    """Charges each target customer; on error retries (blindly unless careful)."""
    charges = [t for t in task.targets if t.kind == "charge"]

    def gen():
        for n, t in enumerate(charges):
            args = {"customer_id": t.match["customer_id"], "amount_cents": t.match["amount_cents"],
                    "currency": "usd", "description": "Invoice"}
            if use_key:
                args["idempotency_key"] = f"k-{n}"
            obs = yield [{"name": "billing_create_charge", "args": args}]
            tries = 0
            while not obs[0]["ok"] and tries < 3:
                tries += 1
                if careful:
                    lst = yield [{"name": "billing_list_charges", "args": {"customer_id": args["customer_id"]}}]
                    if any(c["amount_cents"] == args["amount_cents"] for c in lst[0]["result"]["charges"]):
                        break
                obs = yield [{"name": "billing_create_charge", "args": args}]
        yield [{"name": "mail_send", "args": {"to": ["finance@acme.example"], "subject": f"Invoice {task.instruction.split()[5]} charged",
                                              "body": "done"}}]
        yield [{"name": "finish", "args": {"status": "completed", "summary": "ok"}}]
    return gen


class TestFaultSemantics(unittest.TestCase):
    def _world(self):
        task = invoice_task()
        w = World(seed=1)
        task.setup(w)
        return task, w

    def test_equivalent_pairs_are_observationally_identical(self):
        for pre, post in (("timeout_pre", "timeout_post"), ("http500_pre", "http500_post")):
            outs = []
            for mode in (pre, post):
                task, w = self._world()
                cid = task.targets[0].match["customer_id"]
                rt = ToolRuntime(w, build_tools(), [FaultSpec(mode, "billing_create_charge", {"customer_id": cid})])
                obs = rt.execute("billing_create_charge", {"customer_id": cid, "amount_cents": 100, "description": "x"})
                outs.append((json.dumps(obs, sort_keys=True), w.clock, len(w.ledger)))
            self.assertEqual(outs[0][0], outs[1][0])
            self.assertEqual(outs[0][1], outs[1][1])
            self.assertEqual(outs[0][2], 0)
            self.assertEqual(outs[1][2], 1)

    def test_invalid_request_does_not_consume_fault(self):
        task, w = self._world()
        cid = task.targets[0].match["customer_id"]
        rt = ToolRuntime(w, build_tools(), [FaultSpec("timeout_post", "billing_create_charge", {"customer_id": cid})])
        bad = rt.execute("billing_create_charge", {"customer_id": cid, "amount_cents": -5, "description": "x"})
        self.assertEqual(bad["error"]["status"], 400)
        good = rt.execute("billing_create_charge", {"customer_id": cid, "amount_cents": 100, "description": "x"})
        self.assertEqual(good["error"]["type"], "timeout")
        self.assertEqual(len(w.ledger), 1)

    def test_stale_read_on_weibo(self):
        task = make_task("release_announcement", 0)
        w = World(seed=2)
        task.setup(w)
        rt = ToolRuntime(w, build_tools(), [FaultSpec("timeout_post", "social_publish", {"platform": "weibo"})])
        obs = rt.execute("social_publish", {"platform": "weibo", "text": "hello world launch"})
        self.assertEqual(obs["error"]["type"], "timeout")
        listed = rt.execute("social_list_posts", {"platform": "weibo"})
        self.assertFalse(any("launch" in p["text"] for p in listed["result"]["posts"]))
        rt.execute("wait", {"seconds": 200})
        listed = rt.execute("social_list_posts", {"platform": "weibo"})
        self.assertTrue(any("launch" in p["text"] for p in listed["result"]["posts"]))

    def test_partial_batch(self):
        task = make_task("migration_log", 0)
        w = World(seed=3)
        task.setup(w)
        rt = ToolRuntime(w, build_tools(), [FaultSpec("partial_timeout", "db_insert_many", {"table": "schema_migrations"})])
        rows = [{"version": f"v{k}"} for k in range(4)]
        obs = rt.execute("db_insert_many", {"table": "schema_migrations", "rows": rows})
        self.assertEqual(obs["error"]["type"], "timeout")
        self.assertEqual(len(w.ledger), 2)
        v = rt.verify("db_insert_many", {"table": "schema_migrations", "rows": rows}, 0.0)
        self.assertEqual(v.status, "partial")
        self.assertEqual(len(v.detail["missing_rows"]), 2)

    def test_rate_limit_outage_drift(self):
        task, w = self._world()
        cid = task.targets[0].match["customer_id"]
        args = {"customer_id": cid, "amount_cents": 100, "description": "x"}
        rt = ToolRuntime(w, build_tools(), [FaultSpec("rate_limit", "billing_create_charge", {"customer_id": cid})])
        self.assertEqual(rt.execute("billing_create_charge", args)["error"]["status"], 429)
        self.assertEqual(rt.execute("billing_create_charge", args)["error"]["status"], 429)
        rt.execute("wait", {"seconds": 25})
        self.assertTrue(rt.execute("billing_create_charge", args)["ok"])

        task, w = self._world()
        rt = ToolRuntime(w, build_tools(), [FaultSpec("outage", "billing_create_charge", {"customer_id": cid})])
        for _ in range(4):
            self.assertEqual(rt.execute("billing_create_charge", args)["error"]["status"], 503)
        self.assertEqual(len(w.ledger), 0)

        task, w = self._world()
        rt = ToolRuntime(w, build_tools(), [FaultSpec("schema_drift", "billing_create_charge", {"customer_id": cid})])
        self.assertEqual(rt.execute("billing_create_charge", args)["error"]["status"], 400)
        fixed = {"customer_id": cid, "amount": 100, "description": "x"}
        self.assertTrue(rt.execute("billing_create_charge", fixed)["ok"])
        self.assertEqual(w.ledger[-1].attrs["amount_cents"], 100)

    def test_idempotency_key_semantics(self):
        task, w = self._world()
        cid = task.targets[0].match["customer_id"]
        rt = ToolRuntime(w, build_tools(), [FaultSpec("timeout_post", "billing_create_charge", {"customer_id": cid})])
        args = {"customer_id": cid, "amount_cents": 100, "description": "x", "idempotency_key": "abc"}
        self.assertEqual(rt.execute("billing_create_charge", args)["error"]["type"], "timeout")
        again = rt.execute("billing_create_charge", args)
        self.assertTrue(again.get("idempotent_replay"))
        self.assertEqual(len(w.ledger), 1)
        conflict = rt.execute("billing_create_charge", {**args, "amount_cents": 200})
        self.assertEqual(conflict["error"]["status"], 409)

    def test_late_commit_defeats_verification_but_not_keys(self):
        task, w = self._world()
        cid = task.targets[0].match["customer_id"]
        rt = ToolRuntime(w, build_tools(), [FaultSpec("timeout_late", "billing_create_charge", {"customer_id": cid})])
        args = {"customer_id": cid, "amount_cents": 100, "description": "x"}
        self.assertEqual(rt.execute("billing_create_charge", args)["error"]["type"], "timeout")
        self.assertEqual(len(w.ledger), 0)
        listed = rt.execute("billing_list_charges", {"customer_id": cid})
        self.assertEqual(listed["result"]["charges"], [])  # truly absent at verification time
        self.assertTrue(rt.execute("billing_create_charge", args)["ok"])
        rt.flush()
        self.assertEqual(len(w.ledger), 2)  # the in-flight original lands later: duplicate

        task, w = self._world()
        rt = ToolRuntime(w, build_tools(), [FaultSpec("timeout_late", "billing_create_charge", {"customer_id": cid})])
        keyed = {**args, "idempotency_key": "stable-1"}
        rt.execute("billing_create_charge", keyed)
        self.assertTrue(rt.execute("billing_create_charge", keyed)["ok"])
        rt.flush()
        self.assertEqual(len(w.ledger), 1)  # the late original replays against the key

    def test_duplicate_delivery(self):
        task, w = self._world()
        cid = task.targets[0].match["customer_id"]
        rt = ToolRuntime(w, build_tools(), [FaultSpec("duplicate_delivery", "billing_create_charge", {"customer_id": cid})])
        obs = rt.execute("billing_create_charge", {"customer_id": cid, "amount_cents": 100, "description": "x"})
        self.assertTrue(obs["ok"])
        self.assertEqual(len(w.ledger), 2)
        task, w = self._world()
        rt = ToolRuntime(w, build_tools(), [FaultSpec("duplicate_delivery", "billing_create_charge", {"customer_id": cid})])
        rt.execute("billing_create_charge", {"customer_id": cid, "amount_cents": 100, "description": "x", "idempotency_key": "k"})
        self.assertEqual(len(w.ledger), 1)

    def test_doc_variants(self):
        tools = build_tools()
        self.assertIn("3 minutes", tools["social_list_posts"].schema("neutral")["description"])
        self.assertNotIn("3 minutes", tools["social_list_posts"].schema("no_consistency_docs")["description"])
        self.assertIn("Not idempotent", tools["mail_send"].schema("explicit")["description"])

    def test_search_uses_term_semantics(self):
        task, w = self._world()
        rt = ToolRuntime(w, build_tools(), [])
        rt.execute("mail_send", {"to": ["ava.novak@customer.example"], "subject": "Scheduled maintenance on December 22",
                                 "body": "Hello"})
        rt.execute("wait", {"seconds": 130})
        q = rt.execute("mail_search_sent", {"query": "ava.novak@customer.example Scheduled maintenance on December 22"})
        self.assertEqual(len(q["result"]["messages"]), 1)  # multi-field query matches, as in real search
        self.assertEqual(rt.execute("mail_search_sent", {"query": "maintenance January"})["result"]["messages"], [])

    def test_escalation_reports_truth(self):
        task, w = self._world()
        cid = task.targets[0].match["customer_id"]
        rt = ToolRuntime(w, build_tools(), [FaultSpec("timeout_post", "billing_create_charge", {"customer_id": cid})])
        rt.execute("billing_create_charge", {"customer_id": cid, "amount_cents": 100, "description": "x"})
        rep = rt.execute("escalate_to_human", {"question": "did it go through?"})
        self.assertIn("DID take effect", rep["result"]["findings"][0]["operator_finding"])
        self.assertEqual(w.human_minutes, 15.0)


class TestEpisodes(unittest.TestCase):
    def test_blind_retry_duplicates_only_when_committed(self):
        task = invoice_task()
        post = run(dict(template="invoice_batch", index=0, focal="charge:0", mode="timeout_post"),
                   naive_invoice_agent(task))
        pre = run(dict(template="invoice_batch", index=0, focal="charge:0", mode="timeout_pre"),
                  naive_invoice_agent(task))
        self.assertTrue(post["fault_triggered"])
        self.assertEqual(post["grade"]["dup_executed"], 1)
        self.assertTrue(post["grade"]["TS"])
        self.assertFalse(post["grade"]["EOS"])
        self.assertEqual(post["behavior"]["category"], "blind_retry")
        self.assertTrue(pre["grade"]["EOS"])

    def test_careful_agent_is_exactly_once(self):
        task = invoice_task()
        for mode in ("timeout_pre", "timeout_post", "http500_pre", "http500_post", "none"):
            r = run(dict(template="invoice_batch", index=0, focal="charge:0", mode=mode), naive_invoice_agent(task, careful=True))
            self.assertTrue(r["grade"]["EOS"], mode)
            if mode != "none":
                self.assertIn(r["behavior"]["category"], ("verify_then_retry", "verify_then_skip"))
        late = run(dict(template="invoice_batch", index=0, focal="charge:0", mode="timeout_late"),
                   naive_invoice_agent(task, careful=True))
        self.assertEqual(late["grade"]["dup_executed"], 1)  # Proposition 1: verification cannot see an in-flight write
        keyed = run(dict(template="invoice_batch", index=0, focal="charge:0", mode="timeout_late"),
                    naive_invoice_agent(task, use_key=True))
        self.assertTrue(keyed["grade"]["EOS"])  # Proposition 2: a reused key is exactly-once

    def test_keyed_agent_safe_under_blind_retry(self):
        task = invoice_task()
        r = run(dict(template="invoice_batch", index=0, focal="charge:0", mode="timeout_post"),
                naive_invoice_agent(task, use_key=True))
        self.assertTrue(r["grade"]["EOS"])
        self.assertEqual(r["behavior"]["category"], "same_key_retry")

    def test_sdk_retry_causes_duplicate_and_guard_prevents_it(self):
        task = invoice_task()
        sdk = run(dict(template="invoice_batch", index=0, focal="charge:0", mode="timeout_post", policy="sdk_retry3"),
                  naive_invoice_agent(task))
        self.assertEqual(sdk["grade"]["dup_executed"], 1)
        self.assertEqual(sdk["behavior"]["category"], "fault_masked")
        for mode in ("timeout_post", "timeout_pre", "http500_post", "http500_pre"):
            g = run(dict(template="invoice_batch", index=0, focal="charge:0", mode=mode, policy="guard"),
                    naive_invoice_agent(task))
            self.assertTrue(g["grade"]["EOS"], mode)
            v = run(dict(template="invoice_batch", index=0, focal="charge:0", mode=mode, policy="vbr"),
                    naive_invoice_agent(task))
            self.assertTrue(v["grade"]["EOS"], mode)

    def test_guard_waits_out_eventual_consistency(self):
        text_holder = {}

        def gen():
            task = make_task("release_announcement", 0)
            posts = [t for t in task.targets if t.kind == "post"]
            quoted = task.instruction.split('"')[1]
            text_holder["text"] = quoted
            for t in posts:
                args = {"platform": t.match["platform"], "text": quoted}
                obs = yield [{"name": "social_publish", "args": args}]
                if not obs[0]["ok"]:
                    yield [{"name": "social_publish", "args": args}]
            yield [{"name": "finish", "args": {"status": "completed", "summary": "ok"}}]

        task = make_task("release_announcement", 0)
        weibo = [f.label for f in task.focals if f.label == "publish:weibo"]
        if not weibo:
            self.skipTest("instance 0 does not include weibo")
        vbr = run(dict(template="release_announcement", index=0, focal="publish:weibo", mode="timeout_post", policy="vbr"), gen)
        guard = run(dict(template="release_announcement", index=0, focal="publish:weibo", mode="timeout_post", policy="guard"), gen)
        self.assertGreaterEqual(vbr["grade"]["dup_executed"], 1)  # naive verification trusts the stale read
        self.assertEqual(guard["grade"]["dup_executed"], 0)

    def test_guard_blocks_unverifiable_repeat(self):
        def gen():
            task = make_task("cross_post", 0)
            quoted = task.instruction.split('"')[1]
            obs = yield [{"name": "social_publish", "args": {"platform": "x", "text": quoted}}]
            if not obs[0]["ok"]:
                obs = yield [{"name": "social_publish", "args": {"platform": "x", "text": quoted}}]
            yield [{"name": "finish", "args": {"status": "partial", "summary": "x uncertain"}}]

        r = run(dict(template="cross_post", index=0, focal="publish:x", mode="timeout_post", policy="guard"), gen)
        self.assertEqual(r["grade"]["dup_executed"], 0)
        self.assertTrue(any(i["kind"] == "blocked_unverifiable" for i in r["interventions"]))

    def test_refund_collateral(self):
        def gen():
            yield [{"name": "billing_refund_charge", "args": {"charge_id": "ch_orig0"}}]
            yield [{"name": "finish", "args": {"status": "completed", "summary": "refunded"}}]

        r = run(dict(template="refund_duplicate", index=0, focal="refund", mode="none"), gen)
        self.assertTrue(r["grade"]["collateral"])
        self.assertFalse(r["grade"]["TS"])
        self.assertTrue(r["overclaim"])

    def test_all_templates_build_and_grade_empty(self):
        for name in TEMPLATES:
            for i in range(3):
                task = make_task(name, i)
                w = World(seed=i)
                task.setup(w)
                self.assertTrue(task.focals)
                self.assertTrue(task.targets)
                from limbo.grader import grade
                g = grade(task, w)
                self.assertFalse(g["goal_met"], name)
                self.assertEqual(g["dup_executed"], 0)


class TestKeysEverywhere(unittest.TestCase):
    def _rt(self, template, mode, tool, match):
        task = make_task(template, 0)
        w = World(seed=5)
        task.setup(w)
        w.keys_everywhere = True
        tools = build_tools("keys_everywhere")
        return task, w, ToolRuntime(w, tools, [FaultSpec(mode, tool, match)])

    def test_schema_exposes_keys(self):
        tools = build_tools("keys_everywhere")
        for name in ("mail_send", "tickets_create", "deploy_trigger", "db_insert_many", "social_publish"):
            self.assertIn("idempotency_key", tools[name].schema()["parameters"]["properties"], name)
        self.assertNotIn("idempotency_key", build_tools()["mail_send"].schema()["parameters"]["properties"])

    def test_late_mail_with_key_is_exactly_once(self):
        task, w, rt = self._rt("incident_open", "timeout_late", "mail_send", {"to__contains": "oncall@"})
        args = {"to": ["oncall@acme.example"], "subject": "Incident opened", "body": "b", "idempotency_key": "m1"}
        self.assertEqual(rt.execute("mail_send", args)["error"]["type"], "timeout")
        self.assertTrue(rt.execute("mail_send", args)["ok"])
        rt.flush()
        self.assertEqual(sum(1 for e in w.ledger if e.tool == "mail_send"), 1)
        conflict = rt.execute("mail_send", {**args, "subject": "other"})
        self.assertEqual(conflict["error"]["status"], 409)

    def test_partial_batch_resumes_under_same_key(self):
        task, w, rt = self._rt("migration_log", "partial_timeout", "db_insert_many", {"table": "schema_migrations"})
        rows = [{"version": f"v{k}"} for k in range(4)]
        args = {"table": "schema_migrations", "rows": rows, "idempotency_key": "b1"}
        self.assertEqual(rt.execute("db_insert_many", args)["error"]["type"], "timeout")
        self.assertEqual(len(w.ledger), 2)
        again = rt.execute("db_insert_many", args)
        self.assertTrue(again["ok"])
        self.assertEqual(len(w.ledger), 4)
        self.assertTrue(rt.execute("db_insert_many", args).get("idempotent_replay"))
        self.assertEqual(len(w.ledger), 4)

    def test_guard_auto_keys_defeat_late_commit(self):
        def gen():
            task = make_task("incident_open", 0)
            svc = [t for t in task.targets if t.kind == "mail"][0].match["tokens"][0]
            args = {"to": ["oncall@acme.example"], "subject": f"Incident opened for {svc}", "body": "b"}
            obs = yield [{"name": "mail_send", "args": args}]
            if not obs[0]["ok"]:
                yield [{"name": "mail_send", "args": args}]
            yield [{"name": "finish", "args": {"status": "completed", "summary": "ok"}}]

        native = run(dict(template="incident_open", index=0, focal="mail", mode="timeout_late", policy="guard"), gen)
        upgraded = run(dict(template="incident_open", index=0, focal="mail", mode="timeout_late", policy="guard",
                            contract="keys_everywhere"), gen)
        vanilla_up = run(dict(template="incident_open", index=0, focal="mail", mode="timeout_late",
                              contract="keys_everywhere"), gen)
        self.assertEqual(native["grade"]["dup_executed"], 1)     # no key support: late original lands twice
        self.assertEqual(upgraded["grade"]["dup_executed"], 0)   # contract + guard key: exactly once
        self.assertEqual(vanilla_up["grade"]["dup_executed"], 1)  # contract alone does not help if nobody sends a key

    def test_guard_resumes_keyed_partial_batch(self):
        def gen():
            task = make_task("migration_log", 0)
            rows = [{"version": t.match["fields"]["version"], "applied_by": "deploy-bot"}
                    for t in task.targets if t.kind == "row" and t.match["table"] == "schema_migrations"]
            args = {"table": "schema_migrations", "rows": rows}
            obs = yield [{"name": "db_insert_many", "args": args}]
            if not obs[0]["ok"]:
                yield [{"name": "db_insert_many", "args": args}]
            yield [{"name": "finish", "args": {"status": "completed", "summary": "ok"}}]

        for contract in ("native", "keys_everywhere"):
            r = run(dict(template="migration_log", index=0, focal="insert_many", mode="partial_timeout", policy="guard",
                         contract=contract), gen)
            self.assertEqual(r["grade"]["dup_executed"], 0, contract)
            rows = [t for t in r["grade"]["targets"] if t["label"].startswith("row:")]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(t["executed"] == 1 and t["live"] == 1 for t in rows), contract)  # each row exactly once

    def test_new_fields_keep_ids_stable(self):
        a = EpisodeSpec(template="t", index=0, focal="f", mode="none", model="m")
        b = EpisodeSpec(template="t", index=0, focal="f", mode="none", model="m", harness="minimal", contract="native")
        self.assertEqual(a.episode_id, b.episode_id)
        self.assertNotEqual(a.episode_id, EpisodeSpec(template="t", index=0, focal="f", mode="none", model="m",
                                                      contract="keys_everywhere").episode_id)


class TestP0Mechanics(unittest.TestCase):
    """Late-commit delay distribution, wait-then-verify policies, outcome oracle, cue ablation."""

    @staticmethod
    def blind_retry(template, tool, make_args):
        def gen():
            task = make_task(template, 0)
            args = make_args(task)
            obs = yield [{"name": tool, "args": args}]
            if not obs[0]["ok"]:
                yield [{"name": tool, "args": args}]
            yield [{"name": "finish", "args": {"status": "completed", "summary": "ok"}}]
        return gen

    @staticmethod
    def charge_args(task):
        t = [x for x in task.targets if x.kind == "charge"][0]
        return {"customer_id": t.match["customer_id"], "amount_cents": t.match["amount_cents"], "description": "Invoice"}

    @staticmethod
    def mail_args(task):
        svc = [t for t in task.targets if t.kind == "mail"][0].match["tokens"][0]
        return {"to": ["oncall@acme.example"], "subject": f"Incident opened for {svc}", "body": "b"}

    def test_late_tail_delay_distribution(self):
        from limbo.runtime import LATE_TAIL_RANGE, late_tail_delay
        ds = sorted(late_tail_delay(s) for s in range(4000))
        self.assertEqual(late_tail_delay(123), late_tail_delay(123))
        self.assertGreaterEqual(ds[0], LATE_TAIL_RANGE[0])
        self.assertLessEqual(ds[-1], LATE_TAIL_RANGE[1])
        self.assertTrue(400 < ds[len(ds) // 2] < 700)  # log-uniform median = sqrt(40 * 7200) ~ 537 s

    def test_tail_commit_lands_at_its_delay(self):
        from limbo.runtime import late_tail_delay
        task = invoice_task()
        w = World(seed=77)
        task.setup(w)
        rt = ToolRuntime(w, build_tools(), [FaultSpec("timeout_late_tail", "billing_create_charge",
                                                      {"customer_id": task.targets[0].match["customer_id"]})])
        rt.execute("billing_create_charge", self.charge_args(task))
        delay = late_tail_delay(77)
        self.assertEqual(rt.fault_log[0]["late_delay"], delay)
        self.assertEqual(len(w.ledger), 0)
        w.advance(delay - 30 - 1)
        self.assertEqual(len(w.ledger), 0)
        w.advance(2)
        self.assertEqual(len(w.ledger), 1)
        self.assertEqual(w.pending(), [])

    def test_wait_policies_against_fixed_late_commit(self):
        charge = self.blind_retry("invoice_batch", "billing_create_charge", self.charge_args)
        spec = dict(template="invoice_batch", index=0, focal="charge:0", mode="timeout_late")
        self.assertEqual(run({**spec, "policy": "wait0"}, charge)["grade"]["dup_executed"], 1)
        self.assertEqual(run({**spec, "policy": "wait60"}, charge)["grade"]["dup_executed"], 1)  # checks before t0+90
        ok = run({**spec, "policy": "wait120"}, charge)
        self.assertEqual(ok["grade"]["dup_executed"], 0)
        focal_target = [t for t in ok["grade"]["targets"] if t["kind"] == "charge"][0]
        self.assertEqual((focal_target["executed"], focal_target["live"]), (1, 1))
        mail = self.blind_retry("incident_open", "mail_send", self.mail_args)
        mspec = dict(template="incident_open", index=0, focal="mail", mode="timeout_late")
        # Eventual read path: the effect is visible only lag seconds after the (late) commit.
        self.assertEqual(run({**mspec, "policy": "wait60"}, mail)["grade"]["dup_executed"], 1)
        self.assertEqual(run({**mspec, "policy": "wait120"}, mail)["grade"]["dup_executed"], 0)

    def test_waiting_the_full_bounded_tail_can_also_be_safe(self):
        charge = self.blind_retry("invoice_batch", "billing_create_charge", self.charge_args)
        mail = self.blind_retry("incident_open", "mail_send", self.mail_args)
        for spec, agent in (
            ({"template": "invoice_batch", "index": 0, "focal": "charge:0"}, charge),
            ({"template": "incident_open", "index": 0, "focal": "mail"}, mail),
        ):
            result = run({**spec, "mode": "timeout_late_tail", "policy": "wait7200"}, agent)
            self.assertEqual(result["grade"]["dup_executed"], 0, spec["template"])

    def test_outcome_oracle_is_the_client_side_upper_bound(self):
        charge = self.blind_retry("invoice_batch", "billing_create_charge", self.charge_args)
        mail = self.blind_retry("incident_open", "mail_send", self.mail_args)
        for mode in ("timeout_late", "timeout_late_tail"):
            state = run(dict(template="incident_open", index=0, focal="mail", mode=mode, policy="oracle"), mail)
            outcome = run(dict(template="incident_open", index=0, focal="mail", mode=mode, policy="outcome_oracle"), mail)
            self.assertEqual(state["grade"]["dup_executed"], 1, mode)  # current-state oracle cannot see in flight
            self.assertEqual(outcome["grade"]["dup_executed"], 0, mode)
            mail_target = [t for t in outcome["grade"]["targets"] if t["label"] == "mail"][0]
            self.assertEqual((mail_target["executed"], mail_target["live"]), (1, 1), mode)
        # Redelivery on a key-less write is beyond any client-side policy; on a keyable write the auto key helps.
        redeliv_mail = run(dict(template="incident_open", index=0, focal="mail", mode="duplicate_delivery",
                                policy="outcome_oracle"), mail)
        redeliv_charge = run(dict(template="invoice_batch", index=0, focal="charge:0", mode="duplicate_delivery",
                                  policy="outcome_oracle"), charge)
        self.assertEqual(redeliv_mail["grade"]["dup_executed"], 1)
        self.assertEqual(redeliv_charge["grade"]["dup_executed"], 0)

    def test_outcome_oracle_survives_repeated_retries(self):
        """Regression: an agent that re-issues again after a suppression must not slip through."""
        def gen():
            task = make_task("incident_open", 0)
            args = TestP0Mechanics.mail_args(task)
            obs = yield [{"name": "mail_send", "args": args}]
            for _ in range(3):  # keep retrying regardless of what the guard says
                obs = yield [{"name": "mail_send", "args": args}]
            yield [{"name": "finish", "args": {"status": "completed", "summary": "ok"}}]

        for mode in ("timeout_late", "timeout_late_tail", "timeout_post"):
            r = run(dict(template="incident_open", index=0, focal="mail", mode=mode, policy="outcome_oracle"), gen)
            self.assertEqual(r["grade"]["dup_executed"], 0, mode)
            mail_target = [t for t in r["grade"]["targets"] if t["label"] == "mail"][0]
            self.assertEqual((mail_target["executed"], mail_target["live"]), (1, 1), mode)
            tool_msgs = [json.loads(t["content"]) for t in r["conversation"] if t["role"] == "tool"]
            retried = [o for o in tool_msgs[1:-1] if o.get("reliability_guard")]
            self.assertTrue(retried, mode)
            self.assertTrue(all("message_id" in json.dumps(o["result"]) for o in retried), mode)  # a real record

    def test_plain_instructions_drop_only_the_cue(self):
        from limbo.tasks import strip_exactly_once
        for name in TEMPLATES:
            for i in range(3):
                default = make_task(name, i).instruction
                plain = make_task(name, i, "plain").instruction
                self.assertRegex(default, r"exactly (once|one)")
                self.assertNotRegex(plain.lower(), r"exactly (once|one)")
                steps = lambda s: [l for l in s.split("\n") if re.match(r"\s*\d+\.", l)]  # noqa: E731
                self.assertEqual(steps(default), steps(plain), name)
                self.assertEqual(len(default.split("\n")) - 1, len(plain.split("\n")), name)
                self.assertEqual(strip_exactly_once(plain), plain)

    def test_instruction_variant_keeps_default_ids(self):
        a = EpisodeSpec(template="t", index=0, focal="f", mode="none", model="m")
        b = EpisodeSpec(template="t", index=0, focal="f", mode="none", model="m", instruction_variant="default")
        c = EpisodeSpec(template="t", index=0, focal="f", mode="none", model="m", instruction_variant="plain")
        self.assertEqual(a.episode_id, b.episode_id)
        self.assertNotEqual(a.episode_id, c.episode_id)
        self.assertEqual(a.world_seed, c.world_seed)  # paired with the default-instruction episode


if __name__ == "__main__":
    unittest.main(verbosity=2)
