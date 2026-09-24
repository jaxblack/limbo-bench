"""Analysis of LIMBO episode records: tidy table, summaries, bootstrap CIs,
paired tests, variance decomposition and figures.

    .venv/Scripts/python -m limbo.analysis summary --experiment pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RESULTS = Path(__file__).resolve().parents[1] / "results"

COMMITTED = {"timeout_post", "http500_post", "partial_timeout", "timeout_late"}
POST = COMMITTED | {"duplicate_delivery"}
PRE = {"timeout_pre", "http500_pre"}
BENIGN = {"http503_transient", "rate_limit", "outage", "schema_drift"}


def load(experiments: Iterable[str]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        path = RESULTS / exp / "episodes.jsonl"
        if not path.exists():
            continue
        latest: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[rec["episode_id"]] = rec
        for rec in latest.values():
            rows.append(flatten(rec))
    return pd.DataFrame(rows)


def flatten(r: dict) -> dict:
    s, g, b, f = r.get("spec", {}), r.get("grade") or {}, r.get("behavior") or {}, r.get("focal") or {}
    u = r.get("usage") or {}
    calls = r.get("agent_calls") or []
    late = next((e.get("late_delay") for e in (r.get("fault_log") or []) if e.get("late_delay") is not None), None)
    return {
        "episode_id": r.get("episode_id"), "experiment": s.get("experiment"), "model": s.get("model"),
        "harness": s.get("harness", "minimal"), "contract_variant": s.get("contract", "native"), "policy": s.get("policy"), "template": s.get("template"),
        "instruction_variant": s.get("instruction_variant", "default"), "late_delay": late,
        "index": s.get("index"), "focal": s.get("focal"), "mode": s.get("mode"), "replicate": s.get("replicate"),
        "doc_variant": s.get("doc_variant"), "paraphrase": s.get("paraphrase"), "human_available": s.get("human_available"),
        "tool": f.get("tool"), "idempotency": f.get("idempotency"), "verification": f.get("verification"),
        "reversible": f.get("reversible"), "batch": f.get("batch"), "truth": r.get("fault_truth"),
        "triggered": r.get("fault_triggered"), "stop": r.get("stop_reason"), "error": r.get("error"),
        "TS": g.get("TS"), "EOS": g.get("EOS"), "dup_exec": g.get("dup_executed", 0), "dup_live": g.get("dup_live", 0),
        "dup_comp": g.get("dup_compensated", 0), "collateral": bool(g.get("collateral")),
        "extraneous": g.get("extraneous_writes", 0), "goal_met": g.get("goal_met"),
        "category": b.get("category"), "verified": b.get("verified"), "waited": b.get("waited"),
        "first_action": b.get("first_action"),
        "escalated_b": b.get("escalated"), "redo_intents": b.get("redo_intents", 0),
        "overclaim": r.get("overclaim"), "claimed_complete": r.get("claimed_complete"),
        "key_used": any(isinstance(c.get("args"), dict) and c["args"].get("idempotency_key")
                        for c in (r.get("agent_calls") or [])),
        "uncertain_reported": r.get("uncertain_reported"), "escalations": r.get("escalations", 0),
        "human_minutes": r.get("human_minutes", 0.0), "virtual_s": r.get("virtual_seconds", 0.0),
        # Agent-visible completion time: excludes the end-of-episode settling of in-flight requests.
        "agent_s": max((float(c.get("t") or 0.0) for c in calls), default=0.0),
        "verify_behaviour": b.get("category") in ("verify_then_retry", "verify_then_skip"),
        "n_calls": r.get("n_agent_calls", 0), "n_exec": r.get("n_executions", 0), "turns": r.get("n_turns", 0),
        "in_tok": u.get("input_tokens", 0), "out_tok": u.get("output_tokens", 0), "reason_tok": u.get("reasoning_tokens", 0),
        "llm_s": r.get("llm_latency_s", 0.0), "wall_s": r.get("wall_s", 0.0),
    }


def valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["stop"].isin(["llm_error", "harness_error"])].copy()


# --------------------------------------------------------------------------- intervals
# Episodes that share a task template are correlated. With only 12 templates a cluster
# bootstrap is anti-conservative and collapses to [0, 0] for zero-event cells, so per-cell
# intervals are Wilson intervals on a Kish effective sample size, n / (1 + (m - 1) * ICC),
# where m is the mean number of episodes per template in the cell and the ICC is estimated
# once per experiment from within-cell residuals.
_ICC: dict[str, float] = {}


def pooled_icc(df: pd.DataFrame, col: str, cell_cols: list[str], cluster: str = "template") -> float:
    d = df[cell_cols + [cluster, col]].dropna().copy()
    if len(d) < 10:
        return 0.0
    d["y"] = d[col].astype(float)
    d["r"] = d["y"] - d.groupby(cell_cols)["y"].transform("mean")
    grp = d.groupby(cell_cols + [cluster])["r"]
    k, n_total = grp.ngroups, len(d)
    if k < 2 or n_total <= k:
        return 0.0
    sizes = grp.size().to_numpy(dtype=float)
    means = grp.mean().to_numpy()
    grand = d["r"].mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    msw = float(((d["r"] - grp.transform("mean")) ** 2).sum() / (n_total - k))
    n0 = (n_total - (sizes ** 2).sum() / n_total) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float(min(1.0, max(0.0, (msb - msw) / denom))) if denom > 0 else 0.0


def use_icc(df: pd.DataFrame, cell_cols: list[str],
            cols: Iterable[str] = ("dsr", "EOS", "TS", "dup_left", "overclaim", "key_used", "verify_behaviour")) -> dict:
    for c in cols:
        if c in df.columns:
            _ICC[c] = pooled_icc(df, c, cell_cols)
    return dict(_ICC)


def wilson(p: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def rate_ci(df: pd.DataFrame, col: str, cluster: str = "template") -> tuple[float, float, float]:
    x = df[[cluster, col]].dropna()
    if x.empty:
        return (float("nan"),) * 3
    y = x[col].astype(float)
    n, p = len(y), float(y.mean())
    mbar = n / max(1, x[cluster].nunique())
    deff = max(1.0, 1.0 + (mbar - 1.0) * _ICC.get(col, 0.0))
    lo, hi = wilson(p, n / deff)
    return p, lo, hi


def cluster_bootstrap(df: pd.DataFrame, col: str, cluster: str = "template", n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = df[[cluster, col]].dropna()
    if x.empty:
        return (float("nan"),) * 3
    groups = [g[col].astype(float).to_numpy() for _, g in x.groupby(cluster)]
    rng = np.random.default_rng(seed)
    est = float(np.mean(np.concatenate(groups)))
    k = len(groups)
    boots = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, k, size=k)
        boots[i] = np.mean(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def rate_table(df: pd.DataFrame, by: list[str], metrics=("EOS", "TS", "dsr", "dup_left", "overclaim")) -> pd.DataFrame:
    d = df.copy()
    d["dsr"] = d["dup_exec"] > 0
    d["dup_left"] = d["dup_live"] > 0
    out = []
    for keys, g in d.groupby(by, dropna=False):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row["n"] = len(g)
        for m in metrics:
            est, lo, hi = cluster_bootstrap(g, m)
            row[m] = est
            row[f"{m}_ci"] = f"[{lo:.2f},{hi:.2f}]"
        out.append(row)
    return pd.DataFrame(out)


def mcnemar(a: pd.Series, b: pd.Series) -> float:
    """Exact two-sided McNemar p-value for paired booleans."""
    from scipy.stats import binomtest
    a, b = a.astype(bool).to_numpy(), b.astype(bool).to_numpy()
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def contract_class(row) -> str:
    idem, ver = row.get("idempotency"), row.get("verification")
    if idem in ("idempotent", "naturally_idempotent"):
        return "idempotent"
    if idem == "keyed_optional":
        return "keyable"
    return {"strong": "non-idem/strong", "eventual": "non-idem/eventual", "none": "non-idem/unverifiable"}.get(ver, "other")


def mcfadden_r2(df: pd.DataFrame, y: str, factors: list[str]) -> float:
    import statsmodels.api as sm
    yv = df[y].astype(float).to_numpy()
    p = yv.mean()
    if p in (0.0, 1.0):
        return 0.0
    ll0 = float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    if not factors:
        return 0.0
    X = pd.get_dummies(df[factors].astype(str), drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.Logit(yv, X).fit_regularized(alpha=1e-4, disp=0, maxiter=500)
        ll = float(res.llf) if np.isfinite(res.llf) else ll0
    except Exception:
        return float("nan")
    return max(0.0, 1.0 - ll / ll0)


def shapley_r2(df: pd.DataFrame, y: str, factors: list[str]) -> dict[str, float]:
    """Shapley decomposition of McFadden pseudo-R^2 across factor groups."""
    from itertools import combinations
    from math import factorial
    n = len(factors)
    cache: dict[frozenset, float] = {}

    def v(s: frozenset) -> float:
        if s not in cache:
            cache[s] = mcfadden_r2(df, y, sorted(s))
        return cache[s]

    out = {}
    for f in factors:
        others = [g for g in factors if g != f]
        phi = 0.0
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                S = frozenset(S)
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi += w * (v(S | {f}) - v(S))
        out[f] = phi
    out["total"] = v(frozenset(factors))
    return out


def summary(exps: list[str]) -> None:
    df = load(exps)
    if df.empty:
        print("no data")
        return
    print(f"records={len(df)} invalid={int(df['stop'].isin(['llm_error', 'harness_error']).sum())}")
    df = valid(df)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n== by model x mode (triggered or none)")
    d = df[(df["triggered"]) | (df["mode"] == "none")]
    print(rate_table(d, ["model", "mode"]).round(3).to_string(index=False))
    print("\n== post-commit faults by focal properties (all models)")
    p = d[d["mode"].isin(POST)]
    print(rate_table(p, ["idempotency", "verification"]).round(3).to_string(index=False))
    print("\n== behaviour after ambiguous faults")
    a = d[d["mode"].isin(POST | PRE)]
    print(pd.crosstab([a["model"], a["mode"]], a["category"]).to_string())
    print("\n== trigger rate by model/mode")
    print(df[df["mode"] != "none"].groupby(["model", "mode"])["triggered"].mean().round(2).unstack().to_string())
    print("\n== cost (mean per episode)")
    print(df.groupby("model")[["n_calls", "turns", "in_tok", "out_tok", "reason_tok", "llm_s"]].mean().round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary"])
    ap.add_argument("--experiment", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a.experiment)


if __name__ == "__main__":
    main()
