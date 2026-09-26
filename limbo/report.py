"""Generate paper tables, figures and headline numbers from episode records.

    .venv/Scripts/python -m limbo.report --out paper/generated
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limbo.analysis import (BENIGN, COMMITTED, PRE, contract_class, load, mcnemar, rate_ci,  # noqa: E402
                            shapley_r2, use_icc, valid)

MODEL_ORDER = ["gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol", "claude-opus-5.5", "gemini-3.8-flash", "grok-4.7",
               "gpt-5.4-mini", "gpt-4.1"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
MODE_ORDER = ["none", "timeout_pre", "http500_pre", "timeout_post", "http500_post", "timeout_late", "partial_timeout",
              "duplicate_delivery", "http503_transient", "rate_limit", "outage", "schema_drift"]
MODE_LABEL = {"none": "none", "timeout_pre": "timeout (not run)", "http500_pre": "500 (not run)",
              "timeout_post": "timeout (committed)", "http500_post": "500 (committed)", "timeout_late": "timeout (late commit)",
              "partial_timeout": "partial batch", "duplicate_delivery": "redelivery", "http503_transient": "503 transient",
              "rate_limit": "429 rate limit", "outage": "persistent 503", "schema_drift": "schema drift"}
CONTRACT_ORDER = ["idempotent", "keyable", "non-idem/strong", "non-idem/eventual", "non-idem/unverifiable"]
POLICY_ORDER = ["vanilla", "aware", "reflect", "sdk_retry3", "rules", "vbr", "guard", "oracle", "outcome_oracle"]
POLICY_LABEL = {"vanilla": "vanilla", "aware": "aware", "reflect": "reflect", "sdk_retry3": "sdk-retry",
                "rules": "rules", "vbr": "vbr", "guard": "guard", "oracle": "state oracle",
                "outcome_oracle": "outcome oracle"}
FRONTIER = ["gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol", "claude-opus-5.5", "gemini-3.8-flash", "grok-4.7"]
WEAK = ["gpt-5.4-mini", "gpt-4.1"]


def label(policy: str, tex: bool = True) -> str:
    if policy.startswith("wait") and policy[4:].isdigit():
        s = f"wait {policy[4:]} s"
    else:
        s = POLICY_LABEL.get(policy, policy)
    return s.replace("_", r"\_") if tex else s


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = valid(df).copy()
    df["contract"] = df.apply(contract_class, axis=1)
    df["dsr"] = df["dup_exec"] > 0
    df["dup_left"] = df["dup_live"] > 0
    df["fault_group"] = np.select(
        [df["mode"] == "none", df["mode"].isin(PRE), df["mode"].isin(COMMITTED), df["mode"] == "duplicate_delivery",
         df["mode"].isin(BENIGN)],
        ["none", "not-executed", "committed", "redelivery", "explicit"], "other")
    return df


def fmt(p: float, lo: float | None = None, hi: float | None = None, pct: bool = True) -> str:
    if p != p:
        return "--"
    s = f"{100 * p:.0f}" if pct else f"{p:.2f}"
    return s


def order(values, pref):
    vals = list(dict.fromkeys(values))
    return [v for v in pref if v in vals] + sorted(v for v in vals if v not in pref)


def e1_tables(e1: pd.DataFrame, out: Path, numbers: dict, e1_all: pd.DataFrame | None = None) -> None:
    d = e1[(e1["triggered"]) | (e1["mode"] == "none")]
    use_icc(d[d["mode"] != "none"], ["model", "mode"])
    # The partial-batch column draws on the E1s supplement, which exists because E1 has only one
    # batch focal write per instance (n=2 per model).
    d_all = e1_all[(e1_all["triggered"]) | (e1_all["mode"] == "none")] if e1_all is not None else d
    models = order(d["model"], MODEL_ORDER)
    modes = order(d["mode"], MODE_ORDER)
    # Heatmap of DSR, model x mode.
    mat = np.array([[(d_all if k == "partial_timeout" else d)
                     .pipe(lambda x: x[(x.model == m) & (x["mode"] == k)]["dsr"].mean()) for k in modes] for m in models])
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(modes)), [MODE_LABEL.get(k, k) for k in modes], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(modes)):
            v = mat[i, j]
            if v == v:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, label="duplicate side-effect rate")
    fig.tight_layout()
    fig.savefig(out / "e1_heatmap.pdf")
    fig.savefig(out / "e1_heatmap.png", dpi=160)
    plt.close(fig)

    # Table: per model, grouped fault families with CIs.
    rows = []
    for m in models:
        g = d[d.model == m]
        row = {"model": m}
        for fam, sel in [("committed", g["mode"].isin({"timeout_post", "http500_post", "partial_timeout"})),
                         ("late", g["mode"] == "timeout_late"), ("redelivery", g["mode"] == "duplicate_delivery")]:
            est, lo, hi = rate_ci(g[sel], "dsr")
            row[f"dsr_{fam}"] = (est, lo, hi)
        est, lo, hi = rate_ci(g[g["mode"].isin(PRE)], "TS")
        row["ts_pre"] = (est, lo, hi)
        est, lo, hi = rate_ci(g[g["mode"] != "none"], "EOS")
        row["eos_all"] = (est, lo, hi)
        rows.append(row)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & \multicolumn{3}{c}{Duplicate rate (\%) under committed faults} & TS (\%) when & EOS (\%) \\",
             r" & lost ack / 500 / partial & late commit & redelivery & not executed & all faults \\", r"\midrule"]
    for r in rows:
        cells = []
        for k in ("dsr_committed", "dsr_late", "dsr_redelivery", "ts_pre", "eos_all"):
            est, lo, hi = r[k]
            cells.append("--" if est != est else f"{100 * est:.0f} {{\\scriptsize[{100 * lo:.0f},{100 * hi:.0f}]}}")
        lines.append(f"{r['model']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "e1_models.tex").write_text("\n".join(lines), encoding="utf-8")
    numbers["e1"] = {r["model"]: {k: [round(x, 4) for x in r[k]] for k in r if k != "model"} for r in rows}
    numbers["e1_n"] = int(len(e1))
    numbers["e1_trigger_rate"] = round(float(e1[e1["mode"] != "none"]["triggered"].mean()), 4)


def e1_contract(e1: pd.DataFrame, out: Path, numbers: dict) -> None:
    c = e1[e1["triggered"] & e1["mode"].isin(COMMITTED)]
    use_icc(c, ["model", "contract"])
    models = order(c["model"], MODEL_ORDER)
    classes = order(c["contract"], CONTRACT_ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    width = 0.8 / max(1, len(classes))
    for j, cl in enumerate(classes):
        ests, los, his = [], [], []
        for m in models:
            est, lo, hi = rate_ci(c[(c.model == m) & (c.contract == cl)], "dsr")
            ests.append(est)
            los.append(est - lo if est == est else 0)
            his.append(hi - est if est == est else 0)
        x = np.arange(len(models)) + (j - (len(classes) - 1) / 2) * width
        ax.bar(x, ests, width, yerr=[los, his], capsize=1.5, label=cl, error_kw={"lw": 0.6})
    ax.set_xticks(range(len(models)), models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("duplicate rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=len(classes), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "e1_contract.pdf")
    fig.savefig(out / "e1_contract.png", dpi=160)
    plt.close(fig)
    numbers["e1_contract"] = {}
    for cl in classes:
        est, lo, hi = rate_ci(c[c.contract == cl], "dsr")
        numbers["e1_contract"][cl] = [round(est, 4), round(lo, 4), round(hi, 4), int((c.contract == cl).sum())]
    frontier = [m for m in models if m in ("gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol", "claude-opus-5.5", "gemini-3.8-flash", "grok-4.7")]
    numbers["e1_contract_frontier"] = {}
    for cl in classes:
        est, lo, hi = rate_ci(c[(c.contract == cl) & c.model.isin(frontier)], "dsr")
        numbers["e1_contract_frontier"][cl] = [round(est, 4), round(lo, 4), round(hi, 4)]


def behaviour(e1: pd.DataFrame, out: Path, numbers: dict) -> None:
    a = e1[e1["triggered"] & e1["mode"].isin({"timeout_pre", "timeout_post", "timeout_late", "http500_pre", "http500_post"})]
    cats = ["blind_retry", "new_key_retry", "same_key_retry", "verify_then_retry", "verify_then_skip", "escalate",
            "stop_without_check", "move_on"]
    tab = pd.crosstab([a["model"], a["mode"]], a["category"], normalize="index")
    tab = tab.reindex(columns=[c for c in cats if c in tab.columns], fill_value=0)
    tab.to_csv(out / "behaviour.csv")
    # Observation-equivalence check: first-action distributions of pre vs post timeouts.
    from scipy.stats import chi2_contingency
    res = {}
    for m in order(a["model"], MODEL_ORDER):
        sub = a[(a.model == m) & a["mode"].isin(["timeout_pre", "timeout_post"])]
        ct = pd.crosstab(sub["mode"], sub["first_action"].fillna("none"))
        if ct.shape[0] == 2 and ct.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(ct)
            res[m] = round(float(p), 4)
    numbers["obs_equivalence_first_action_p"] = res
    models = order(a["model"], MODEL_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, mode in zip(axes, ["timeout_pre", "timeout_post", "timeout_late"]):
        sub = a[a["mode"] == mode]
        t = pd.crosstab(sub["model"], sub["category"], normalize="index").reindex(index=models).fillna(0)
        t = t.reindex(columns=[c for c in cats if c in t.columns], fill_value=0)
        left = np.zeros(len(t))
        for col in t.columns:
            ax.barh(range(len(t)), t[col].to_numpy(), left=left, label=col)
            left += t[col].to_numpy()
        ax.set_title(MODE_LABEL[mode], fontsize=9)
        ax.set_yticks(range(len(t)), t.index, fontsize=8)
        ax.set_xlim(0, 1)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "behaviour.pdf")
    fig.savefig(out / "behaviour.png", dpi=160)
    plt.close(fig)


def _gee(formula: str, d: pd.DataFrame, term: str, binomial: bool = True) -> dict:
    """GEE with exchangeable within-template correlation and bias-reduced (Mancl-DeRouen) SEs;
    t reference distribution with G-1 degrees of freedom because there are only ~12 clusters."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import t as tdist
    fam = sm.families.Binomial() if binomial else sm.families.Gaussian()
    res = smf.gee(formula, "template", d, family=fam, cov_struct=sm.cov_struct.Exchangeable()).fit(
        cov_type="bias_reduced", maxiter=200)
    b, se = float(res.params[term]), float(res.bse[term])
    if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
        raise ValueError(f"non-finite or non-positive GEE estimate for {term}")
    dfree = max(1, d["template"].nunique() - 1)
    q975, q95 = tdist.ppf(0.975, dfree), tdist.ppf(0.95, dfree)
    p = float(2 * tdist.sf(abs(b / se), dfree)) if se > 0 else float("nan")
    return {"b": b, "se": se, "lo95": b - q975 * se, "hi95": b + q975 * se, "lo90": b - q95 * se, "hi90": b + q95 * se,
            "p": p, "df": dfree, "n": int(len(d))}


def duplicate_reporting_metrics(e1: pd.DataFrame) -> dict:
    dup = e1[e1["triggered"] & (e1["dup_exec"] > 0)].copy()
    if dup.empty:
        raise ValueError("H5 requires duplicate-producing, fault-triggered episodes")
    dup["cnu"] = (dup["claimed_complete"].astype(bool) &
                  ~dup["uncertain_reported"].fillna(False).astype(bool)).astype(float)
    use_icc(dup, ["model", "mode"], cols=("claimed_complete", "cnu", "overclaim"))

    def metric(col: str) -> tuple[float, list[float]]:
        estimate, lo, hi = rate_ci(dup, col)
        return round(estimate, 4), [round(lo, 4), round(hi, 4)]

    completed, completed_ci = metric("claimed_complete")
    no_uncertainty, no_uncertainty_ci = metric("cnu")
    overclaim, overclaim_ci = metric("overclaim")
    return {"completed_given_dup": completed, "completed_ci": completed_ci,
            "complete_no_uncertainty_given_dup": no_uncertainty,
            "complete_no_uncertainty_ci": no_uncertainty_ci,
            "overclaim_given_dup": overclaim, "overclaim_ci": overclaim_ci, "n": int(len(dup))}


def hypotheses(e1: pd.DataFrame, numbers: dict) -> None:
    from scipy.stats import fisher_exact
    c = e1[e1["triggered"] & e1["mode"].isin(COMMITTED)]
    # H3: among episodes where the agent verified before re-issuing, eventual vs strong read paths.
    # Restricted to lost-acknowledgement faults: under late commits verification fails on any read path.
    v = c[(c["verified"] == True) & c["contract"].isin(["non-idem/strong", "non-idem/eventual"])  # noqa: E712
          & c["mode"].isin(["timeout_post", "http500_post"])].copy()
    t = pd.crosstab(v["contract"], v["dsr"])
    if t.shape == (2, 2):
        odds, p = fisher_exact(t.to_numpy())
        v["y"], v["eventual"] = v["dsr"].astype(int), (v["contract"] == "non-idem/eventual").astype(int)
        g = _gee("y ~ eventual", v, "eventual")
        numbers["H3"] = {"dsr_verified_eventual": round(float(v[v.contract == "non-idem/eventual"]["dsr"].mean()), 4),
                         "dsr_verified_strong": round(float(v[v.contract == "non-idem/strong"]["dsr"].mean()), 4),
                         "p": round(float(p), 6), "p_gee": g["p"], "n": int(len(v))}
    else:
        raise ValueError("H3 requires both contract classes and both duplicate outcomes")
    # H4 (preregistered TOST, +/-5 pp): blind re-issue of irreversible vs reversible non-idempotent writes.
    amb = e1[e1["triggered"] & e1["mode"].isin(PRE | COMMITTED) & (e1["idempotency"] == "non_idempotent")
             & ~e1["category"].isin(["fault_masked", "focal_not_reached", "no_fault"])].copy()
    amb["blind"] = amb["category"].isin(["blind_retry", "new_key_retry"]).astype(float)
    amb["irreversible"] = (amb["reversible"] == False).astype(int)  # noqa: E712
    rev, irr = amb[amb.irreversible == 0]["blind"], amb[amb.irreversible == 1]["blind"]
    h4 = {"blind_reversible": round(float(rev.mean()), 4), "blind_irreversible": round(float(irr.mean()), 4),
          "n_rev": int(len(rev)), "n_irr": int(len(irr))}
    g = _gee("blind ~ irreversible", amb, "irreversible", binomial=False)
    h4.update({"diff": g["b"], "lo90": g["lo90"], "hi90": g["hi90"], "lo95": g["lo95"], "hi95": g["hi95"],
               "p_diff": g["p"], "equivalent_5pp": bool(g["lo90"] > -0.05 and g["hi90"] < 0.05)})
    numbers["H4"] = h4
    # H5: overclaim among episodes with a duplicate.
    numbers["H5"] = duplicate_reporting_metrics(e1)
    # H1 (preregistered): mixed-effects logistic regression with a contract fixed effect.
    hh = c[c["contract"].isin(CONTRACT_ORDER)].copy()
    hh["y"] = hh["dsr"].astype(int)
    hh["risky"] = hh["contract"].isin(["non-idem/eventual", "non-idem/unverifiable"]).astype(int)
    hh["tpl_inst"] = hh["template"].astype(str) + ":" + hh["index"].astype(str)
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    md = BinomialBayesMixedGLM.from_formula("y ~ risky + C(model) + C(mode)",
                                            {"template": "0 + C(template)", "inst": "0 + C(tpl_inst)"}, hh)
    fit = md.fit_vb()
    i = list(md.exog_names).index("risky")
    mu, sd = float(fit.fe_mean[i]), float(fit.fe_sd[i])
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("H1 mixed-effects estimate is not finite")
    numbers["H1_mixed"] = {"coef": mu, "sd": sd, "or": float(np.exp(mu)), "or_lo": float(np.exp(mu - 1.96 * sd)),
                           "or_hi": float(np.exp(mu + 1.96 * sd)), "n": int(len(hh))}
    g = _gee("y ~ risky + C(model) + C(mode)", hh, "risky")
    numbers["H1_gee"] = {"or": float(np.exp(g["b"])), "or_lo": float(np.exp(g["lo95"])),
                         "or_hi": float(np.exp(g["hi95"])), "p": g["p"], "n": g["n"]}


# Faults an immediate read-back resolves (the effect is already visible, or its absence is informative)
# versus faults it cannot resolve (the request is still in flight, or the transport delivered it twice).
RESOLVABLE = {"timeout_post", "http500_post", "partial_timeout"}
UNRESOLVABLE = {"timeout_late", "duplicate_delivery"}
STRATA = {"pooled": COMMITTED | {"duplicate_delivery"}, "resolvable": RESOLVABLE, "unresolvable": UNRESOLVABLE}


def shapley(df: pd.DataFrame, factors: list[str], label: str, numbers: dict, out: Path) -> None:
    """Shapley decomposition of McFadden pseudo-R^2, pooled (as preregistered) and by stratum."""
    res = {}
    for stratum, modes in STRATA.items():
        d = df[df["triggered"] & df["mode"].isin(modes)].copy()
        if d.empty or d["dsr"].nunique() < 2:
            continue
        d["dsr_i"] = d["dsr"].astype(int)
        fs = [f for f in factors if d[f].nunique() > 1]
        s = shapley_r2(d, "dsr_i", fs)
        res[stratum] = {k: round(v, 4) for k, v in s.items()}
        res[stratum]["n"] = int(len(d))
        res[stratum]["dsr"] = round(float(d["dsr"].mean()), 4)
    numbers[f"shapley_{label}"] = res.get("pooled", {})
    numbers[f"shapley_{label}_strata"] = res
    cols = [s for s in ("pooled", "resolvable", "unresolvable") if s in res]
    head = {"pooled": "pooled (planned)", "resolvable": "read-back resolves", "unresolvable": "read-back cannot"}
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Factor & " + " & ".join(head[c] for c in cols) + r" \\", r"\midrule"]
    for f in factors:
        cells = []
        for c_ in cols:
            r = res[c_]
            cells.append(f"{100 * r[f] / r['total']:.0f}" if f in r and r.get("total") else "--")
        lines.append(f"{f} share (\\%) & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("total pseudo-$R^2$ & " + " & ".join(f"{res[c_]['total']:.2f}" for c_ in cols) + r" \\")
    lines.append("duplicate rate (\\%) & " + " & ".join(f"{100 * res[c_]['dsr']:.0f}" for c_ in cols) + r" \\")
    lines.append("episodes & " + " & ".join(f"{res[c_]['n']:,}" for c_ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"shapley_{label}.tex").write_text("\n".join(lines), encoding="utf-8")


def obs_equivalence(e1: pd.DataFrame, e2: pd.DataFrame, numbers: dict, n_perm: int = 2000) -> None:
    """Do agents act differently after a timeout depending on the hidden outcome?

    (a) total variation distance between first-action distributions in the not-executed and committed
        worlds, against a permutation null; (b) paired first-action agreement across the two worlds,
        against agreement between two independent runs of the same world (E1 vs E2-vanilla).
    """
    rng = np.random.default_rng(0)
    a = e1[e1["triggered"] & e1["mode"].isin(["timeout_pre", "timeout_post"])].copy()
    a["fa"] = a["first_action"].fillna("none")

    def tvd(x: pd.Series, y: pd.Series) -> float:
        px, py = x.value_counts(normalize=True), y.value_counts(normalize=True)
        idx = px.index.union(py.index)
        return 0.5 * float((px.reindex(idx, fill_value=0) - py.reindex(idx, fill_value=0)).abs().sum())

    per_model, obs_all, null_all = {}, [], []
    key = ["model", "template", "index", "focal"]
    for m in order(a["model"], MODEL_ORDER):
        s = a[a.model == m]
        pairs = pd.concat([s[s["mode"] == "timeout_pre"].set_index(key)["fa"].rename("x"),
                           s[s["mode"] == "timeout_post"].set_index(key)["fa"].rename("y")], axis=1, join="inner").dropna()
        if len(pairs) < 5:
            continue
        x, y = pairs["x"].to_numpy(), pairs["y"].to_numpy()
        obs = tvd(pd.Series(x), pd.Series(y))
        null = []
        for _ in range(n_perm):
            # The two worlds of a pair share the task; under "no leakage" their labels are exchangeable.
            flip = rng.random(len(x)) < 0.5
            null.append(tvd(pd.Series(np.where(flip, y, x)), pd.Series(np.where(flip, x, y))))
        null = np.array(null)
        per_model[m] = {"tvd": round(obs, 4), "null_mean": round(float(null.mean()), 4),
                        "null_95": round(float(np.percentile(null, 95)), 4),
                        "p_perm": round(float((null >= obs - 1e-12).mean()), 4), "pairs": int(len(pairs))}
        obs_all.append(obs)
        null_all.append(null.mean())
    pre = a[a["mode"] == "timeout_pre"].set_index(key)["fa"]
    post = a[a["mode"] == "timeout_post"].set_index(key)["fa"]
    cross = pd.concat([pre.rename("x"), post.rename("y")], axis=1, join="inner").dropna()
    b = e2[(e2.policy == "vanilla") & e2["triggered"] & e2["mode"].isin(["timeout_pre", "timeout_post"])].copy()
    b["fa"] = b["first_action"].fillna("none")
    k2 = key + ["mode"]
    same = pd.concat([a.set_index(k2)["fa"].rename("x"), b.set_index(k2)["fa"].rename("y")], axis=1, join="inner").dropna()
    models2 = set(b["model"])
    cross2 = cross[cross.index.get_level_values("model").isin(models2) & (cross.index.get_level_values("index") == 0)]
    numbers["obs_equivalence"] = {
        "per_model": per_model,
        "mean_tvd": round(float(np.mean(obs_all)), 4) if obs_all else None,
        "mean_null_tvd": round(float(np.mean(null_all)), 4) if null_all else None,
        "max_p_perm_min": min((v["p_perm"] for v in per_model.values()), default=None),
        "cross_world_agreement": round(float((cross2["x"] == cross2["y"]).mean()), 4) if len(cross2) else None,
        "same_world_agreement": round(float((same["x"] == same["y"]).mean()), 4) if len(same) else None,
        "n_cross": int(len(cross2)), "n_same": int(len(same)),
    }


def late_wait_analysis(numbers: dict, experiments: tuple[str, ...] = ("e1",), exclude: tuple[str, ...] = ()) -> None:
    """Late commits (fixed 90 s): how long agents waited before re-issuing, by read-path class."""
    import json as _json
    from limbo.analysis import RESULTS
    from limbo.policies import same_intent
    from limbo.runtime import _arg_matches
    from limbo.services import build_tools
    tools = build_tools()
    rows = []
    for exp in experiments:
        path = RESULTS / exp / "episodes.jsonl"
        if not path.exists():
            continue
        latest = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            latest[r["episode_id"]] = r
        for r in latest.values():
            s = r["spec"]
            if (s["mode"] != "timeout_late" or not r.get("fault_triggered") or s["model"] in exclude
                    or r.get("stop_reason") in ("llm_error", "harness_error")):
                continue
            spec = tools[r["focal"]["tool"]]
            if spec.contract.identity is None:
                continue
            calls = r["agent_calls"]
            k = next((i for i, c in enumerate(calls) if c["name"] == spec.name
                      and _arg_matches(c["args"], r["focal"]["match"]) and not c["ok"]), None)
            if k is None:
                continue
            ident = spec.contract.identity(calls[k]["args"])
            redo = next((i for i in range(k + 1, len(calls)) if calls[i]["name"] == spec.name
                         and same_intent(spec.contract.identity(calls[i]["args"]), ident)), None)
            end = redo if redo is not None else len(calls)
            waited = sum(float((c["args"] or {}).get("seconds") or 0) for c in calls[k + 1:end] if c["name"] == "wait")
            rows.append({"verification": r["focal"]["verification"], "redo": redo is not None,
                         "waited60": waited >= 60, "dup": r["grade"]["dup_executed"] > 0})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    out = {"n": int(len(t))}
    for name, sel in (("quick_redo", t.redo & ~t.waited60), ("waited_redo", t.redo & t.waited60), ("no_redo", ~t.redo)):
        g = t[sel]
        out[name] = {"n": int(len(g)), "dup": round(float(g["dup"].mean()), 4) if len(g) else None,
                     "eventual_share": round(float((g["verification"] == "eventual").mean()), 4) if len(g) else None}
    numbers["late_wait"] = out


def load_many(names: list[str]) -> pd.DataFrame:
    frames = [load([n]) for n in names]
    frames = [f for f in frames if not f.empty]
    return prep(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def ci_cell(g: pd.DataFrame, col: str) -> str:
    est, lo, hi = rate_ci(g, col)
    return "--" if est != est else f"{100 * est:.0f} {{\\scriptsize[{100 * lo:.0f},{100 * hi:.0f}]}}"


def test_retest(e1: pd.DataFrame, e2: pd.DataFrame, numbers: dict) -> None:
    """E1 and E2-vanilla share world seeds: two independent runs of the same episode."""
    key = ["model", "template", "index", "focal", "mode"]
    a = e1[e1["triggered"] | (e1["mode"] == "none")].set_index(key)[["dsr", "EOS"]]
    b = e2[(e2.policy == "vanilla") & (e2["triggered"] | (e2["mode"] == "none"))].set_index(key)[["dsr", "EOS"]]
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner").dropna()
    out = {"n_pairs": int(len(j))}
    for col in ("dsr", "EOS"):
        x, y = j[f"{col}_1"].astype(bool), j[f"{col}_2"].astype(bool)
        po = float((x == y).mean())
        px, py = float(x.mean()), float(y.mean())
        pe = px * py + (1 - px) * (1 - py)
        out[col] = {"agreement": round(po, 4), "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None,
                    "rate_run1": round(px, 4), "rate_run2": round(py, 4)}
    numbers["test_retest"] = out


def e2_tables(e2: pd.DataFrame, out: Path, numbers: dict) -> None:
    d = e2[(e2["triggered"]) | (e2["mode"] == "none")].copy()
    models = order(d["model"], MODEL_ORDER)
    pols = order(d["policy"], POLICY_ORDER)
    faulted = d[d["mode"] != "none"]
    use_icc(faulted, ["model", "policy"])
    lines = [r"\begin{tabular}{l" + "c" * len(pols) + "}", r"\toprule",
             "Model & " + " & ".join(label(p) for p in pols) + r" \\", r"\midrule"]
    numbers["e2"] = {}
    for metric, lbl in (("EOS", "EOS (\\%)"), ("dsr", "Duplicate rate (\\%)"), ("TS", "TS (\\%)")):
        lines.append(rf"\multicolumn{{{len(pols) + 1}}}{{l}}{{\emph{{{lbl}}}}} \\")
        for m in models:
            cells = []
            for p in pols:
                g = faulted[(faulted.model == m) & (faulted.policy == p)]
                est, lo, hi = rate_ci(g, metric)
                numbers["e2"].setdefault(m, {}).setdefault(p, {})[metric] = [round(est, 4), round(lo, 4), round(hi, 4)]
                cells.append("--" if est != est else f"{100 * est:.0f}")
            lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for col, lbl in (("n_calls", "tool calls"), ("tokens", "tokens (k)"), ("agent_s", "sim. time (min)"),
                     ("human_minutes", "human (min)")):
        cells = []
        for p in pols:
            g = faulted[faulted.policy == p]
            v = g[col].mean() if col != "tokens" else (g["in_tok"] + g["out_tok"]).mean() / 1000
            if col == "agent_s":
                v = v / 60
            cells.append(f"{v:.1f}")
            numbers["e2"].setdefault("_cost", {}).setdefault(p, {})[col] = round(float(v), 3)
        lines.append(f"mean {lbl} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "e2_conditions.tex").write_text("\n".join(lines), encoding="utf-8")

    # Paired McNemar: guard vs each other condition on EOS (same model, template, focal, mode).
    key = ["model", "template", "index", "focal", "mode"]
    g = faulted[faulted.policy == "guard"].set_index(key)["EOS"]
    tests = {}
    raw = []
    for p in pols:
        if p in ("guard", "oracle", "outcome_oracle"):
            continue
        o = faulted[faulted.policy == p].set_index(key)["EOS"]
        j = pd.concat([g.rename("g"), o.rename("o")], axis=1, join="inner").dropna()
        pv = mcnemar(j["o"], j["g"])
        raw.append((p, pv, len(j), float(j["g"].mean() - j["o"].mean())))
    # Holm correction.
    raw.sort(key=lambda x: x[1])
    k = len(raw)
    running = 0.0
    for i, (p, pv, n, diff) in enumerate(raw):
        adj = min(1.0, max(running, (k - i) * pv))
        running = adj
        tests[p] = {"p_holm": adj, "n_pairs": n, "eos_gain_pp": round(100 * diff, 1)}
    numbers["e2_mcnemar_guard_vs"] = tests
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Guard vs. & EOS gain (pp) & paired episodes & Holm-adjusted $p$ \\",
             r"\midrule"]
    for p in pols:
        if p in tests:
            t = tests[p]
            lines.append(f"{label(p)} & {t['eos_gain_pp']:+.1f} & {t['n_pairs']} & "
                         f"{t['p_holm']:.2g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "e2_mcnemar.tex").write_text("\n".join(lines), encoding="utf-8")

    # Pareto: EOS vs mean tool calls per policy, per model.
    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for p in pols:
            gg = faulted[(faulted.model == m) & (faulted.policy == p)]
            if gg.empty:
                continue
            ax.scatter(gg["n_calls"].mean(), gg["EOS"].mean(), s=20 + 200 * gg["dsr"].mean(), alpha=0.8)
            ax.annotate(label(p, tex=False), (gg["n_calls"].mean(), gg["EOS"].mean()), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("mean tool calls", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("exactly-once success", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "e2_pareto.pdf")
    fig.savefig(out / "e2_pareto.png", dpi=160)
    plt.close(fig)


def e2k_tables(e2: pd.DataFrame, e2k: pd.DataFrame, out: Path, numbers: dict) -> None:
    both = pd.concat([e2[e2.policy.isin(["vanilla", "guard"])], e2k], ignore_index=True)
    d = both[both["triggered"] & both["mode"].isin(COMMITTED | {"duplicate_delivery"})]
    use_icc(d, ["contract_variant", "policy", "mode"])
    modes = order(d["mode"], MODE_ORDER)
    lines = [r"\begin{tabular}{ll" + "c" * len(modes) + "}", r"\toprule",
             "Contract & Condition & " + " & ".join(MODE_LABEL[m] for m in modes) + r" \\", r"\midrule"]
    numbers["e2k"] = {}
    for contract in ("native", "keys_everywhere"):
        for p in ("vanilla", "guard"):
            cells = []
            for mo in modes:
                g = d[(d.contract_variant == contract) & (d.policy == p) & (d["mode"] == mo)]
                est, lo, hi = rate_ci(g, "dsr")
                numbers["e2k"].setdefault(contract, {}).setdefault(p, {})[mo] = [round(est, 4), round(lo, 4), round(hi, 4), int(len(g))]
                cells.append("--" if est != est else f"{100 * est:.0f}")
            lines.append(f"{contract.replace('_', '-')} & {label(p)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "e2k_contract.tex").write_text("\n".join(lines), encoding="utf-8")
    # How often do models send a key when the contract offers one (vanilla, keys_everywhere)?
    kv = e2k[(e2k.policy == "vanilla") & e2k["triggered"] & e2k["key_used"].notna()]
    numbers["e2k_key_use"] = {}
    for m in order(kv["model"], MODEL_ORDER):
        numbers["e2k_key_use"][m] = round(float(kv[kv.model == m]["key_used"].mean()), 4)


def e3_tables(e3: pd.DataFrame, out: Path, numbers: dict) -> None:
    d = e3[(e3["triggered"]) | (e3["mode"] == "none")].copy()
    use_icc(d[d["mode"] != "none"], ["harness", "model", "policy"])
    harnesses = [h for h in ("minimal", "copilot", "hermes", "codex") if h in set(d["harness"])]
    models = order(d["model"], MODEL_ORDER)
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Harness & Model & \multicolumn{3}{c}{Duplicate rate (\%), native} & EOS (\%) native & EOS (\%) guard & n \\",
             r" & & committed & late & redelivery & & & \\", r"\midrule"]
    numbers["e3"] = {}
    for h in harnesses:
        for m in models:
            g = d[(d.harness == h) & (d.model == m)]
            if g.empty:
                continue
            nat, grd = g[g.policy == "vanilla"], g[g.policy == "guard"]
            fam = {"committed": nat["mode"].isin({"timeout_post", "http500_post", "partial_timeout"}),
                   "late": nat["mode"] == "timeout_late", "redelivery": nat["mode"] == "duplicate_delivery"}
            cells = [ci_cell(nat[sel], "dsr") for sel in fam.values()]
            cells.append(ci_cell(nat[nat["mode"] != "none"], "EOS"))
            cells.append(ci_cell(grd[grd["mode"] != "none"], "EOS"))
            lines.append(f"{h} & {m} & " + " & ".join(cells) + f" & {len(g)} \\\\")
            numbers["e3"].setdefault(h, {})[m] = {
                "dsr_committed": round(float(nat[fam['committed']]["dsr"].mean()), 4) if fam["committed"].any() else None,
                "dsr_late": round(float(nat[fam['late']]["dsr"].mean()), 4) if fam["late"].any() else None,
                "dsr_redelivery": round(float(nat[fam['redelivery']]["dsr"].mean()), 4) if fam["redelivery"].any() else None,
                "eos_native": round(float(nat[nat['mode'] != 'none']["EOS"].mean()), 4) if len(nat) else None,
                "eos_guard": round(float(grd[grd['mode'] != 'none']["EOS"].mean()), 4) if len(grd) else None,
                "dsr_guard": round(float(grd[grd['mode'] != 'none']["dsr"].mean()), 4) if len(grd) else None,
                "dsr_native": round(float(nat[nat['mode'] != 'none']["dsr"].mean()), 4) if len(nat) else None,
                "ts_native": round(float(nat[nat['mode'] != 'none']["TS"].mean()), 4) if len(nat) else None,
                "ts_guard": round(float(grd[grd['mode'] != 'none']["TS"].mean()), 4) if len(grd) else None,
                "n": int(len(g)),
            }
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (out / "e3_harness.tex").write_text("\n".join(lines), encoding="utf-8")
    nat = d[d.policy == "vanilla"]
    shapley(nat, ["contract", "mode", "model", "harness"], "e3", numbers, out)
    # Figure: duplicate rate by harness and fault family (native), models pooled.
    fams = [("committed", {"timeout_post", "http500_post", "partial_timeout"}), ("late", {"timeout_late"}),
            ("redelivery", {"duplicate_delivery"})]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    w = 0.8 / max(1, len(harnesses))
    for j, h in enumerate(harnesses):
        vals, errs_lo, errs_hi = [], [], []
        for _, ms in fams:
            est, lo, hi = rate_ci(nat[(nat.harness == h) & nat["mode"].isin(ms)], "dsr")
            vals.append(est)
            errs_lo.append(est - lo if est == est else 0)
            errs_hi.append(hi - est if est == est else 0)
        x = np.arange(len(fams)) + (j - (len(harnesses) - 1) / 2) * w
        ax.bar(x, vals, w, yerr=[errs_lo, errs_hi], capsize=2, label=h, error_kw={"lw": 0.6})
    ax.set_xticks(range(len(fams)), [f for f, _ in fams])
    ax.set_ylabel("duplicate rate (native)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, ncol=len(harnesses))
    fig.tight_layout()
    fig.savefig(out / "e3_harness.pdf")
    fig.savefig(out / "e3_harness.png", dpi=160)
    plt.close(fig)


def e3k_tables(e3: pd.DataFrame, e3k: pd.DataFrame, out: Path, numbers: dict) -> None:
    """Same model (gpt-5.6-sol) in every harness: native vs keys-everywhere contract, vanilla vs guard."""
    modes = COMMITTED | {"duplicate_delivery"}
    nat = e3[e3["triggered"] & e3["mode"].isin(modes) & (e3.model == "gpt-5.6-sol")]
    k = e3k[e3k["triggered"] & e3k["mode"].isin(modes)]
    use_icc(pd.concat([nat, k], ignore_index=True), ["harness", "contract_variant", "policy"])
    harnesses = [h for h in ("minimal", "copilot", "hermes", "codex") if h in set(k["harness"]) | set(nat["harness"])]
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{native contract} & \multicolumn{2}{c}{keys everywhere} & EOS (\%) \\",
             r"Harness & vanilla & guard & vanilla & guard & keys + guard \\", r"\midrule"]
    res = {}
    for h in harnesses:
        cells = []
        r = {}
        for tag, df in (("native", nat), ("keys", k)):
            for p in ("vanilla", "guard"):
                g = df[(df.harness == h) & (df.policy == p)]
                est, lo, hi = rate_ci(g, "dsr")
                r[f"{tag}_{p}_dsr"] = round(est, 4) if est == est else None
                r[f"{tag}_{p}_n"] = int(len(g))
                cells.append("--" if est != est else f"{100 * est:.0f}")
        g = k[(k.harness == h) & (k.policy == "guard")]
        est, lo, hi = rate_ci(g, "EOS")
        r["keys_guard_eos"] = round(est, 4) if est == est else None
        keyed = k[(k.harness == h) & (k.policy == "vanilla") & k["key_used"].notna()]
        r["keys_vanilla_keyuse"] = round(float(keyed["key_used"].mean()), 4) if len(keyed) else None
        cells.append("--" if est != est else f"{100 * est:.0f}")
        lines.append(f"{h} & " + " & ".join(cells) + r" \\")
        res[h] = r
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "e3k_harness.tex").write_text("\n".join(lines), encoding="utf-8")
    numbers["e3k"] = res


E5_MODELS = ["claude-opus-5.5", "gpt-6-sol", "gemini-3.8-flash"]


def key_stability(numbers: dict, experiments: tuple[str, ...] = ("e2k", "e5k")) -> None:
    """Keys-everywhere contract, late commits: do agents reuse the key when they re-issue a write?"""
    import json as _json
    from limbo.analysis import RESULTS
    from limbo.policies import same_intent
    from limbo.runtime import _arg_matches
    from limbo.services import build_tools
    tools = build_tools("keys_everywhere")
    rows = []
    for exp in experiments:
        path = RESULTS / exp / "episodes.jsonl"
        if not path.exists():
            continue
        latest = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            latest[r["episode_id"]] = r
        for r in latest.values():
            s = r["spec"]
            if (s["mode"] not in ("timeout_late", "timeout_late_tail") or not r.get("fault_triggered")
                    or r.get("stop_reason") in ("llm_error", "harness_error")):
                continue
            spec = tools[r["focal"]["tool"]]
            if spec.contract.identity is None or spec.contract.idempotent:
                continue
            calls = r["agent_calls"]
            k = next((i for i, c in enumerate(calls) if c["name"] == spec.name
                      and _arg_matches(c["args"], r["focal"]["match"]) and not c["ok"]), None)
            if k is None:
                continue
            if not spec.contract.supports_key(calls[k]["args"]):
                continue
            ident = spec.contract.identity(calls[k]["args"])
            redo = next((i for i in range(k + 1, len(calls)) if calls[i]["name"] == spec.name
                         and same_intent(spec.contract.identity(calls[i]["args"]), ident)), None)
            if redo is None:
                continue
            k0, k1 = calls[k]["args"].get("idempotency_key"), calls[redo]["args"].get("idempotency_key")
            kind = "same" if (k0 and k0 == k1) else ("no_first" if not k0 else "changed")
            rows.append({"kind": kind, "dup": r["grade"]["dup_executed"] > 0, "policy": s["policy"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    out = {"n": int(len(t))}
    for kind in ("same", "no_first", "changed"):
        g = t[t.kind == kind]
        out[kind] = {"n": int(len(g)), "dup": round(float(g["dup"].mean()), 4) if len(g) else None}
    dups = t[t.dup]
    out["share_of_dups_not_reused"] = round(float((dups.kind != "same").mean()), 4) if len(dups) else None
    numbers["key_stability"] = out


def e5_tables(e2: pd.DataFrame, e2k: pd.DataFrame, e5: pd.DataFrame, e5k: pd.DataFrame, out: Path,
              numbers: dict) -> None:
    """Waiting versus keys under late commits: fixed 90 s delay and heavy-tailed delay."""
    frames = []

    def take(df, mode, policies, contract="native"):
        if df is None or df.empty:
            return
        g = df[df["triggered"] & (df["mode"] == mode) & (df["index"] == 0) & df.model.isin(E5_MODELS)
               & df.policy.isin(policies) & (df.contract_variant == contract)].copy()
        g["panel"] = "fixed" if mode == "timeout_late" else "tail"
        g["condition"] = g["policy"] if contract == "native" else "keys:" + g["policy"]
        frames.append(g)

    take(e2, "timeout_late", ["vanilla", "guard", "oracle", "outcome_oracle"])
    take(e5, "timeout_late", ["wait0", "wait60", "wait120", "wait300"])
    take(e2k, "timeout_late", ["vanilla", "guard"], "keys_everywhere")
    take(e5, "timeout_late_tail", ["vanilla", "wait0", "wait60", "wait300", "wait900", "wait3600", "guard",
                                   "outcome_oracle"])
    take(e5k, "timeout_late_tail", ["vanilla", "guard"], "keys_everywhere")
    if not frames:
        return
    d = pd.concat(frames, ignore_index=True)
    use_icc(d, ["panel", "condition"])
    conds = ["vanilla", "wait0", "wait60", "wait120", "wait300", "wait900", "wait3600", "guard", "oracle",
             "outcome_oracle", "keys:vanilla", "keys:guard"]
    delays = d[d.panel == "tail"].drop_duplicates(["template", "focal"])["late_delay"].dropna().to_numpy()
    res: dict = {"tail_delay_median": float(np.median(delays)) if len(delays) else None,
                 "tail_delay_max": float(delays.max()) if len(delays) else None, "n_tail_worlds": int(len(delays))}
    for panel in ("fixed", "tail"):
        for c in conds:
            g = d[(d.panel == panel) & (d.condition == c)]
            if g.empty:
                continue
            eos, dsr = rate_ci(g, "EOS"), rate_ci(g, "dsr")
            r = {"n": int(len(g)), "eos": [round(x, 4) for x in eos], "dsr": [round(x, 4) for x in dsr],
                 "ts": round(float(g["TS"].mean()), 4), "latency_min": round(float(g["agent_s"].mean() / 60), 2),
                 "latency_p90_min": round(float(np.percentile(g["agent_s"], 90) / 60), 2)}
            if c.startswith("wait") and panel == "tail" and len(delays):
                r["ceiling"] = round(float((delays <= float(c[4:])).mean()), 4)
            res.setdefault(panel, {})[c] = r
    numbers["e5"] = res

    def cond_label(c: str) -> str:
        return ("keys: " + label(c.split(":", 1)[1])) if c.startswith("keys:") else label(c)

    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{fixed delay (90 s)} & \multicolumn{4}{c}{heavy-tailed delay (40 s--2 h)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
             r"Condition & EOS (\%) & dup. (\%) & time (min) & EOS (\%) & dup. (\%) & time (min) & $P(\delta\le\Delta)$ \\",
             r"\midrule"]
    for c in conds:
        f, t = res.get("fixed", {}).get(c), res.get("tail", {}).get(c)
        if not f and not t:
            continue

        def cells(r, ceiling=False):
            if not r:
                return ["--"] * (4 if ceiling else 3)
            x = [f"{100 * r['eos'][0]:.0f}", f"{100 * r['dsr'][0]:.0f}", f"{r['latency_min']:.1f}"]
            if ceiling:
                x.append(f"{100 * r['ceiling']:.0f}" if "ceiling" in r else "--")
            return x
        lines.append(cond_label(c) + " & " + " & ".join(cells(f) + cells(t, True)) + r" \\")
        if c in ("wait300", "wait3600", "outcome_oracle"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "e5_waiting.tex").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, panel, title in zip(axes, ("fixed", "tail"), ("in-flight delay fixed at 90 s",
                                                           "in-flight delay log-uniform 40 s to 2 h")):
        pr = res.get(panel, {})
        waits = [c for c in conds if c.startswith("wait") and c in pr]
        if waits:
            xs = [max(pr[c]["latency_min"], 0.05) for c in waits]
            ys = [pr[c]["eos"][0] for c in waits]
            ax.plot(xs, ys, "-o", color="tab:blue", ms=4, lw=1, label="wait Δ, then verify")
            for c, x, y in zip(waits, xs, ys):
                ax.annotate(f"Δ={c[4:]}s", (x, y), fontsize=6, xytext=(3, -8), textcoords="offset points")
        style = {"vanilla": ("tab:gray", "s"), "guard": ("tab:orange", "D"), "oracle": ("tab:purple", "v"),
                 "outcome_oracle": ("tab:red", "^"), "keys:vanilla": ("tab:green", "P"), "keys:guard": ("darkgreen", "*")}
        for c, (col, mk) in style.items():
            if c not in pr:
                continue
            r = pr[c]
            ax.errorbar(max(r["latency_min"], 0.05), r["eos"][0], yerr=[[r["eos"][0] - r["eos"][1]], [r["eos"][2] - r["eos"][0]]],
                        fmt=mk, color=col, ms=6 if mk != "*" else 9, capsize=2, lw=0.8, label=cond_label(c).replace("\\", ""))
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100)]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlim(0.4, 120)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("mean agent-visible episode time (min, log)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("exactly-once success", fontsize=8)
    seen, handles, labels_ = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels_.append(l)
    fig.legend(handles, labels_, loc="lower center", ncol=7, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "e5_pareto.pdf")
    fig.savefig(out / "e5_pareto.png", dpi=160)
    plt.close(fig)


def e6_tables(e1_core: pd.DataFrame, e6: pd.DataFrame, out: Path, numbers: dict) -> None:
    """Cue ablation: identical worlds with and without the closing 'exactly once' sentence."""
    if e6.empty:
        return
    key = ["model", "template", "index", "focal", "mode"]
    base = e1_core[(e1_core["index"] == 0) & e1_core.model.isin(set(e6.model)) & (e1_core.policy == "vanilla")]
    cols = ["dsr", "EOS", "TS", "verify_behaviour", "escalations", "triggered"]
    j = base.set_index(key)[cols].join(e6.set_index(key)[cols], lsuffix="_d", rsuffix="_p", how="inner").reset_index()
    fams = [("not executed (TS)", {"timeout_pre"}, "TS"), ("read-back resolves", RESOLVABLE, "dsr"),
            ("late commit", {"timeout_late"}, "dsr"), ("redelivery", {"duplicate_delivery"}, "dsr"),
            ("no fault (EOS)", {"none"}, "EOS")]
    res = {"n_pairs": int(len(j))}
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Fault family & metric & default (\%) & plain (\%) & $\Delta$ (pp) & McNemar $p$ & pairs \\", r"\midrule"]
    for name, modes, metric in fams:
        g = j[j["mode"].isin(modes)]
        if name != "no fault (EOS)":
            g = g[g["triggered_d"].astype(bool) & g["triggered_p"].astype(bool)]
        if g.empty:
            continue
        a, b = g[f"{metric}_d"].astype(bool), g[f"{metric}_p"].astype(bool)
        p = mcnemar(a, b)
        va = float(g["verify_behaviour_d"].mean()) if modes & (RESOLVABLE | {"timeout_late", "timeout_pre"}) else None
        vb = float(g["verify_behaviour_p"].mean()) if va is not None else None
        res[name] = {"default": round(float(a.mean()), 4), "plain": round(float(b.mean()), 4),
                     "diff_pp": round(100 * float(b.mean() - a.mean()), 1), "p": p, "n": int(len(g)),
                     "verify_default": None if va is None else round(va, 4),
                     "verify_plain": None if vb is None else round(vb, 4)}
        mname = {"dsr": "dup.", "TS": "TS", "EOS": "EOS"}[metric]
        lines.append(f"{name} & {mname} & {100 * a.mean():.0f} & {100 * b.mean():.0f} & "
                     f"{100 * (b.mean() - a.mean()):+.1f} & {p:.2g} & {len(g)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "e6_cue.tex").write_text("\n".join(lines), encoding="utf-8")
    per_model = {}
    g = j[j["mode"].isin(RESOLVABLE) & j["triggered_d"].astype(bool) & j["triggered_p"].astype(bool)]
    for m in order(g["model"], MODEL_ORDER):
        s = g[g.model == m]
        per_model[m] = {"default": round(float(s["dsr_d"].mean()), 4), "plain": round(float(s["dsr_p"].mean()), 4),
                        "n": int(len(s))}
    res["resolvable_per_model"] = per_model
    weak = set(WEAK)
    for tier, sel in (("frontier", ~g.model.isin(weak)), ("weak", g.model.isin(weak))):
        s = g[sel]
        if len(s):
            res[f"resolvable_{tier}"] = {"default": round(float(s["dsr_d"].mean()), 4),
                                         "plain": round(float(s["dsr_p"].mean()), 4), "n": int(len(s)),
                                         "p": mcnemar(s["dsr_d"].astype(bool), s["dsr_p"].astype(bool)),
                                         "models": order(s["model"], MODEL_ORDER)}
    numbers["e6"] = res

    def pc(x):
        return f"{100 * x:.0f}"

    def pv(p):
        return f"$p={fmt_p(p)}$"

    r, lt, rd = res.get("read-back resolves"), res.get("late commit"), res.get("redelivery")
    fr, wk = res.get("resolvable_frontier"), res.get("resolvable_weak")
    parts = []
    if r:
        verb = "rises" if r["plain"] > r["default"] else ("falls" if r["plain"] < r["default"] else "stays")
        parts.append(f"Without the cue, the duplicate rate on faults that a read-back resolves {verb} from "
                     f"{pc(r['default'])}\\% to {pc(r['plain'])}\\% ({pv(r['p'])})")
    if lt:
        parts.append(f"on late commits from {pc(lt['default'])}\\% to {pc(lt['plain'])}\\% ({pv(lt['p'])})")
    if rd:
        parts.append(f"and on redelivery stays at {pc(rd['default'])}\\% ({pv(rd['p'])})" if pc(rd["default"]) == pc(rd["plain"])
                     else f"and on redelivery from {pc(rd['default'])}\\% to {pc(rd['plain'])}\\% ({pv(rd['p'])})")
    text = ", ".join(parts) + "." if parts else ""
    if fr and wk:
        who = ("the weaker models" if len(wk["models"]) > 1 else
               "the weaker model, \\texttt{" + wk["models"][0] + "}")
        text += (f" On resolvable faults the change is concentrated in {who} ({pc(wk['default'])}\\% to "
                 f"{pc(wk['plain'])}\\%, {pv(wk['p'])}); the frontier models move from {pc(fr['default'])}\\% to "
                 f"{pc(fr['plain'])}\\% ({pv(fr['p'])}).")
    if r and r["plain"] > r["default"] and r["p"] < 0.05:
        text += (" The explicit instruction therefore makes agents more careful, and the duplicate rates we report with "
                 "it are conservative for instructions that leave exactly-once execution implicit.")
    elif r and r["p"] >= 0.05:
        text += " The explicit instruction does not measurably change behaviour on these faults."
    numbers["e6_summary"] = text


def e4_tables(frames: dict[str, pd.DataFrame], out: Path, numbers: dict) -> None:
    """Each E4 variant is compared with its matched E2 baseline (same models, modes, instance, focal writes)."""
    base = frames.get("base")
    res: dict = {}

    def matched(df: pd.DataFrame, ref: pd.DataFrame, policies: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = ["model", "template", "index", "focal", "mode"]
        a = df[df["triggered"] & (df["mode"] != "none") & df.policy.isin(policies)]
        b = ref[ref["triggered"] & (ref["mode"] != "none") & ref.policy.isin(policies)]
        common = a[key].merge(b[key], on=key).drop_duplicates()
        return a.merge(common, on=key), b.merge(common, on=key)

    def stats(g: pd.DataFrame) -> dict:
        return {"dsr": round(float(g["dsr"].mean()), 4) if len(g) else None,
                "eos": round(float(g["EOS"].mean()), 4) if len(g) else None,
                "ts": round(float(g["TS"].mean()), 4) if len(g) else None,
                "escal": round(float((g["escalations"] > 0).mean()), 4) if len(g) else None,
                "n": int(len(g))}

    text = []

    def pc(x):
        return "--" if x is None else f"{100 * x:.0f}"

    if base is not None and not base.empty:
        docs = frames.get("docs")
        if docs is not None and not docs.empty:
            for variant in ("no_consistency_docs", "explicit"):
                a, b = matched(docs[docs.doc_variant == variant], base, ["vanilla"])
                res[f"docs_{variant}"] = {"variant": stats(a), "base": stats(b)}
            r1, r2 = res.get("docs_no_consistency_docs"), res.get("docs_explicit")
            if r1 and r2:
                text.append(
                    f"\\paragraph{{Documentation.}} Removing every statement about read-path lag and consistency from the "
                    f"tool descriptions changes the duplicate rate from {pc(r1['base']['dsr'])}\\% to "
                    f"{pc(r1['variant']['dsr'])}\\% (matched episodes, $n={r1['variant']['n']}$); adding an explicit "
                    f"``not idempotent'' warning to each write changes it from {pc(r2['base']['dsr'])}\\% to "
                    f"{pc(r2['variant']['dsr'])}\\% ($n={r2['variant']['n']}$).")
        para = frames.get("para")
        if para is not None and not para.empty:
            parts = []
            for k in (1, 2):
                a, b = matched(para[para.paraphrase == k], base, ["vanilla"])
                res[f"para_{k}"] = {"variant": stats(a), "base": stats(b)}
                parts.append(f"{pc(res[f'para_{k}']['variant']['dsr'])}\\%")
            text.append(
                f"\\paragraph{{Prompt wording.}} Two paraphrases of the system prompt yield duplicate rates of "
                f"{' and '.join(parts)}, against {pc(res['para_1']['base']['dsr'])}\\% for the original on the same "
                f"episodes.")
        nh = frames.get("nohuman")
        if nh is not None and not nh.empty:
            for p in ("vanilla", "guard"):
                a, b = matched(nh[nh.policy == p], base, [p])
                res[f"nohuman_{p}"] = {"variant": stats(a), "base": stats(b)}
            v, g = res["nohuman_vanilla"], res["nohuman_guard"]
            text.append(
                f"\\paragraph{{Unresponsive operator.}} When escalation never receives an answer, vanilla agents' duplicate "
                f"rate is {pc(v['variant']['dsr'])}\\% (vs.\\ {pc(v['base']['dsr'])}\\% with an operator) and the "
                f"guard's is {pc(g['variant']['dsr'])}\\% (vs.\\ {pc(g['base']['dsr'])}\\%); escalation occurred in "
                f"{pc(g['variant']['escal'])}\\% of guard episodes.")
        ab = frames.get("ablate")
        if ab is not None and not ab.empty:
            parts = []
            for p in ("guard-no-key", "guard-no-consistency", "guard-no-block", "guard-no-annotate"):
                sub = ab[ab.policy == p].assign(policy="guard")
                a, b = matched(sub, base, ["guard"])
                res[f"ablate_{p}"] = {"variant": stats(a), "base": stats(b)}
                parts.append(f"without {p.split('-', 2)[2]} {pc(res[f'ablate_{p}']['variant']['dsr'])}\\%")
            ref = res["ablate_guard-no-key"]["base"]["dsr"]
            ab_models = order(ab["model"], MODEL_ORDER)
            who = (f"For \\texttt{{{ab_models[0]}}}, the E2 model the guard helps most," if len(ab_models) == 1 else
                   f"For the {NUMBER_WORDS[len(ab_models)]} models the guard helps most,")
            text.append(
                f"\\paragraph{{Guard ablations.}} {who} the full guard's duplicate rate "
                f"on the ablation episodes is {pc(ref)}\\%; removing one component at a time gives: "
                + "; ".join(parts) + ".")
    numbers["e4"] = res
    (out / "e4_text.tex").write_text("\n\n".join(text) + "\n", encoding="utf-8")


def appendix_artifacts(out: Path) -> None:
    from limbo.prompts import NUDGE_MESSAGE, REFLECT_MESSAGE, RELIABILITY_RULES, SYSTEM_VANILLA
    from limbo.harness import PREAMBLE
    from limbo.services import build_tools
    tools = build_tools()
    lines = [r"\begin{tabular}{llp{9.2cm}}", r"\toprule", r"Tool & Kind & Description shown to the agent \\", r"\midrule"]
    for name, spec in tools.items():
        kind = "write" if spec.contract.write else "read"
        desc = spec.schema("neutral")["description"].replace("_", r"\_").replace("%", r"\%")
        lines.append(rf"\texttt{{{name.replace('_', chr(92) + '_')}}} & {kind} & {desc} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tools.tex").write_text("\n".join(lines), encoding="utf-8")

    def verb(title: str, text: str) -> str:
        return f"\\paragraph{{{title}}}\n\\begin{{small}}\\begin{{verbatim}}\n{wrap(text)}\n\\end{{verbatim}}\\end{{small}}\n"

    def wrap(text: str, width: int = 100) -> str:
        import textwrap
        return "\n".join(textwrap.fill(p, width) if p.strip() else "" for p in text.split("\n"))

    body = (verb("Scaffold system prompt (vanilla)", SYSTEM_VANILLA)
            + verb("Reliability rules appended in the aware condition", RELIABILITY_RULES.strip())
            + verb("Reflection message (reflect condition)", REFLECT_MESSAGE)
            + verb("Continuation nudge", NUDGE_MESSAGE)
            + verb("Harness preamble (prepended to the task in E3)", PREAMBLE.strip()))
    (out / "prompts.tex").write_text(body, encoding="utf-8")


def headline(e1: pd.DataFrame, e1_core: pd.DataFrame, e2: pd.DataFrame, e2k: pd.DataFrame, e3: pd.DataFrame,
             numbers: dict) -> None:
    """Numbers quoted in the prose; each is recomputed from episode records."""
    H: dict = {}
    frontier = ["gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol", "claude-opus-5.5", "gemini-3.8-flash", "grok-4.7"]
    weak = list(WEAK)

    def rate(df, col="dsr"):
        return float(df[col].mean()) if len(df) else float("nan")

    if not e1.empty:
        t1 = e1_core[e1_core["triggered"]]
        fr, wk = t1[t1.model.isin(frontier)], t1[t1.model.isin(weak)]
        for mo in ("timeout_post", "http500_post", "timeout_late", "duplicate_delivery"):
            H[f"fr_{mo}"] = rate(fr[fr["mode"] == mo])
            H[f"wk_{mo}"] = rate(wk[wk["mode"] == mo])
        tp = e1[e1["triggered"] & (e1["mode"] == "partial_timeout")]
        H["fr_partial"] = rate(tp[tp.model.isin(frontier)])
        H["wk_partial"] = rate(tp[tp.model.isin(weak)])
        pre = t1[t1["mode"].isin(PRE)]
        H["pre_ts"] = rate(pre, "TS")
        H["pre_dsr"] = rate(pre)
        H["explicit_dsr"] = rate(t1[t1["mode"].isin(BENIGN)])
        H["none_dsr"] = rate(e1_core[e1_core["mode"] == "none"])
        c = e1[e1["triggered"] & e1["mode"].isin(COMMITTED)]
        for cl, name in (("keyable", "keyable"), ("idempotent", "idem"), ("non-idem/strong", "strong"),
                         ("non-idem/eventual", "eventual"), ("non-idem/unverifiable", "unver")):
            H[f"fr_c_{name}"] = rate(c[(c.contract == cl) & c.model.isin(frontier)])
            H[f"all_c_{name}"] = rate(c[c.contract == cl])
        ks = c[c.contract == "keyable"]
        H["keyable_keyuse_fr"] = rate(ks[ks.model.isin(frontier)], "key_used")
        H["keyable_keyuse_wk"] = rate(ks[ks.model.isin(weak)], "key_used")
        cf = c[c.model.isin(frontier)]
        for cl, name in (("non-idem/unverifiable", "unver"), ("non-idem/strong", "strong"), ("non-idem/eventual", "eventual")):
            g = cf[cf.contract == cl]
            H[f"fr_c_{name}_escalate"] = float((g["category"] == "escalate").mean()) if len(g) else float("nan")
            H[f"fr_c_{name}_verify"] = float(g["category"].isin(["verify_then_retry", "verify_then_skip"]).mean()) \
                if len(g) else float("nan")
    if not e2.empty:
        f = e2[e2["triggered"] & (e2["mode"] != "none")]
        for p in POLICY_ORDER:
            g = f[f.policy == p]
            H[f"e2_{p}_eos"] = rate(g, "EOS")
            H[f"e2_{p}_dsr"] = rate(g)
            H[f"e2_{p}_ts"] = rate(g, "TS")
            H[f"e2_{p}_calls"] = float(g["n_calls"].mean()) if len(g) else float("nan")
            H[f"e2_{p}_human"] = float(g["human_minutes"].mean()) if len(g) else float("nan")
            H[f"e2_{p}_minutes"] = float(g["virtual_s"].mean() / 60) if len(g) else float("nan")
        for m in order(f["model"], MODEL_ORDER):
            for p in ("vanilla", "aware", "guard", "oracle", "sdk_retry3"):
                g = f[(f.model == m) & (f.policy == p)]
                H[f"e2_{m}_{p}_eos"] = rate(g, "EOS")
                H[f"e2_{m}_{p}_ts"] = rate(g, "TS")
        # Complacency check: does the guard's promise to verify displace the model's own caution?
        oo = f[(f.policy == "outcome_oracle") & (f["dup_exec"] > 0)]
        H["oo_dup_redeliv_share"] = float((oo["mode"] == "duplicate_delivery").mean()) if len(oo) else float("nan")
        H["oo_dup_n"] = float(len(oo))
        late = f[(f["mode"] == "timeout_late") & (f.model == "gpt-6-sol")]
        for p in ("vanilla", "guard"):
            g = late[late.policy == p]
            H[f"cmp_{p}_dsr"] = rate(g)
            H[f"cmp_{p}_vretry"] = float((g["category"] == "verify_then_retry").mean()) if len(g) else float("nan")
            H[f"cmp_{p}_escal"] = float((g["category"] == "escalate").mean()) if len(g) else float("nan")
    if not e2k.empty:
        k = e2k[e2k["triggered"] & (e2k["mode"] != "none")]
        for p in ("vanilla", "guard"):
            g = k[k.policy == p]
            H[f"e2k_{p}_eos"] = rate(g, "EOS")
            H[f"e2k_{p}_dsr"] = rate(g)
            H[f"e2k_{p}_late"] = rate(g[g["mode"] == "timeout_late"])
            H[f"e2k_{p}_redeliv"] = rate(g[g["mode"] == "duplicate_delivery"])
        H["e2k_vanilla_keyuse"] = rate(k[(k.policy == "vanilla") & k["key_used"].notna()], "key_used")
        if not e2.empty:
            f = e2[e2["triggered"] & (e2["mode"] != "none")]
            for p in ("vanilla", "guard"):
                g = f[f.policy == p]
                H[f"e2n_{p}_late"] = rate(g[g["mode"] == "timeout_late"])
                H[f"e2n_{p}_redeliv"] = rate(g[g["mode"] == "duplicate_delivery"])
    if not e3.empty:
        f = e3[e3["triggered"] & (e3["mode"] != "none")]
        for h in ("minimal", "copilot", "hermes", "codex"):
            for p in ("vanilla", "guard"):
                g = f[(f.harness == h) & (f.policy == p)]
                H[f"e3_{h}_{p}_dsr"] = rate(g)
                H[f"e3_{h}_{p}_eos"] = rate(g, "EOS")
                H[f"e3_{h}_{p}_ts"] = rate(g, "TS")
            g = e3[(e3.harness == h)]
            H[f"e3_{h}_tokens_k"] = float((g["in_tok"] + g["out_tok"]).mean() / 1000) if len(g) else float("nan")
    numbers["headline"] = {k: (round(v, 4) if isinstance(v, float) and v == v else None) for k, v in H.items()}


def fmt_p(p) -> str:
    """p-value for LaTeX math mode."""
    if p is None or p != p:
        return "--"
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        mant = round(p / 10 ** e)
        if mant >= 10:
            mant, e = 1, e + 1
        return f"{mant:.0f}\\times 10^{{{e}}}"
    return f"{p:.3f}" if p < 0.05 else f"{p:.2f}"


def write_macros(numbers: dict, out: Path) -> None:
    def pct(x):
        return "--" if x is None or x != x else f"{100 * x:.0f}"

    def pct1(x):
        return "--" if x is None or x != x else f"{100 * x:.1f}"

    H = numbers.get("headline", {})
    m = {}
    excl = numbers.get("pooled_exclude") or []
    m["NModels"] = NUMBER_WORDS[len(MODEL_ORDER)]
    weak_pooled = [w for w in WEAK if w not in excl]
    weak_names = [rf"\texttt{{{w}}}" for w in weak_pooled]
    m["WeakList"] = (", ".join(weak_names[:-1]) + " and " + weak_names[-1]) if len(weak_names) > 1 else weak_names[0]
    m["GptFourOneCoverage"] = f"{100 * (numbers.get('gpt41_coverage') or 0):.0f}"
    m["PooledNote"] = ((r"Pooled statistics exclude \texttt{gpt-4.1}, which is served from a separate low quota and "
                        rf"completed only {100 * (numbers.get('gpt41_coverage') or 0):.0f}\% of its E1 design; it is "
                        r"shown per model with this caveat.") if "gpt-4.1" in excl else "")
    m["NEpisodesAll"] = f"{numbers.get('n_total', 0):,}"
    m["NEOne"] = f"{numbers.get('e1_n', 0):,}"
    m["NETwo"] = f"{numbers.get('e2_n', 0):,}"
    m["NEThree"] = f"{numbers.get('e3_n', 0):,}"
    m["NEFour"] = f"{numbers.get('e4_n', 0):,}"
    m["FrontierPostDSR"] = pct1(H.get("fr_timeout_post"))
    m["FrontierHttpPostDSR"] = pct(H.get("fr_http500_post"))
    m["FrontierLateDSR"] = pct(H.get("fr_timeout_late"))
    m["FrontierRedelivDSR"] = pct(H.get("fr_duplicate_delivery"))
    m["FrontierPartialDSR"] = pct(H.get("fr_partial"))
    m["WeakPostDSR"] = pct(H.get("wk_timeout_post"))
    m["WeakHttpPostDSR"] = pct(H.get("wk_http500_post"))
    m["WeakLateDSR"] = pct(H.get("wk_timeout_late"))
    m["WeakPartialDSR"] = pct(H.get("wk_partial"))
    m["PreTS"] = pct(H.get("pre_ts"))
    m["PreDSR"] = pct1(H.get("pre_dsr"))
    m["ExplicitDSR"] = pct1(H.get("explicit_dsr"))
    m["NoneDSR"] = pct1(H.get("none_dsr"))
    m["FrontierKeyableDSR"] = pct1(H.get("fr_c_keyable"))
    m["FrontierIdemDSR"] = pct1(H.get("fr_c_idem"))
    m["FrontierStrongDSR"] = pct(H.get("fr_c_strong"))
    m["FrontierEventualDSR"] = pct(H.get("fr_c_eventual"))
    m["FrontierUnverDSR"] = pct(H.get("fr_c_unver"))
    m["KeyUseFrontier"] = pct1(H.get("keyable_keyuse_fr"))
    m["OutcomeOracleRedelivShare"] = pct(H.get("oo_dup_redeliv_share"))
    m["OutcomeOracleDupN"] = f"{int(H['oo_dup_n'])}" if H.get("oo_dup_n") is not None else "--"
    m["KeyUseWeak"] = pct1(H.get("keyable_keyuse_wk"))
    m["UnverEscalate"] = pct(H.get("fr_c_unver_escalate"))
    m["StrongVerify"] = pct(H.get("fr_c_strong_verify"))
    m["EventualVerify"] = pct(H.get("fr_c_eventual_verify"))
    for p in ("vanilla", "guard"):
        tag = p.capitalize()
        m[f"Cmp{tag}DSR"] = pct(H.get(f"cmp_{p}_dsr"))
        m[f"Cmp{tag}VRetry"] = pct(H.get(f"cmp_{p}_vretry"))
        m[f"Cmp{tag}Escal"] = pct(H.get(f"cmp_{p}_escal"))
    h1 = numbers.get("H1_mixed") or {}
    g1 = numbers.get("H1_gee") or {}
    m["HOneOR"] = f"{h1['or']:.1f}" if "or" in h1 else "--"
    m["HOneLo"] = f"{h1['or_lo']:.1f}" if "or_lo" in h1 else "--"
    m["HOneHi"] = f"{h1['or_hi']:.1f}" if "or_hi" in h1 else "--"
    m["HOneGeeOR"] = f"{g1['or']:.1f}" if "or" in g1 else "--"
    m["HOneGeeLo"] = f"{g1['or_lo']:.1f}" if "or_lo" in g1 else "--"
    m["HOneGeeHi"] = f"{g1['or_hi']:.1f}" if "or_hi" in g1 else "--"
    m["HOneGeeP"] = fmt_p(g1.get("p"))
    h3 = numbers.get("H3") or {}
    m["HThreeEventual"] = pct1(h3.get("dsr_verified_eventual"))
    m["HThreeStrong"] = pct1(h3.get("dsr_verified_strong"))
    m["HThreeN"] = f"{h3.get('n', 0):,}"
    m["HThreeGeeP"] = fmt_p(h3.get("p_gee"))
    h4 = numbers.get("H4") or {}
    m["HFourIrr"] = pct1(h4.get("blind_irreversible"))
    m["HFourRev"] = pct1(h4.get("blind_reversible"))
    m["HFourDiff"] = f"{100 * h4['diff']:+.1f}" if "diff" in h4 else "--"
    m["HFourLoNinety"] = f"{100 * h4['lo90']:+.1f}" if "lo90" in h4 else "--"
    m["HFourHiNinety"] = f"{100 * h4['hi90']:+.1f}" if "hi90" in h4 else "--"
    m["HFourP"] = fmt_p(h4.get("p_diff"))
    m["HFourVerdict"] = ("equivalence within $\\pm$5\\,pp is established" if h4.get("equivalent_5pp")
                         else "equivalence within $\\pm$5\\,pp cannot be established")
    h5 = numbers.get("H5") or {}
    m["CompletedGivenDup"] = pct(h5.get("completed_given_dup"))
    m["CompletedLo"] = pct((h5.get("completed_ci") or [None, None])[0])
    m["CompletedHi"] = pct((h5.get("completed_ci") or [None, None])[1])
    m["OverclaimGivenDup"] = pct(h5.get("overclaim_given_dup"))
    m["OverclaimLo"] = pct((h5.get("overclaim_ci") or [None, None])[0])
    m["OverclaimHi"] = pct((h5.get("overclaim_ci") or [None, None])[1])
    m["CompleteNoUncertain"] = pct(h5.get("complete_no_uncertainty_given_dup"))
    m["CompleteNoUncertainLo"] = pct((h5.get("complete_no_uncertainty_ci") or [None, None])[0])
    m["CompleteNoUncertainHi"] = pct((h5.get("complete_no_uncertainty_ci") or [None, None])[1])
    m["NDup"] = f"{h5.get('n', 0):,}"
    for tag, key in (("One", "shapley_e1"), ("Three", "shapley_e3")):
        s = numbers.get(key) or {}
        tot = s.get("total") or float("nan")
        for f in ("contract", "mode", "model", "harness"):
            m[f"Shap{f.capitalize()}{tag}"] = f"{s[f]:.2f}" if f in s else "--"
            m[f"Share{f.capitalize()}{tag}"] = pct(s[f] / tot) if f in s and tot == tot and tot else "--"
        m[f"ShapTotal{tag}"] = f"{tot:.2f}" if tot == tot else "--"
        strata = numbers.get(f"{key}_strata") or {}
        for st, stag in (("resolvable", "Res"), ("unresolvable", "Unres")):
            r = strata.get(st) or {}
            t_ = r.get("total") or float("nan")
            for f in ("contract", "mode", "model", "harness"):
                m[f"Share{f.capitalize()}{tag}{stag}"] = pct(r[f] / t_) if f in r and t_ == t_ and t_ else "--"
            m[f"ShapTotal{tag}{stag}"] = f"{t_:.2f}" if t_ == t_ else "--"
    oe = numbers.get("obs_equivalence") or {}
    m["ObsTVD"] = f"{oe['mean_tvd']:.2f}" if oe.get("mean_tvd") is not None else "--"
    m["ObsNullTVD"] = f"{oe['mean_null_tvd']:.2f}" if oe.get("mean_null_tvd") is not None else "--"
    m["ObsMinPerm"] = f"{oe['max_p_perm_min']:.2f}" if oe.get("max_p_perm_min") is not None else "--"
    m["ObsCrossAgree"] = pct(oe.get("cross_world_agreement"))
    m["ObsSameAgree"] = pct(oe.get("same_world_agreement"))
    m["ObsNCross"] = f"{oe.get('n_cross', 0):,}"
    m["ObsNSame"] = f"{oe.get('n_same', 0):,}"
    lw = numbers.get("late_wait") or {}
    for key, tag in (("quick_redo", "Quick"), ("waited_redo", "Waited"), ("no_redo", "NoRedo")):
        r = lw.get(key) or {}
        m[f"Late{tag}N"] = f"{r.get('n', 0):,}"
        m[f"Late{tag}Dup"] = pct(r.get("dup"))
        m[f"Late{tag}Eventual"] = pct(r.get("eventual_share"))
    e5 = numbers.get("e5") or {}
    words = {"0": "Zero", "60": "Sixty", "120": "OneTwenty", "300": "FiveMin", "900": "FifteenMin", "3600": "Hour"}

    def cond_tag(c: str) -> str:
        if c.startswith("wait") and c[4:] in words:
            return "Wait" + words[c[4:]]
        return "".join(x.capitalize() for x in re.split(r"[:_]", c))

    for panel, ptag in (("fixed", "Fixed"), ("tail", "Tail")):
        for c in ("vanilla", "wait0", "wait60", "wait120", "wait300", "wait900", "wait3600", "guard", "oracle",
                  "outcome_oracle", "keys:vanilla", "keys:guard"):
            for suffix in ("EOS", "DSR", "Min", "Ceiling"):
                m.setdefault(f"EFive{ptag}{cond_tag(c)}{suffix}", "--")
        for c, r in (e5.get(panel) or {}).items():
            ctag = cond_tag(c)
            m[f"EFive{ptag}{ctag}EOS"] = pct(r["eos"][0])
            m[f"EFive{ptag}{ctag}DSR"] = pct(r["dsr"][0])
            m[f"EFive{ptag}{ctag}Min"] = f"{r['latency_min']:.1f}"
            if "ceiling" in r:
                m[f"EFive{ptag}{ctag}Ceiling"] = pct(r["ceiling"])
    m["EFiveTailMedian"] = f"{e5['tail_delay_median'] / 60:.0f}" if e5.get("tail_delay_median") else "--"
    ks = numbers.get("key_stability") or {}
    for kind, tag in (("same", "Reused"), ("no_first", "NoFirst"), ("changed", "Changed")):
        r = ks.get(kind) or {}
        m[f"Key{tag}N"] = f"{r.get('n', 0):,}"
        m[f"Key{tag}Dup"] = pct(r.get("dup"))
    m["KeyDupsNotReused"] = pct(ks.get("share_of_dups_not_reused"))
    m["NEFive"] = f"{numbers.get('e5_n', 0):,}"
    e6 = numbers.get("e6") or {}
    for key, tag in (("not executed (TS)", "Pre"), ("read-back resolves", "Res"), ("late commit", "Late"),
                     ("redelivery", "Redeliv"), ("no fault (EOS)", "None")):
        r = e6.get(key) or {}
        m[f"ESix{tag}Default"] = pct(r.get("default"))
        m[f"ESix{tag}Plain"] = pct(r.get("plain"))
        m[f"ESix{tag}Diff"] = f"{r['diff_pp']:+.1f}" if "diff_pp" in r else "--"
        m[f"ESix{tag}P"] = fmt_p(r.get("p"))
        m[f"ESix{tag}VerifyDefault"] = pct(r.get("verify_default"))
        m[f"ESix{tag}VerifyPlain"] = pct(r.get("verify_plain"))
    m["NESix"] = f"{numbers.get('e6_n', 0):,}"
    m["ESixSummary"] = numbers.get("e6_summary", "")
    for p in POLICY_ORDER:
        tag = p.replace("_", "").replace("3", "three").capitalize()
        m[f"ETwo{tag}EOS"] = pct(H.get(f"e2_{p}_eos"))
        m[f"ETwo{tag}DSR"] = pct(H.get(f"e2_{p}_dsr"))
        m[f"ETwo{tag}TS"] = pct(H.get(f"e2_{p}_ts"))
        m[f"ETwo{tag}Calls"] = f"{H[f'e2_{p}_calls']:.1f}" if H.get(f"e2_{p}_calls") is not None else "--"
        m[f"ETwo{tag}Human"] = f"{H[f'e2_{p}_human']:.1f}" if H.get(f"e2_{p}_human") is not None else "--"
    for mm, tag in (("gemini-3.8-flash", "Gemini"), ("claude-opus-5.5", "Claude"), ("gpt-6-sol", "GptSix")):
        for p in ("vanilla", "aware", "guard", "oracle", "sdk_retry3"):
            ptag = p.replace("_", "").replace("3", "three").capitalize()
            m[f"ETwo{tag}{ptag}EOS"] = pct(H.get(f"e2_{mm}_{p}_eos"))
            m[f"ETwo{tag}{ptag}TS"] = pct(H.get(f"e2_{mm}_{p}_ts"))
    for p in ("vanilla", "guard"):
        tag = p.capitalize()
        m[f"EKeys{tag}EOS"] = pct(H.get(f"e2k_{p}_eos"))
        m[f"EKeys{tag}DSR"] = pct1(H.get(f"e2k_{p}_dsr")) if 0 < (H.get(f"e2k_{p}_dsr") or 0) < 0.01 else \
            pct(H.get(f"e2k_{p}_dsr"))
        m[f"EKeys{tag}Late"] = pct(H.get(f"e2k_{p}_late"))
        m[f"EKeys{tag}Redeliv"] = pct(H.get(f"e2k_{p}_redeliv"))
        m[f"ENative{tag}Late"] = pct(H.get(f"e2n_{p}_late"))
        m[f"ENative{tag}Redeliv"] = pct(H.get(f"e2n_{p}_redeliv"))
    m["EKeysVanillaKeyUse"] = pct1(H.get("e2k_vanilla_keyuse"))
    for h in ("minimal", "copilot", "hermes", "codex"):
        tag = h.capitalize()
        for p in ("vanilla", "guard"):
            m[f"EThree{tag}{p.capitalize()}DSR"] = pct(H.get(f"e3_{h}_{p}_dsr"))
            m[f"EThree{tag}{p.capitalize()}EOS"] = pct(H.get(f"e3_{h}_{p}_eos"))
            m[f"EThree{tag}{p.capitalize()}TS"] = pct(H.get(f"e3_{h}_{p}_ts"))
        v = H.get(f"e3_{h}_tokens_k")
        m[f"EThree{tag}TokensK"] = f"{v:.0f}" if v is not None else "--"
    tr = numbers.get("test_retest") or {}
    for h, r in (numbers.get("e3k") or {}).items():
        tag = h.capitalize()
        for key, val in r.items():
            if key.endswith("_n"):
                continue
            name = "".join(part.capitalize() for part in key.replace("dsr", "DSR").replace("eos", "EOS").split("_"))
            m[f"EThreeK{tag}{name}"] = pct(val)
    m["RetestN"] = f"{tr.get('n_pairs', 0):,}"
    m["RetestAgree"] = pct((tr.get("dsr") or {}).get("agreement"))
    m["RetestKappa"] = f"{(tr.get('dsr') or {}).get('kappa') or 0:.2f}"
    m["GuardSummary"] = numbers.get("guard_summary", "")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "generated"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}
    appendix_artifacts(out)
    e1 = load_many(["e1", "e1_gpt41", "e1s", "e1s_gpt41"])
    e1_core = load_many(["e1", "e1_gpt41"])
    # gpt-4.1 runs on a separate quota; if its coverage is incomplete it is shown per model but kept
    # out of pooled statistics so that a template-skewed subset cannot bias them.
    designed = 692
    cov = float(len(e1_core[e1_core.model == "gpt-4.1"])) / designed if not e1_core.empty else 0.0
    numbers["gpt41_coverage"] = round(cov, 3)
    pooled_exclude = [] if cov >= 0.9 else ["gpt-4.1"]
    numbers["pooled_exclude"] = pooled_exclude
    e1_pool = e1[~e1.model.isin(pooled_exclude)] if not e1.empty else e1
    e1_core_pool = e1_core[~e1_core.model.isin(pooled_exclude)] if not e1_core.empty else e1_core
    if not e1.empty:
        e1_tables(e1_core, out, numbers, e1)
        e1_contract(e1, out, numbers)
        behaviour(e1_core, out, numbers)
        hypotheses(e1_pool, numbers)
        shapley(e1_pool, ["contract", "mode", "model"], "e1", numbers, out)
        late_wait_analysis(numbers, exclude=tuple(pooled_exclude))
        numbers["e1_n"] = int(len(e1))
        fm = ["gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol", "claude-opus-5.5", "gemini-3.8-flash", "grok-4.7"]
        f = e1_core[e1_core.model.isin(fm) & e1_core["triggered"]]
        numbers["frontier"] = {mo: round(float(f[f["mode"] == mo]["dsr"].mean()), 4)
                               for mo in ("timeout_post", "http500_post", "timeout_late", "duplicate_delivery",
                                          "partial_timeout") if (f["mode"] == mo).any()}
    e2 = load_many(["e2"])
    e2k = load_many(["e2k"])
    if not e2.empty:
        e2_tables(e2, out, numbers)
        numbers["e2_n"] = int(len(e2) + len(e2k))
        if not e1.empty:
            test_retest(e1_core, e2, numbers)
            obs_equivalence(e1_core_pool, e2, numbers)
        if not e2k.empty:
            e2k_tables(e2, e2k, out, numbers)
    e5, e5k = load_many(["e5"]), load_many(["e5k"])
    if not e5.empty:
        e5_tables(e2, e2k, e5, e5k, out, numbers)
        key_stability(numbers)
        numbers["e5_n"] = int(len(e5) + len(e5k))
    e6 = load_many(["e6"])
    if not e6.empty:
        e6_tables(e1_core_pool, e6, out, numbers)
        numbers["e6_n"] = int(len(e6))
    e3 = load_many(["e3_minimal", "e3_copilot", "e3_hermes", "e3_codex"])
    e3k = load_many(["e3k_minimal", "e3k_copilot", "e3k_hermes", "e3k_codex"])
    if not e3.empty:
        e3_tables(e3, out, numbers)
        numbers["e3_n"] = int(len(e3) + len(e3k))
        if not e3k.empty:
            e3k_tables(e3, e3k, out, numbers)
    e4 = {"base": e2 if not e2.empty else None, "docs": load_many(["e4_docs"]), "para": load_many(["e4_para"]),
          "nohuman": load_many(["e4_nohuman"]), "ablate": load_many(["e4_ablate"])}
    if any(v is not None and not v.empty for k, v in e4.items() if k != "base"):
        e4_tables(e4, out, numbers)
        numbers["e4_n"] = int(sum(len(v) for k, v in e4.items() if k != "base" and v is not None))
    numbers["n_total"] = int(sum(numbers.get(k, 0) for k in ("e1_n", "e2_n", "e3_n", "e4_n", "e5_n", "e6_n")))
    headline(e1_pool, e1_core_pool, e2, e2k, e3, numbers)
    H = numbers["headline"]

    def p(x):
        return "--" if x is None else f"{100 * x:.0f}"

    e3k_eos = [r.get("keys_guard_eos") for r in (numbers.get("e3k") or {}).values() if r.get("keys_guard_eos") is not None]
    e3k_van = [r.get("keys_vanilla_dsr") for r in (numbers.get("e3k") or {}).values() if r.get("keys_vanilla_dsr") is not None]
    harness_clause = (f" The same holds in every harness: with keys everywhere, the duplicate rate of a shared model is at "
                      f"most {p(max(e3k_van))}\\% even without the guard, and exactly-once success is at least "
                      f"{p(min(e3k_eos))}\\% with it." if e3k_eos and e3k_van else "")
    numbers["guard_summary"] = (
        f"Transparent client-side retries cut exactly-once success from {p(H.get('e2_vanilla_eos'))}\\% to "
        f"{p(H.get('e2_sdk_retry3_eos'))}\\%. In the native contract even an outcome oracle that sees in-flight "
        f"requests reaches only {p(H.get('e2_outcome_oracle_eos'))}\\%, because redelivery on key-less writes is beyond "
        f"any client-side policy; with keys on every write and a guard that attaches them, exactly-once success "
        f"reaches {p(H.get('e2k_guard_eos'))}\\%." + harness_clause)
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")
    write_macros(numbers, out)
    brief = {k: numbers[k] for k in numbers if k in ("frontier", "e1_contract_frontier", "H1_logit", "H3", "H4", "H5",
                                                   "shapley_e1", "shapley_e3", "e2_mcnemar_guard_vs", "n_total",
                                                   "obs_equivalence_first_action_p")}
    print(json.dumps(brief, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
