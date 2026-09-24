"""Extract short, readable trajectories for the paper's case studies.

    python -m limbo.cases --experiment e1 --out paper/generated/cases.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# (label, predicate over a record) — the first matching episode of each kind is shown.
CASES = [
    ("Verification defeated by a stale read path",
     lambda r: r["spec"]["mode"] == "timeout_post" and r["focal"]["verification"] == "eventual"
     and r["grade"]["dup_executed"] > 0 and r["behavior"]["category"] == "verify_then_retry"
     and r["spec"]["model"] in ("claude-opus-5.5", "gpt-6-sol", "gpt-5.6-sol")),
    ("A late commit lands after a careful retry",
     lambda r: r["spec"]["mode"] == "timeout_late" and r["focal"]["verification"] == "strong"
     and r["grade"]["dup_executed"] > 0 and r["behavior"]["category"] == "verify_then_retry"
     and r["spec"]["model"] in ("claude-opus-5.5", "gpt-6-sol", "gpt-6-astra")),
    ("Proactive idempotency key makes a blind retry safe",
     lambda r: r["spec"]["mode"] in ("timeout_post", "timeout_late") and r["behavior"]["category"] == "same_key_retry"
     and r["grade"]["EOS"]),
    ("Escalation instead of guessing",
     lambda r: r["spec"]["mode"] == "timeout_late" and r["behavior"]["category"] == "escalate" and r["grade"]["EOS"]),
    ("Misleading 500 treated as a clean failure",
     lambda r: r["spec"]["mode"] == "http500_post" and r["behavior"]["category"] == "blind_retry"
     and r["grade"]["dup_executed"] > 0),
]


def esc(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("$", r"\$").replace("{", r"\{").replace("}", r"\}").replace("~", r"\~{}")
            .replace("^", r"\^{}"))


def short(obj, n=110) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 3] + "..."


def render(label: str, r: dict) -> str:
    s = r["spec"]
    lines = [rf"\paragraph{{{esc(label)}.}} \texttt{{{esc(s['model'])}}}, template \texttt{{{esc(s['template'])}}}, "
             rf"focal write \texttt{{{esc(s['focal'])}}}, fault \texttt{{{esc(s['mode'])}}} "
             rf"(outcome: {'duplicate' if r['grade']['dup_executed'] else 'exactly once'}; agent reported "
             rf"\texttt{{{esc(str((r.get('finish') or {}).get('status', 'nothing')))}}})."]
    lines.append(r"\begin{small}\begin{enumerate}[leftmargin=1.6em,itemsep=0pt,topsep=2pt]")
    for c in r["agent_calls"][:14]:
        args = {k: v for k, v in (c.get("args") or {}).items() if k not in ("body",)}
        res = "ok" if c["ok"] else (c.get("error_type") or "error") + (f" {c['status']}" if c.get("status") else "")
        lines.append(rf"\item \texttt{{{esc(c['name'])}}} \texttt{{{esc(short(args, 95))}}} $\rightarrow$ {esc(res)}")
    late = [e for e in r["events"] if e.get("source") == "late_commit"]
    if late:
        lines.append(rf"\item[] \emph{{(the in-flight original commits at t={late[0]['t_start']:.0f}\,s)}}")
    lines.append(r"\end{enumerate}\end{small}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", nargs="+", default=["e1"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "generated" / "cases.tex"))
    a = ap.parse_args()
    found: dict[str, dict] = {}
    for exp in a.experiment:
        p = RESULTS / exp / "episodes.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("stop_reason") in ("llm_error", "harness_error") or not r.get("fault_triggered"):
                continue
            for label, pred in CASES:
                if label not in found:
                    try:
                        if pred(r):
                            found[label] = r
                    except (KeyError, TypeError):
                        pass
            if len(found) == len(CASES):
                break
    body = "\n\n".join(render(lbl, found[lbl]) for lbl, _ in CASES if lbl in found)
    Path(a.out).write_text(body + "\n", encoding="utf-8")
    print(f"wrote {len(found)} cases to {a.out}")


if __name__ == "__main__":
    main()
