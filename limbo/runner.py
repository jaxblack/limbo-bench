"""Experiment runner: expands a design into episodes, runs them in parallel,
and appends one JSON record per episode. Re-running skips finished episodes.

    python -m limbo.runner --experiment pilot --models gpt-6-sol claude-opus-5.5 \
        --policies vanilla --modes none timeout_pre timeout_post --indices 0 --workers 12
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limbo.agent import EpisodeSpec, run_episode  # noqa: E402
from limbo.runtime import FAULT_MODES  # noqa: E402
from limbo.services import build_tools  # noqa: E402
from limbo.tasks import TEMPLATES, make_task  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
_TOOLS = build_tools()


def compatible(mode: str, tool: str, batch: bool) -> bool:
    if mode == "partial_timeout":
        return batch
    if mode == "schema_drift":
        return _TOOLS[tool].contract.drift is not None
    return True


def expand(experiment: str, templates: Iterable[str], indices: Iterable[int], modes: Iterable[str],
           models: Iterable[str], policies: Iterable[str], replicates: int = 1, focal_filter: str | None = None,
           max_focals: int | None = None, **extra) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    for tpl in templates:
        for i in indices:
            task = make_task(tpl, i)
            focals = [f for f in task.focals if not focal_filter or focal_filter in f.label]
            for mode in modes:
                chosen = focals[:1] if mode == "none" else [f for f in focals if compatible(mode, f.tool, f.batch)]
                if max_focals and len(chosen) > max_focals:
                    step = (len(chosen) - 1) / (max_focals - 1) if max_focals > 1 else 0
                    chosen = [chosen[round(k * step)] for k in range(max_focals)]
                for f in chosen:
                    for model in models:
                        for pol in policies:
                            for rep in range(replicates):
                                specs.append(EpisodeSpec(template=tpl, index=i, focal=f.label, mode=mode, model=model,
                                                         policy=pol, replicate=rep, experiment=experiment, **extra))
    return specs

def done_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("stop_reason") not in ("llm_error", "harness_error"):
                ids.add(rec["episode_id"])
    return ids


def run(specs: list[EpisodeSpec], out_dir: Path, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "episodes.jsonl"
    finished = done_ids(out)
    todo = [s for s in specs if s.episode_id not in finished]
    lock = threading.Lock()
    log = out_dir / "progress.log"
    t0 = time.time()
    counts = {"ok": 0, "err": 0}

    def work(spec: EpisodeSpec) -> dict:
        if spec.harness != "minimal":
            from limbo.harness import run_harness_episode
            return run_harness_episode(spec)
        return run_episode(spec)

    with log.open("a", encoding="utf-8") as lf:
        lf.write(f"{time.strftime('%H:%M:%S')} start: {len(todo)} to run ({len(specs) - len(todo)} already done)\n")
        lf.flush()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, s): s for s in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # defensive: run_episode already catches
                    rec = {"episode_id": s.episode_id, "spec": s.__dict__, "stop_reason": "harness_error", "error": str(exc)}
                bad = rec.get("stop_reason") in ("llm_error", "harness_error")
                with lock:
                    counts["err" if bad else "ok"] += 1
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                    n = counts["ok"] + counts["err"]
                    if n % 10 == 0 or bad or n == len(todo):
                        g = rec.get("grade") or {}
                        lf.write(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} ok={counts['ok']} err={counts['err']} "
                                 f"elapsed={time.time() - t0:.0f}s last={s.model}/{s.policy}/{s.template}/{s.focal}/{s.mode} "
                                 f"EOS={g.get('EOS')} dup={g.get('dup_executed')} {('ERR ' + str(rec.get('error'))[:160]) if bad else ''}\n")
                        lf.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--templates", nargs="*", default=list(TEMPLATES))
    ap.add_argument("--indices", nargs="*", type=int, default=[0])
    ap.add_argument("--modes", nargs="*", default=list(FAULT_MODES))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--policies", nargs="*", default=["vanilla"])
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--focal-filter", default=None)
    ap.add_argument("--max-focals", type=int, default=None)
    ap.add_argument("--doc-variant", default="neutral")
    ap.add_argument("--paraphrase", type=int, default=0)
    ap.add_argument("--no-human", action="store_true")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--harness", default="minimal", choices=["minimal", "copilot", "hermes", "codex"])
    ap.add_argument("--contract", default="native", choices=["native", "keys_everywhere"])
    ap.add_argument("--instruction-variant", default="default", choices=["default", "plain"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-name", default=None, help="results directory (default: experiment name)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    extra = {"doc_variant": a.doc_variant, "paraphrase": a.paraphrase, "human_available": not a.no_human,
             "reasoning_effort": a.reasoning_effort, "harness": a.harness, "contract": a.contract,
             "instruction_variant": a.instruction_variant}
    specs = expand(a.experiment, a.templates, a.indices, a.modes, a.models, a.policies, a.replicates,
                   a.focal_filter, a.max_focals, **extra)
    print(f"{len(specs)} episodes in design")
    if a.dry_run:
        return
    run(specs, RESULTS / (a.out_name or a.experiment), a.workers)


if __name__ == "__main__":
    main()
