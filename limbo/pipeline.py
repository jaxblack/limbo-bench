"""Sequential experiment pipeline: waits for a running process, then runs the
remaining designs one after another. Logs to results/pipeline.log.

    python -m limbo.pipeline --wait-pid 1234
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOG = RESULTS / "pipeline.log"

ALL_MODELS = ["gpt-6-sol", "gpt-6-astra", "gpt-5.6-sol", "claude-opus-5.5", "gemini-3.8-flash", "grok-4.7",
              "gpt-5.4-mini", "gpt-4.1"]
# gpt-4.1 sits on a separate, small "utility model" quota; it runs in its own slow lane.
MAIN = [m for m in ALL_MODELS if m != "gpt-4.1"]
# The E1 design's fault modes (fixed before later modes such as timeout_late_tail were added).
E1_MODES = ["none", "timeout_pre", "timeout_post", "timeout_late", "http500_pre", "http500_post", "http503_transient",
            "outage", "rate_limit", "schema_drift", "partial_timeout", "duplicate_delivery"]
E2_MODELS = ["claude-opus-5.5", "gpt-6-sol", "gemini-3.8-flash"]
E2_MODES = ["none", "timeout_pre", "timeout_post", "timeout_late", "http500_post", "partial_timeout",
            "duplicate_delivery", "http503_transient"]
E4_MODELS = ["claude-opus-5.5", "gpt-6-sol", "gemini-3.8-flash"]


def log(msg: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def archive_mode(experiment: str, mode: str, tag: str) -> int:
    """Move records of one fault mode aside so they are re-run with the current code."""
    path = RESULTS / experiment / "episodes.jsonl"
    if not path.exists():
        return 0
    keep, moved = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        (moved if (rec.get("spec") or {}).get("mode") == mode else keep).append(line)
    if moved:
        (RESULTS / experiment / f"episodes_{mode}_{tag}.jsonl").write_text("\n".join(moved) + "\n", encoding="utf-8")
        path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return len(moved)


def runner(*args: str, workers: int = 48) -> None:
    cmd = [sys.executable, "-m", "limbo.runner", *args, "--workers", str(workers)]
    exp = args[args.index("--out-name") + 1] if "--out-name" in args else args[args.index("--experiment") + 1]
    (RESULTS / exp).mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (RESULTS / exp / "COMMIT").write_text(commit, encoding="utf-8")
    log("run " + " ".join(args))
    t0 = time.time()
    with (RESULTS / exp / "stdout.txt").open("a", encoding="utf-8") as out:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"done rc={rc} in {time.time() - t0:.0f}s: {exp}")


def count_errors(out_name: str) -> int:
    path = RESULTS / out_name / "episodes.jsonl"
    if not path.exists():
        return 0
    latest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec["episode_id"]] = rec.get("stop_reason")
    return sum(1 for v in latest.values() if v in ("llm_error", "harness_error"))


def slow_lane() -> None:
    """gpt-4.1: few workers, and wait out quota windows between passes."""
    for attempt in range(16):
        runner("--experiment", "e1", "--out-name", "e1_gpt41", "--models", "gpt-4.1", "--indices", "0", "1",
               "--modes", *E1_MODES, workers=3)
        runner("--experiment", "e1s", "--out-name", "e1s_gpt41", "--models", "gpt-4.1", "--templates", "migration_log",
               "cross_post", "--indices", *map(str, range(2, 10)), "--modes", "timeout_pre", "timeout_post",
               "timeout_late", "http500_post", "partial_timeout", "duplicate_delivery", workers=3)
        errs = count_errors("e1_gpt41") + count_errors("e1s_gpt41")
        log(f"slow lane pass {attempt}: {errs} errored episodes remain")
        if errs == 0:
            break
        time.sleep(1500)


def p0_lane(lane: str) -> None:
    """P0 revision experiments (see PREREGISTRATION.md deviations 8-11)."""
    E5 = ["claude-opus-5.5", "gpt-6-sol", "gemini-3.8-flash"]
    CUE = ["gpt-6-astra", "gpt-6-sol", "claude-opus-5.5", "gemini-3.8-flash", "gpt-5.4-mini"]
    steps = {
        "A": [
            # True client-side upper bound on the full E2 design.
            ("--experiment", "e2", "--models", *E2_MODELS, "--policies", "outcome_oracle", "--modes", *E2_MODES,
             "--indices", "0"),
            # Waiting vs keys, fixed 90 s in-flight delay: the threshold is between 60 and 120 s.
            ("--experiment", "e5", "--models", *E5, "--policies", "wait0", "wait60", "wait120", "wait300",
             "--modes", "timeout_late", "--indices", "0"),
            # Waiting vs keys, heavy-tailed in-flight delay (log-uniform 40 s to 2 h).
            ("--experiment", "e5", "--models", *E5, "--policies", "vanilla", "wait0", "wait60", "wait300", "wait900",
             "wait3600", "guard", "outcome_oracle", "--modes", "timeout_late_tail", "--indices", "0"),
            ("--experiment", "e5k", "--contract", "keys_everywhere", "--models", *E5, "--policies", "vanilla", "guard",
             "--modes", "timeout_late_tail", "--indices", "0"),
        ],
        "B": [
            # Cue ablation: identical worlds, instructions without the closing "exactly once" sentence.
            ("--experiment", "e6", "--instruction-variant", "plain", "--models", *CUE, "--modes", "none",
             "timeout_pre", "timeout_post", "http500_post", "timeout_late", "partial_timeout", "duplicate_delivery",
             "--indices", "0"),
        ],
    }[lane]
    for attempt in (1, 2):
        log(f"p0 lane {lane} pass {attempt}")
        for args in steps:
            try:
                runner(*args, workers=24)
            except Exception as exc:
                log(f"p0 step failed: {exc}")
    log(f"p0 lane {lane} finished")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--slow-lane", action="store_true")
    ap.add_argument("--p0-lane", choices=["A", "B"], default=None)
    a = ap.parse_args()
    if a.p0_lane:
        p0_lane(a.p0_lane)
        return
    if a.slow_lane:
        slow_lane()
        log("slow lane finished")
        return
    if a.wait_pid:
        log(f"waiting for pid {a.wait_pid}")
        while pid_alive(a.wait_pid):
            time.sleep(20)
    steps = {
        "e1": lambda: runner("--experiment", "e1", "--models", *MAIN, "--indices", "0", "1", "--modes", *E1_MODES),
        "e1s": lambda: runner("--experiment", "e1s", "--models", *MAIN, "--templates", "migration_log", "cross_post",
                              "--indices", *map(str, range(2, 10)), "--modes", "timeout_pre", "timeout_post",
                              "timeout_late", "http500_post", "partial_timeout", "duplicate_delivery"),
        "e2": lambda: runner("--experiment", "e2", "--models", *E2_MODELS, "--policies", "vanilla", "aware", "reflect",
                             "sdk_retry3", "rules", "vbr", "guard", "oracle", "--modes", *E2_MODES, "--indices", "0"),
        "e2k": lambda: runner("--experiment", "e2k", "--contract", "keys_everywhere", "--models", *E2_MODELS,
                              "--policies", "vanilla", "guard", "--modes", *E2_MODES, "--indices", "0"),
        "e4_docs_none": lambda: runner("--experiment", "e4_docs", "--doc-variant", "no_consistency_docs", "--models",
                                       *E4_MODELS, "--modes", "timeout_post", "timeout_late", "http500_post",
                                       "--indices", "0", "--max-focals", "2"),
        "e4_docs_explicit": lambda: runner("--experiment", "e4_docs", "--doc-variant", "explicit", "--models",
                                           *E4_MODELS, "--modes", "timeout_post", "timeout_late", "http500_post",
                                           "--indices", "0", "--max-focals", "2"),
        "e4_para1": lambda: runner("--experiment", "e4_para", "--paraphrase", "1", "--models", *E4_MODELS, "--modes",
                                   "timeout_post", "timeout_late", "http500_post", "--indices", "0", "--max-focals", "2"),
        "e4_para2": lambda: runner("--experiment", "e4_para", "--paraphrase", "2", "--models", *E4_MODELS, "--modes",
                                   "timeout_post", "timeout_late", "http500_post", "--indices", "0", "--max-focals", "2"),
        "e4_nohuman": lambda: runner("--experiment", "e4_nohuman", "--no-human", "--models", *E4_MODELS, "--policies",
                                     "vanilla", "guard", "--modes", "timeout_post", "timeout_late", "--indices", "0",
                                     "--max-focals", "2"),
        "e4_ablate": lambda: runner("--experiment", "e4_ablate", "--models", "gemini-3.8-flash",
                                    "--policies", "guard-no-key", "guard-no-consistency", "guard-no-block",
                                    "guard-no-annotate", "--modes", "timeout_post", "timeout_late", "http500_post",
                                    "partial_timeout", "duplicate_delivery", "--indices", "0", "--max-focals", "2"),
    }
    for attempt in (1, 2):  # second pass only re-runs episodes that errored or were interrupted
        log(f"pass {attempt}")
        for name, step in steps.items():
            if name in a.skip:
                continue
            try:
                step()
            except Exception as exc:  # keep the pipeline going
                log(f"step {name} failed: {exc}")
    log("pipeline finished")


if __name__ == "__main__":
    main()
