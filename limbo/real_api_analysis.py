"""Descriptive, offline analysis of the exploratory real GitHub Issues cases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .analysis import wilson
from .real_api import MODES, OPERATIONS, OUTPUT_NAME, RESULTS

REFERENCE_MODELS = ("gpt-6-sol", "gpt-5.4-mini")
MODE_LABELS = {"none": "no fault", "timeout_pre": "lost request", "timeout_post": "lost acknowledgement",
               "http500_pre": "500 before write", "http500_post": "500 after write",
               "timeout_late": "late commit", "duplicate_delivery": "redelivery"}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"real API traces not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [record["episode_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode IDs in real API trace; refusing to double-count")
    return records


def check_design(records: list[dict], models: tuple[str, ...], replicates: int,
                 operations: tuple[str, ...] = OPERATIONS, modes: tuple[str, ...] = MODES) -> None:
    expected = {(model, operation, mode, i) for model in models for operation in operations
                for mode in modes for i in range(replicates)}
    actual = {(r["spec"]["model"], r["spec"]["operation"], r["spec"]["mode"], r["spec"]["replicate"])
              for r in records}
    if actual != expected:
        raise ValueError(f"real API matrix incomplete or contaminated: missing={len(expected - actual)}, "
                         f"unexpected={len(actual - expected)}")


def rate(records: list[dict], key: str) -> dict:
    n = len(records)
    if not n:
        raise ValueError(f"cannot calculate {key} on an empty cell")
    observed = sum(bool(record["grade"][key]) if key != "dup_executed" else
                   record["grade"]["dup_executed"] > 0 for record in records)
    estimate = observed / n
    lo, hi = wilson(estimate, n)
    return {"rate": round(estimate, 4), "events": observed, "n": n,
            "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no real API records")
    for record in records:
        grade = record.get("grade")
        if grade is None:
            if record["stop_reason"] != "infrastructure_error":
                raise ValueError("missing persisted-effect grade on a non-infrastructure case")
            continue
        count, ids = grade["n_committed"], grade["committed_ids"]
        if (type(count) is not int or count < 0 or not isinstance(ids, list) or len(ids) != len(set(ids))
                or count != len(ids) or grade["EOS"] != (count == 1) or grade["TS"] != (count >= 1)
                or grade["dup_executed"] != max(0, count - 1)):
            raise ValueError(f"inconsistent persisted-effect grade for case {record['episode_id']}")
    invalid = [record for record in records if record["stop_reason"] == "infrastructure_error"
               or record.get("grade") is None]
    invalid_ids = {id(record) for record in invalid}
    valid = [record for record in records if id(record) not in invalid_ids]
    late_records = [r for r in records if r["spec"]["mode"] == "timeout_late" and r["fault_triggered"]]
    late_delays = {float(entry["late_delay_s"]) for r in late_records for entry in r["fault_log"]
                   if entry["mode"] == "timeout_late" and "late_delay_s" in entry}
    if late_records and len(late_delays) != 1:
        raise ValueError("missing or mixed late-commit delays in real API trace")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    operations: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in valid:
        spec = record["spec"]
        groups[(spec["model"], spec["mode"])].append(record)
        operations[(spec["model"], spec["operation"], spec["mode"])].append(record)

    def metrics(rows: list[dict]) -> dict:
        first_reads = []
        for row in rows:
            first = next((call["observation"]["result"] for call in row["tool_calls"]
                          if call["name"] in ("issues_find", "comments_find") and
                          call["observation"].get("ok")), None)
            if first is not None:
                if not isinstance(first, list):
                    raise ValueError("read-back tool returned a non-list result")
                first_reads.append(bool(first))
        first_rate = None
        if first_reads:
            p = sum(first_reads) / len(first_reads)
            lo, hi = wilson(p, len(first_reads))
            first_rate = {"rate": round(p, 4), "events": sum(first_reads), "n": len(first_reads),
                          "ci95": [round(lo, 4), round(hi, 4)]}
        return {"n": len(rows), "EOS": rate(rows, "EOS"), "TS": rate(rows, "TS"),
                "duplicate": rate(rows, "dup_executed"),
                "first_read_visible": first_rate,
                "completed": sum(row.get("finish", {}).get("status") == "completed" for row in rows),
                "mean_wall_s": round(sum(row["wall_s"] for row in rows) / len(rows), 2)}

    return {
        "n_attempted": len(records), "n_valid": len(valid), "n_infrastructure_errors": len(invalid),
        "n_untriggered": sum(r["spec"]["mode"] != "none" and not r["fault_triggered"] for r in valid),
        "late_delay_s": next(iter(late_delays), None),
        "model_transport": sorted({r["model_transport"] for r in records}),
        "usage": {"input_tokens": sum(r["usage"]["input_tokens"] for r in records),
                  "output_tokens": sum(r["usage"]["output_tokens"] for r in records)},
        "model_mode": {f"{model}/{mode}": metrics(rows) for (model, mode), rows in sorted(groups.items())},
        "model_operation_mode": {f"{model}/{operation}/{mode}": metrics(rows)
                                 for (model, operation, mode), rows in sorted(operations.items())},
    }


def latex_table(summary: dict, models: tuple[str, ...], modes: tuple[str, ...] = MODES,
                operations: tuple[str, ...] | None = None) -> str:
    lines = ([r"\begin{tabular}{lllrcc}", r"\toprule",
              r"Model & Operation & Proxy-injected fault & $n$ & EOS (\%) & Duplicate (\%) \\"]
             if operations is not None else
             [r"\begin{tabular}{llrcc}", r"\toprule",
              r"Model & Proxy-injected fault & $n$ & EOS (\%) & Duplicate (\%) \\"])
    lines.append(r"\midrule")

    def cell(metric: dict) -> str:
        p, (lo, hi) = metric["rate"], metric["ci95"]
        return f"{100 * p:.0f} [{100 * lo:.0f},{100 * hi:.0f}]"

    for model in models:
        for operation in (operations if operations is not None else (None,)):
            for mode in modes:
                key = f"{model}/{operation}/{mode}" if operation is not None else f"{model}/{mode}"
                groups = summary["model_operation_mode"] if operation is not None else summary["model_mode"]
                group = groups.get(key)
                if group is None:
                    raise ValueError(f"no valid real API cases for {key}")
                label = MODE_LABELS[mode]
                if mode == "timeout_late":
                    if summary["late_delay_s"] is None:
                        raise ValueError("late-commit table needs a measured delay")
                    label += f" ({summary['late_delay_s']:g} s)"
                prefix = f"{model} & {operation} & " if operation is not None else f"{model} & "
                lines.append(f"{prefix}{label} & {group['n']} & "
                             f"{cell(group['EOS'])} & {cell(group['duplicate'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="real_github", help="results subdirectory")
    parser.add_argument("--input", type=Path, default=None, help="explicit JSONL path (overrides --run-name)")
    parser.add_argument("--expected-replicates", type=int, default=6)
    parser.add_argument("--models", nargs="+", default=list(REFERENCE_MODELS))
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--out-dir", type=Path, default=RESULTS.parent / "paper" / "generated")
    parser.add_argument("--output-prefix", default="real_api")
    args = parser.parse_args()
    if args.expected_replicates < 1:
        parser.error("--expected-replicates must be positive")
    if not OUTPUT_NAME.fullmatch(args.output_prefix):
        parser.error("--output-prefix must be a simple name")
    if not OUTPUT_NAME.fullmatch(args.run_name):
        parser.error("--run-name must be a simple name")
    records = load_records(args.input or RESULTS / args.run_name / "episodes.jsonl")
    check_design(records, tuple(args.models), args.expected_replicates, tuple(args.operations), tuple(args.modes))
    summary = summarize(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.output_prefix}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / f"{args.output_prefix}.tex").write_text(
        latex_table(summary, tuple(args.models), tuple(args.modes),
                    tuple(args.operations) if tuple(args.modes) == ("timeout_late",) else None), encoding="utf-8")
    print(f"real API cases: attempted={summary['n_attempted']}, valid={summary['n_valid']}, "
          f"infrastructure_errors={summary['n_infrastructure_errors']}, "
          f"untriggered={summary['n_untriggered']}")
    print(f"model transport: {', '.join(summary['model_transport'])}")


if __name__ == "__main__":
    main()
