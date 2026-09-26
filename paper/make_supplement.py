"""Create an anonymized, <=100 MB TMLR supplementary ZIP of LIMBO code and data.

The original arXiv/preprint package is *not* anonymous. Do not upload it or
the named public repository as a TMLR supplement. This builder includes only
explicitly selected code and canonical episode traces; it selects only the
paper's reported models and fails closed on personal identifiers and secrets.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
EXPERIMENTS = (
    "e1", "e1s", "e1_gpt41", "e1s_gpt41", "e2", "e2k",
    "e3_minimal", "e3_copilot", "e3_hermes", "e3_codex",
    "e3k_minimal", "e3k_copilot", "e3k_hermes", "e3k_codex",
    "e4_docs", "e4_para", "e4_nohuman", "e4_ablate", "e5", "e5k", "e6",
)
REAL_EXPERIMENTS = ("real_github", "real_github_late90")
PAPER_MODELS = frozenset(("gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol", "gpt-5.4-mini", "gpt-4.1",
                          "claude-opus-5.5", "gemini-3.8-flash", "grok-4.7"))
FORBIDDEN = re.compile(
    r"(?i)(jaxblack|jiapengli|microsoft\.com|skypool|github_pat_|gh[opurs]_[A-Za-z0-9_]{16,}"
    r"|sk-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9_.-]{20,})"
)
README = """LIMBO — anonymous TMLR supplementary material

Code, tests and canonical traces for the manuscript. This is a sanitized
review copy; it intentionally does not link to the author's public repository
or contain an author-named license header. After review, the named code/data
release supplies the attribution and license details.

1. Python >=3.11; install: python -m pip install -r requirements.txt
2. Environment tests: python -m unittest discover -s tests -p "test_*.py"
3. Recompute E1-E6: python -m limbo.report
4. Recompute real-service case study:
   python -m limbo.real_api_analysis --expected-replicates 6
   python -m limbo.real_api_analysis --run-name real_github_late90 --modes timeout_late --expected-replicates 3 --output-prefix real_api_late90

The canonical records are in results/<experiment>/episodes.jsonl. A single
episode may have multiple attempts; the report uses its last recorded result.
The real GitHub Issues REST case used a private synthetic test repository.
Repository owner and source-commit IDs are redacted here; model calls used a
local proxy backed by the SAME gateway as the main study, not an independent
provider-direct API. Its injected faults are produced by our local transport
shim, not failures attributed to GitHub's infrastructure. The 4 s and 90 s
late-commit runs have separate result directories and must not be pooled.
A scripted, model-free check of GitHub Issues' handling of a repeated
Idempotency-Key header is in results/real_github/key_contract_probe.json.

One model in the internal prospective protocol was withdrawn before public
release. The file PREREGISTRATION.md explains the deviations; the original
timestamp came from private Git history and is not independently verifiable
from this anonymous export. This archive contains no withdrawn-model data.
"""


def check_text(content: str, source: str) -> None:
    match = FORBIDDEN.search(content)
    if match:
        raise ValueError(f"private identity or credential marker in {source}: {match.group(0)[:20]}")


def anonymize_real_record(record: dict) -> dict:
    clean = copy.deepcopy(record)
    clean.pop("repo", None)
    clean.pop("source_commit", None)
    return clean


def _write_records(zipfile_out: zipfile.ZipFile, source: Path, member: str, *, real: bool = False) -> int:
    count = 0
    with source.open(encoding="utf-8") as rows, zipfile_out.open(member, "w", force_zip64=True) as dest:
        for line in rows:
            if not line.strip():
                continue
            record = json.loads(line)
            model = (record.get("spec") or {}).get("model")
            if not isinstance(model, str):
                raise ValueError(f"episode has no model identifier in {source}")
            if model not in PAPER_MODELS:
                continue
            if real:
                record = anonymize_real_record(record)
            text = json.dumps(record, ensure_ascii=True)
            check_text(text, str(source))
            dest.write((text + "\n").encode("utf-8"))
            count += 1
    return count


def build(output: Path, include_real: bool = True) -> dict[str, int]:
    sources = {exp: RESULTS / exp / "episodes.jsonl" for exp in EXPERIMENTS}
    if include_real:
        sources.update({exp: RESULTS / exp / "episodes.jsonl" for exp in REAL_EXPERIMENTS})
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing canonical experiment data: " + ", ".join(missing))
    tests = (source for source in (ROOT / "tests").glob("test_*.py") if source.name != "test_supplement.py")
    code = sorted((ROOT / "limbo").glob("*.py")) + sorted(tests)
    code += [ROOT / "requirements.txt", ROOT / "PREREGISTRATION.md", ROOT / "fleet" / "replicate.sh"]
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    try:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            check_text(README, "anonymous README")
            archive.writestr("README.txt", README)
            for source in code:
                text = source.read_text(encoding="utf-8")
                check_text(text, str(source))
                archive.writestr(str(source.relative_to(ROOT)).replace("\\", "/"), text)
            for exp, source in sources.items():
                member = f"results/{exp}/episodes.jsonl"
                counts[exp] = _write_records(archive, source, member, real=(exp in REAL_EXPERIMENTS))
            if include_real:
                failed = RESULTS / "real_github" / "episodes_gateway_error_v0.jsonl"
                if failed.exists():
                    counts["real_github_infrastructure_attempts"] = _write_records(
                        archive, failed, "results/real_github/episodes_gateway_error_v0.jsonl", real=True)
                key_probe = RESULTS / "real_github" / "key_contract_probe.json"
                probe_text = key_probe.read_text(encoding="utf-8")
                check_text(probe_text, str(key_probe))
                archive.writestr("results/real_github/key_contract_probe.json", probe_text)
                for exp in REAL_EXPERIMENTS:
                    meta = json.loads((RESULTS / exp / "run.json").read_text(encoding="utf-8"))
                    sanitized = {"batch_id": meta["batch_id"], "late_delay_s": meta["late_delay_s"],
                                 "model_transport_note": meta["model_transport_note"]}
                    archive.writestr(f"results/{exp}/run.json", json.dumps(sanitized, indent=2) + "\n")
        if output.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("TMLR supplementary archive exceeds 100 MB")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "supplement_build" / "limbo_tmlr_supplement.zip")
    args = parser.parse_args()
    counts = build(args.out)
    print(f"supplement: {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MiB, "
          f"{sum(counts.values())} canonical records)")


if __name__ == "__main__":
    main()
