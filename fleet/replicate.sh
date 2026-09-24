#!/usr/bin/env bash
# LIMBO cross-machine replication slice (macOS/Linux).
#
#   bash fleet/replicate.sh <slice>
#
# Slices (instance index 1 complements the reference run on index 0):
#   copilot  - Copilot CLI harness, claude-opus-5.5, vanilla + guard
#   hermes   - Hermes harness, claude-opus-5.5, vanilla + guard
#   minimal  - minimal scaffold, claude-opus-5.5 + gpt-6-sol, vanilla
#
# Writes results/fleet_<slice>_<host>/ and prints a summary.
# Never prints credentials; the model endpoint key is read by limbo/llm.py at runtime.
set -euo pipefail

slice="${1:?usage: replicate.sh copilot|hermes|minimal}"
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"
host="$(hostname -s 2>/dev/null || hostname)"
exp="fleet_${slice}_${host}"

py="$(command -v python3.12 || command -v python3.11 || command -v python3)"
"$py" - <<'PY'
import sys
assert sys.version_info >= (3, 11), f"python >= 3.11 required, found {sys.version}"
print("python", sys.version.split()[0])
PY

"$py" -m unittest discover -s tests -p 'test_*.py'

modes=(none timeout_pre timeout_post timeout_late http500_post partial_timeout duplicate_delivery)
case "$slice" in
  copilot)
    command -v copilot >/dev/null || { echo "copilot CLI not found" >&2; exit 2; }
    args=(--harness copilot --models claude-opus-5.5 --policies vanilla guard --max-focals 2 --workers 3) ;;
  hermes)
    command -v hermes >/dev/null || { echo "hermes CLI not found" >&2; exit 2; }
    # Hermes loads MCP servers only when its optional 'mcp' SDK is installed in its own venv.
    hpy="$(head -1 "$(command -v hermes)" | sed 's/^#!//')"
    if [ -x "$hpy" ] && ! "$hpy" -c 'import mcp' 2>/dev/null; then
      "$hpy" -m pip install -q mcp 2>/dev/null || uv pip install --python "$hpy" mcp
    fi
    args=(--harness hermes --models claude-opus-5.5 --policies vanilla guard --max-focals 2 --workers 3) ;;
  minimal)
    args=(--models claude-opus-5.5 gpt-6-sol --policies vanilla --workers 8) ;;
  *) echo "unknown slice $slice" >&2; exit 2 ;;
esac

mkdir -p "results/$exp"
git rev-parse HEAD > "results/$exp/COMMIT"
"$py" -m limbo.runner --experiment "$exp" --indices 1 --modes "${modes[@]}" "${args[@]}"

# Summary without third-party packages.
"$py" - "$exp" <<'PY'
import json, sys, collections
exp = sys.argv[1]
rows = [json.loads(l) for l in open(f"results/{exp}/episodes.jsonl", encoding="utf-8")]
by = collections.defaultdict(list)
for r in rows:
    s = r["spec"]
    if r.get("stop_reason") in ("llm_error", "harness_error"):
        by[(s.get("harness", "minimal"), s["model"], s["policy"], "ERROR")].append(r)
        continue
    if s["mode"] != "none" and not r.get("fault_triggered"):
        continue
    by[(s.get("harness", "minimal"), s["model"], s["policy"], s["mode"])].append(r)
print("harness,model,policy,mode,n,dup_rate,eos_rate,ts_rate")
for k in sorted(by):
    g = by[k]
    if k[3] == "ERROR":
        print(",".join(k) + f",{len(g)},,,")
        continue
    dup = sum(1 for r in g if r["grade"]["dup_executed"] > 0) / len(g)
    eos = sum(1 for r in g if r["grade"]["EOS"]) / len(g)
    ts = sum(1 for r in g if r["grade"]["TS"]) / len(g)
    print(",".join(k) + f",{len(g)},{dup:.3f},{eos:.3f},{ts:.3f}")
PY

gzip -kf "results/$exp/episodes.jsonl"
echo "results: $here/results/$exp/episodes.jsonl.gz"
