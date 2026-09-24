# LIMBO: agents under tool failures with uncertain side effects

LIMBO measures what tool-using LLM agents do when a tool call fails but the
action may already have taken effect: timeouts after commit, misleading 500s,
requests still in flight when the client gives up, partial batches, and
at-least-once redelivery. Every outcome is graded from a ground-truth effect
ledger, not by an LLM judge.

The same sandbox can be driven three ways:

- a minimal function-calling scaffold (any OpenAI-compatible endpoint),
- production agent CLIs (GitHub Copilot CLI, Hermes, Codex CLI) through a
  stdio MCP proxy, and
- scripted agents, used by the unit tests.

## Quick start

Python 3.11+. The benchmark itself uses only the standard library; analysis
needs the packages in `requirements.txt`. From this directory:

```powershell
python -m unittest discover -s tests -p "test_*.py"         # deterministic checks, no network
python -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt

# One model, one fault mode, every focal write of instance 0:
python -m limbo.runner --experiment demo --models gpt-6-sol --modes timeout_post --indices 0 --workers 4
.venv\Scripts\python -m limbo.analysis summary --experiment demo
```

Credentials are read at runtime and never written to results: set
`LIMBO_API_KEY` and, for endpoints other than the OpenAI API, `LIMBO_BASE_URL`
(any OpenAI-compatible endpoint); `LIMBO_EXTRA_HEADERS` (a JSON object) adds
request headers that a gateway requires. The paper's runs reached every model
through one such gateway.

Harness runs add `--harness copilot|hermes|codex`. Each episode starts a fresh
sandbox on a random localhost port, launches the CLI in an empty temporary
directory with that sandbox as its only MCP server, and uses a throwaway
harness home so memory cannot leak across episodes. The CLIs are taken from
`PATH`; `LIMBO_HERMES_CLI`, `LIMBO_HERMES_HOME` and `LIMBO_HERMES_PROVIDER`
configure Hermes (the paper used its built-in `copilot` provider), and on
Windows `LIMBO_NODE`, `LIMBO_COPILOT_JS` and `LIMBO_CODEX_JS` point at the Node
entry points so that long prompts bypass `cmd.exe` quoting.

## What is in the sandbox

| Service | Writes (idempotency) | Read-back path |
| --- | --- | --- |
| social | `social_publish` (key honored on mastodon only) | `social_list_posts` — weibo lags 180 s, x has none |
| billing | `billing_create_charge` (Stripe-style key), `billing_refund_charge` (naturally idempotent) | `billing_list_charges` |
| tickets | `tickets_create`, `tickets_add_comment` (non-idempotent), `tickets_update_status` (conditional) | `tickets_list_recent` (strong), `tickets_search` (lags 120 s) |
| mail | `mail_send` (irreversible) | `mail_search_sent` (lags 120 s) |
| data | `db_insert`, `db_insert_many` (non-atomic batch), `db_upsert` (idempotent) | `db_query` |
| deploy | `deploy_trigger` | `deploy_list_runs`, `deploy_get_run` |

General tools: `wait`, `escalate_to_human` (a simulated operator who reports
the truth after 15 simulated minutes, or never answers when disabled) and
`finish` (structured status, summary and uncertain operations).

Twelve task templates (one long-horizon composite) are parameterized into
instances. Each template declares focal writes, the calls a fault can attach
to, annotated with their idempotency, verification and reversibility.

## Fault modes

| Mode | Agent observes | What actually happened |
| --- | --- | --- |
| `timeout_pre` / `timeout_post` | identical timeout | not executed / executed, response lost |
| `timeout_late` | identical timeout | executed 90 s later (request was still in flight) |
| `timeout_late_tail` | identical timeout | executed after a log-uniform 40 s–2 h delay (fixed per world) |
| `http500_pre` / `http500_post` | identical 500 | not executed / executed |
| `partial_timeout` | timeout on a batch insert | first half of the rows committed |
| `duplicate_delivery` | success | the request reached the service twice |
| `http503_transient`, `rate_limit`, `outage`, `schema_drift` | explicit errors | not executed |

Faults fire at the service boundary on the n-th call matching the focal write,
so every model and condition faces the same world. Requests the service would
reject anyway get their real 4xx and do not consume the trigger, which keeps
the pre/post pairs observation-equivalent.

## Metrics

- `TS` task success; `EOS` exactly-once success (TS and no duplicate ever executed).
- `dup_executed`, `dup_live`, `dup_compensated`; collateral damage; extraneous writes.
- `overclaim`: finished as `completed` although TS fails or a duplicate remains.
- Recovery category from the model's own calls (blind retry, verify then retry/skip,
  same-key or new-key retry, escalate, stop without check, harness-masked).
- Tool calls, tokens, simulated seconds, simulated human minutes.

## Recovery conditions

`vanilla`, `aware` (reliability rules in the system prompt), `reflect` (a
reflection turn after every error), `sdk_retry3` (transparent client retries),
`rules` (retry 429/503 only), `vbr` (verify an unknown-outcome write before an
identical retry), `wait{N}` (vbr that first waits N seconds plus the documented
read-path lag after the original request, i.e. assumes an in-flight bound of N s),
`guard` (vbr plus automatic idempotency keys, consistency-aware re-verification,
blocking of unverifiable repeats and outcome annotations), `oracle` (the "state
oracle": guard verifying against the current ground truth; it cannot see requests
still in flight, so it is not an upper bound under late commits) and
`outcome_oracle` (also sees in-flight requests: the upper bound for any
client-side policy). Harness policies read tool contracts, never the ground
truth. In harness runs the same policies execute inside the MCP tool server, so
they apply to any harness unchanged.

`--contract keys_everywhere` gives every non-idempotent write Stripe-style key
semantics; `--instruction-variant plain` removes the closing "exactly once"
sentence from every task instruction (same worlds, same grading).

## Reproducing the paper

Each experiment is a `limbo.runner` invocation; `limbo.pipeline` chains the
scaffold experiments exactly as run for the paper, and `--slow-lane` runs the
low-quota model separately. Harness experiments (E3) need Copilot CLI, Hermes
(with its optional `mcp` package) and Codex CLI on the machine.

```powershell
python -m limbo.pipeline                 # E1, E1s, E2, E2k, E4 (+ a retry pass)
python -m limbo.pipeline --slow-lane     # gpt-4.1 on its own quota
python -m limbo.pipeline --p0-lane A     # outcome oracle on E2, E5 waiting vs keys
python -m limbo.pipeline --p0-lane B     # E6 cue ablation
python -m limbo.runner --experiment e3_copilot --harness copilot --models claude-opus-5.5 gpt-5.6-sol gemini-3.8-flash `
  --policies vanilla guard --modes none timeout_pre timeout_post timeout_late http500_post partial_timeout duplicate_delivery `
  --indices 0 --max-focals 2 --workers 8  # likewise e3_hermes, e3_codex, e3_minimal (--harness minimal)
.venv\Scripts\python -m limbo.report     # tables, figures, numbers.json and LaTeX macros in paper/generated
python -m limbo.cases --experiment e1 e1s
cd paper; tectonic main.tex
```

`bash fleet/replicate.sh copilot|hermes|minimal` runs a cross-machine
replication slice on macOS/Linux. `PREREGISTRATION.md` lists the preregistered
hypotheses and every later deviation.

Per-cell intervals in the paper are Wilson intervals on Kish effective sample
sizes (intra-class correlation by task template), regressions use a binomial
mixed model and GEE with bias-reduced cluster-robust errors, and the Shapley
decomposition is reported pooled and split into faults an immediate read-back
resolves versus faults it cannot (`limbo/analysis.py`, `limbo/report.py`).

## Layout

```
limbo/world.py         clock, entities, ground-truth effect ledger
limbo/services.py      services, tool schemas and machine-readable contracts
limbo/runtime.py       tool execution with fault injection, escalation, deferred commits
limbo/tasks.py         task templates, targets, focal writes
limbo/grader.py        programmatic grading
limbo/policies.py      recovery conditions
limbo/agent.py         minimal scaffold episode loop
limbo/llm.py           chat/completions + responses client
limbo/sandbox_http.py  sandbox session behind a token-protected localhost endpoint
limbo/mcp_proxy.py     stdio MCP server forwarding to that endpoint
limbo/harness.py       Copilot CLI / Hermes / Codex CLI episode runners
limbo/gateway_proxy.py local retrying proxy for harness model traffic (never tool calls)
limbo/runner.py        experiment expansion, parallel execution, resume
limbo/pipeline.py      the paper's experiment sequence
limbo/analysis.py      tidy tables, bootstrap CIs, Shapley decomposition
limbo/report.py        paper tables, figures, numbers.json, LaTeX macros
limbo/cases.py         case-study trajectories
fleet/replicate.sh     cross-machine replication slices
paper/                 TMLR manuscript (sections/, generated/)
PREREGISTRATION.md     preregistered hypotheses and every later deviation
```

Results are written to `results/<experiment>/episodes.jsonl` (one full record
per episode, including the conversation and every tool execution) and are not
committed.

## Data

All episode records behind the paper are attached to the
[v1.0 release](https://github.com/jaxblack/limbo-bench/releases/tag/v1.0) as
`limbo-episodes-v1.0.tar.gz`: for every experiment, `episodes.jsonl` (the
report keeps the latest record per episode, so attempts that ended in a gateway
error and were re-run are included but not counted), the records that
deviations 1, 5, 14, 15 and 17 in `PREREGISTRATION.md` archived and replaced
(`episodes_*_v0.jsonl`, `episodes_not_in_design.jsonl`), and the development
commit each experiment ran on (`COMMIT`). Extract it in this
directory, which creates `results/<experiment>/`, and run
`python -m limbo.report` to regenerate every table, figure and number in the
paper. One preregistered model was withdrawn before publication; its episodes
are not included (deviation 16).

## Licence and citation

Code: MIT (`LICENSE`). Episode records and the paper's generated tables and
figures: CC BY 4.0 (`DATA_LICENSE`). The TMLR style files in `paper/` keep their
original licences.

```bibtex
@misc{li2026limbo,
  title        = {{LIMBO}: Benchmark code, harness adapters and episode traces},
  author       = {Li, Jiapeng},
  howpublished = {GitHub repository, version 1.0},
  year         = {2026},
  url          = {https://github.com/jaxblack/limbo-bench}
}
```
