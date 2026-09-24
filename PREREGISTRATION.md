# LIMBO — Preregistration (frozen before E1)

Frozen on 2026-09-23 before any E1–E4 episode was run. The pilot (E0, 3 models,
345 episodes) was used only to debug the harness and to add two fault stages
(`timeout_late`, `duplicate_delivery`) and one documentation variant
(`no_consistency_docs`). No threshold below was tuned on pilot data.

## Primary outcomes

- **DSR** — duplicate side-effect rate: fraction of episodes with at least one
  committed write beyond what a target requires (`dup_executed > 0`), including
  duplicates the agent later compensated.
- **EOS** — exactly-once success: task goals met, no collateral damage, and
  `dup_executed == 0`.
- **TS** — task success (goals met, no collateral), reported to expose
  over-caution.

Secondary: residual duplicates (`dup_live > 0`), overclaim (finished as
`completed` while TS fails or a duplicate remains), escalations and simulated
human minutes, tool calls, tokens, virtual latency, recovery-behaviour category.

## Unit of analysis and pairing

Episode = (template, instance, focal write, fault mode, model, condition,
harness, replicate). The world seed depends only on (template, instance, focal,
mode), so every model and condition faces the identical world and fault.
Episodes whose fault never triggered (focal write never attempted) are excluded
from fault-conditional analyses and reported as trigger rates. Episodes that end
in a transport/LLM error after 8 attempts are excluded and counted.

## Hypotheses and tests (α = 0.05, Holm-corrected within each family)

| # | Prediction | Test |
| --- | --- | --- |
| H1 | Under committed-but-unacknowledged faults (`timeout_post`, `http500_post`, `timeout_late`, `partial_timeout`), DSR for non-idempotent tools that are only eventually verifiable or unverifiable is ≥ 10% and higher than for keyed or strongly verifiable tools | cluster-bootstrap CI of DSR by contract class; mixed-effects logistic with contract fixed effect |
| H2 | The tool contract explains more DSR variance than model identity; harness identity explains ≥ 10% | Shapley decomposition of McFadden pseudo-R² over {contract, fault, model, harness} |
| H3 | Given that the agent verified before re-issuing, DSR is higher for eventually consistent read paths than strongly consistent ones | Fisher exact / logistic on verified subset |
| H4 | Blind-retry rate does not differ between irreversible (email) and reversible writes | two-proportion test, equivalence margin ±5pp (TOST) |
| H5 | Among episodes whose fault committed, `completed` with no uncertainty listed occurs in ≥ 20% of episodes with a duplicate | proportion with CI |
| H6 | The MCP-deployed guard reduces DSR to ≤ 2% in every harness with TS loss ≤ 3pp | per-harness CI; paired McNemar vs. native |

Mitigation family (E2), primary comparison: `guard` vs. each of `vanilla`,
`aware`, `reflect`, `sdk_retry3`, `rules`, `vbr` on EOS (paired McNemar, Holm).
`oracle` is an upper bound only and is excluded from tests.

## Experiments

- **E1** 9 models × 12 fault modes × all focal writes × 2 instances per template, `vanilla`, minimal scaffold (6,228 episodes).
- **E2** 4 models × 8 conditions × core fault modes × 2 instances.
- **E3** Copilot CLI, Hermes, Codex CLI (+ minimal scaffold) × shared models × core faults × {native, guard proxy}.
- **E4** documentation variants, three system-prompt paraphrases, operator unavailable, guard ablations.

## Stopping and reporting

Each experiment runs to completion of its design; there is no optional stopping.
All models and conditions are reported, including negative results. Every LLM
request and response is recorded in the episode trace for re-grading. Code state
for each experiment is the git commit recorded alongside its results.

## Deviations log (added after freezing, in chronological order)

1. **Late-commit timing.** The first E1 build committed in-flight requests at the
   next tool call rather than at their due time, which only matters when an agent
   waits longer than the read-path lag. All E1 `timeout_late` episodes were
   archived and re-run with the corrected clock (commit 32398bb).
2. **E1 supplement.** Partial batches and the unverifiable endpoint had one focal
   write per instance; E1s adds instances 2–9 of the two templates that carry them.
3. **gpt-4.1 quota.** The gateway serves gpt-4.1 from a small separate quota
   (HTTP 429 with a one-hour Retry-After). It runs the identical E1 design in a
   low-concurrency lane; in E2 and E4 it is replaced by [withdrawn model; see deviation 16], the
   other model with high duplicate rates in E1.
4. **Harness infrastructure failures.** Harness sessions that ended because the
   harness gave up on sustained gateway 429s (after at least one tool call) were
   first recorded as ordinary exits. They are infrastructure failures and are now
   excluded and re-run. Hermes' per-episode home raises `agent.api_max_retries` to
   12 and Codex model traffic passes through a local retrying proxy; tool calls are
   unaffected.
5. **Search semantics.** `mail_search_sent` and `tickets_search` originally
   required the whole query to be a substring of a single field, so realistic
   multi-term queries could miss a visible record. Both now use term (AND)
   matching across fields. Every episode in which the agent called either tool
   was archived and re-run; episodes that never called them saw identical
   environment responses and remain valid samples.
6. **Added contract experiments.** The keys-everywhere contract (E2k) was added
   while E1 was running, after interim E1 data showed near-zero duplicates on
   keyable writes; E3k (keys-everywhere in every harness with one shared model)
   was added after interim E3 data showed that residual harness duplicates came
   from late commits and redelivery. Both are post-registration additions and are
   labelled as such in the paper; their hypotheses were not preregistered.
7. **E4 scope.** To fit the gateway budget, the E4 robustness runs use two focal
   writes per template (evenly spaced, as in E3) instead of all focal writes.
   Comparisons with E2 are made on matched episodes only.

### Revision after internal review (P0), logged before the new runs

8. **Stratified variance decomposition.** The preregistered Shapley analysis pools
   lost-acknowledgement, misleading-500, partial, late-commit and redelivery faults.
   Late commits (with an unknown in-flight bound) and redelivery cannot be fixed by
   any verification-only policy, so pooling guarantees that the fault stage and the
   contract explain variance. The revision reports the preregistered pooled analysis
   and, in addition, separate decompositions for faults that verification can
   resolve and for faults it cannot.
9. **Oracle.** The preregistered `oracle` verifies against the *current* ground-truth
   state; it cannot see requests that are still in flight and is therefore not an
   upper bound under late commits. It is renamed "state oracle". An `outcome_oracle`
   that also sees in-flight requests (the upper bound for any client-side policy;
   only redelivery on key-less writes remains) is added and run on the full E2 design.
10. **E5, waiting versus keys.** With a fixed 90 s in-flight delay, "wait long
    enough, then verify" is a viable policy that E1–E4 did not test. E5 adds a
    heavy-tailed in-flight delay (`timeout_late_tail`: log-uniform between 40 s and
    2 h, drawn deterministically per world so that all policies see the same delay)
    and a family of wait-then-verify harness policies with an assumed in-flight bound
    Δ ∈ {0, 60, 120, 300, 900, 3600} s, compared with keys (guard, keys-everywhere
    contract) and the outcome oracle, on four models.
11. **E6, cue ablation.** Every task instruction ends with a sentence asking for
    exactly-once execution, which may prime caution. E6 removes that sentence (same
    worlds, same grading) for six models on instance 0 and is compared pairwise with E1.
12. **Statistics.** The preregistered mixed-effects logistic regression had been
    replaced by a fixed-effects logit without being logged. The revision reports the
    preregistered model (Bayesian binomial mixed GLM with random intercepts for
    template and template×instance) and, as a check, GEE with bias-reduced standard
    errors clustered by template. H4 is now tested with TOST (±5 pp) as preregistered.
    Observation equivalence is now assessed with equivalence-style statistics (total
    variation distance with bootstrap bounds, and paired first-action agreement across
    worlds compared with test–retest agreement within a world) instead of a
    non-significant χ² test. Per-cell 95% intervals are now Wilson intervals on
    Kish-effective sample sizes (template intra-class correlation), because a cluster
    bootstrap over 12 templates is anti-conservative and degenerate at 0%.
13. **gpt-4.1.** Its separate quota reset; the remaining E1/E1s episodes were run in
    the low-concurrency lane until the quota was exhausted again. It reached 87% of
    its E1 design, below the 90% threshold, so it is reported per model with its
    coverage and excluded from pooled statistics. `python -m limbo.pipeline
    --slow-lane` resumes it; the report includes it in pooled statistics
    automatically once coverage reaches 90%.
14. **Outcome oracle corrected before analysis.** The first implementation answered
    an identical re-issue of an in-flight write with a placeholder ("still
    processing") and then stopped intercepting; agents that retried again slipped
    through, so it was beaten by `wait120` and was not an upper bound. The corrected
    oracle waits for the in-flight request to commit, returns its record, and keeps
    suppressing identical re-issues (regression test
    `test_outcome_oracle_survives_repeated_retries`). All outcome-oracle episodes run
    with the first implementation were archived and re-run; no other policy changed.
15. **gpt-4.1 lane accidentally ran `timeout_late_tail`.** The E1 command for the
    gpt-4.1 lane used the default mode list, which by then included the new mode. The
    37 resulting episodes are outside the E1 design and were archived; the E1 mode list
    is now pinned.

### Before publication

16. **Withdrawn model.** One of the nine preregistered models was withdrawn from the
    study before publication. All of its episodes (E1, E1s, E2, E2k, E4 guard ablations,
    E5, E5k and E6) are excluded from every analysis and from the released data, and
    every reported number was recomputed without it; its name is redacted in this
    document. E1 therefore reports eight models, E2, E2k, E5 and E5k three, E6 five and
    the E4 guard ablations one, and the pooled weaker-model tier consists of
    `gpt-5.4-mini` alone while `gpt-4.1` remains below its coverage threshold.
17. **Guard on keyed partial batches (logged late).** Under the keys-everywhere contract the
    first guard implementation rewrote a partially committed batch to its missing rows before
    re-sending it under the original key, which the keyed contract treats as a different
    request; the fix re-sends the identical request so that the service resumes it (regression
    test `test_guard_resumes_keyed_partial_batch`). The eight affected guard episodes (four in
    E2k, one per harness in E3k) were archived and re-run with the fix; no other policy changed.
