# Final Evaluation v2 Stage 2 Real-Run Handoff

## Outcome

Stage 2 completed successfully using only the frozen Stage 1 validation packets. The v2 CLI
screened the complete 36-cell Hybrid-RAG grid on a deterministic category-balanced subset of 50
validation cases, then evaluated six eligible finalists on all 300 validation cases.

The frozen selection is:

```text
name: hybrid_w35_r5_i2_item_first
word budget: 35
rule count: 5
item count: 2
evidence order: item_first
```

No test cases or legacy v1 packets were used. No Stage 3 explanation generation ran.

## CLI implementation

The following command is now registered:

```powershell
uv run --extra cuda efr --config configs/final_eval_v2.yaml run-hybrid-validation-v2 `
  --input outputs/final_eval_v2/prepared/validation/locked_packets.csv `
  --output-dir outputs/final_eval_v2/hybrid_validation
```

The implementation:

- validates the `validation` split, `final_eval_v2_selected` packet protocol, and one non-empty
  Stage 1 packet hash;
- binds the Stage 1 packet hash and phase to generation/judge cache fingerprints;
- uses the full 3 x 2 x 3 x 2 grid, including labelled rule-only diagnostic candidates;
- excludes `item_count=0` candidates from final Hybrid selection;
- uses the frozen priority hierarchy and never calls the legacy weighted selector;
- writes screening and finalist generations, judgments, per-case metrics, summaries, the frozen
  selection, a stage manifest, and a report;
- resumes completed local-model calls from content-addressed caches.

The shared rule-packet parser was also corrected for v2: it now reads the structured
`rule_evidence_packet` JSON, with a parallel ID/text fallback. This prevents v2 prompts from
receiving valid rule IDs paired with blank rule text.

## Real-run details

```text
runtime: 11,914.9 seconds (3 h 18 m 34.9 s)
screening cases: 50 (10 per target category)
screening configurations: 36
screening explanations/judgments: 1,800 / 1,800
finalists: 6
full validation cases: 300
finalist explanations/judgments: 1,800 / 1,800
generator: llama3.2@a80c4f17acd5
judge: qwen3:8b@500a1f067a9f
```

Packet and artifact bindings:

```text
input file hash: 91e31b422acc71ac6f5c1fc55ddfa684a0c8cc306c2d47219ed6c4e3e753b722
Stage 1 packet hash: 251eb99e65aea61d5612aec3f2e465302b25bfe90fa2dc7f26fd51c65a357cb1
Stage 2 artifact fingerprint: be91b6e184c4ba6093b70a55ae2c6d7544cb763a19bb4890ae85b025f73e937c
```

## Selected full-validation metrics

| Metric | Value |
|---|---:|
| Hallucinated fashion-claim rate | 0.000000 |
| Rule-supported styling-claim rate | 1.000000 |
| Evidence misuse rate | 0.000000 |
| Candidate substitution rate | 0.000000 |
| Rule evidence overlap | 0.401969 |
| Item evidence overlap | 0.201599 |
| General clarity | 4.150000 |
| Mean explanation words | 34.886667 |

These are validation-selection measurements from the declared llama3.2/qwen3:8b model pair, not
held-out test estimates or claims of global optimality.

## Outputs

Primary artifacts:

```text
outputs/final_eval_v2/hybrid_validation/grid_results.csv
outputs/final_eval_v2/hybrid_validation/selected_hybrid_config.json
outputs/final_eval_v2/hybrid_validation/stage_manifest.json
reports/final_eval_v2/hybrid_validation_report.md
```

Audit/resume artifacts in the same output directory include `screening_*` and `finalist_*` CSVs.

## Verification

```text
focused Hybrid v2 tests: 6 passed
full project suite: 63 passed
Ruff: all checks passed
CLI help/registration: passed
real-run row-count audit: passed
```

The first full-suite attempt used the inaccessible Windows user temp directory and produced 15
fixture-setup permission errors; rerunning with `--basetemp .tmp/pytest_stage2` produced the valid
`63 passed` result.

## Stage 3 readiness

Stage 2 selection is frozen and bound to the selected Stage 1 packet hash. Stage 3 may now
implement the decision-gated all-four-variant v2 generation path using this selected Hybrid
configuration. The Stage 1 decision remains `regenerate_all_variants`.
