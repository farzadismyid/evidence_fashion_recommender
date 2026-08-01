# Final Evaluation v2 implementation handoff

## Executive status

The repository contains tested v2 building blocks for configuration, modality/fusion scoring,
the Stage 1 change gate, Hybrid grid construction/selection, two-stage claim evaluation,
cross-model judging, clustered inference, reporting semantics, and final provenance freezing.

It does **not** yet contain end-to-end v2 CLI orchestration. In particular, no new v2 command
has been registered in `cli.py`, the modality evaluator is not connected to dataset/candidate
loading, the held-out reranker still uses hard-coded weights, and the freeze helper has no CLI
entry point. Therefore no real Stage 1, Stage 2, or final workflow should be launched yet.

No retrieval, generation, judging, validation, or final evaluation workflow was run during
implementation. Existing `outputs/robustness/` artifacts were not modified.

## 1. Changed files by commit

### `0ceba05` — Add final evaluation v2 configuration schema

- `configs/final_eval_v2.yaml`
- `src/evidence_fashion_recommender/config.py`
- `tests/test_final_eval_config.py`

Adds validated v2 roots, eleven fusion weights, the 36-cell Hybrid grid dimensions, practical
tie threshold, screening/finalist settings, Stage 1 packet requirement, schema versions, and
primary comparison families.

### `35a5d0b` — Add matched modality and fusion evaluation

- `src/evidence_fashion_recommender/models/multimodal.py`
- `src/evidence_fashion_recommender/evaluation/modality.py`
- `tests/test_modality.py`

Adds explicit-weight normalized CLIP fusion, matched-candidate MiniLM/CLIP image/CLIP text/
fused scoring, metric aggregation, and hierarchical fusion selection.

### `dcdc8a7` — Add final evaluation protocol decision gate

- `src/evidence_fashion_recommender/evaluation/protocol_gate.py`
- `tests/test_protocol_gate.py`

Adds paired legacy/v2 locked-item and evidence-packet comparison with conservative decisions:
`regenerate_all_variants` or `legacy_generation_v2_judging`.

### `6d54800` — Add final evaluation Hybrid validation grid

- `src/evidence_fashion_recommender/generation.py`
- `src/evidence_fashion_recommender/evaluation/robustness.py`
- `tests/test_hybrid_v2.py`

Adds `item_limit`, the 36-cell factorial grid, rule-only-candidate labeling, selected Stage 1
validation-packet enforcement, and priority/tie-based finalist selection.

### `0301724` — Add two-stage claim evaluation

- `src/evidence_fashion_recommender/evaluation/claim_evaluation.py`
- `tests/test_claim_evaluation_v2.py`

Adds exhaustive typed atomic-claim extraction, explicit extraction failure, structured
reference packets, source-specific support labels, and separate claim verification.

### `338e1d0` — Add cross-model judging and clustered analysis

- `src/evidence_fashion_recommender/evaluation/final_judging.py`
- `src/evidence_fashion_recommender/evaluation/final_reporting.py`
- `src/evidence_fashion_recommender/evaluation/statistics.py`
- `tests/test_final_eval_analysis.py`

Adds anchored general judging, normalized same-family exclusion, primary/sensitivity summaries,
outfit-clustered paired bootstrap, predefined-comparison support, and external-grounding N/A
semantics.

### `241897a` and corrective `aa53ca6` — Add fail-closed freeze and correct commit scope

Net implementation files:

- `src/evidence_fashion_recommender/final_freeze.py`
- `tests/test_final_freeze.py`

The first commit accidentally included already-staged documentation and the dependency-file
deletion. `aa53ca6` removed those unrelated paths from that implementation slice without
discarding their working-tree content. The net source change is the freeze helper and its test.

### `361aed9` — Document final evaluation v2 implementation plan

- `reports/final_evaluation_implementation_plan_v2.md`

Contains the revised Stage 0A → Stage 1/2 selection → final-freeze ordering and binds Hybrid
validation to Stage 1-selected validation evidence packets.

### `dbf981a` — Add final evaluation protocols and audit

- `docs/final_evaluation_fix_protocol.md`
- `docs/final_evaluation_stage_design.md`
- `reports/reviewer_risk_audit.md`
- `requirements.txt` deleted intentionally

### Handoff commit

- `reports/final_eval_v2_implementation_handoff.md`

This document records implemented scope, gaps, safe next steps, and protocol compliance.

## 2. New CLI commands and exact usage

### Implemented CLI commands

**None.** No new v2 subcommand is currently registered in
`src/evidence_fashion_recommender/cli.py`.

The following names appear in the approved plan but remain TODOs and will currently fail as
unknown commands:

```text
tune-clip-fusion
evaluate-final-retrieval-v2
compare-locked-artifacts-v2
run-hybrid-validation-v2
freeze-final-eval-v2
run-final-explanations-v2
extract-claims-v2
verify-claims-v2
judge-general-quality-v2
analyze-final-eval-v2
build-final-report-v2
```

The reusable Python functions exist, but calling them manually would bypass partition guards,
artifact manifests, output-root checks, and resume handling. Do not use ad hoc Python calls for
real validation.

Currently valid development-only preflight commands are:

```powershell
uv run efr --config configs/final_eval_v2.yaml validate-config
uv run pytest -q
uv run ruff check src tests
```

These are not Stage 1 experimental runs.

## 3. New configuration path and key settings

Configuration: `configs/final_eval_v2.yaml`, extending `configs/robustness.yaml`.

Key settings:

```text
experiment_name: final-eval-v2
output_root: outputs/final_eval_v2
report_root: reports/final_eval_v2
fusion_image_weights: 1.0, 0.9, ..., 0.0
hybrid_word_budgets: 35, 55, 75
hybrid_rule_counts: 3, 5
hybrid_item_counts: 0, 2, 5
hybrid_evidence_orders: rules_first, item_first
hybrid_practical_tie: 0.01
hybrid_screening_cases_per_category: 10
hybrid_finalist_count: 6
require_stage1_validation_packets: true
claim_schema_version: v2
judge_schema_version: v2
```

The config contains the four predefined explanation comparisons and four predefined retrieval
comparisons. It validates the exact versioned roots and the range/uniqueness of fusion weights.

## 4. Tests added or updated

New tests:

- `tests/test_final_eval_config.py`: roots, fusion grid, 36-cell configuration, invalid root.
- `tests/test_modality.py`: normalized fusion, matched methods/candidates, tie-break selection.
- `tests/test_protocol_gate.py`: unchanged legacy labeling and evidence-change regeneration.
- `tests/test_hybrid_v2.py`: 36 specs, 12 rule-only candidates, Stage 1 packet binding,
  exclusion of rule-only candidates from final Hybrid selection.
- `tests/test_claim_evaluation_v2.py`: reference-packet flags, separate extraction/verification,
  source labels, empty extraction as failure/N/A.
- `tests/test_final_eval_analysis.py`: model-family exclusion, cross-model primary summaries,
  clustered case/outfit counts, N/A rule grounding.
- `tests/test_final_freeze.py`: complete validation selections, clean source state, selected
  Stage 1 packet binding, rejection of legacy-only Hybrid packets.

Verification performed before handoff:

```text
pytest: 50 passed
ruff: all checks passed across src and tests
```

Only unit/development checks ran. The pytest cache emitted a permissions warning but tests
completed successfully using a workspace-local temporary directory.

## 5. Current Git status

At handoff creation, the intended state after committing this file and pushing is:

```text
branch: main
working tree: clean
remote: origin
```

The final exact status and pushed commit are recorded in the assistant handoff message after
the push. No output directories are expected to be changed.

## 6. Remaining TODOs

### Blocking Stage 1

1. Register v2 CLI parsers and handlers in `cli.py`.
2. Connect `evaluation/modality.py` to prepared datasets, identical cached candidate sets,
   separate target/query CLIP image/text embeddings, validation schedules, and v2 paths.
3. Ensure embedding production/cache exposes separate CLIP image and text arrays; current
   historical artifacts may contain fused vectors only.
4. Write validation per-case results, summaries, and `selected_fusion.json` with
   `selected_on=validation`, schedule/config/model hashes, and metric hierarchy.
5. Change held-out reranking to load a frozen selection artifact instead of
   `cli.py`'s hard-coded `[0.9, 1.0]` test list.
6. Write selected Stage 1 validation recommendation/evidence packets with one packet hash for
   Stage 2 binding.
7. Wire `compare_locked_packets` to versioned inputs and output the decision-gate artifacts.
8. Add development-only CLI integration tests using tiny synthetic fixtures.

### Blocking Stage 2

1. Add resumable Hybrid screening/finalist CLI orchestration.
2. Produce rule/item overlap, claim-based hallucination, evidence misuse, substitution, and
   clarity fields required by `select_hybrid_finalists`.
3. Resolve naming compatibility: old code uses `candidate_first`; v2 uses `item_first`.
4. Persist selected Hybrid metadata using `item_count` consistently; the internal dataclass
   currently calls it `item_limit`.
5. Ensure the old weighted selector is not invoked by the v2 path.

### Blocking Stages 3–5

1. Implement the decision-gated all-four-variant v2 generation path and legacy-generation/
   v2-judging labeling.
2. Add resumable CLI runners for claim extraction, claim verification, and anchored judging.
3. Add retry/error tables and completeness/cardinality checks.
4. Add an explicit anchored `rule_grounded_faithfulness` score or establish that claim-source
   rates are the sole primary rule-grounding construct.
5. Rename/contextualize historical `faithfulness_to_available_information` in generated v2
   tables; old robustness code and artifacts retain the historical field name.
6. Add conditional citation metrics and aggregate citation-to-claim entailment.

### Blocking Stage 6/freeze

1. Register analysis/report/freeze commands.
2. Add multiple-comparison corrections to the clustered predefined-family output; the new
   clustered helper currently produces intervals but does not calculate p-values/corrections.
3. Build required v2 general-quality, grounding, retrieval, statistics, sensitivity, and
   report files under the exact versioned roots.
4. Reconcile stage-design path names with the revised plan before runtime. The stage-design
   document uses some older paths such as `manifest/pre_run_manifest.json`; the implemented
   helper requires `outputs/final_eval_v2/freeze/FINAL_FREEZE_MANIFEST.json`.
5. Wire `create_final_eval_v2_freeze` to the CLI and include final command-list generation,
   run-tag creation, and report-side freeze documentation.
6. Add the resumable PowerShell stage orchestrator only after commands exist.

## 7. Exact next safe command for Stage 1 retrieval validation

### Current status: blocked / no valid Stage 1 command exists

Do **not** run the legacy `evaluate-ranking`, `tune-reranking`, or
`evaluate-heldout-ranking` commands as a substitute; they do not implement the full v2
modality/fusion protocol and the held-out reranking path remains hard-coded.

The exact next safe command today is only a non-experimental configuration preflight:

```powershell
uv run efr --config configs/final_eval_v2.yaml validate-config
```

After the Stage 1 CLI TODOs are implemented and development-smoke-tested, the intended first
real validation command is:

```powershell
uv run efr --config configs/final_eval_v2.yaml tune-clip-fusion `
  --input outputs/robustness/schedules/validation_schedule.csv `
  --output-dir outputs/final_eval_v2/validation/fusion_tuning
```

That command is **planned, not currently implemented**, and must not be run until its parser,
handler, partition checks, candidate-set reuse, and selection manifest tests exist.

## 8. Estimated Stage 1 runtime

Expected category: **moderate retrieval-only run**.

- If separate query/target MiniLM and CLIP image/text embeddings already exist and fingerprint
  correctly: approximately **15–45 minutes** on the recorded RTX 5070 Ti environment. The
  eleven fusion points are vector arithmetic and should not re-encode images.
- If separate CLIP image/text embeddings must be built because only fused caches exist:
  approximately **1–3 hours**, dominated by dataset image loading and CLIP encoding.
- Evidence scoring/reranking and artifact comparison may add roughly **10–30 minutes**.
- No explanation generation or LLM judging belongs in Stage 1.

These estimates must be refined after a development-only timed smoke run; no such timed model
run has been performed.

## Protocol compliance checklist

Status values: implemented, partially implemented, not implemented, intentionally deferred,
or blocked / needs decision.

| Planned item | Status | Evidence | Tests | Runtime still required | Stage |
|---|---|---|---|---|---|
| Versioned `outputs/final_eval_v2/` and `reports/final_eval_v2/` roots | **Partially implemented** | `0ceba05`; `config.py`, `configs/final_eval_v2.yaml` | Yes, `test_final_eval_config.py` | CLI/output writers still required | 0A/1–6 |
| Preserve old outputs/no overwrite | **Partially implemented** | Exact roots enforced in config; freeze destination guard in `241897a` | Root/freeze tests | All future handlers need fail-on-existing semantics | All |
| Stage 0A development-only implementation before freeze | **Implemented** | `361aed9`; implementation plan | Documentation only | No | 0A |
| Final freeze after Stage 1 and Stage 2 selection | **Partially implemented** | `241897a`; `final_freeze.py` requires all three selections | Yes | CLI wiring and actual selections required | Freeze |
| Clean freeze/provenance fail-closed | **Partially implemented** | `241897a`; `final_freeze.py` | Yes, including dirty/legacy constraints indirectly | Run tag, CLI, report freeze, full runtime metadata integration remain | Freeze/6 |
| Clean MiniLM/CLIP image/CLIP text/fused comparison | **Partially implemented** | `35a5d0b`; `evaluation/modality.py` | Synthetic matched-candidate tests | Dataset/cache/CLI retrieval run required | 1 |
| Eleven-point fusion validation curve | **Partially implemented** | `0ceba05`, `35a5d0b`; config grid and scoring/selector | Yes | CLI and validation retrieval run required | 1 |
| Validation selection hierarchy for fusion | **Implemented as library logic** | `35a5d0b`; `select_fusion_weight` | Yes | Validation results required | 1 |
| Selected reranking artifact loaded instead of hard-coded | **Not implemented** | Existing `cli.py:1034-1036` still hard-codes test weights | Existing old tuning tests only | Code/CLI fix plus retrieval-only run | 1 |
| Honest reranking trade-off reporting | **Partially implemented** | Historical reports already cautious; no v2 reporter wired | No new end-to-end test | Stage 1 and Stage 6 outputs | 1/6 |
| Stage 1 locked recommendation/evidence decision gate | **Implemented as library logic** | `dcdc8a7`; `evaluation/protocol_gate.py` | Yes | CLI plus real Stage 1 packet comparison required | 1 |
| Full regeneration on material packet/prompt/policy change | **Partially implemented** | Gate returns `regenerate_all_variants`; `dcdc8a7` | Yes for decision | Generation orchestrator required | 1/3 |
| Legacy-generation/v2-judging labeling when unchanged | **Partially implemented** | Gate returns explicit label; `dcdc8a7` | Yes | Row-level propagation/reporting required | 1/3–6 |
| Hybrid must use selected Stage 1 validation packets | **Implemented as validation guard** | `6d54800`; `validate_stage1_validation_packets` | Yes | Packet production and CLI binding required | 1/2 |
| Legacy-only Hybrid packets ineligible | **Implemented as guard** | `6d54800`; `robustness.py` | Yes | Runtime selection still required | 2 |
| Hybrid 36-grid: word/rule/item/order | **Implemented as library logic** | `6d54800`; `full_hybrid_specs` | Yes, exactly 36 | Validation generation+judge run required | 2 |
| `item_count=0` labelled rule-only and not final Hybrid | **Implemented** | `6d54800`; `HybridPromptSpec` and selector | Yes | Validation run required | 2 |
| Optional staged 36-grid screening/finalists | **Partially implemented** | Config contains subset/finalist settings; selector returns finalists | Partial | Screening sampler/resume CLI required | 2 |
| Priority Hybrid selection without weighted composite | **Partially implemented** | `6d54800`; `select_hybrid_finalists` uses ordered thresholds | Yes | Metrics pipeline and CLI required; old selector still exists for v1 | 2 |
| Same/matched output budget policy across variants | **Not implemented** | V2 Hybrid budgets exist; no final four-variant v2 generation policy | No | Decision and generation code required | 2/3 |
| Separate claim extraction and verification | **Implemented as library logic** | `0301724`; `claim_evaluation.py` | Yes | Judge-only or full v2 runtime required | 4/5 |
| Extract all atomic claims/no cap | **Implemented in prompt/parser path** | `0301724`; extraction prompt | Yes with multiple claims | LLM extraction run required | 4/5 |
| Empty claim list not perfect support | **Implemented** | `0301724`; failure row has N/A support | Yes | LLM extraction run required | 4/5 |
| Explicit structured reference packet | **Implemented** | `0301724`; `ReferencePacket` | Yes | Verification run required | 4/5 |
| Distinguish generation evidence from evaluation references | **Implemented at row/packet level** | `0301724`; shown-to-generator flags and packet hash | Yes | Report integration required | 4/5/6 |
| Six source-specific verification labels | **Implemented** | `0301724`; `SUPPORT_LABELS` and parser | Yes | Verification run required | 4/5 |
| Citation-to-claim entailment | **Partially implemented** | Verification schema records `citation_entails_claim` | Parser tests exercise values | Citation aggregation/reporting required | 4/5/6 |
| Cross-model-only primary judging | **Implemented as library logic** | `338e1d0`; family normalization and eligibility filter | Yes | Judge run and CLI integration required | 4/5 |
| All-judge results sensitivity only | **Implemented as summary logic** | `338e1d0`; `primary_and_sensitivity_summaries` | Yes | Judge run/report integration required | 4/5/6 |
| Remove false “generator is different” wording | **Implemented only in new v2 prompt** | `338e1d0`; `anchored_general_judge_prompt` omits it | Prompt exercised indirectly | Old v1 prompt remains intentionally historical | 4/5 |
| Anchored judge prompts | **Partially implemented** | `338e1d0`; six anchored dimensions | Score/parser tests are indirect | Rule-grounded anchor and runtime judge validation remain | 4/5 |
| Input consistency construct | **Implemented in v2 judge** | `338e1d0`; anchored `input_consistency` | Summary tests | Judge run required | 4/5 |
| General quality/clarity/specificity constructs | **Implemented in v2 judge** | `338e1d0`; anchored fields | Summary tests | Judge run required | 4/5 |
| Hallucination-risk construct | **Partially implemented** | Anchored judge dimension plus source-specific claim labels | Unit parsing/summary coverage | Claim-derived primary metric aggregation remains | 4/5/6 |
| Evidence-misuse construct | **Partially implemented** | Anchored judge field; verification supports contradiction | Unit summary coverage | Wrong-rule/category/item aggregation remains | 4/5/6 |
| Rule-grounded faithfulness construct | **Partially implemented** | Rule-supported claim labels/rates in `0301724` and `338e1d0` reporting helper | N/A/rule-rate tests | Explicit anchored score or documented claim-rate-only decision needed | 4/5/6 |
| Rename old faithfulness to contextual faithfulness | **Not implemented in code/report output** | Only specified in `361aed9` plan and protocol docs | No | Reporting transformation required; old artifacts remain unchanged | 6 |
| Describe No-RAG as input-grounded consistency | **Partially implemented** | V2 uses `input_consistency`; documentation specifies wording | Judge summary tests | Final report wording required | 4/5/6 |
| External rule grounding N/A for No-RAG/Item-RAG | **Implemented as table logic** | `338e1d0`; `external_grounding_table` | Yes | Claim verification data/report build required | 6 |
| Citation metrics N/A/conditional rather than vacuous 1 | **Not implemented** | Old `study.py` behavior unchanged; v2 reporter lacks citation table | No | Analysis/report code required | 6 |
| Separate rule/item evidence overlap | **Not implemented** | Config/selection expects columns but no v2 calculator exists | No | Analysis implementation and runtime data required | 2/6 |
| Outfit-clustered paired bootstrap | **Implemented as library logic** | `338e1d0`; `statistics.py` | Yes, case/outfit counts | Analysis run required | 6 |
| Preserve pairing across variants/generators/judges | **Partially implemented** | Pivot supports paired ID/outfit and averages duplicate rows | Basic test only | Crossed-model design validation needed | 6 |
| Primary comparison families predefined | **Implemented in config** | `0ceba05`; `configs/final_eval_v2.yaml` | Config test | Analysis wiring required | 6 |
| Multiple-comparison correction within primary families | **Not implemented for clustered v2 output** | Old correction functions exist; clustered comparator does not apply them | No v2 correction test | Analysis implementation required | 6 |
| Case and unique-outfit counts | **Implemented in clustered result** | `338e1d0`; `statistics.py` | Yes | Analysis run required | 6 |
| Counterfactual tautology removed/replaced | **Intentionally deferred** | Protocol permits removal; no v2 reporter exists yet | No | Ensure it is omitted from v2 main report | 6 |
| Resumable/checkpointed long-running stages | **Not implemented** | Underlying per-prompt cache exists, but no v2 orchestration/checkpoints | No | Required before real Stage 2–5 runs | 2–5 |
| V2 tables, figures and report | **Not implemented** | Only `final_reporting.py` helper exists | One table-transform test | Report builder/runtime required | 6 |
| Human evaluation | **Intentionally deferred** | Protocol limitation | N/A | No | Limitation |
| Full-catalogue/hard-negative evaluation | **Intentionally deferred** | Protocol limitation | N/A | No | Limitation |

## Go/no-go conclusion

**No-go for real Stage 1 validation at this commit.** The mathematical and validation
building blocks are present, but the safe v2 command, artifact plumbing, selected-reranker
loading, and integration tests are incomplete. The next implementation slice should be CLI
and artifact orchestration for Stage 1 only, followed by development-only synthetic smoke
tests. Real validation should remain blocked until that slice is reviewed.
