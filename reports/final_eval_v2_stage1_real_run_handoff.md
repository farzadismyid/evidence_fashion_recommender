# Final Evaluation v2 Stage 1 Real-Run Handoff

## Outcome

Stage 1 completed successfully after correcting the CUDA PyTorch environment and one true
decision-gate alignment bug. All primary artifacts are fresh `final_eval_v2` outputs. No Stage 2,
Hybrid validation, explanation generation, claim extraction, judging, or final reporting ran.
`outputs/robustness/` was read only.

The final decision gate selects **Gate A: regenerate all four explanation variants** under the
frozen v2 recommendations, packets, prompts, and settings.

## Dependency fix

The failed environment contained `torch 2.12.1` from PyPI (non-CUDA build), paired with
`torchvision 0.26.0+cu128`, whose locked dependency is `torch 2.11.0+cu128`. Torchaudio was absent.
This binary mismatch caused `torchvision::nms` registration to fail and then caused the downstream
Transformers `PreTrainedModel` import failure.

`pyproject.toml` now pins and sources the complete CUDA/CPU triplet consistently:

```text
torch==2.11.0
torchvision==0.26.0
torchaudio==2.11.0
CUDA index: https://download.pytorch.org/whl/cu128
```

`uv.lock` was regenerated and the environment was synchronized with:

```powershell
uv lock
uv sync --extra cuda --extra dev
```

Final versions:

| Package | Version |
|---|---:|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| transformers | 5.12.1 |
| sentence-transformers | 5.6.0 |

CUDA smoke test:

```text
torch CUDA runtime: 12.8
torch.cuda.is_available(): True
device: NVIDIA GeForce RTX 5070 Ti
torchvision import: PASS
torchaudio import: PASS
sentence_transformers import: PASS
transformers import: PASS
CLIPModel / CLIPProcessor import: PASS
```

After the dependency change, the project suite passed (`59 passed`) and Ruff was clean.

## Commands run and runtimes

The user had already run target-item materialization successfully (67,524 rows); its runtime was
not captured in this session. Commands below were run with `uv run --extra cuda efr --config
configs/final_eval_v2.yaml` followed by the shown arguments.

| Step | Command arguments | Runtime |
|---|---|---:|
| Initial preflight | `inspect-final-eval-v2-readiness` | 1.68 s |
| Validation queries | `materialize-final-retrieval-v2-query-embeddings --split validation --schedule outputs/robustness/schedules/validation_schedule.csv --approve-compute-query-embeddings` | 7.88 s |
| Validation candidates | `produce-final-eval-v2-candidates --split validation --schedule outputs/robustness/schedules/validation_schedule.csv --target-items outputs/final_eval_v2/sources/target_items.parquet --output outputs/final_eval_v2/sources/validation/candidate_sets.csv` | 32.73 s |
| Validation materialization | `materialize-final-retrieval-v2-inputs --split validation ... --output-root outputs/final_eval_v2/materialized` | 2.18 s |
| Validation bundle | `prepare-final-retrieval-v2-bundle --split validation ... --output-dir outputs/final_eval_v2/prepared/validation` | 2.68 s |
| Fusion validation | `tune-clip-fusion --bundle outputs/final_eval_v2/prepared/validation --output-dir outputs/final_eval_v2/validation/fusion_tuning` | 2.68 s |
| Reranking validation | `tune-reranking-v2 --bundle outputs/final_eval_v2/prepared/validation --fusion-selection .../selected_fusion.json --output-dir outputs/final_eval_v2/validation/reranking_tuning` | 3.73 s |
| Validation selected cases | `produce-final-eval-v2-selected-cases --split validation ...` | 7.42 s |
| Validation locked packets | `create-locked-packets-v2 --split validation ... --output outputs/final_eval_v2/prepared/validation/locked_packets.csv` | 1.78 s |
| Test queries | `materialize-final-retrieval-v2-query-embeddings --split test --schedule outputs/robustness/schedules/test_schedule.csv --approve-compute-query-embeddings` | 6.80 s |
| Test candidates | `produce-final-eval-v2-candidates --split test ...` | 32.86 s |
| Test materialization | `materialize-final-retrieval-v2-inputs --split test ...` | 2.29 s |
| Test bundle | `prepare-final-retrieval-v2-bundle --split test ...` | 2.66 s |
| Test selected cases | `produce-final-eval-v2-selected-cases --split test ...` | 7.42 s |
| Test locked packets | `create-locked-packets-v2 --split test ...` | 1.75 s |
| Held-out retrieval | `evaluate-final-retrieval-v2 --bundle outputs/final_eval_v2/prepared/test ... --output-dir outputs/final_eval_v2/retrieval/test` | 3.01 s |
| Corrected decision gate | `compare-locked-artifacts-v2 --legacy-packets outputs/robustness/test_cases.csv --v2-packets outputs/final_eval_v2/retrieval/test/locked_recommendation_evidence_packets.csv --output-dir outputs/final_eval_v2/decision_gate` | 1.87 s |
| Final preflight | `inspect-final-eval-v2-readiness` | 2.93 s |

Exact commands whose table entries are abbreviated:

```powershell
uv run --extra cuda efr --config configs/final_eval_v2.yaml materialize-final-retrieval-v2-inputs --split validation --schedule outputs/robustness/schedules/validation_schedule.csv --target-items outputs/final_eval_v2/sources/target_items.parquet --candidate-source outputs/final_eval_v2/sources/validation/candidate_sets.csv --output-root outputs/final_eval_v2/materialized
uv run --extra cuda efr --config configs/final_eval_v2.yaml prepare-final-retrieval-v2-bundle --split validation --schedule outputs/final_eval_v2/materialized/validation/schedule.csv --candidate-sets outputs/final_eval_v2/materialized/validation/candidate_sets.csv --target-embedding-dir outputs/final_eval_v2/materialized/target_embeddings --query-embedding-dir outputs/final_eval_v2/materialized/validation/query_embeddings --output-dir outputs/final_eval_v2/prepared/validation
uv run --extra cuda efr --config configs/final_eval_v2.yaml produce-final-eval-v2-selected-cases --split validation --schedule outputs/robustness/schedules/validation_schedule.csv --target-items outputs/final_eval_v2/sources/target_items.parquet --candidate-sets outputs/final_eval_v2/sources/validation/candidate_sets.csv --bundle outputs/final_eval_v2/prepared/validation --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json --output outputs/final_eval_v2/sources/validation/selected_cases.csv
uv run --extra cuda efr --config configs/final_eval_v2.yaml create-locked-packets-v2 --split validation --source-cases outputs/final_eval_v2/sources/validation/selected_cases.csv --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json --output outputs/final_eval_v2/prepared/validation/locked_packets.csv
uv run --extra cuda efr --config configs/final_eval_v2.yaml produce-final-eval-v2-candidates --split test --schedule outputs/robustness/schedules/test_schedule.csv --target-items outputs/final_eval_v2/sources/target_items.parquet --output outputs/final_eval_v2/sources/test/candidate_sets.csv
uv run --extra cuda efr --config configs/final_eval_v2.yaml materialize-final-retrieval-v2-inputs --split test --schedule outputs/robustness/schedules/test_schedule.csv --target-items outputs/final_eval_v2/sources/target_items.parquet --candidate-source outputs/final_eval_v2/sources/test/candidate_sets.csv --output-root outputs/final_eval_v2/materialized
uv run --extra cuda efr --config configs/final_eval_v2.yaml prepare-final-retrieval-v2-bundle --split test --schedule outputs/final_eval_v2/materialized/test/schedule.csv --candidate-sets outputs/final_eval_v2/materialized/test/candidate_sets.csv --target-embedding-dir outputs/final_eval_v2/materialized/target_embeddings --query-embedding-dir outputs/final_eval_v2/materialized/test/query_embeddings --output-dir outputs/final_eval_v2/prepared/test
uv run --extra cuda efr --config configs/final_eval_v2.yaml produce-final-eval-v2-selected-cases --split test --schedule outputs/robustness/schedules/test_schedule.csv --target-items outputs/final_eval_v2/sources/target_items.parquet --candidate-sets outputs/final_eval_v2/sources/test/candidate_sets.csv --bundle outputs/final_eval_v2/prepared/test --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json --output outputs/final_eval_v2/sources/test/selected_cases.csv
uv run --extra cuda efr --config configs/final_eval_v2.yaml create-locked-packets-v2 --split test --source-cases outputs/final_eval_v2/sources/test/selected_cases.csv --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json --output outputs/final_eval_v2/prepared/test/locked_packets.csv
uv run --extra cuda efr --config configs/final_eval_v2.yaml evaluate-final-retrieval-v2 --bundle outputs/final_eval_v2/prepared/test --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json --locked-packets outputs/final_eval_v2/prepared/test/locked_packets.csv --output-dir outputs/final_eval_v2/retrieval/test
uv run --extra cuda efr --config configs/final_eval_v2.yaml compare-locked-artifacts-v2 --legacy-packets outputs/robustness/test_cases.csv --v2-packets outputs/final_eval_v2/retrieval/test/locked_recommendation_evidence_packets.csv --output-dir outputs/final_eval_v2/decision_gate
```

## Outputs created

```text
outputs/final_eval_v2/materialized/query_embeddings/validation/<fingerprint>/
outputs/final_eval_v2/materialized/query_embeddings/test/<fingerprint>/
outputs/final_eval_v2/materialized/target_embeddings/
outputs/final_eval_v2/materialized/validation/
outputs/final_eval_v2/materialized/test/
outputs/final_eval_v2/sources/validation/candidate_sets.csv
outputs/final_eval_v2/sources/test/candidate_sets.csv
outputs/final_eval_v2/sources/validation/selected_cases.csv
outputs/final_eval_v2/sources/test/selected_cases.csv
outputs/final_eval_v2/prepared/validation/
outputs/final_eval_v2/prepared/test/
outputs/final_eval_v2/validation/fusion_tuning/
outputs/final_eval_v2/validation/reranking_tuning/
outputs/final_eval_v2/retrieval/test/
outputs/final_eval_v2/decision_gate/
```

The first gate attempt exposed a case-ID alignment bug and is preserved for audit at
`outputs/final_eval_v2/decision_gate_invalid_case_id_alignment/`. It must not be used as a result.

## Frozen selections and validation metrics

Selected fusion weight:

```text
CLIP image weight = 0.40
CLIP text weight  = 0.60
```

Validation fusion metrics at the selected weight:

| Metric | Value |
|---|---:|
| Hit rate@10 | 0.266667 |
| NDCG@10 | 0.144803 |
| Reciprocal rank | 0.147810 |

Selected reranking weight:

```text
CLIP weight     = 1.00
Evidence weight = 0.00
```

The selected reranker therefore equals fused CLIP on this validation split. Adding evidence at
every tested nonzero weight reduced the primary validation hierarchy relative to CLIP-only.

## Held-out test metrics

| Method | Hit rate@10 | NDCG@10 | Reciprocal rank |
|---|---:|---:|---:|
| MiniLM text | 0.160000 | 0.061150 | 0.066535 |
| CLIP image | 0.210000 | 0.102359 | 0.105912 |
| CLIP text | 0.196667 | 0.111687 | 0.121798 |
| Fused CLIP (0.40 image) | 0.233333 | 0.124016 | 0.131588 |
| Evidence-reranked | 0.233333 | 0.124016 | 0.131588 |

## Decision gate

Legacy and v2 schedules contain the same 300 semantic query/target cases but use different
protocol-specific case IDs. The original gate incorrectly outer-joined on those IDs and reported
600 missing cases. The true bug fix pairs uniquely by `(query_item_id, target_category)` when ID
sets differ, and records `comparison_schema_version=2` to prevent stale reuse.

Corrected result:

```text
cases: 300
alignment: query_item_id + target_category
changed recommendation rate: 0.983333
changed evidence packet rate: 1.000000
material change: true
decision: regenerate_all_variants
```

## Stage 2 readiness

Stage 1 artifact readiness is complete: final preflight reports all Stage 1 inputs, selections,
selected cases, and locked packets as `READY`, no query computation required, and no v1 primary
reference.

Stage 2 is **blocked on missing CLI orchestration**, not on Stage 1 data. The planned command
`run-hybrid-validation-v2` is documented but is not registered in the current CLI. The existing
`run-hybrid-ablations` command is the legacy v1 path and must not be used for primary v2.

Planned next Stage 2 command (do not run until implemented/reviewed):

```powershell
uv run --extra cuda efr --config configs/final_eval_v2.yaml run-hybrid-validation-v2 `
  --input outputs/final_eval_v2/prepared/validation/locked_packets.csv `
  --output-dir outputs/final_eval_v2/hybrid_validation
```

## Git status

The dependency lock, gate alignment bug fix, test, and this handoff are the intended tracked
changes. Experimental outputs under `outputs/final_eval_v2/` remain untracked/ignored artifacts.
After committing this handoff slice, the intended tracked working tree is clean.
