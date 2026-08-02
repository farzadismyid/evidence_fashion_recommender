# Final Evaluation v2 Sources and Preflight Handoff

## Outcome

This slice adds fresh v2 producers for the target-item table, deterministic candidate sets with
fresh evidence scoring, and selected recommendation/evidence cases. It also adds a read-only
preflight command. No real producer, retrieval, validation, generation, or judging command was
run, and `outputs/robustness/` was not modified.

All primary artifacts require `final_eval_v2_*` protocols and content hashes. V1 candidates and
selected cases are rejected rather than copied or relabelled.

## Commands

Read-only readiness inspection (the exact next safe command):

```powershell
uv run efr --config configs/final_eval_v2.yaml inspect-final-eval-v2-readiness
```

Target items (cache resolution, hashing, and Parquet write; no model run):

```powershell
uv run efr --config configs/final_eval_v2.yaml `
  materialize-final-eval-v2-target-items `
  --output outputs/final_eval_v2/sources/target_items.parquet
```

Fresh validation candidates (loads/reuses KB embeddings and computes candidate evidence scores):

```powershell
uv run efr --config configs/final_eval_v2.yaml `
  produce-final-eval-v2-candidates `
  --split validation `
  --schedule outputs/robustness/schedules/validation_schedule.csv `
  --target-items outputs/final_eval_v2/sources/target_items.parquet `
  --output outputs/final_eval_v2/sources/validation/candidate_sets.csv
```

Use the same command with the test schedule, `--split test`, and the test output path only after
the validation workflow is approved.

Fresh selected cases (only after validation-selected fusion and reranking artifacts exist):

```powershell
uv run efr --config configs/final_eval_v2.yaml `
  produce-final-eval-v2-selected-cases `
  --split validation `
  --schedule outputs/robustness/schedules/validation_schedule.csv `
  --target-items outputs/final_eval_v2/sources/target_items.parquet `
  --candidate-sets outputs/final_eval_v2/sources/validation/candidate_sets.csv `
  --bundle outputs/final_eval_v2/prepared/validation `
  --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json `
  --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json `
  --output outputs/final_eval_v2/sources/validation/selected_cases.csv
```

## Inputs and outputs

Stable read-only inputs:

```text
outputs/cache/datasets/<fingerprint>.parquet
outputs/cache/embeddings/<fingerprint>.npy
outputs/cache/knowledge_base_embeddings/<fingerprint>.npy
outputs/robustness/schedules/validation_schedule.csv
outputs/robustness/schedules/test_schedule.csv
data/knowledge_base/... (configured KB)
```

Fresh outputs:

```text
outputs/final_eval_v2/sources/target_items.parquet
outputs/final_eval_v2/sources/target_items.manifest.json
outputs/final_eval_v2/sources/validation/candidate_sets.csv
outputs/final_eval_v2/sources/validation/candidate_sets.manifest.json
outputs/final_eval_v2/sources/test/candidate_sets.csv
outputs/final_eval_v2/sources/test/candidate_sets.manifest.json
outputs/final_eval_v2/sources/validation/selected_cases.csv
outputs/final_eval_v2/sources/validation/selected_cases.manifest.json
outputs/final_eval_v2/sources/test/selected_cases.csv
outputs/final_eval_v2/sources/test/selected_cases.manifest.json
```

Target manifests bind row order to all three cached target modalities. Candidate manifests bind
the schedule, target table, full config, evidence/KB representation, deterministic seed, and
output. Selected-case manifests bind both validation selection artifacts, candidates, schedule,
target table, and every CLIP array used for locking.

## Real-run order

1. Run the read-only readiness inspection.
2. Materialize the target table and review its row-order/embedding compatibility hashes.
3. If preflight says query embeddings are absent, obtain explicit approval and run the existing
   query-only command for validation. It never rebuilds targets.
4. Produce fresh validation candidate sets and review their manifest.
5. Materialize and prepare the validation Stage 1 bundle.
6. After approval, run fusion validation and reranking validation.
7. Produce validation selected cases from the frozen selections for Hybrid validation packets.
8. Repeat source/bundle preparation for test without tuning on test.
9. Produce test selected cases and locked packets under the same frozen validation settings.

## Runtime and approval status

- Preflight: seconds, read-only.
- Target table: typically under two minutes; hashing large cached arrays may take several
  minutes, but no model is loaded.
- Candidate production: moderate/compute-heavy because it embeds candidate-specific evidence
  queries. Expected roughly 10–40 minutes per 300-case split depending on cache/GPU throughput.
- Selected cases: moderate; CLIP scoring is cached-array arithmetic, while rule evidence retrieval
  embeds the selected candidate packet. Expected roughly 5–20 minutes per split.
- Query-only embeddings: expected 10–30 minutes per split and still require explicit approval.
- Fusion/reranking selection remains intentionally unrun.

These are planning estimates, not timings measured by a real v2 run.

## Preflight behavior

`inspect-final-eval-v2-readiness` reports `READY` or `BLOCKED` for target sources/embeddings,
split-specific query embeddings/candidates/bundles, frozen selections, selected cases, and locked
packets. It validates roots, protocols, manifests, and hashes where applicable; reports missing or
mismatched artifacts; flags query computation and v1 primary references; and prints the next safe
command.

## Tests and safety

Synthetic tests cover target row-order compatibility, embedding compatibility hashes, fresh
candidate scoring/schema/manifests, selected fusion/reranking consumption, v1 rejection,
selected evidence packets, preflight transitions, resume, tamper detection, and no overwrite.

All producer destinations are checked by the CLI to remain under `outputs/final_eval_v2/`.
Historical robustness artifacts are read only as frozen schedule inputs. The code never promotes
v1 candidates or selected cases to a v2 protocol.

## Remaining blockers before the first real validation command

1. Review this slice and run the read-only preflight.
2. Approve target-table and validation-candidate materialization.
3. Decide whether validation query embeddings need and are approved for query-only computation.
4. Review resulting manifests and hashes before preparing the bundle.
5. Explicitly approve real `tune-clip-fusion`; it remains outside this slice.

## Git status

At handoff authoring, only this implementation slice was uncommitted. After the recorded slice
commit, the intended working tree is clean. See the accompanying assistant response for the final
commit hash and verified status.
