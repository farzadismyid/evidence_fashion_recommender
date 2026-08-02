# Final Evaluation v2 — Stage 1 preparation handoff

## Outcome

This slice adds idempotent CLI orchestration for:

1. assembling validation/test Stage 1 bundles from frozen schedules, fixed candidate tables,
   and already-existing embedding arrays;
2. selecting the v2 CLIP/evidence reranking weight on validation; and
3. freezing validation/test locked recommendation/evidence packets produced by the selected
   v2 configuration.

Preparation never invokes an embedding model. Missing arrays cause a hard failure stating
that embeddings will not be rebuilt. All implementation tests use tiny synthetic fixtures.
No real validation, retrieval, generation, judging, or final evaluation was run, and
`outputs/robustness/` was not modified.

## Exact commands

### Prepare validation bundle

```powershell
uv run efr --config configs/final_eval_v2.yaml prepare-final-retrieval-v2-bundle `
  --split validation `
  --schedule outputs/robustness/schedules/validation_schedule.csv `
  --candidate-sets outputs/final_eval_v2/materialized/validation/candidate_sets.csv `
  --target-embedding-dir outputs/final_eval_v2/materialized/target_embeddings `
  --query-embedding-dir outputs/final_eval_v2/materialized/validation/query_embeddings `
  --output-dir outputs/final_eval_v2/prepared/validation
```

### Prepare test bundle

```powershell
uv run efr --config configs/final_eval_v2.yaml prepare-final-retrieval-v2-bundle `
  --split test `
  --schedule outputs/robustness/schedules/test_schedule.csv `
  --candidate-sets outputs/final_eval_v2/materialized/test/candidate_sets.csv `
  --target-embedding-dir outputs/final_eval_v2/materialized/target_embeddings `
  --query-embedding-dir outputs/final_eval_v2/materialized/test/query_embeddings `
  --output-dir outputs/final_eval_v2/prepared/test
```

The source directories must already contain the exact named files listed below. These commands
copy and validate them; they do not encode data.

### Select v2 reranking weight on validation

Run only after the validation bundle and fusion selection exist:

```powershell
uv run efr --config configs/final_eval_v2.yaml tune-reranking-v2 `
  --bundle outputs/final_eval_v2/prepared/validation `
  --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json `
  --output-dir outputs/final_eval_v2/validation/reranking_tuning
```

Writes:

```text
outputs/final_eval_v2/validation/reranking_tuning/validation_results.csv
outputs/final_eval_v2/validation/reranking_tuning/validation_summary.csv
outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json
outputs/final_eval_v2/validation/reranking_tuning/stage_manifest.json
```

`selected_weight.json` records `selected_on=validation`, CLIP/evidence weights, config hash,
schedule hash, bundle fingerprint, metric hierarchy, and validation-results path.

### Freeze validation locked packets

```powershell
uv run efr --config configs/final_eval_v2.yaml create-locked-packets-v2 `
  --split validation `
  --source-cases outputs/final_eval_v2/materialized/validation/selected_cases.csv `
  --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json `
  --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json `
  --output outputs/final_eval_v2/prepared/validation/locked_packets.csv
```

### Freeze test locked packets

```powershell
uv run efr --config configs/final_eval_v2.yaml create-locked-packets-v2 `
  --split test `
  --source-cases outputs/final_eval_v2/materialized/test/selected_cases.csv `
  --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json `
  --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json `
  --output outputs/final_eval_v2/prepared/test/locked_packets.csv
```

The source cases must explicitly contain
`packet_source_protocol=final_eval_v2_selected`. Legacy cases are rejected rather than
relabeled. Outputs contain the locked recommended item, item evidence, rule IDs/text,
per-packet hash, complete packet-set hash, and
`stage1_packet_protocol=final_eval_v2_selected`.

## Expected inputs

Frozen read-only inputs already present:

```text
outputs/robustness/schedules/validation_schedule.csv
outputs/robustness/schedules/test_schedule.csv
```

Inputs that must be materialized and reviewed before real preparation:

```text
outputs/final_eval_v2/materialized/target_embeddings/
  target_minilm.npy
  target_clip_image.npy
  target_clip_text.npy

outputs/final_eval_v2/materialized/<split>/query_embeddings/
  query_minilm.npy
  query_clip_image.npy
  query_clip_text.npy

outputs/final_eval_v2/materialized/<split>/candidate_sets.csv
outputs/final_eval_v2/materialized/<split>/selected_cases.csv
```

Candidate sets require one stable `candidate_position` per case, aligned `target_row`,
`is_positive`, and cached `evidence_score`. Selected cases require recommendation and evidence
packet columns plus the explicit v2 source-protocol marker.

## Expected prepared outputs

Both `outputs/final_eval_v2/prepared/validation/` and `.../test/` contain:

```text
schedule.csv
candidate_sets.csv
target_minilm.npy
target_clip_image.npy
target_clip_text.npy
query_minilm.npy
query_clip_image.npy
query_clip_text.npy
preparation_manifest.json
locked_packets.csv                 # after selected-v2 packet freezing
locked_packets.manifest.json       # after selected-v2 packet freezing
```

The preparation manifest records all source hashes and `embeddings_rebuilt=false`. A matching
manifest resumes; mismatched, incomplete, or unmanifested outputs fail without overwrite.

## Real embedding availability

A read-only inspection found 16 files in `outputs/cache/embeddings/`, including large arrays
and metadata consistent with cached target MiniLM/CLIP modalities. The existing artifact code
supports separate `minilm_text`, `clip_image`, and `clip_text` target arrays. Those arrays appear
available but still need fingerprint-to-name resolution and controlled materialization into
the named v2 source directory.

The cache contains `robustness_query_embeddings`, but the historical robustness command cached
only fused query CLIP vectors. Separate validation/test query MiniLM, CLIP-image, and CLIP-text
arrays were not established by this inspection. They likely require one approved query-only
materialization pass; the new preparation command will not trigger it implicitly.

`outputs/final_eval_v2/` did not exist at inspection time, confirming that no real v2
experimental artifacts have been created.

## Runtime estimate

- Bundle copying, hashing, and validation with existing arrays: approximately **2–10 minutes**,
  dominated by copying/hashing several hundred MB.
- Reranking selection once the bundle exists: approximately **minutes**, using cached candidate
  evidence and vector arithmetic; no model calls.
- Locked-packet hashing: under **one minute** for the current study scale.
- If separate query embeddings need approved materialization: approximately **10–45 minutes**
  on the recorded GPU, depending mainly on image loading. This is not performed by this slice.

## Tests

`tests/test_stage1_preparation_cli_v2.py` verifies with synthetic arrays that:

- validation bundles contain all required files;
- preparation is idempotent;
- manifests declare that embeddings were not rebuilt;
- missing query embeddings fail rather than invoking a model;
- reranking selection writes validation results, summary, and the fully attributed artifact;
- locked packets reject implicit/legacy relabeling and contain 64-character hashes;
- packet creation resumes only for matching inputs.

The full unit suite and Ruff are run before this slice is committed.

## Current Git status

After the implementation commit, the intended working tree is clean. The exact commit and
status are reported in the assistant handoff message.

## Remaining prerequisite and exact next safe command

The real named embedding/candidate sources do not yet exist under
`outputs/final_eval_v2/materialized/`. Therefore the preparation command above is not yet the
safe next action.

The exact next safe command remains the non-experimental config check:

```powershell
uv run efr --config configs/final_eval_v2.yaml validate-config
```

The next implementation/approval decision is whether to add a read-only cache-resolution and
query-only materialization command. Only after its development smoke test and source-hash review
should the real validation bundle command be approved.
