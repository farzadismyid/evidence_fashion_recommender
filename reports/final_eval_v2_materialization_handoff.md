# Final Evaluation v2 — real-run materialization handoff

## Outcome

This slice implements safe materialization for a **fresh final_eval_v2 primary evaluation**.
Old v1 recommendations, evidence packets, explanations, judgments, reports, and statistics are
rejected as primary v2 sources. They may later be read only by the audit decision gate.

No real materialization, validation, retrieval, generation, or judging was run. All smoke tests
used tiny temporary arrays. `outputs/robustness/` was not modified.

## New commands

### Explicit query-only embedding materialization

This is the only command in the slice that may invoke embedding models. It requires an explicit
approval flag and never reads, writes, or rebuilds target embeddings.

```powershell
uv run efr --config configs/final_eval_v2.yaml `
  materialize-final-retrieval-v2-query-embeddings `
  --split validation `
  --schedule outputs/robustness/schedules/validation_schedule.csv `
  --approve-compute-query-embeddings
```

Repeat with `--split test` and the test schedule after validation materialization is approved.
The cache location is content-derived under:

```text
outputs/final_eval_v2/materialized/query_embeddings/<split>/<fingerprint>/
```

It contains separate `query_minilm.npy`, `query_clip_image.npy`, `query_clip_text.npy`, and a
manifest with schedule/config/model hashes, output hashes, and
`target_embeddings_rebuilt=false`.

### Materialize frozen retrieval inputs

Validation:

```powershell
uv run efr --config configs/final_eval_v2.yaml materialize-final-retrieval-v2-inputs `
  --split validation `
  --schedule outputs/robustness/schedules/validation_schedule.csv `
  --target-items outputs/final_eval_v2/sources/target_items.parquet `
  --candidate-source outputs/final_eval_v2/sources/validation/candidate_sets.csv `
  --output-root outputs/final_eval_v2/materialized
```

Test:

```powershell
uv run efr --config configs/final_eval_v2.yaml materialize-final-retrieval-v2-inputs `
  --split test `
  --schedule outputs/robustness/schedules/test_schedule.csv `
  --target-items outputs/final_eval_v2/sources/target_items.parquet `
  --candidate-source outputs/final_eval_v2/sources/test/candidate_sets.csv `
  --output-root outputs/final_eval_v2/materialized
```

The command resolves target arrays through the repository's content-addressed embedding records
using the supplied target table and v2 config. It fails if any matching target modality is
missing. It copies—never computes—target and query arrays.

Candidate sources require companion `.manifest.json` files with:

```text
protocol=final_eval_v2_candidate_sets
schedule_hash=<frozen schedule SHA-256>
output_hash=<candidate CSV SHA-256>
```

This prevents existing v1 candidate outputs from being silently promoted to primary v2 data.

### Materialize selected v2 cases

Run only after validation has selected and frozen fusion/reranking settings and a fresh selected-
case producer has run under those settings:

```powershell
uv run efr --config configs/final_eval_v2.yaml `
  materialize-final-retrieval-v2-selected-cases `
  --split validation `
  --schedule outputs/robustness/schedules/validation_schedule.csv `
  --source-cases outputs/final_eval_v2/sources/validation/selected_cases.csv `
  --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json `
  --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json `
  --output outputs/final_eval_v2/materialized/validation/selected_cases.csv
```

Repeat for test only after selections are frozen. Source cases require a matching manifest with
`protocol=final_eval_v2_selected_cases` and must contain
`packet_source_protocol=final_eval_v2_selected`. V1 rows are rejected.

## Expected input paths

Stable read-only inputs:

```text
outputs/robustness/schedules/validation_schedule.csv
outputs/robustness/schedules/test_schedule.csv
outputs/cache/embeddings/<fingerprinted target arrays>
outputs/final_eval_v2/materialized/query_embeddings/<split>/<fingerprint>/ # if approved
```

Fresh v2 sources still requiring a producer/orchestration step:

```text
outputs/final_eval_v2/sources/target_items.parquet
outputs/final_eval_v2/sources/validation/candidate_sets.csv
outputs/final_eval_v2/sources/test/candidate_sets.csv
outputs/final_eval_v2/sources/validation/selected_cases.csv
outputs/final_eval_v2/sources/test/selected_cases.csv
```

Each candidate/selected-case CSV requires its own provenance manifest and frozen schedule hash.

## Expected outputs

```text
outputs/final_eval_v2/materialized/target_embeddings/
  target_minilm.npy
  target_clip_image.npy
  target_clip_text.npy
  target_embedding_manifest.json

outputs/final_eval_v2/materialized/validation/
  schedule.csv
  candidate_sets.csv
  query_minilm.npy
  query_clip_image.npy
  query_clip_text.npy
  query_embeddings/
  materialization_manifest.json
  selected_cases.csv                 # after selections
  selected_cases.manifest.json

outputs/final_eval_v2/materialized/test/
  <same structure>
```

Manifests record source paths/hashes, schedule and normalized-case hashes, target/query embedding
fingerprints, copied/computed status, model/config hashes for computed query arrays, output
hashes, `primary_output_protocol=fresh_final_eval_v2`, and
`v1_primary_outputs_used=false`.

## Cache reuse versus fresh v2 work

May be reused when fingerprints match:

- raw dataset/cache references;
- frozen schedules;
- target MiniLM, CLIP-image, and CLIP-text arrays;
- separately cached v2 query arrays;
- KB/rule embeddings;
- deterministic candidate pools only when a v2 source manifest validates them.

Must be freshly produced for primary v2:

- fusion and reranking selections;
- locked recommendations and item/rule packets;
- Hybrid validation and selection;
- all four explanation variants;
- claim extraction and verification;
- anchored judging;
- statistics, reports, and final manifests.

V1 material is audit-only and never receives a v2 primary protocol marker.

## Query embedding status

The historical query cache was designed around fused robustness query vectors. A prior read-only
inspection did not establish valid separate MiniLM/image/text query arrays for both v2 splits.
Therefore real query-only computation should currently be assumed necessary unless the new
resolver finds the exact content-derived v2 cache directory complete.

The explicit query-only command is implemented but **not approved for real execution by this
handoff**. Its synthetic builder path is tested for resume, output hashes, array alignment, and
the no-target-rebuild invariant.

## Safety and resume behavior

- Missing target arrays fail; target computation is never attempted.
- Missing query arrays fail with the exact approved query-only command as remediation.
- Query computation fails without `--approve-compute-query-embeddings`.
- Existing unmanifested or mismatched outputs fail without overwrite.
- Matching manifests resume without recomputation/copy.
- Candidate and selected-case sources require schedule-bound hashes and v2 protocols.
- Every CLI output is constrained under `outputs/final_eval_v2/`.
- `outputs/robustness/` is used only for read-only frozen schedules/audit data.

## Runtime estimate

- Cache resolution and manifest validation: under **1 minute**.
- Copy/hash cached target arrays (several hundred MB): approximately **2–10 minutes**.
- Query-only validation embeddings if required: approximately **10–30 minutes** on the recorded
  RTX 5070 Ti, dominated by loading 300 query images and model initialization.
- Query-only test embeddings: similar **10–30 minutes**.
- Candidate/selected-case production is not part of this command and must be estimated once its
  fresh v2 producer is wired; evidence scoring may add tens of minutes.

No runtime estimate above was measured with a real run in this slice.

## Tests

`tests/test_materialization_v2.py` covers:

- fingerprint-compatible target cache resolution and copying;
- separate query-cache detection;
- explicit approval failure;
- tiny query-only computation and resume without recomputation;
- missing-query failure;
- candidate schema and source-manifest validation;
- selected-case schema and selection hashes;
- source/output/embedding manifest hashes;
- mismatched-output no-overwrite behavior;
- rejection of v1 candidate and selected-case protocols;
- explicit manifest confirmation that v1 primary outputs were not used.

The full unit suite and Ruff are run before commit.

## Remaining blockers before real Stage 1 validation

1. Approve or reject real query-only embedding computation after reviewing its command and
   estimated runtime.
2. Add/wire the fresh v2 candidate-set producer that supplies evidence scores and its source
   manifest. Materialization intentionally does not derive primary candidates from v1 outputs.
3. Add/wire the fresh selected-case producer under selected fusion/reranking settings.
4. Materialize the target item table whose row order matches cached target arrays.
5. Review all source manifests and hashes before running `tune-clip-fusion`.
6. Freeze a clean commit before any real validation output is selected.

## Current Git status and exact next safe command

After commit, the intended working tree is clean; the assistant handoff records the exact commit.

No real query computation is approved yet. The exact safe command remains:

```powershell
uv run efr --config configs/final_eval_v2.yaml validate-config
```

The next approval decision is whether to run the explicit query-only command for validation.
Do not run real `tune-clip-fusion` until candidate sources, query arrays, and materialization
manifests have been reviewed.
