# Final Evaluation v2 — Stage 1 orchestration handoff

## Outcome

Stage 1 now has three partition-guarded CLI commands backed by fingerprinted, resumable
artifacts. Implementation and verification used only tiny synthetic fixtures. No real dataset,
retrieval, generation, judging, validation, or final-evaluation workflow was run, and
`outputs/robustness/` was not modified.

## Commands

### 1. Select CLIP fusion weight on validation

```powershell
uv run efr --config configs/final_eval_v2.yaml tune-clip-fusion `
  --bundle outputs/final_eval_v2/prepared/validation `
  --output-dir outputs/final_eval_v2/validation/fusion_tuning
```

Writes:

```text
modality_results.csv
fusion_weight_validation.csv
selected_fusion.json
stage_manifest.json
```

The per-case table includes MiniLM text-only, CLIP image-only, CLIP text-only, and every
configured fused CLIP weight on identical candidate rows. Selection uses validation NDCG@10,
then HR@10, reciprocal rank, and balanced-weight preference.

### 2. Evaluate frozen Stage 1 settings on test

```powershell
uv run efr --config configs/final_eval_v2.yaml evaluate-final-retrieval-v2 `
  --bundle outputs/final_eval_v2/prepared/test `
  --fusion-selection outputs/final_eval_v2/validation/fusion_tuning/selected_fusion.json `
  --reranking-selection outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json `
  --locked-packets outputs/final_eval_v2/prepared/test/locked_packets.csv `
  --output-dir outputs/final_eval_v2/retrieval/test
```

Writes:

```text
test_ranking_results.csv
selected_reranking.json
locked_recommendation_evidence_packets.csv
stage_manifest.json
```

The command reads both selected artifacts and never substitutes hard-coded v2 weights.
Locked packets receive per-case generation-packet hashes, one complete Stage 1 packet-set hash,
and `stage1_packet_protocol=final_eval_v2_selected` for Stage 2 binding.

### 3. Run the legacy/v2 decision gate

```powershell
uv run efr --config configs/final_eval_v2.yaml compare-locked-artifacts-v2 `
  --legacy-packets outputs/robustness/final_study/explanations.csv `
  --v2-packets outputs/final_eval_v2/retrieval/test/locked_recommendation_evidence_packets.csv `
  --output-dir outputs/final_eval_v2/decision_gate
```

Writes:

```text
packet_comparison.csv
decision.json
stage_manifest.json
```

The conservative decision is either `regenerate_all_variants` or
`legacy_generation_v2_judging`.

## Prepared-bundle contract

Each validation/test bundle contains:

```text
schedule.csv
candidate_sets.csv
target_minilm.npy
target_clip_image.npy
target_clip_text.npy
query_minilm.npy
query_clip_image.npy
query_clip_text.npy
```

`schedule.csv` must contain unique `paper_case_id`, `query_outfit_id`, `target_category`, and
the single expected `research_split`. `candidate_sets.csv` must contain `paper_case_id`,
`candidate_position`, `target_row`, and `is_positive`; test evaluation also requires
`evidence_score`. Candidate rows are sorted once by case/position and reused across every
method and fusion weight.

The existing embedding cache already supports separate target MiniLM, CLIP-image, and CLIP-text
arrays. Query arrays and fixed candidate sets must be materialized into this bundle by an
approved preparation step before real validation. This slice deliberately does not run that
preparation over the real dataset.

## Resume and overwrite behavior

- Every stage fingerprints all material input files and method settings.
- A matching complete output directory is treated as a resume hit.
- A mismatched manifest fails rather than overwriting the directory.
- An existing directory without a manifest fails.
- Incomplete outputs fail visibly instead of silently resuming.
- All CLI output paths must resolve under `outputs/final_eval_v2/`.
- Legacy outputs remain read-only inputs to the decision gate.

## Held-out reranking correction

The historical `evaluate-heldout-ranking` command now accepts `--selection` and loads the
frozen validation-selected `clip_weight`; its former hard-coded `[0.9, 1.0]` behavior was
removed. The v2 test command independently requires its explicit reranking-selection path and
requires `selected_on=validation`.

## Tests

`tests/test_stage1_cli_v2.py` builds validation/test bundles containing two cases and two
candidates entirely under pytest's temporary directory. It invokes all three CLI commands
twice, proving artifact creation and idempotent resume, then verifies selected artifacts,
test results, packet hashes, and the unchanged-packet gate decision.

Related unit coverage remains in:

- `tests/test_modality.py`
- `tests/test_protocol_gate.py`
- `tests/test_tuning.py`
- `tests/test_final_eval_config.py`

## Remaining items before real validation

1. Materialize the real prepared validation/test bundles from the frozen schedules and cached
   target/query embeddings. This is a retrieval-preparation workflow and was not run here.
2. Produce/freeze the v2 validation reranking artifact at
   `outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json`. The v2 retrieval
   command consumes but does not tune this artifact.
3. Produce `locked_packets.csv` from the selected fusion/reranking configuration. The test
   command hashes and freezes this prepared input; it does not itself reconstruct catalogue
   metadata or retrieve KB rules.
4. Review the bundle preparation provenance and schedule hashes before approving Stage 1.
5. Run a development-only bundle-preparation smoke test before touching real validation.

## Safe next command

No real Stage 1 command should run until the prepared validation bundle and selected reranking
artifact are reviewed. The safe non-experimental check remains:

```powershell
uv run efr --config configs/final_eval_v2.yaml validate-config
```

Once bundle preparation is approved and complete, the first Stage 1 validation command is the
`tune-clip-fusion` invocation shown above.
