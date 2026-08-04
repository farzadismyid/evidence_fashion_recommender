# Reproducibility protocol

## Environment and immutable inputs

- Python dependencies are resolved by `uv.lock`; use Python 3.11 or 3.12.
- `configs/final_eval_v2.yaml` is the authoritative final-evaluation configuration.
- Every run records the resolved configuration, Git revision and dirty state, runtime and GPU
  details, model identity/digest/settings, source identities, and artifact schema.
- Content-addressed caches include model, prompt, configuration, and source fingerprints.
- Large data, embeddings, model caches, and evaluation checkpoints under `outputs/` are local
  artifacts. Compact final reports and recovery audits are committed.

Setup and preflight:

```powershell
uv sync --extra dev --extra cuda
uv run --extra cuda efr --config configs/final_eval_v2.yaml validate-config
uv run --extra cuda efr --config configs/final_eval_v2.yaml doctor
uv run --extra cuda efr --config configs/final_eval_v2.yaml inspect-final-eval-v2-readiness
```

CPU-only reviewers can use `--extra cpu`. Dataset/model downloads require internet access.
Generation and LLM evaluation require the configured Ollama models; reproducing recorded outputs
also requires matching their stored digests.

## Final-evaluation order

Do not select settings on the test split. The detailed preparation commands and input contracts
are recorded in the Stage 1 handoffs under `reports/`.

1. Materialize validation inputs and select CLIP fusion on validation:

   ```powershell
   uv run --extra cuda efr --config configs/final_eval_v2.yaml tune-clip-fusion
   uv run --extra cuda efr --config configs/final_eval_v2.yaml tune-reranking-v2
   uv run --extra cuda efr --config configs/final_eval_v2.yaml select-evidence-in-loop-reranking-v2
   ```

2. Freeze validation packets and select Hybrid-RAG using validation only:

   ```powershell
   uv run --extra cuda efr --config configs/final_eval_v2.yaml run-hybrid-validation-v2
   ```

3. Materialize test selected cases/packets, evaluate frozen retrieval settings, run the decision
   gate, and freeze the final protocol. Use the explicit paths in
   `reports/final_eval_v2_evidence_in_loop_correction_handoff.md` and
   `reports/final_eval_v2_stage1_orchestration_handoff.md`.

4. Generate and evaluate the 3,600 explanations:

   ```powershell
   uv run --extra cuda efr --config configs/final_eval_v2.yaml freeze-final-eval-v2
   uv run --extra cuda efr --config configs/final_eval_v2.yaml run-final-explanations-v2
   uv run --extra cuda efr --config configs/final_eval_v2.yaml extract-claims-v2
   uv run --extra cuda efr --config configs/final_eval_v2.yaml verify-claims-v2
   uv run --extra cuda efr --config configs/final_eval_v2.yaml judge-general-quality-v2
   ```

5. Build the immutable pre-recovery analysis once:

   ```powershell
   uv run --extra cuda efr --config configs/final_eval_v2.yaml analyze-final-eval-v2 `
     --artifact-root outputs/final_eval_v2 `
     --output-dir reports/final_eval_v2/pre_recovery
   ```

## Resume and failure semantics

The three Stage 4 commands keep separate checkpoints and resume completed keys. Never delete a
checkpoint merely to repeat failures. Row-level malformed output follows the command's bounded
retry/repair policy; unresolved rows are logged and retained as N/A. In particular:

- extract every atomic claim with no claim-count cap;
- never infer perfect support from an empty extraction;
- never map N/A to unsupported, support=0, support=1, or a judge score;
- never truncate, rewrite, or regenerate Stage 3 text for evaluation;
- evaluate word count and the 35-word limit separately from quality;
- use cross-model-only judging for primary claims and self-judge-inclusive results only for
  sensitivity analysis.

## Targeted recovery and deterministic merge

Stage 4D should run only after preserving the pre-recovery analysis. Its source and destination
roots must differ:

```powershell
uv run --extra cuda efr --config configs/final_eval_v2.yaml recover-stage4d-v2 `
  --artifact-root outputs/final_eval_v2 `
  --recovery-root outputs/final_eval_v2/recovery/stage4d `
  --post-root outputs/final_eval_v2/post_recovery `
  --max-tokens 1600
```

The merge starts from the original completed tables and matches existing unique extraction,
verification, and judgment keys. Successful rows must retain identical canonical hashes. An N/A
row is replaced only if the recovered output passes the existing schema, key, claim-ID, label,
and semantic validation; otherwise the original N/A record remains. Every attempted replacement
records its old/new status, method, and original/recovered hashes.

Generate the post-recovery analysis exactly once from the new merged root:

```powershell
uv run --extra cuda efr --config configs/final_eval_v2.yaml analyze-final-eval-v2 `
  --artifact-root outputs/final_eval_v2/post_recovery `
  --output-dir reports/final_eval_v2/post_recovery `
  --analysis-label "POST-RECOVERY FINAL ANALYSIS" `
  --stage4d-recovery-run
```

## Required validation

```powershell
uv run --extra cuda --extra dev ruff check src tests
uv run --extra cuda --extra dev pytest -q
```

Also verify:

- 3,600 unique explanation/extraction/verification keys;
- 10,800 unique judgment keys;
- unchanged Stage 3 explanation hash;
- unchanged canonical hashes for every originally successful row;
- artifact inventory hashes and manifest bindings;
- exact extraction, verification, and judgment N/A coverage;
- clean `git status` after committing compact reproducibility artifacts.

The completed post-recovery run and its exact counts are documented in
`reports/final_eval_v2/post_recovery/STAGE4D_HANDOFF.md`.
