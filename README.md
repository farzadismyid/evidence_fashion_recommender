# Evidence-Constrained Multimodal Fashion Recommendation

A reproducible research pipeline for complementary fashion recommendation and faithful,
evidence-grounded explanations. It combines text and image retrieval, validation-selected
CLIP fusion, evidence-in-the-loop reranking, controlled RAG variants, atomic-claim
verification, and cross-model explanation judging.

The original notebook implementation is preserved under [`archive/`](archive/README.md).
The completed modular final evaluation is implemented under `src/` and configured by
[`configs/final_eval_v2.yaml`](configs/final_eval_v2.yaml).

## Final evaluation status

The v2 evaluation is complete through Stage 4D targeted recovery:

- Stage 1: frozen validation/test retrieval and evidence packets.
- Stage 2: validation-only Hybrid-RAG configuration selection.
- Stage 3: 3,600 preserved explanations: 300 cases x 4 variants x 3 generators.
- Stage 4A: separate extraction of all atomic claims, with no three-claim cap.
- Stage 4B: separate claim verification; unavailable rows remain N/A.
- Stage 4C: 10,800 general judgments from three judges. Cross-model-only results are
  primary; self-judge-inclusive results are sensitivity diagnostics.
- Stage 4D: key-targeted recovery of explicit failed/N/A rows followed by one deterministic
  post-recovery merge and analysis.

The proposed reranker is the validation-selected evidence-in-loop operating point:

```text
selection policy: evidence_in_loop_pareto_v2
CLIP weight:      0.75
evidence weight:  0.25
```

CLIP 1.00 / evidence 0.00 is retained only as the accuracy-optimal baseline, not as the
proposed method.

## Installation

Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/) are required. The exact resolved
environment is committed in `uv.lock`.

```powershell
uv sync --extra dev --extra cuda
uv run --extra cuda efr --config configs/final_eval_v2.yaml validate-config
uv run --extra cuda efr --config configs/final_eval_v2.yaml doctor
```

For CPU-only inspection and tests, replace `cuda` with `cpu`. Explanation generation and
LLM evaluation require the configured Ollama models and their recorded model digests.

## Reproducing the final pipeline

Generated datasets, embeddings, caches, and full checkpoints are intentionally excluded from
Git. Restore the inputs at the paths recorded by the manifests before running expensive
commands. Then follow [`docs/reproducibility.md`](docs/reproducibility.md), which documents
the stage order, resume behavior, integrity rules, and exact final commands.

The final Stage 3-4 command sequence is:

```powershell
uv run --extra cuda efr --config configs/final_eval_v2.yaml freeze-final-eval-v2
uv run --extra cuda efr --config configs/final_eval_v2.yaml run-final-explanations-v2
uv run --extra cuda efr --config configs/final_eval_v2.yaml extract-claims-v2
uv run --extra cuda efr --config configs/final_eval_v2.yaml verify-claims-v2
uv run --extra cuda efr --config configs/final_eval_v2.yaml judge-general-quality-v2
```

Each costly evaluation command is checkpointed and resumes completed rows. Stage 3 explanation
text is never rewritten during evaluation or recovery.

Stage 4D is a recovery operation, not part of a clean first run. It targets only explicit
failed/N/A keys and uses a larger output allowance without changing prompts or scoring:

```powershell
uv run --extra cuda efr --config configs/final_eval_v2.yaml recover-stage4d-v2 `
  --artifact-root outputs/final_eval_v2 `
  --recovery-root outputs/final_eval_v2/recovery/stage4d `
  --post-root outputs/final_eval_v2/post_recovery `
  --max-tokens 1600
```

## Evaluation contract

- Validation selects fusion, evidence-reranking, and Hybrid-RAG settings; test does not.
- The proposed method always has evidence in the reranking loop.
- Extraction and verification remain separate model calls and artifacts.
- Empty or failed extraction is N/A, never perfect support.
- Failed extraction, verification, or judging is N/A, never zero, one, unsupported, or a
  fabricated score.
- Length compliance (`word_count` and `over_35_words`) is reported separately from explanation
  quality. Over-length explanations are retained unchanged.
- Cross-model-only judging is the primary analysis. All-judge results, including self-judging,
  are sensitivity diagnostics only.
- Recovery copies successful source rows byte-for-byte at the record level and validates their
  canonical hashes before accepting a merged table.

## Results and audit artifacts

The committed compact result bundles are:

- [`reports/final_eval_v2/pre_recovery`](reports/final_eval_v2/pre_recovery): immutable
  pre-recovery analysis.
- [`reports/final_eval_v2/post_recovery`](reports/final_eval_v2/post_recovery): final
  post-recovery tables, figures, artifact inventory, and handoff.
- [`reports/final_eval_v2/post_recovery/STAGE4D_HANDOFF.md`](reports/final_eval_v2/post_recovery/STAGE4D_HANDOFF.md):
  recovery counts, integrity checks, and headline results.
- `outputs/final_eval_v2/recovery/stage4d`: committed compact source hashes, completion manifest,
  and per-key recovery audits. Large merged tables remain ignored under `outputs/`.

## Repository structure

```text
configs/     validated experiment configurations
data/        local datasets and knowledge base (large derived data ignored)
docs/        methodology, protocols, architecture, and reproducibility
outputs/     generated runs, caches, checkpoints, and merged tables (ignored by default)
reports/     committed compact tables, figures, manifests, and handoffs
src/         installable evidence_fashion_recommender package
tests/       unit and synthetic integration tests
archive/     preserved exploratory notebook implementation
```

## Validation

```powershell
uv run --extra cuda --extra dev ruff check src tests
uv run --extra cuda --extra dev pytest -q
```

The completed Stage 4D commit passed Ruff and all 79 tests. Run manifests additionally bind
configuration, source artifacts, model identities, row counts, and hashes.

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). The project is released under the
[`MIT License`](LICENSE).
