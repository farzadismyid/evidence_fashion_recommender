# Reproducibility protocol

## What is fixed

- Python and dependency versions are locked by `uv.lock`.
- Model names and optional revisions are stored in configuration.
- Seeds and deterministic settings are configured centrally.
- Every run records the resolved configuration, Git revision, dirty-worktree state,
  Python version, PyTorch/CUDA versions, and GPU name.
- Cache fingerprints include the model identity, source identity, configuration, and
  artifact schema.

## What reviewers need

```powershell
uv sync --extra dev --extra cuda
uv run efr --config configs/paper_baseline.yaml doctor
uv run pytest
uv run efr --config configs/paper_baseline.yaml prepare-data
```

CPU-only reviewers should use `uv sync --extra dev --extra cpu`.

Downloading the Polyvore dataset and Hugging Face models requires internet access.
Explanation experiments additionally require the configured Ollama model.

## Fair final evaluation

Development, validation, and final test cases must be separated. Model selection, prompt
changes, knowledge-base changes, thresholds, and weights should use development and
validation data only. The final test configuration should be committed before the held-out
run, and all attempted final runs should be retained.

Human-review sheets must be completed by raters who are blind to the method identity where
possible. An LLM judge is supporting evidence, not a substitute for human assessment.
