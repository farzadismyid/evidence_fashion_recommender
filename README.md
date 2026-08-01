# Evidence-Constrained Multimodal Fashion Recommendation

A reproducible research package for complementary fashion recommendation and faithful,
evidence-grounded explanations. It combines multimodal item retrieval with expert styling
rules, evidence-aware reranking, and controlled explanation ablations.

The original exploratory implementation and its results are preserved in
[`archive/`](archive/README.md). New experiments use the modular package under `src/`.

## Research question

Can expert-authored styling rules improve the faithfulness and usefulness of explanations
for multimodal fashion recommendations without hiding the trade-off with ranking accuracy?

## Quick start

Requirements:

- Python 3.11
- `uv`
- Internet access for the dataset and Hugging Face models
- Optional NVIDIA GPU for practical multimodal experiments
- Ollama and the configured local model for generation and LLM-judge stages

```powershell
uv sync --extra dev --extra cuda
uv run efr --config configs/default.yaml validate-config
uv run efr --config configs/default.yaml doctor
uv run efr --config configs/default.yaml audit-kb
uv run pytest
```

For a CPU-only reviewer environment, replace `--extra cuda` with `--extra cpu`. The two
profiles are intentionally mutually exclusive.

Prepare and cache the Polyvore metadata:

```powershell
uv run efr --config configs/default.yaml prepare-data
```

Build or reuse all target embeddings:

```powershell
uv run efr --config configs/paper_baseline.yaml build-embeddings
uv run efr --config configs/paper_baseline.yaml build-indexes
```

Run the controlled paper baseline and improved light-rerank experiment:

```powershell
uv run efr --config configs/paper_baseline.yaml evaluate-ranking
uv run efr --config configs/paper_improved.yaml evaluate-ranking
```

Run the complete 400-explanation systematic study:

```powershell
uv run efr --config configs/paper_improved.yaml build-study-cases `
  --output outputs/modular_study_cases.csv
uv run efr --config configs/paper_baseline.yaml run-explanation-study `
  --input outputs/modular_study_cases.csv
```

This regenerates and caches No-RAG, Item-RAG, Rule-RAG, and Hybrid-RAG explanations,
evaluates citations, evidence overlap, unsupported claims, occasion drift, candidate
substitution and prompt leakage, evaluates RAG retrieval, runs an independent Qwen3 judge,
and performs corrected paired statistical tests. The study input is rebuilt by the modular
retrieval, evidence, and reranking pipeline rather than read directly from notebook results.

Optional Florence captioning is available independently:

```powershell
uv run efr --config configs/default.yaml caption-image --image path/to/image.jpg
```

Run one complete recommendation, evidence, reranking, and explanation workflow:

```powershell
uv run efr --config configs/paper_improved.yaml `
  recommend `
  --query-item-id 100002074_1 `
  --target-category shoes `
  --request "recommend smart casual shoes" `
  --generate
```

The convenience script [`scripts/reproduce_baseline.ps1`](scripts/reproduce_baseline.ps1)
runs environment checks, KB audit, tests, embedding preparation, and both ranking
experiments.

Preview the configured methodology:

```powershell
uv run efr --config configs/paper_baseline.yaml show-plan
```

Override any declared experimental variable without editing Python:

```powershell
uv run efr --config configs/paper_baseline.yaml `
  --set models.generator.name=llama3.3 `
  --set retrieval.final_top_k=10 `
  --set evaluation.controlled_cases=500 `
  show-plan
```

Unknown keys and incompatible settings are rejected before a costly run begins.

## Configuration

[`configs/default.yaml`](configs/default.yaml) declares dataset versions, model providers
and names, batch sizes, fusion weights, retrieval depth, evidence settings, reranking
weights, generation variants, temperatures, evaluation sizes, seeds, cache policy, and
output behaviour.

[`configs/paper_baseline.yaml`](configs/paper_baseline.yaml) inherits from the defaults and
freezes the first modular paper baseline. Create a new named configuration for every
reported experiment; do not overwrite a historical experiment definition.

Model revisions should be pinned before a final paper or thesis run. Changing a model name
is sufficient when the provider and model architecture are compatible. Changing model
families may also require changing the provider adapter.

## Outputs and caching

Every run receives a new immutable directory:

```text
outputs/runs/<experiment>_<UTC timestamp>/
├── config_resolved.yaml
├── run_manifest.json
├── logs/
├── embeddings/
├── indexes/
├── predictions/
├── metrics/
├── figures/
└── reports/
```

Large reusable artifacts live in `outputs/cache/` under content-derived fingerprints.
Matching artifacts are reused; incompatible models or settings receive different keys.
Generated artifacts and caches are excluded from Git.

## Methodology

The project supports:

- MiniLM text retrieval
- CLIP image, text, and fused multimodal retrieval
- Category-aware FAISS indexes
- Candidate and image-quality filtering
- Versioned expert-rule knowledge bases
- Query-level and candidate-specific evidence retrieval
- Candidate-type-aware evidence filtering
- Weighted evidence reranking and weight ablations
- Florence-2 visual captions
- No-RAG, Item-RAG, Rule-RAG, and Hybrid-RAG explanations
- Candidate-locked, leakage-safe prompts
- Controlled candidate-set recommendation evaluation
- Citation, overlap, unsupported-claim, retrieval, and independent LLM-judge evaluation
- Paired bootstrap intervals and multiple-comparison corrections

See [methodology](docs/methodology.md), [architecture](docs/architecture.md),
[caching](docs/caching.md), [reproducibility](docs/reproducibility.md), and
[human evaluation](docs/human_evaluation.md).

## Verified modular results

The 300-case modular run reproduces the archived text baseline exactly and the CLIP
HitRate@10 within one case. KB v3 evidence reranking improves the archived evidence
HitRate@10 from 0.210 to 0.240. The light 0.90/0.10 reranker reaches NDCG@10 of 0.1168,
close to pure CLIP at 0.1209. See the
[result report](reports/modular_baseline_results.md) for the complete comparison and
limitations.

The completed 400-explanation systematic results—including faithfulness, retrieval,
independent-judge, and corrected statistical evaluation—are reported in
[the final systematic results](reports/final_systematic_results.md).

[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) provides a short,
notebook-style introduction without duplicating implementation code.

## Research integrity

Improvements should be selected on development and validation data. The final held-out
protocol must be frozen before evaluation. LLM judges are supporting tools, and unsuccessful
variants must be reported as well as favourable results.

The current modular report intentionally uses systematic evaluation only. Human evaluation
is excluded from the present scope and documented as future work; the project does not
claim human preference or human-perceived explanation quality.

## Robustness phase

The extended protocol freezes the original report, assigns complete outfits—not
individual items—to disjoint development, validation, and test partitions, tunes only on
validation, and evaluates 300 balanced held-out explanation cases with three generators
and three independent judges.

Run the cached end-to-end robustness workflow:

```powershell
.\scripts\reproduce_robustness.ps1 -Profile cuda
```

The workflow includes Hybrid-RAG prompt and evidence-budget ablations, validation-only
reranking selection, claim-level verification, corrected substitution detection,
cross-model judge sensitivity, agreement statistics, KB-proxy and consensus retrieval
evaluation, counterfactual retrieval tests, and a final before-versus-after report. See
the [robustness protocol](docs/robustness_protocol.md) for the leakage and interpretation
rules. Human review is still future work.

The completed frozen study contains 3,600 explanations, 10,800 explanation judgments,
and 4,500 rule-relevance labels, with zero final parsing errors. Its main result is a
trade-off: Hybrid-RAG gives the highest evidence overlap and 78% fewer deterministic
unsupported claims than No-RAG, while No-RAG has the highest aggregate model-judge score;
Rule-RAG has the lowest deterministic unsupported-claim count. Recommendation reranking
raises held-out HitRate@10 from 0.260 to 0.270 but lowers NDCG@10 from 0.1377 to 0.1348.
See the [final robustness results](reports/final_robustness_results.md) and generated
`outputs/robustness/final_report/FINAL_ROBUSTNESS_REPORT.md`.
