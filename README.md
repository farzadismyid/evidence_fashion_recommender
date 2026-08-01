# Evidence-Constrained Multimodal Fashion Recommendation

A reproducible research package for complementary fashion recommendation and faithful,
evidence-grounded explanations. It combines multimodal item retrieval with expert styling
rules, evidence-aware reranking, and controlled explanation ablations.

The original exploratory implementation and its results are preserved in
[`archive/`](archive/README.md). New experiments use the modular package under `src/`.

## Project Goal

The project investigates whether retrieved fashion knowledge can improve both recommendation quality and explanation faithfulness in a multimodal fashion recommendation pipeline.
The system takes:

* a query fashion item image or text description
* a user request
* a target recommendation category
* retrieved fashion rules from a knowledge base

and produces:

* top-K complementary item recommendations
* evidence-constrained reranking
* explanations using only retrieved fashion evidence

## Research Focus

The main research focus is:

> Can a fashion recommender produce useful complementary recommendations while keeping explanations faithful to retrieved fashion knowledge?

This project treats explanation faithfulness as a core part of the recommendation process, not as a post-hoc description added after retrieval.

## Current Pipeline

The planned pipeline is:

1. Load and inspect a Polyvore-style fashion dataset.
2. Represent fashion items using image/text embeddings.
3. Retrieve candidate complementary items from the dataset.
4. Retrieve relevant fashion rules from a curated knowledge base.
5. Rerank candidates using visual-textual compatibility and retrieved evidence.
6. Generate explanations constrained only to retrieved evidence.
7. Use a critic/validation step to check whether the explanation is supported by the evidence.
8. Evaluate recommendations and explanations using ablation experiments.

## Current Implementation Status

Completed so far:

* Project repository initialised.
* Python environment configured using `uv`.
* PyTorch CUDA setup tested.
* Polyvore-style dataset loaded and inspected.
* Basic exploratory data analysis completed.
* Outfit/category distribution inspected.
* Fashion item text fields prepared.
* Initial embedding-based similarity testing completed.
* Same-outfit and different-outfit item comparisons tested.
* Early baseline recommendation logic started.

In progress:

* Source-grounded fashion rules knowledge base.
* RAG-ready CSV format for fashion rules.
* Evidence retrieval design.
* Recommendation reranking with retrieved rules.

Planned next:

* Add `data/kb/fashion_rules.csv`.
* Build a rule retriever using semantic search.
* Combine candidate retrieval with evidence-aware scoring.
* Add faithful explanation generation.
* Add critic/validator step.
* Run ablation experiments.

## Project Structure

```text
evidence_fashion_recommender/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── notebook.ipynb
│
├── data/
│   ├── kb/
│   │   ├── README.md
│   │   └── fashion_rules.csv
│   │
│   └── sample/
│
├── outputs/
│
└── .gitignore
```

Some folders may be empty during early development. Dataset files and generated outputs should not be committed unless they are small, necessary, and reproducible.

## Environment Setup

This project uses `uv` for dependency management.

Create and activate the environment:

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
tops, bottoms, shoes, outerwear, accessories
```

## Evidence-Constrained Explanation

The explanation module should only use retrieved evidence from the KB.

For example, if the system recommends shoes for a dress, the explanation must be based on retrieved rules about dress-shoe pairing, formality, colour harmony, occasion, or silhouette.

The explanation should not invent fashion logic that is not present in the retrieved evidence.

## Planned Ablation Study

The project will compare several system variants:

| Variant               | Description                                                                           |
| --------------------- | ------------------------------------------------------------------------------------- |
| Baseline              | Recommendation using item similarity only                                             |
| Prompt-only           | Recommendation with fixed styling instructions                                        |
| RAG                   | Recommendation using retrieved fashion rules                                          |
| Prompt + RAG          | Recommendation using instructions and retrieved evidence                              |
| Prompt + RAG + Critic | Recommendation with evidence validation                                               |
| Full system           | Multimodal retrieval, evidence reranking, faithful explanation, and critic validation |

## Evaluation Plan

Recommendation evaluation may include:

* top-K retrieval quality
* category correctness
* compatibility comparison
* same-outfit vs different-outfit similarity
* ranking inspection
* qualitative recommendation analysis

Explanation evaluation may include:

* evidence usage
* citation overlap
* unsupported claim detection
* faithfulness scoring
* human inspection
* ablation comparison

## Reproducibility Notes

To keep the project reproducible:

* Keep notebooks clean and sequential.
* Use fixed random seeds where possible.
* Avoid committing large datasets.
* Document dataset source and preprocessing steps.
* Save generated outputs under `outputs/`.
* Keep the fashion KB in CSV format.
* Track major experiments with clear section headings.

Project title:

**Evidence-Constrained Multimodal Fashion Recommendation with Faithful Explanations**

Research area:

* multimodal recommendation
* fashion recommendation
* retrieval-augmented generation
* faithful explanation generation
* evidence-constrained AI systems
