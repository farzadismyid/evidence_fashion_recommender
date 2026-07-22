# Evidence-Constrained Multimodal Fashion Recommendation with Faithful Explanations

This repository contains the implementation for an MPhil Artificial Intelligence thesis project on evidence-constrained fashion recommendation.

The system is designed to recommend complementary fashion items from a Polyvore-style dataset using both multimodal item understanding and retrieved fashion knowledge. The main goal is not only to recommend visually compatible items, but also to generate faithful explanations that are grounded in retrieved evidence rather than unsupported LLM reasoning.

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

```bash
uv venv
```

Install dependencies:

```bash
uv sync
```

If PyTorch CUDA installation causes dependency conflicts with `uv`, install the CUDA build separately using `pip` inside the virtual environment.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Core Dependencies

Main expected libraries:

* Python
* PyTorch
* Transformers
* sentence-transformers
* pandas
* numpy
* scikit-learn
* matplotlib
* tqdm
* Jupyter

Additional libraries may be added as the RAG and explanation modules are implemented.

## Dataset

The project uses a Polyvore-style fashion dataset containing outfit items, item categories, item descriptions, and outfit-level groupings.

The dataset is used for:

* item retrieval
* outfit compatibility testing
* category-based recommendation
* complementary item recommendation
* baseline evaluation

The current early-stage work has focused on loading the dataset, checking category distributions, inspecting outfit composition, and testing whether embedding similarity can identify meaningful item relationships.

## Fashion Knowledge Base

The fashion knowledge base will be stored at:

```text
data/kb/fashion_rules.csv
```

The KB contains source-grounded fashion rules collected from reputable fashion editorials, retailer style guides, museum or institutional sources, academic papers, and industry sources.

Each rule is designed to be:

* atomic
* source-grounded
* useful for retrieval
* suitable for explanation generation
* linked to input and recommended categories
* tagged by style, colour, occasion, season, fit, gender context, and formality

The KB is not intended to represent universal scientific laws. It is a curated evidence-grounded styling resource for recommendation and explanation.

## KB Schema

The required CSV columns are:

```text
rule_id
rule_text
input_category
recommended_category
style_tags
colour_tags
occasion_tags
season_tags
gender_context
body_fit_context
formality_level
evidence_keywords
source_type
source_title
source_author_or_org
source_year
source_url_or_reference
source_reliability
notes
```

Valid input categories:

```text
tops, bottoms, shoes, outerwear, accessories, dresses
```

Valid recommended categories:

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

## Current Research Limitation

Fashion styling rules are partly subjective and context-dependent. Therefore, the KB should be described as a source-grounded styling heuristic resource, not as an objective universal truth.

The academic contribution is not that each fashion rule is scientifically proven. The contribution is that the recommendation and explanation pipeline is constrained by retrieved evidence, making the explanation process more transparent and faithful.

## Thesis Context

Project title:

**Evidence-Constrained Multimodal Fashion Recommendation with Faithful Explanations**

Research area:

* multimodal recommendation
* fashion recommendation
* retrieval-augmented generation
* faithful explanation generation
* evidence-constrained AI systems
