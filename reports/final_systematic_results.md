# Final modular systematic results

The definitive modular run uses 300 controlled recommendation cases (99
same-category negatives per case), 100 freshly rebuilt recommendation/evidence cases,
and 400 newly generated explanations. Human review is intentionally excluded from this
study and is future work.

## Recommendation quality

| Method | HR@1 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| MiniLM text | 0.0233 | 0.0900 | 0.1400 | 0.0585 |
| CLIP multimodal | 0.0533 | 0.1600 | 0.2467 | 0.1209 |
| Light evidence rerank (0.90/0.10) | 0.0467 | 0.1533 | 0.2367 | 0.1168 |

Evidence reranking presents an accuracy/grounding trade-off; it is not claimed to
outperform pure CLIP.

## Explanation and faithfulness quality

| Variant | Unsupported claims | Evidence overlap | Citation presence | Judge faithfulness | Judge usefulness | Overall judge |
|---|---:|---:|---:|---:|---:|---:|
| No-RAG | 0.19 | 0.000 | 0.00 | 3.64 | 4.12 | 4.182 |
| Item-RAG | 0.24 | 0.108 | 0.00 | 4.76 | 4.75 | 4.828 |
| Rule-RAG | 0.08 | 0.389 | 1.00 | 4.02 | 4.23 | 4.392 |
| Hybrid-RAG | 0.04 | 0.470 | 1.00 | 4.10 | 4.24 | 4.460 |

All supplied citations had 1.00 measured correctness and precision. Prompt leakage was
0.00 for three variants and 0.01 for Rule-RAG. Occasion drift ranged from 0.00 to 0.02
claims per explanation. The independent judge is pinned Qwen3:8b and is supporting
model-based evidence, not human evidence.

## Retrieval quality

Rule retrieval returned five rules and achieved evidence coverage of 1.00 for every
target category. HitRate@5 was 0.60 for accessories, 0.50 for bottoms, and 0.90 for
outerwear, shoes, and tops. MRR ranged from 0.214 for accessories to 0.750 for tops.
Precision, recall, hit rate, NDCG, reciprocal rank, category compatibility, reliability,
and unique-rule usage are all exported by category.

The full machine-readable tables, figures, resolved configurations, and manifests are in
`outputs/final/`. Pairwise tests use 5,000 paired bootstrap resamples and include Holm
and Benjamini-Hochberg corrections.

## Scope limitation

Human evaluation is excluded from the present work. It should be added later to measure
human preference, perceived usefulness, and agreement; none of those claims are made
from the current systematic metrics.
