# Modular baseline results

Run date: 2026-07-30

The modular evaluation uses the same 300 controlled cases, 99 same-category negatives,
query construction, target embeddings, and random seed as the archived notebook. The
case table matches the archived table exactly.

## Ranking comparison

| Configuration | Model | HitRate@1 | HitRate@5 | HitRate@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| Archived notebook | Text baseline | 0.0233 | 0.0900 | 0.1400 | 0.0585 |
| Modular | Text baseline | 0.0233 | 0.0900 | 0.1400 | 0.0585 |
| Archived notebook | CLIP multimodal | 0.0533 | 0.1633 | 0.2433 | 0.1200 |
| Modular | CLIP multimodal | 0.0533 | 0.1600 | 0.2467 | 0.1209 |
| Archived notebook, default KB | Evidence reranked, 0.65/0.35 | 0.0400 | 0.1367 | 0.2100 | 0.0999 |
| Modular, KB v3 | Evidence reranked, 0.65/0.35 | 0.0400 | 0.1133 | 0.2400 | 0.1082 |
| Modular, KB v3 | Evidence reranked, 0.90/0.10 | 0.0467 | 0.1533 | 0.2367 | 0.1168 |

The text baseline is reproduced exactly. CLIP differs by one case at HitRate@10, consistent
with the Transformers 4-to-5 compatibility path used for fresh query embeddings. Target
embeddings and all evaluation cases are identical.

KB v3 materially closes the archived evidence-reranking gap at HitRate@10. A lighter
evidence weight improves early ranking and NDCG relative to the 0.65/0.35 v3 setting,
although it does not surpass pure CLIP. This remains an accuracy-grounding trade-off and
must not be reported as evidence reranking beating CLIP.

## Reproduction runs

- `outputs/runs/paper-baseline-v3_20260730T190439Z`
- `outputs/runs/paper-improved-light-rerank_20260730T190522Z`
- `outputs/runs/paper-improved-light-rerank_20260730T190808Z`

Generated output directories are intentionally not tracked. Each contains its resolved
configuration, environment manifest, logs, predictions, evidence, and metrics.

## Explanation status

The end-to-end modular command successfully generated all four variants for a two-item
smoke test. The archived 400-explanation type-filtered v3 study remains preserved as the
historical paper baseline. A new full generation run should only be launched after the
prompt and evaluation protocol are frozen because sampling 400 new responses constitutes
a new experimental run, not a mechanical migration check.

Human review remains an external requirement. The archived 120-row review sheet contains
no completed ratings and must not be described as completed human evaluation.

