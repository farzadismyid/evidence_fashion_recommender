# Expanded recommendation and evidence analysis

This is the frozen 3,000-case **expanded confirmatory evaluation**, not an independent replication. It retains the historical 300 cases, adds 2,700 outcome-blind frozen-schedule cases, and uses only the validation-selected fusion plus the fixed 0.75 CLIP / 0.25 evidence reranker.

## Scope and integrity

- Methods: MiniLM text-only, CLIP image-only, CLIP text-only, validation-selected fused CLIP, and fused CLIP + evidence reranking (0.75 / 0.25).
- Historical 300 candidate evidence scores and query embeddings were reused unchanged. Only 2,700 new query embeddings and candidate evidence scores were computed.
- Every case uses all same-outfit target-category positives and up to 99 deterministic negatives from other outfits; this is controlled-pool ranking, not full-catalogue ranking.
- The frozen run manifest binds the protocol, schedule, candidate, and result hashes in `outputs/recommendation_eval_expanded/run_manifest.json`.

## Micro results

| Cohort | Method | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Historical 300 | Fused CLIP | .070 | .147 | .233 | .070 | .098 | .124 | .132 |
| Historical 300 | Evidence reranked | .073 | .130 | .230 | .073 | .092 | .121 | .128 |
| New 2,700 | Fused CLIP | .046 | .141 | .235 | .046 | .083 | .110 | .115 |
| New 2,700 | Evidence reranked | .040 | .135 | .224 | .040 | .075 | .102 | .107 |
| Expanded 3,000 | MiniLM | .026 | .088 | .161 | .026 | .049 | .069 | .080 |
| Expanded 3,000 | CLIP image | .045 | .154 | .247 | .045 | .084 | .112 | .117 |
| Expanded 3,000 | CLIP text | .038 | .119 | .195 | .038 | .070 | .092 | .099 |
| Expanded 3,000 | Fused CLIP | .049 | .141 | .235 | .049 | .085 | .111 | .116 |
| Expanded 3,000 | Evidence reranked | .043 | .134 | .225 | .043 | .077 | .104 | .109 |

Category-level and category-macro tables are saved alongside the machine-readable micro table. The expanded category-macro HR@10 is .231 for fused CLIP and .219 for evidence reranking; category macro metrics are reported so the large accessories stratum does not determine the conclusion.

## Pool and cluster composition

| Category | Cases | Query outfits | Mean relevant items | Candidate rows | Mean candidates |
|---|---:|---:|---:|---:|---:|
| accessories | 973 | 838 | 1.82 | 98,098 | 100.82 |
| bottoms | 497 | 443 | 1.02 | 49,710 | 100.02 |
| outerwear | 291 | 257 | 1.00 | 29,101 | 100.00 |
| shoes | 679 | 603 | 1.04 | 67,929 | 100.04 |
| tops | 560 | 487 | 1.09 | 56,051 | 100.09 |

There are 1,829 query-outfit clusters: mean 1.64 cases/outfit, median 1, maximum 8. The distribution is 1 case: 1,093 outfits; 2: 454; 3: 174; 4: 75; 5: 25; 6: 5; 7: 2; 8: 1.

## Paired outfit-clustered bootstrap

All intervals use 5,000 query-outfit-clustered resamples, seed 42, and 95% percentiles. Differences are first method minus second method.

| Metric | Contrast | Difference | 95% CI | p |
|---|---|---:|---:|---:|
| HR@10 | Evidence reranked − fused CLIP | −.010 | [−.021, .001] | .0948 |
| NDCG@10 | Evidence reranked − fused CLIP | −.0077 | [−.0127, −.0028] | .0032 |
| MRR | Evidence reranked − fused CLIP | −.0072 | [−.0118, −.0026] | .0020 |
| HR@10 | Fused CLIP − MiniLM | +.074 | [.056, .093] | < 1/5000 |
| HR@10 | Fused CLIP − CLIP image | −.012 | [−.031, .007] | .2248 |
| HR@10 | Fused CLIP − CLIP text | +.040 | [.029, .051] | < 1/5000 |
| NDCG@10 | Fused CLIP − CLIP image | −.0007 | [−.0108, .0093] | .8864 |
| MRR | Fused CLIP − CLIP image | −.0006 | [−.0100, .0087] | .9168 |

The reranker is **statistically distinguishable from fused CLIP on NDCG@10 and MRR** (lower), while HR@10 is inconclusive. It should therefore be described as close on HR@10 but not equivalent, and as worse on the rank-sensitive NDCG@10/MRR outcomes. Fused CLIP remains clearly above MiniLM and CLIP text; relative to CLIP image, the expanded comparison is inconclusive for HR@10, NDCG@10, and MRR.

## Interpretation versus the historical 300

The original historical evaluation found fused CLIP and evidence reranking close on HR@10 (.233 versus .230). The 2,700 new cases and full expanded cohort preserve the broad conclusion that evidence reranking is near fused CLIP on HR@10, but they change its strength: the expanded data identify lower NDCG@10 and MRR for the fixed evidence reranker. The confirmatory conclusion is therefore that the frozen 0.75/0.25 operating point remains retrieval-competitive in hit rate but does not preserve fused-CLIP ranking quality on all metrics.

## Outputs

- `outputs/recommendation_eval_expanded/micro_results.csv`
- `outputs/recommendation_eval_expanded/category_results.csv`
- `outputs/recommendation_eval_expanded/category_macro_results.csv`
- `outputs/recommendation_eval_expanded/candidate_and_relevance_by_category.csv`
- `outputs/recommendation_eval_expanded/outfit_cluster_summary.csv`
- `outputs/recommendation_eval_expanded/outfit_cluster_distribution.csv`
- `outputs/recommendation_eval_expanded/outfit_clustered_bootstrap_comparisons.csv`
