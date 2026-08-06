# Post-recovery claim-level clustered bootstrap

Claim rates use only `verification_status == complete`; extraction and verification N/A rows are excluded from those denominators. N/A coverage is separately the fraction of explanation-level checkpoint records with `verification_status == N/A`.

Case-level clusters: 300; replicates: 5000; seed: 42; intervals: 95% percentile.

## Variant estimates

| Metric | Variant | Estimate (95% CI) | Numerator / denominator |
|---|---|---:|---:|
| any_permitted_evidence_support | no_rag | 56.72% (53.75%, 59.66%) | 1131 / 1994 |
| any_permitted_evidence_support | item_rag | 58.87% (55.78%, 61.97%) | 1092 / 1855 |
| any_permitted_evidence_support | rule_rag | 79.14% (77.09%, 81.12%) | 1552 / 1961 |
| any_permitted_evidence_support | hybrid_rag | 81.81% (79.80%, 83.86%) | 1565 / 1913 |
| generation_available_evidence_support | no_rag | N/A (not applicable) | 0 / 1994 |
| generation_available_evidence_support | item_rag | 38.81% (36.10%, 41.72%) | 720 / 1855 |
| generation_available_evidence_support | rule_rag | 70.42% (67.76%, 72.92%) | 1381 / 1961 |
| generation_available_evidence_support | hybrid_rag | 77.99% (75.79%, 80.21%) | 1492 / 1913 |
| rule_support | no_rag | 36.51% (33.45%, 39.55%) | 728 / 1994 |
| rule_support | item_rag | 16.17% (14.00%, 18.36%) | 300 / 1855 |
| rule_support | rule_rag | 70.42% (67.76%, 72.92%) | 1381 / 1961 |
| rule_support | hybrid_rag | 53.37% (50.77%, 55.84%) | 1021 / 1913 |
| item_support | no_rag | 15.30% (12.84%, 17.77%) | 305 / 1994 |
| item_support | item_rag | 38.81% (36.10%, 41.72%) | 720 / 1855 |
| item_support | rule_rag | 6.07% (4.53%, 7.70%) | 119 / 1961 |
| item_support | hybrid_rag | 24.62% (22.40%, 26.89%) | 471 / 1913 |
| query_locked_item_support | no_rag | 4.91% (3.83%, 6.01%) | 98 / 1994 |
| query_locked_item_support | item_rag | 3.88% (2.96%, 4.88%) | 72 / 1855 |
| query_locked_item_support | rule_rag | 2.65% (1.83%, 3.58%) | 52 / 1961 |
| query_locked_item_support | hybrid_rag | 3.82% (2.98%, 4.70%) | 73 / 1913 |
| unsupported | no_rag | 36.86% (33.97%, 39.99%) | 735 / 1994 |
| unsupported | item_rag | 34.88% (31.73%, 37.97%) | 647 / 1855 |
| unsupported | rule_rag | 18.20% (16.35%, 20.05%) | 357 / 1961 |
| unsupported | hybrid_rag | 16.10% (14.20%, 18.03%) | 308 / 1913 |
| contradicted | no_rag | 0.35% (0.14%, 0.63%) | 7 / 1994 |
| contradicted | item_rag | 0.05% (0.00%, 0.17%) | 1 / 1855 |
| contradicted | rule_rag | 0.25% (0.05%, 0.50%) | 5 / 1961 |
| contradicted | hybrid_rag | 0.47% (0.20%, 0.80%) | 9 / 1913 |
| not_verifiable | no_rag | 6.07% (4.93%, 7.23%) | 121 / 1994 |
| not_verifiable | item_rag | 6.20% (4.88%, 7.55%) | 115 / 1855 |
| not_verifiable | rule_rag | 2.40% (1.77%, 3.07%) | 47 / 1961 |
| not_verifiable | hybrid_rag | 1.62% (1.09%, 2.20%) | 31 / 1913 |
| claim_evaluation_na_coverage | no_rag | 2.11% (1.22%, 3.11%) | 19 / 900 |
| claim_evaluation_na_coverage | item_rag | 1.56% (0.78%, 2.56%) | 14 / 900 |
| claim_evaluation_na_coverage | rule_rag | 3.33% (2.22%, 4.56%) | 30 / 900 |
| claim_evaluation_na_coverage | hybrid_rag | 2.44% (1.44%, 3.56%) | 22 / 900 |

## Paired case-clustered differences

| Metric | Contrast | Difference pp (95% CI) | Bootstrap p-value |
|---|---|---:|---:|
| any_permitted_evidence_support | item_rag minus no_rag | 2.15 (-0.64, 4.91) | 0.1224 |
| any_permitted_evidence_support | rule_rag minus no_rag | 22.42 (19.45, 25.37) | < 1/5000 |
| any_permitted_evidence_support | hybrid_rag minus no_rag | 25.09 (22.07, 28.14) | < 1/5000 |
| any_permitted_evidence_support | rule_rag minus item_rag | 20.28 (17.14, 23.38) | < 1/5000 |
| any_permitted_evidence_support | hybrid_rag minus item_rag | 22.94 (19.88, 26.02) | < 1/5000 |
| any_permitted_evidence_support | hybrid_rag minus rule_rag | 2.67 (0.48, 4.78) | 0.0192 |
| unsupported | item_rag minus no_rag | -1.98 (-5.04, 1.14) | 0.1988 |
| unsupported | rule_rag minus no_rag | -18.66 (-21.70, -15.74) | < 1/5000 |
| unsupported | hybrid_rag minus no_rag | -20.76 (-23.91, -17.82) | < 1/5000 |
| unsupported | rule_rag minus item_rag | -16.67 (-19.77, -13.65) | < 1/5000 |
| unsupported | hybrid_rag minus item_rag | -18.78 (-21.85, -15.76) | < 1/5000 |
| unsupported | hybrid_rag minus rule_rag | -2.10 (-4.11, -0.03) | 0.0464 |

Machine-readable CSV and JSON outputs are in `outputs/final_eval_v2/post_recovery/statistics/`. The script preserves complete case clusters, including all generators, variants, and valid claims for every sampled test case.
