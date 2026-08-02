# Final Evaluation v2 Evidence-in-the-Loop Correction Handoff

## Outcome

The proposed reranker is now frozen at **CLIP weight 0.75 / evidence weight 0.25** under
`selection_policy=evidence_in_loop_pareto_v2`. The unconstrained **1.00 / 0.00** selection is
retained and explicitly labelled as the accuracy-optimal baseline; it is no longer the proposed
method.

Only the requested downstream Stage 1 products were regenerated, followed by a complete Stage 2
Hybrid validation rerun. Stage 3 was cancelled before any freeze or final explanation generation
ran and remains paused.

## Conceptual correction

The original validation selector optimized ranking metrics without requiring evidence to
participate. It therefore selected `clip_weight=1.00`, making `evidence_weight=0.00`. That result is
valid as an unconstrained fused-CLIP accuracy baseline, but it cannot represent an
evidence-in-the-loop proposed method because evidence has no influence on the ranking.

The corrected proposed-method policy selects `0.75 / 0.25` as the declared Pareto/knee operating
point. Evidence has a substantial 25% role, while the validation degradation remains materially
smaller than at the next heavier tested evidence setting (`0.65 / 0.35`). This is a
method-definition constraint and trade-off selection, not a claim that `0.75 / 0.25` maximizes
unconstrained accuracy.

## Frozen artifacts

Proposed method:

```text
outputs/final_eval_v2/validation/reranking_tuning/selected_weight.json
selection_policy: evidence_in_loop_pareto_v2
method_role: proposed_evidence_in_loop_reranker
clip_weight: 0.75
evidence_weight: 0.25
selected_on: validation
```

Accuracy-optimal baseline:

```text
outputs/final_eval_v2/validation/reranking_tuning/selected_weight_accuracy_optimal.json
selection_policy: accuracy_optimal_unconstrained_baseline_v2
method_role: accuracy_optimal_baseline
clip_weight: 1.00
evidence_weight: 0.00
superseded_as_proposed_method: true
```

The complete original `1.00 / 0.00` downstream Stage 1/2 slice is preserved under:

```text
outputs/final_eval_v2/superseded_accuracy_optimal_evidence0/
```

It contains the original selected cases, locked packets, test retrieval evaluation, decision
gate, Hybrid validation outputs, and Hybrid report. These are audit artifacts only.

## Validation trade-off

`delta_vs_clip_only` is NDCG@10 minus the `1.00 / 0.00` validation NDCG@10.

| CLIP | Evidence | HR@10 | NDCG@10 | Reciprocal rank | Delta vs CLIP-only |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | 0.266667 | 0.144803 | 0.147810 | 0.000000 |
| 0.95 | 0.05 | 0.256667 | 0.140038 | 0.145219 | -0.004765 |
| 0.90 | 0.10 | 0.246667 | 0.136512 | 0.144923 | -0.008291 |
| 0.85 | 0.15 | 0.246667 | 0.134291 | 0.140017 | -0.010511 |
| **0.75** | **0.25** | **0.253333** | **0.135715** | **0.140235** | **-0.009088** |
| 0.65 | 0.35 | 0.236667 | 0.129638 | 0.135220 | -0.015165 |

Compared with evidence weight 0.35, the selected 0.25 setting improves HR@10 by 0.016667,
NDCG@10 by 0.006077, and reciprocal rank by 0.005015, while retaining a substantial evidence
contribution.

## Held-out test trade-off

| Method | HR@10 | NDCG@10 | Reciprocal rank |
|---|---:|---:|---:|
| Fused CLIP accuracy baseline (`1.00 / 0.00`) | 0.233333 | 0.124016 | 0.131588 |
| Proposed evidence-in-loop reranker (`0.75 / 0.25`) | 0.230000 | 0.121231 | 0.128400 |
| Difference | -0.003333 | -0.002785 | -0.003188 |

The held-out result shows a small ranking cost for making evidence operational in the proposed
reranker. Test data was used only for evaluation, not selection.

## Downstream Stage 1 rerun and changes

Regenerated:

1. validation selected cases;
2. validation locked packets;
3. test selected cases;
4. test locked packets;
5. held-out test retrieval evaluation;
6. decision gate.

Direct comparison against the superseded `1.00 / 0.00` v2 artifacts:

| Split | Cases | Recommendations changed | Rate | Evidence packets changed | Rate |
|---|---:|---:|---:|---:|---:|
| Validation | 300 | 148 | 0.493333 | 148 | 0.493333 |
| Test | 300 | 150 | 0.500000 | 150 | 0.500000 |

New packet hashes:

```text
validation: c561b2016ba8ad7b8d06aac7cb3dd8a56793e91adfa5f4f70186a448ae3188f9
test:       445dca5a3e513d6e425e110d32dd4323097a12544ec8a13a1af07977b93df6b3
```

The corrected legacy-vs-v2 decision gate remains material:

```text
cases: 300
changed recommendation rate: 0.986667
changed evidence packet rate: 1.000000
decision: regenerate_all_variants
```

## Corrected Stage 2 result

The complete 36-cell screening and six-finalist validation procedure was rerun using the new
evidence-in-loop validation packets. Runtime was 14,202.5 seconds (3 h 56 m 42.5 s), with 1,800
screening generations/judgments and 1,800 finalist generations/judgments.

The selected configuration is unchanged in settings but newly bound to the corrected packet hash:

```text
name: hybrid_w35_r5_i2_item_first
word budget: 35
rule count: 5
item count: 2
evidence order: item_first
Stage 1 packet hash: c561b2016ba8ad7b8d06aac7cb3dd8a56793e91adfa5f4f70186a448ae3188f9
Stage 2 artifact fingerprint: f46765496ed8b9a55e78d94e688dc53bc35b5b061cf250104fe5269245c6d2b3
```

Selected full-validation metrics:

| Metric | Value |
|---|---:|
| Hallucinated fashion-claim rate | 0.000000 |
| Rule-supported styling-claim rate | 1.000000 |
| Evidence misuse rate | 0.000000 |
| Candidate substitution rate | 0.000000 |
| Rule evidence overlap | 0.393880 |
| Item evidence overlap | 0.225539 |
| General clarity | 4.150000 |
| Mean explanation words | 34.783333 |

## Verification

```text
full project suite: 64 passed
Ruff: all checks passed
Stage 1 corrected artifact row counts: 300 validation / 300 test
Stage 2 screening rows: 1,800 generations / 1,800 judgments
Stage 2 finalist rows: 1,800 generations / 1,800 judgments
Stage 3 freeze/generation: not run
```

## Exact next step for Stage 3

Stage 3 remains paused. When it is explicitly resumed, the first action is **not generation**:
commit the correction slice so the repository is clean, then implement/review and run the
fail-closed `freeze-final-eval-v2` CLI so its immutable manifest binds the new `0.75 / 0.25`
reranking artifact, corrected Stage 1 packet hashes, corrected decision gate, and corrected Stage 2
Hybrid selection. Only after that freeze passes may `run-final-explanations-v2` be implemented/run
under Gate A for all four variants.
