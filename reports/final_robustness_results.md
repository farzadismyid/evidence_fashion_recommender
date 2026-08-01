# Final robustness results

This is the frozen, systematic, non-human evaluation. Human review is excluded from the
current experiment and remains future work. Model judges are proxy evaluators rather than
human preference measurements.

## Integrity and scale

- 300 balanced held-out cases, with 60 cases for each target category.
- Development, validation, and test outfits are disjoint.
- 3 generators x 4 grounding variants x 300 cases = 3,600 explanations.
- 3 judges x 3,600 explanations = 10,800 parsed explanation judgments; zero final errors.
- 300 cases x 5 retrieved rules x 3 judges = 4,500 rule-relevance labels; zero final errors.
- Prompt and reranker choices were selected on validation before opening the test set.
- The original baseline is retained under `outputs/robustness/before_baseline` with a
  SHA-256 manifest.

## Recommendation results

| Evaluation | Method | HitRate@10 | NDCG@10 |
|---|---:|---:|---:|
| Original baseline | Pure CLIP | 0.2467 | 0.1209 |
| Frozen held-out test | Pure CLIP | 0.2600 | 0.1377 |
| Frozen held-out test | Validation-selected 0.9 CLIP / 0.1 evidence | 0.2700 | 0.1348 |

On the same frozen test set, reranking improves HitRate@10 by 0.0100 but reduces NDCG@10
by 0.0029. It is therefore a recall/rank-position trade-off, not a uniform improvement.
The original and new test samples differ, so their absolute difference is not treated as
a causal improvement estimate.

## Explanation results averaged across generators and judges

| Variant | Overall judge /5 | Judge faithfulness /5 | Claim support | Label compliance | Deterministic unsupported claims | Evidence overlap |
|---|---:|---:|---:|---:|---:|---:|
| No-RAG | 4.117 | 4.098 | 0.704 | 0.806 | 0.177 | 0.000 |
| Item-RAG | 4.055 | 3.666 | 0.577 | 0.741 | 0.174 | 0.100 |
| Rule-RAG | 4.098 | 4.059 | 0.727 | 0.747 | **0.034** | 0.376 |
| Hybrid-RAG | 4.087 | 3.977 | 0.687 | 0.756 | 0.039 | **0.425** |

Cross-model scores excluding self-judges preserve the same overall ordering: No-RAG
4.160, Rule-RAG 4.139, Hybrid-RAG 4.123, and Item-RAG 4.092.

Hybrid-RAG reduces the deterministic unsupported-claim count by 78.0% relative to No-RAG
and produces the greatest evidence overlap, but it does not achieve the highest overall
judge score. Rule-RAG is marginally stronger on deterministic unsupported claims and
claim support. Thus, the central result is a trade-off: evidence-grounded methods greatly
improve measurable grounding, while No-RAG remains slightly preferred by aggregate model
judges for general explanation quality.

## Retrieval, faithfulness, and reliability

- Consensus rule retrieval: Precision@1 0.663, Precision@5 0.347, HitRate@5 0.963,
  MRR 0.792.
- Counterfactual category false-match rate: 0.000.
- Rule-relevance agreement: 0.790 mean pairwise agreement and 0.540 mean Cohen's kappa.
- Mean pairwise judge Spearman correlation across reported dimensions: 0.209; for the
  overall score alone: 0.373.
- Candidate-substitution detector frozen benchmark: precision/recall/F1 = 1.000/1.000/1.000.
- The validation ablation retained the original 55-word, five-rule, candidate-first Hybrid
  prompt. Longer or reordered prompts did not improve the selection objective.

Judge agreement is modest, and grounding-safety scores are nearly constant. In addition,
not every judge followed the categorical claim-label format. Canonical variants were
normalized deterministically; descriptive responses were conservatively mapped to
`not_verifiable`, and label compliance is reported in the table. These observations limit
claims of model-independent explanation quality and motivate human evaluation as future
work.

The generated report and complete machine-readable tables are in
`outputs/robustness/final_report` and `outputs/robustness/final_study`.
