# Final Evaluation Stage Design

## Purpose

This file defines how the final evaluation should be executed after the protocol fixes are implemented.

The goal is to avoid one uncontrolled long run. The workflow should be broken into resumable stages so the PC can be safely shut down between stages.

Do not overwrite previous outputs. Use versioned directories, for example:

```text
outputs/final_eval_v2/
reports/final_eval_v2/
```

---

## Stage 0: Pre-run freeze

Run only after code/config changes are approved.

Required before any final test run:

```text
git status is clean
final protocol committed
final configs committed
run tag created
selected validation settings saved
resolved config saved
prompt hashes saved
KB hash saved
split/case schedule hashes saved
model/runtime metadata saved
```

Output:

```text
outputs/final_eval_v2/manifest/pre_run_manifest.json
reports/final_eval_v2/pre_run_freeze.md
```

Do not continue to final testing if the repo is dirty.

---

## Stage 1: Recommendation / retrieval

Purpose:

```text
Validate and test retrieval methods before explanation generation.
```

Run on identical validation/test cases:

```text
MiniLM text-only
CLIP image-only
CLIP text-only
CLIP fused image+text
fusion-weight validation curve
evidence-reranking validation/test
```

Primary outputs:

```text
outputs/final_eval_v2/recommendation/modality_comparison.csv
outputs/final_eval_v2/recommendation/fusion_weight_validation.csv
outputs/final_eval_v2/recommendation/reranking_validation.csv
outputs/final_eval_v2/recommendation/test_ranking_results.csv
reports/final_eval_v2/recommendation_results.md
```

Selection rules:

```text
Fusion weight: select on validation NDCG@10, then HR@10, then MRR/mean rank.
Reranking weight: select on validation NDCG@10, then HR@10, then MRR/mean rank.
```

Freeze selected recommendation settings before Stage 2.

Do not regenerate explanations during this stage.

---

## Stage 2: Hybrid-RAG validation

Purpose:

```text
Select the final Hybrid-RAG evidence budget/order before final explanation generation.
```

Suggested validation grid:

```text
word_budget: 35, 55, 75
rule_count: 3, 5
item_count: 0, 2, 5
evidence_order: rules_first, item_first
```

Selection rule:

```text
1. Lowest hallucinated fashion-claim rate.
2. Highest rule-supported styling-claim rate.
3. Lowest evidence misuse / candidate substitution.
4. Highest evidence overlap, split into rule and item overlap.
5. Acceptable general clarity.
6. Shorter valid explanation if practically tied.
```

Use validation only. Do not use test outputs for selecting Hybrid-RAG settings.

Primary outputs:

```text
outputs/final_eval_v2/hybrid_validation/grid_results.csv
outputs/final_eval_v2/hybrid_validation/selected_hybrid_config.json
reports/final_eval_v2/hybrid_validation_report.md
```

Freeze selected Hybrid-RAG settings before Stage 3.

---

## Stage 3: Explanation generation

Purpose:

```text
Generate final explanations using frozen retrieval and frozen generation settings.
```

Variants:

```text
No-RAG
Item-RAG
Rule-RAG
Hybrid-RAG
```

Rules:

```text
Use same word/token budget policy across variants unless explicitly justified.
Use locked recommended item.
Use frozen selected Hybrid-RAG config.
Use frozen selected retrieval/reranking config.
Do not change prompts after generation starts.
```

Primary outputs:

```text
outputs/final_eval_v2/explanations/explanations.csv
outputs/final_eval_v2/explanations/generation_errors.csv
outputs/final_eval_v2/explanations/generation_manifest.json
reports/final_eval_v2/generation_summary.md
```

This stage should be resumable.

---

## Stage 4: Judging and evaluation

Purpose:

```text
Judge cached explanations with the final rubric and compute final statistics.
```

Judge tasks:

```text
atomic claim extraction
claim verification / entailment
input consistency
rule-grounded faithfulness
hallucination / unsupported fashion-claim rate
evidence misuse
general explanation quality
citation-to-claim support
```

Primary analysis:

```text
cross-model-only judge results
```

Self-judge rows may be retained only as sensitivity analysis.

Statistical analysis:

```text
outfit-clustered paired bootstrap
predefined primary comparison families
multiple-comparison correction within families
case count and unique-outfit count
```

Primary outputs:

```text
outputs/final_eval_v2/judging/claim_extraction.csv
outputs/final_eval_v2/judging/claim_verification.csv
outputs/final_eval_v2/judging/judge_scores.csv
outputs/final_eval_v2/statistics/clustered_bootstrap.csv
outputs/final_eval_v2/statistics/primary_comparisons.csv
reports/final_eval_v2/final_evaluation_report.md
```

---

## Checkpoint and resume policy

Long-running jobs must save progress regularly.

Preferred checkpoint interval:

```text
about every 3 hours
```

Each stage should:

```text
write partial CSV/JSONL outputs
skip already completed rows on resume
record errors without stopping the full run when safe
write a progress summary
allow safe shutdown between stages
```

Do not require the full pipeline to run in one uninterrupted session.

---

## Final run rule

Validation choices must be frozen before test evaluation.

After final test evaluation starts, do not change:

```text
retrieval settings
fusion weight
reranking weight
Hybrid-RAG config
prompts
judges
metrics
statistics procedure
```

If any of these change, create a new versioned run directory instead of overwriting outputs.
