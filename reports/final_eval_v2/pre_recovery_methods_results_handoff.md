# Pre-recovery methods/results handoff

## Completed

The frozen recommendation evaluation, 3,600 original explanations, separate claim extraction and
verification, cross-model-primary judging, sensitivity judging, length audit, paired bootstrap
comparisons, figures, qualitative examples, limitations, and artifact validation are complete.

Stage 4D targeted recovery was not run. The analysis uses the recorded N/A values unchanged.

## Produced files

The paper-ready bundle is in `reports/final_eval_v2/pre_recovery/`:

- recommendation and reranking trade-off tables;
- claim-support, primary judging, sensitivity judging, and length tables;
- paired bootstrap comparisons with paired effect sizes;
- four figures and a qualitative-example table;
- analysis manifest and SHA-256 artifact inventory.

The narrative result is `reports/final_eval_v2/PRE_RECOVERY_FINAL_ANALYSIS.md`.

## Main results

- Proposed fused+evidence test HR@10=.230, NDCG@10=.121, MRR=.128.
- Fused CLIP test HR@10=.233, NDCG@10=.124, MRR=.132; paired differences versus the proposed
  reranker are small and their 95% bootstrap intervals include zero.
- Supported-claim rates are 52.3% No RAG, 55.1% Item RAG, 78.7% Rule RAG, and 80.3% Hybrid RAG.
- Cross-model general-quality means are 3.817, 3.980, 3.928, and 3.924, respectively.
- Overall 35-word compliance is 63.56% and is reported independently of quality.

## Exact pre-recovery N/A counts

- Claim extraction N/A: 4 explanations.
- Claim verification N/A: 276 explanations.
- Total explanation-level claim table N/A: 280.
- General judgment N/A: 262 of 10,800 explanation-judge pairs (260 cross-model, 2 self-family).

N/A values were not mapped to unsupported, zero, one, or any judge score.

## Next stage

Stage 4D targeted recovery can safely run next because checkpoints are complete, uniquely keyed,
hash-bound to the unchanged Stage 3 explanation file, and failures are explicitly enumerated. It
must write new recovery artifacts without overwriting this pre-recovery bundle or successful rows.

