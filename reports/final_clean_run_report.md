# Final Clean-Run Report

## Project status

**EXPERIMENTAL PROJECT: CLOSED**

This report documents the canonical final clean run of *Evidence-Constrained Multimodal Fashion Recommendation with Expert-Rule-Grounded Explanations*. It uses the frozen Stage 1–5 artifacts only. No model calls were made after Stage 4.

The central result is that, under generator-specific complete-case pairing and case-clustered inference, supplying the exact expert-rule trace used by reranking increased both reranking-trace support and full-KB support of generated explanations.

## Canonical result locations

| Material | Canonical location |
| --- | --- |
| Final provenance and release state | [`artifacts/manifests/final_stage5_manifest.json`](../artifacts/manifests/final_stage5_manifest.json) |
| Explanation paired contrasts | [`artifacts/tables/final_explanation_paired_contrasts.csv`](../artifacts/tables/final_explanation_paired_contrasts.csv) |
| Per-explanation metric source data | [`artifacts/tables/final_explanation_record_metrics.csv`](../artifacts/tables/final_explanation_record_metrics.csv) |
| Recommendation metrics and CIs | [`artifacts/tables/final_recommendation_metrics_with_ci.csv`](../artifacts/tables/final_recommendation_metrics_with_ci.csv) |
| Terminal-failure table | [`artifacts/tables/final_terminal_failures.csv`](../artifacts/tables/final_terminal_failures.csv) |
| Reproducibility package index | [`artifacts/release/release_manifest.json`](../artifacts/release/release_manifest.json) |

## Study design

The frozen experiment contains five recommendation categories: bags, bottoms, outerwear, shoes, and tops. The final knowledge base is [`data/kb/fashion_rules.csv`](../data/kb/fashion_rules.csv): 200 rules, exactly 40 rules per target category, with unique rule IDs, complete provenance, no exact duplicates, and no near-duplicate pairs at the frozen audit threshold.

The recommendation evaluation uses 1,000 test cases (200 per category) and controlled candidate pools. The final confirmatory operating point is image/text fusion `0.40/0.60`, CLIP/evidence reranking `0.75/0.25`, and `rule_top_k=5`. The recommendation trace stored during reranking is the same trace supplied as Rule-RAG evidence; no second retrieval occurs after locking the recommendation.

The explanation study uses 500 deterministic evidence-eligible cases (100 per category). For each case, the locked recommendation and common context are held constant across conditions:

- No-RAG: common case context only.
- Rule-RAG: the same common context plus the exact stored reranking trace.

Three generators were evaluated: Gemma 4 12B, Llama 3.1 8B Instruct Q8_0, and Ministral 3 14B Instruct Q4_K_M. Qwen 3.5 9B extracted atomic claims; Phi-4 14B performed the final claim verification.

## Stage-by-stage execution record

| Stage | Frozen output | Main result |
| --- | --- | --- |
| 1 | [`final_stage1_preflight_manifest.json`](../artifacts/manifests/final_stage1_preflight_manifest.json) | Dataset, splits, embeddings, 200-rule KB, validation-only sensitivity grids, prompts, and model identities frozen. |
| 2 | [`final_stage2_manifest.json`](../artifacts/manifests/final_stage2_manifest.json) | 1,000 recommendations; 500 explanation cases; 3,000 explanation cells attempted, 2,969 accepted. |
| 3 | [`final_stage3_manifest.json`](../artifacts/manifests/final_stage3_manifest.json) | 2,969 accepted explanations processed; 2,965 accepted extractions containing 17,710 atomic claims. |
| 4 | [`final_stage4_manifest.json`](../artifacts/manifests/final_stage4_manifest.json) | 2,861 accepted verifications covering 16,804 claims; 104 terminal verification failures retained. |
| 5 | [`final_stage5_manifest.json`](../artifacts/manifests/final_stage5_manifest.json) | Deterministic final metrics, paired bootstrap inference, figures, release package, and closure. |

### Provenance amendment

The Stage-1 manifest records an authorized post-preflight provenance amendment for the Stage-4 verifier-contract correction. It preserves the original frozen prompt-configuration hash (`3bc600…`) and explicitly binds release provenance to the actual final verifier prompt configuration (`570b3d…`), Stage-4 configuration hash, and final verifier prompt hashes. This was a transparent contract/interface correction; it did not rerun any generator, extractor, or verifier model.

### Stage-4 logical-consistency correction

After verification, 163 claims had the logically impossible combination `trace_support = supported` and `full_kb_support = not_supported`. Because every exact reranking trace is contained in that record's full-KB candidate packet, the canonical verification file was corrected in place by setting only `full_kb_support` to `supported` for those claims. No model calls, other claim fields, or evidence packets were changed. The final canonical verification SHA-256 is:

```text
0f554e58be51c0529c59814f3c5de379ec66c02afbb8fa2c5e48249a32ae9b3e
```

## Recommendation results

All recommendation confidence intervals use 5,000 percentile bootstrap replicates and cluster on the underlying query outfit (734 unique query outfits across 1,000 cases). Full results, including category rows, are in the canonical recommendation table.

| Method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM text | 4.8% | 12.0% | 17.7% | 8.4% | 10.3% | 10.2% |
| CLIP image | 3.2% | 12.6% | 22.0% | 7.9% | 10.9% | 10.0% |
| CLIP text | 3.5% | 13.8% | 21.2% | 8.6% | 11.0% | 10.1% |
| Fused CLIP | 4.5% | 14.3% | 23.1% | 9.4% | 12.2% | 11.4% |
| Evidence rerank | 3.8% | 13.2% | 22.5% | 8.5% | 11.4% | 10.6% |

The reranking diagnostics report a 26.5% top-1 change rate, mean top-5 overlap of 3.859, mean top-1 evidence-score gain of 0.1473, and mean pre-to-post rank shift of 0.566. Of the final KB's 200 rules, 148 appeared at least once in a locked reranking trace.

## Explanation-grounding results

The primary explanation comparison uses complete No-RAG/Rule-RAG pairs only. For overall inference, generator values are averaged within each underlying case before case-clustered resampling, so case-generator rows are not treated as independent observations. Each contrast uses 5,000 paired bootstrap replicates, 95% percentile intervals, absolute differences, and Holm adjustment across the four prespecified overall metrics.

| Metric (Rule-RAG minus No-RAG) | Paired cases | Estimate | 95% CI | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| Reranking-Trace Claim Support Rate | 498 | +21.02 pp | +19.72 to +22.37 pp | 0.0016 |
| Full-KB Claim Support Rate | 498 | +21.40 pp | +20.11 to +22.71 pp | 0.0016 |
| Unsupported Item-Fact Rate (lower is better) | 53 eligible pairs | +0.63 pp | −5.03 to +5.66 pp | 0.9042 |
| Trace-Supported Claims per 100 Words | 498 | +1.60 | +1.48 to +1.72 | 0.0016 |

Interpretation: Rule-RAG had substantially higher alignment with both the exact trace used in reranking and the final KB. The UIFR comparison is inconclusive because common-reference eligibility is sparse (53 paired eligible cases) and its interval spans zero.

### Generator-specific complete-pair results

| Generator | Complete pairs | Trace support difference | Full-KB support difference | Trace-supported claims / 100 words difference |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 12B | 474 | +25.09 pp | +25.22 pp | +1.79 |
| Llama 3.1 8B | 438 | +26.74 pp | +27.26 pp | +1.91 |
| Ministral 3 14B | 456 | +11.35 pp | +11.77 pp | +1.13 |

All three generator-level confidence intervals for the two support rates and supported-claim density exclude zero. Generator-level and category-level rows, claim-level source data, and the paired-contrast figure are available from the canonical artifacts.

## Failure and missingness reporting

No failed output was replaced, rewritten, or silently excluded from the stage manifests. Explanation comparisons use only complete pairs, rather than raw unequal condition totals.

| Stage | Generator / condition | Terminal failures |
| --- | --- | ---: |
| 2 | Llama Rule-RAG | 31 |
| 3 | Gemma No-RAG | 1 |
| 3 | Ministral No-RAG | 3 |
| 4 | Gemma No-RAG / Rule-RAG | 22 / 3 |
| 4 | Llama No-RAG / Rule-RAG | 18 / 16 |
| 4 | Ministral No-RAG / Rule-RAG | 22 / 23 |

The condition asymmetry is why raw condition totals are not used for paired inference. The final paired sample sizes are Gemma 474, Llama 438, and Ministral 456; the overall case-clustered analysis contains 498 cases with at least one complete generator pair.

## Verification totals

Among the 16,804 verified claims:

- Exact-trace support: 2,058 supported; 14,746 not supported.
- Full-KB support: 2,095 supported; 14,709 not supported.
- Common-reference factual support: 961 supported, 13 not supported, and 15,830 N/A.
- Citation entailment: 1,820 entails, 5,502 does not entail, and 9,482 N/A.

`not_supported` means the supplied evidence did not directly entail the claim under the frozen closed-world verification protocol. It must not be read as proof that a claim is false.

## Reproducibility and quality checks

- Every active runtime manifest's declared output hashes were rechecked before Stage 5.
- Stage 2 → 3 → 4 joins were exact: 3,000 explanation records, 2,969 Stage-2 accepted explanation records, 2,965 Stage-3 accepted extraction records, and 2,861 Stage-4 accepted verification records.
- Claim IDs were preserved exactly from extraction into every accepted verification record.
- Exact trace hashes, full-KB candidate packet hashes, citation conventions, and the invariant `trace_support = supported ⇒ full_kb_support = supported` passed.
- Final code quality checks passed: `ruff check .` and 55 tests.

## Scope and limitations

The conclusions are limited to the frozen dataset, five categories, sampled candidate pools, evidence-eligible explanation cases, the final 200-rule KB, and automated extraction/verification. The project does not claim access to hidden model reasoning, universal fashion correctness, full-catalogue production performance, or human preference superiority. Citation syntax and citation entailment are evaluated separately.

