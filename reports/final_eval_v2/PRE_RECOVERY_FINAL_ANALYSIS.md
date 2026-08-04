# PRE-RECOVERY FINAL ANALYSIS

This is the first complete end-to-end analysis of the frozen final evaluation. Stage 4D targeted
recovery has **not** been run. All extraction, verification, and judging failures remain N/A and are
excluded from metric denominators; they are never converted to zero, one, unsupported, or a judge
score. Original Stage 3 explanations and all successful Stage 4 rows remain unchanged.

## Recommendation results

| Method | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text-only (MiniLM) | .017 | .037 | .063 | .160 | .017 | .024 | .032 | .061 | .067 |
| CLIP image-only | .037 | .097 | .147 | .210 | .037 | .063 | .081 | .102 | .106 |
| CLIP text-only | .063 | .110 | .127 | .197 | .063 | .085 | .092 | .112 | .122 |
| Fused CLIP | .070 | .100 | .147 | .233 | .070 | .079 | .098 | .124 | .132 |
| **Fused CLIP + evidence (proposed)** | **.073** | **.093** | **.130** | **.230** | **.073** | **.078** | **.092** | **.121** | **.128** |

The proposed evidence-in-loop reranker uses the validation-frozen knee point CLIP=.75 and
evidence=.25. On test it is essentially tied with fused CLIP at HR@10 (-.0033, 95% bootstrap CI
[-.0467, .0400]), NDCG@10 (-.0028, [-.0177, .0123]), and MRR (-.0032, [-.0121, .0056]). This is
the intended trade-off: evidence materially participates without the larger validation degradation
at evidence=.35. CLIP=1/evidence=0 remains the accuracy-optimal baseline, not the proposed method.

Fused CLIP clearly improves on text-only for HR@10 (+.0733, [.0133, .1333]), NDCG@10 (+.0629,
[.0326, .0931]), and MRR (+.0651, [.0384, .0937]). Effects against the individual CLIP branches
are smaller.

## Explanation grounding

| Variant | Verified claims | N/A explanations | Any supported | Rule-supported* | Item-supported* | Unsupported | Contradicted | Not verifiable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_rag | 1,571 | 82 | 52.3% | N/A | N/A | 41.4% | 0.19% | 6.05% |
| item_rag | 1,527 | 62 | 55.1% | N/A | 36.5% | 39.2% | 0.00% | 5.76% |
| rule_rag | 1,696 | 70 | 78.7% | 72.8% | N/A | 18.6% | 0.24% | 2.42% |
| hybrid_rag | 1,608 | 66 | 80.3% | 54.7% | 21.6% | 17.5% | 0.50% | 1.62% |

\* Generation-grounding rates are N/A when that evidence source was not supplied to the generator.
Raw verifier labels against the full reference packet remain available in the CSV.

Rule and Hybrid RAG substantially reduce unsupported claims relative to No RAG. Hybrid has the
highest overall supported-claim rate and lowest not-verifiable rate. Contradiction rates are low for
all variants, though Hybrid has eight contradicted claims and should not be described as error-free.

## General quality: primary cross-model judging

| Variant | Input consistency | General quality | Clarity | Specificity | Hallucination risk | Evidence misuse |
|---|---:|---:|---:|---:|---:|---:|
| no_rag | 4.553 | 3.817 | 4.333 | 3.449 | 3.821 | 4.958 |
| item_rag | 4.729 | 3.980 | 4.321 | 3.647 | 3.708 | 4.845 |
| rule_rag | 4.535 | 3.928 | 4.274 | 3.658 | 3.800 | 4.299 |
| hybrid_rag | 4.641 | 3.924 | 4.294 | 3.645 | 3.829 | 4.379 |

Relative to No RAG, Rule and Hybrid improve general quality by .125 [.071, .181] and .122 [.069,
.175], and specificity by .226 [.179, .274] and .211 [.167, .254], respectively. Their evidence
misuse scores are lower than No RAG because evidence-bearing outputs create opportunities for
misapplication; Hybrid is better than Rule by .085 [.059, .112]. Hybrid improves input consistency
over No RAG by .075 [.027, .123]. All-judge results, including self-judging, are sensitivity
diagnostics only and are provided separately.

## Length compliance

Length is a separate outcome, not part of explanation quality. Overall compliance with 35 words is
63.56% (2,288/3,600), with mean length 34.56 words. Mistral Rule RAG has the weakest compliance
(21.33%) and Gemma Item RAG the strongest (94.00%). No over-length output was altered.

## Qualitative interpretation

The selected examples show both directions of the Hybrid-versus-No-RAG quality difference and are
provided without rewriting in `pre_recovery/qualitative_examples.csv`. They should be used as
illustrations, not substituted for the quantitative analysis.

## Limitations

- Stage 4D targeted recovery remains pending: 4 extraction, 276 verification, and 262 judgment rows
  are N/A at their respective evaluation levels.
- Qwen accounts for 259/262 judgment failures; primary cross-model denominators therefore vary by
  variant, especially Item RAG.
- Claim support is measured only among successfully verified claims and must be read together with
  N/A coverage.
- The test set contains 300 cases from frozen outfit-disjoint schedules; generalization beyond the
  evaluated catalogue and evidence base is not established.
- Automated judges provide anchored comparative evidence, not human preference judgments.
- Longer responses are reported separately and are not evidence of better quality.

## Conclusion

The evidence-in-loop system preserves retrieval accuracy close to fused CLIP while meaningfully
participating in recommendation selection. At explanation time, Rule and Hybrid grounding sharply
improve claim support and specificity, with Hybrid giving the strongest overall support profile and
less evidence misuse than Rule-only RAG. These conclusions are pre-recovery and must retain their
reported N/A coverage.

