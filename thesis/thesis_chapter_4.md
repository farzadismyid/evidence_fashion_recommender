# Chapter 4

# Results

## 4.1 Introduction

This chapter reports the completed recommendation and explanation experiments. Results are presented in the order of the frozen pipeline: data and retrieval validation, recommendation effectiveness, evidence participation, explanation-generation integrity, atomic-claim assessment, study-specific explanation metrics, general quality judgments, association analysis, heterogeneity, robustness, and deterministic qualitative examples. The main explanation results are Decision-Trace Alignment (DTA) and Unsupported Item-Attribute Rate (UIAR). Visible-evidence grounding and the shared A+B reference are retained as secondary analyses.

All figures and tables in this chapter were derived from saved artifacts. The additional publication analysis made no language-model calls and did not alter Stage 7 generation, Stage 8 extraction, verification, or judging. Percentages are rounded for presentation; computations use the full stored precision. “Unsupported” means unsupported by supplied evidence. It does not mean factually false. For No-RAG, agreement with hidden B is described as post-hoc alignment, whereas Rule-RAG agreement with visible B is decision-trace faithfulness.

## 4.2 Data preparation and representation checks

The pinned source contained 94,096 catalogue items and 21,587 outfits. The deterministic outfit split assigned 15,267 outfits to development, 3,147 to validation, and 3,173 to test. No outfit crossed these partitions. An exact-image audit identified 21 duplicate-image groups and eleven groups that initially crossed research splits. Component-based reassignment moved eleven duplicate-linked outfits and two singleton outfits used to restore exact quotas. After this correction, no exact image crossed the research boundary.

The broad-category mapping supported five balanced confirmatory strata. The Stage 6 prepared universe contained 69,725 items and 20,225 outfits. From the held-out test split, 1,000 recommendation cases were constructed, exactly 200 each for accessories, bottoms, outerwear, shoes, and tops. These cases came from 762 distinct query outfits. Primary controlled pools contained 99,238 candidate rows, with a mean of 99.238 candidates and a range from 60 to 102. Retaining all positives explains the small number of pools above 100.

Representation validation found the expected dimensionalities: 384 for MiniLM and 512 for CLIP image, CLIP text, and fused CLIP. All cached vectors were normalised. The largest observed norm deviation was \(1.19\times10^{-7}\), which is consistent with float32 numerical error rather than a substantive defect. Minimum pairwise distances were positive for every representation family. Repeated ranking with identical inputs produced the same item order, including deterministic item-ID resolution of tied scores.

These checks matter for the subsequent interpretation. A cross-split duplicate could inflate held-out ranking, while an unnormalised fused vector could change the meaning of cosine scores. Neither issue remained in the confirmatory input. The results nevertheless concern controlled pools rather than catalogue-scale retrieval.

## 4.3 Validation-stage findings

### 4.3.1 Fusion and candidate-pool sensitivity

The validation analyses showed that ranking difficulty increased sharply with candidate-pool size. On a 300-case sensitivity cohort, fused CLIP HR@10 was 0.2367 at approximately 100 candidates, 0.0933 at approximately 500, and 0.0800 at approximately 1,000. NDCG@10 moved from 0.1264 to 0.0492 and 0.0374. The equivalent evidence-reranker values were HR@10 0.1967, 0.0833, and 0.0567, with NDCG@10 0.1018, 0.0444, and 0.0321. This monotonic deterioration is expected when many more same-category negatives compete for the same top ten.

The pool sensitivity demonstrates why absolute values cannot be detached from the evaluation protocol. It also supports retaining the approximately 100-candidate setting as the primary, pre-specified controlled task while treating larger pools as stress tests. Larger-pool results do not reverse the interpretation that evidence reranking can change ranks without a demonstrated effectiveness gain.

### 4.3.2 Rule-RAG prompt optimisation and pilot

Six validation-only Rule-RAG configurations all produced structurally valid outputs, but varied in evidence use, length, and quality. The selected five-rule configuration `rag_c3` achieved a support rate of 0.9322, unsupported rate of 0.0678, citation-entailment rate of 0.9544, general quality 4.50, clarity 4.63, specificity 4.53, and mean length 57.07 words on its primary validation cohort. Its 13.33% length-violation rate was accepted because its support and quality profile lay on the Pareto frontier and the prompt retained all five decision rules.

In the subsequent 50-case, three-generator pilot, Rule-RAG support averaged 0.9352 compared with 0.2114 for No-RAG under the pilot verifier. Rule-RAG general quality was 4.52 versus 2.21, clarity 4.62 versus 2.79, and specificity 4.52 versus 2.14. The automated pairwise pilot preferred Rule-RAG in 99.33% of comparisons. These values justified technical progression but are not pooled with the final estimates: the pilot used its own prompts, samples, and preliminary metric framing.

## 4.4 Confirmatory recommendation effectiveness

Table 4.1 summarises the primary micro results. Intervals are query-outfit-clustered 95% percentile bootstrap intervals.

| Method | HR@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| MiniLM text | 0.1530 [0.131, 0.176] | 0.0829 [0.0696, 0.0973] | 0.0854 [0.0739, 0.0977] |
| CLIP image | 0.2040 [0.179, 0.230] | 0.1022 [0.0882, 0.1172] | 0.0991 [0.0873, 0.1121] |
| CLIP text | 0.1800 [0.157, 0.205] | 0.0914 [0.0781, 0.1052] | 0.0901 [0.0789, 0.1019] |
| Fused CLIP 0.40/0.60 | 0.2190 [0.193, 0.247] | 0.1100 [0.0949, 0.1248] | 0.1027 [0.0906, 0.1154] |
| Evidence rerank 0.75/0.25 | 0.2200 [0.194, 0.246] | 0.1078 [0.0931, 0.1231] | 0.0995 [0.0877, 0.1123] |

Fused CLIP produced the highest NDCG@10 and MRR; evidence reranking produced the numerically highest HR@10 by 0.001. Against MiniLM, fused CLIP improved HR@10 by 0.066 and NDCG@10 by 0.0271. Both contrasts remained significant after Holm correction across 28 comparisons (adjusted \(p=0.0112\) and \(p=0.0400\), respectively). Against CLIP text, fused CLIP improved HR@10 by 0.039 and NDCG@10 by 0.0185; both adjusted \(p=0.0112\). Fused CLIP did not significantly outperform CLIP image after multiplicity correction, although its point estimates were higher.

The main fused-versus-reranked contrasts were all non-significant. Expressed as fused minus reranked, HR@10 was −0.001 (95% CI −0.0207 to 0.0182; adjusted \(p=1.000\)); NDCG@10 was +0.00216 (−0.00617 to 0.01064; adjusted \(p=1.000\)); and MRR was +0.00315 (−0.00475 to 0.01093; adjusted \(p=1.000\)). At ranks one and five, intervals likewise crossed zero. The evidence component therefore neither significantly improved nor significantly degraded recommendation effectiveness relative to fused CLIP under this controlled task.

This null contrast is substantively informative. The evidence score was not harmless because it was too weak to affect the order, nor was it an accuracy booster. Instead, it introduced a different ranking signal while preserving broadly similar aggregate effectiveness. The next section quantifies that participation directly.

## 4.5 Evidence participation and rule use

Evidence reranking changed the top recommendation in 505 of 1,000 cases (50.5%) and changed the ordered top five in 991 cases (99.1%). The mean top-five set overlap between fused CLIP and evidence reranking was 0.6526, and top-ten overlap was 0.7088. Across the union of top-ten items, the mean absolute rank shift was 5.552 positions. Thus, even when the same candidates often remained near the top, their ordering changed materially.

The selected items also had higher evidence scores than the positions chosen by fused CLIP. Mean evidence-score gain was 0.01495 at rank one, 0.01177 across the top five, and 0.01001 across the top ten. These gains show that the reranker operated in its intended direction: it promoted candidates better aligned with its expert-rule retrieval function.

Every locked Stage 6 recommendation retained exactly five scoring rules. Across the 1,000 traces, 112 of the 126 available rules appeared at least once. The distribution had Shannon entropy 5.5815, indicating that evidence participation was not concentrated in a negligible handful of rules. Coverage was not complete because category filtering and semantic relevance excluded some rules from all final traces. The stored rule-frequency table permits inspection of every rule’s retrieval and selection counts.

Taken together, the recommendation results answer the second research question with a qualified finding. Evidence reranking changed which items were favoured and increased the rule-derived score of selected items, but did not significantly alter held-out outfit-relevance metrics relative to fused CLIP. “Evidence alignment” and “recommendation accuracy” are empirically separable in this system.

## 4.6 Stage 7 generation integrity and length

The corrected explanation matrix contained exactly 3,000 texts: 500 cases, three generators, and two conditions. Rule-RAG’s 1,500 previously generated texts were reused only after hashes confirmed that their prompts, recommendations, traces, model digests, and 75-word instruction were unchanged. No-RAG’s 1,500 texts were regenerated with the same explicit instruction to use at most 75 words. This correction removes the original avoidable prompt-length asymmetry while preserving the locked recommendation input.

No-RAG averaged 52.84 words and Rule-RAG 60.55. By generator, the corresponding No-RAG/Rule-RAG means were 46.86/43.14 for Gemma, 63.83/59.73 for Llama, and 47.83/78.77 for Mistral. Hence the aggregate residual difference is 7.71 words rather than the original 137.22-word gap, and its direction is not consistent across generators. Two No-RAG outputs and 243 reused Rule-RAG outputs exceeded 75 words. Outputs were retained intact because post-hoc truncation could selectively remove qualifications or citations. The Stage 5 follow-up independently confirmed the same pattern on its validation cohort: 53.00 No-RAG words against 59.73 Rule-RAG words, with no No-RAG cap violations.

The correction is a prompt-level length control, not an exact realised-length randomisation. Models can comply differently with the same instruction, particularly the reused Mistral Rule-RAG outputs. Accordingly, the full paired analysis remains the main estimand, and a 30-pair observed-length sensitivity is reported separately. The key methodological improvement is that any remaining verbosity difference is model behaviour under a common limit rather than an experimentally imposed advantage for one condition.

## 4.7 Claim extraction, refusals, and normalisation

Qwen3 extracted 20,838 atomic claims. After separating 1,469 identity/context claims, 19,369 substantive explanatory claims remained: 10,703 No-RAG and 8,666 Rule-RAG. The lower and much closer claim totals than in the original run are consistent with the common 75-word instruction. Two extraction records failed after retry and were retained as missing, rather than silently replaced. No explanation was classified as a refusal in the revised corpus.

Verification was performed by Mistral, not by the Qwen3 extractor. This cross-model separation reduces the most direct form of self-confirmation: the model that proposes the atomic decomposition is not the model deciding whether those claims are supported by A or B. Of 3,000 expected verification records, 2,980 returned verifier outputs; 20 exhausted retry or parsing recovery and remain explicitly missing. There were 20,618 claims in the successfully verified records. The Qwen3 judge completed all 1,500 condition pairs.

Deterministic schema normalisation affected 1,280 of the 2,980 actual verifier outputs (42.95%). The code records both the raw response and each normalisation reason. This is still a material evaluator limitation: it demonstrates that local structured output was not perfectly reliable. Normalisation enforced the declared schema and conservative source rules; it did not add new semantic support. All denominators therefore remain visible, and no human audit is claimed.

## 4.8 Decision-Trace Alignment

### 4.8.1 Overall results

The revised cross-model verification reverses the earlier DTA result. No-RAG contained 9,641 B-aligned substantive claims out of 10,703, giving micro post-hoc DTA of 90.08%. Rule-RAG contained 7,362 B-aligned claims out of 8,666, giving micro DTA of 84.95%. Macro rates were 91.45% and 86.36%, respectively. Across 1,480 paired eligible explanations, Rule-RAG minus No-RAG macro DTA was −5.13 percentage points (95% case-cluster bootstrap CI −6.51 to −3.77; two-sided \(p=0.0004\)).

This is a negative answer to the narrow claim that exposing B increases alignment with B. It does not imply that No-RAG accessed the hidden trace. Rather, the selected rules encode common fashion relations that the generators can often reconstruct from the item context, and the cross-model verifier accepted those post-hoc matches at a high rate. Rule-RAG also made more case-specific statements whose strict entailment was not always accepted. The result makes the distinction between post-hoc trace agreement and visible-evidence grounding essential.

### 4.8.2 Generator results

Generator and category tables are retained as robustness descriptions rather than used to rescue the rejected directional hypothesis. The overall paired estimate is based on 1,480 explanation pairs: 498 Gemma pairs, 491 Llama pairs, and 491 Mistral pairs. Small differences in eligibility arise from the two extraction and 20 verification failures. The central point is stable at the pooled level: the common-reference DTA metric does not support a Rule-RAG advantage under the cross-model pipeline.

### 4.8.3 Category results

The five categories contributed between 292 and 299 paired eligible observations each. These subgroup estimates are exploratory because the primary correction changed both generated text and verifier. They are useful for identifying category-specific behaviour, but the thesis places the inferential weight on the pre-declared overall paired contrast and reports the unfavourable direction without selective subgroup emphasis.

## 4.9 Unsupported Item-Attribute Rate

### 4.9.1 Claim-level and explanation-level results

The verifier identified 925 eligible item-attribute claims in No-RAG and 776 in Rule-RAG. Micro UIAR was 62.38% for No-RAG and 37.11% for Rule-RAG. Among explanations containing at least one eligible assertion, macro UIAR was 67.17% and 44.72%, respectively. Thus, the Rule-RAG advantage remains after the equal prompt limit and after changing the verifier model.

The paired estimate used the 192 explanations for which both conditions had an eligible item-attribute denominator. Rule-RAG minus No-RAG macro UIAR was −17.45 percentage points (95% case-cluster bootstrap CI −24.61 to −10.56; \(p=0.0004\)). This is narrower than a comparison of all outputs: it asks whether the unsupported fraction differs when both paired texts choose to make concrete item assertions. Texts without an eligible claim are N/A, not automatically successful.

### 4.9.2 Unsupported Attribute Density

No-RAG produced 0.728 unsupported item-attribute claims per 100 words; Rule-RAG produced 0.317. Explanation-macro densities were 0.745 and 0.298. This approximately 56% micro reduction is important because both prompts carried the same word limit and because density additionally normalises realised output length. It does not establish world truth: “unsupported” means not established by the supplied evidence boundary.

### 4.9.3 Subgroup and robustness results

UIAR eligibility remained sparse relative to all substantive claims: 547 No-RAG and 459 Rule-RAG explanations had at least one eligible attribute claim. The paired intersection was smaller still. Generator and category estimates are therefore reported with denominators and treated as exploratory. The overall result is nevertheless supported by three views with the same direction: claim-micro UIAR, eligible-explanation macro UIAR, and unsupported-attribute density.

## 4.10 Citation integrity

Citation metrics apply only to Rule-RAG. Under the strict cross-model schema, citation precision was 87.5% at claim-micro level (21 valid of 24 evaluated citation relations) and 88.14% macro. However, only 19 of 8,275 rule-required claims received a validly verified citation, giving micro coverage of 0.23% and macro coverage of 0.28%. No-RAG remains N/A because it was neither shown nor asked to cite B.

These values must be read together. High precision on 24 evaluated relations is too sparse to support a broad claim that citations were generally valid, while near-zero strict coverage conflicts with the visible prevalence of bracketed rule identifiers. The result primarily exposes a limitation of the cross-model citation-relation parser and verifier under this schema. The thesis therefore reports citation presence as useful for auditability but treats strict citation validity as evaluator-dependent and inconclusive rather than “substantial”.

## 4.11 Secondary visible-evidence and shared-reference analyses

Visible-evidence support supplies the clearest positive grounding result. Against what each generator could actually see, No-RAG support was 2.31% micro and 1.79% macro, whereas Rule-RAG support was 89.12% micro and 89.47% macro. The paired macro difference was +87.73 percentage points. This comparison is intentionally asymmetric in evidence availability: it measures whether a user-facing reason is grounded in the material supplied to that condition.

The source decomposition explains why DTA and visible support disagree. No-RAG had 9,632 B-only claims and only 247 claims supported by A (A-only or A+B). Those B-only matches are post-hoc alignment, not evidence use, because B was hidden. Rule-RAG had 7,334 B-only and 28 A+B claims, reflecting direct access to the trace. Against a common A+B reference, No-RAG support was 92.30% and Rule-RAG 89.12% micro. Thus Rule-RAG does not win the common-reference entailment contest; it wins the operational grounding contest because its accepted reasons are tied to information the model actually received.

## 4.12 General quality judgments

Rule-RAG scored higher on all six 1–5 dimensions. Mean input consistency rose from 3.947 to 4.987; general quality from 4.148 to 4.811; clarity from 4.535 to 4.961; specificity from 3.207 to 4.713; hallucination control from 3.806 to 4.961; and evidence-use correctness from 1.646 to 4.981.

Paired differences and 95% case-cluster intervals were:

| Dimension | Rule-RAG − No-RAG | 95% CI | Paired \(d_z\) |
|---|---:|---:|---:|
| Input consistency | +1.040 | +0.985 to +1.095 | 0.99 |
| General quality | +0.663 | +0.625 to +0.701 | 0.93 |
| Clarity | +0.427 | +0.396 to +0.459 | 0.70 |
| Specificity | +1.506 | +1.458 to +1.551 | 1.75 |
| Hallucination control | +1.155 | +1.085 to +1.223 | 0.91 |
| Evidence-use correctness | +3.335 | +3.287 to +3.385 | 3.56 |

All Holm-adjusted paired tests were highly significant. Evidence-use correctness shows the largest difference because the scale explicitly rewards correct use of displayed sources. The result is meaningful for system behaviour but is not an independent human preference estimate. The judge was Qwen3 and had access to the declared evidence boundary; cross-model separation applies to claim verification, not to this holistic judge.

Ceiling concentration is also visible. Rule-RAG means exceed 4.93 for four dimensions, leaving little range for within-condition correlation. General quality and specificity retained more spread. This ceiling effect helps explain why targeted metrics and judge scores agree strongly across conditions but only weakly among Rule-RAG texts.

## 4.13 Association between targeted metrics and general judgments

The revised results make agreement at the aggregate condition level and disagreement at the metric-definition level especially clear. Rule-RAG is strongly preferred by the holistic judge and has much higher visible support and lower UIAR, yet it has lower post-hoc B alignment than No-RAG. A single broad score would conceal that distinction. Conversely, DTA alone would label the hidden-trace condition superior even though it could not have grounded its wording in the unavailable trace.

This pattern validates the multidimensional design more strongly than a uniformly positive result would. “Faithfulness” cannot be reduced to semantic compatibility with a reference after generation. The relevant questions are whether the reference was available, whether concrete assertions are supported, whether citations can be traced, and whether the explanation is useful and coherent. Those constructs overlap but are not interchangeable.

## 4.14 Length-matched sensitivity

The ten closest realised-length pairs per generator yielded 30 pairs with means of 54.17 No-RAG and 54.37 Rule-RAG words; the mean absolute paired gap was only 0.33 words. Micro DTA was 85.59% for No-RAG and 83.43% for Rule-RAG; macro rates were 89.13% and 83.62%. The paired macro difference was −5.51 points, but its 95% interval crossed zero (−16.40 to +5.22; \(p=0.309\)). This small sensitivity therefore does not contradict the full-corpus negative estimate, but it lacks precision.

UIAR remained too sparse for a stable paired sensitivity: only eight pairs had eligible attributes on both sides. Descriptively, micro UIAR was 65.63% No-RAG and 45.45% Rule-RAG, and unsupported-attribute density was 1.292 versus 0.613 per 100 words. The common prompt cap and density result reduce the original length concern, but this selected 30-pair subset is not a substitute for a separate factorial experiment that randomises length regime.

## 4.15 Deterministic qualitative examples

Deterministically selected median-distance examples were regenerated and re-evaluated with the corrected pipeline. Their principal diagnostic pattern is conceptual rather than a showcase of large gains. No-RAG often gives fluent, conventionally plausible fashion relations that align with hidden B after the fact. The same text may still introduce unsupported colour, material, comfort, construction, brand-quality, or seasonal attributes. Rule-RAG more often anchors its rationale in named category, formality, and compatibility rules and avoids some item-specific embellishment, but a cited generic rule can still be over-applied to an instance.

These examples explain the apparently conflicting aggregate results. A No-RAG explanation can score highly on DTA because ordinary fashion knowledge overlaps with the rule bank, yet score poorly on UIAR because its concrete details are not in A. A Rule-RAG explanation can have excellent visible support and judge scores while losing strict B entailment on a proposition that extends beyond a rule’s wording. Citation markers expose the intended provenance, but they do not guarantee that every attached proposition follows from the cited rule.

### 4.15.1 Cross-metric diagnostic synthesis

Four diagnostic regions remain important. High DTA with low UIAR is desirable trace compatibility without unsupported item detail. High DTA with high UIAR is plausible styling logic accompanied by ungrounded concrete attributes. Lower DTA with no UIAR denominator can describe a cautious relational explanation that makes no concrete item assertion; its UIAR is correctly N/A. Finally, high holistic quality can coexist with a targeted defect because a 1–5 scale averages over propositions.

Claim volume is now much better balanced—10,703 versus 8,666 substantive claims—because the prompt cap is shared. Rates still prevent different numbers of claims from being mistaken for performance, while density directly addresses exposure per 100 words. Missing extractions and verifications are never imputed as successes. Similarly, the scarcity of contradiction labels does not establish truth; A and B often lack evidence capable of proving the opposite. The supported conclusion is a reduction in unverified item-specific assertions, not correction of world-factual falsehoods.

The recommendation and explanation layers therefore form a non-circular chain. Stage 6 shows that B affected rank order without improving the original general-CLIP effectiveness measures. Stage 8 shows that displaying B makes the resulting language visibly grounded and less prone to unsupported item attributes, but not more semantically aligned with B than post-hoc No-RAG text under this verifier. These are different propositions, and the revised thesis keeps them separate.

## 4.16 Answers to the research questions

RQ1 asked whether exact trace access improves alignment with the reranking decision. Under the revised cross-model operationalisation, the answer is no. Micro DTA was 90.08% No-RAG and 84.95% Rule-RAG; macro rates were 91.45% and 86.36%; and the paired difference was −5.13 points (95% CI −6.51 to −3.77, \(p=0.0004\)). No-RAG’s value is post-hoc agreement, not evidence-grounded use.

RQ2 asked whether trace access reduces unsupported concrete item assertions. The answer is yes. Micro UIAR fell from 62.38% to 37.11%, macro UIAR from 67.17% to 44.72%, and the jointly eligible paired difference was −17.45 points (95% CI −24.61 to −10.56, \(p=0.0004\)). Unsupported Attribute Density fell from 0.728 to 0.317 per 100 words.

RQ3 asked whether broad quality judgments agree with evidence-focused metrics. All six judge dimensions favoured Rule-RAG, as did visible support and UIAR, but DTA favoured post-hoc No-RAG alignment. The measures therefore answer meaningfully different questions. Strict citation coverage was too sparse and evaluator-dependent to supply an additional positive claim.

RQ4 asked whether conclusions remain credible after controls and robustness checks. The shared 75-word instruction removes the main design asymmetry; cross-model verification reduces direct self-confirmation; and the length-matched DTA estimate is compatible with the full result but imprecise. UIAR density and holistic judgments remain favourable. Robustness is therefore metric-specific rather than uniformly positive.

## 4.17 Chapter summary

The original general-CLIP fused ranker outperformed text-oriented baselines on key top-ten measures, while evidence reranking changed ranks materially without a significant effectiveness gain. The new FashionCLIP baseline adds an important positive result: its image tower achieved HR@10 0.266 and NDCG@10 0.1387, improvements of 0.062 and 0.0365 over general CLIP image, both Holm-adjusted \(p=0.0084\). FashionCLIP’s fixed 0.40/0.60 fusion reached HR@10 0.225 and NDCG@10 0.1140, but none of its paired fused gains over general fused CLIP was significant. Domain adaptation therefore strengthened image retrieval, while the inherited fusion diluted that advantage.

At the explanation layer, Rule-RAG produced much higher visible grounding, substantially lower UIAR and unsupported-attribute density, and higher automated quality on all six dimensions under a shared word-limit instruction. It did not improve common-reference DTA; the cross-model verifier found higher post-hoc B alignment in No-RAG. Citation validity was too sparsely recognised for a strong conclusion. The resulting contribution is not that every metric favours retrieval, but that trace compatibility, evidence availability, unsupported specificity, and perceived quality can diverge—and that a reproducible evaluation must report those divergences.

## References

[1] Järvelin, K. and Kekäläinen, J. (2002) ‘Cumulated gain-based evaluation of IR techniques’, *ACM Transactions on Information Systems*, 20(4), pp. 422–446. https://doi.org/10.1145/582415.582418.

[2] Efron, B. and Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*. New York: Chapman & Hall/CRC.

[3] Holm, S. (1979) ‘A simple sequentially rejective multiple test procedure’, *Scandinavian Journal of Statistics*, 6(2), pp. 65–70.

[4] Spearman, C. (1904) ‘The proof and measurement of association between two things’, *The American Journal of Psychology*, 15(1), pp. 72–101. https://doi.org/10.2307/1412159.

[5] Jacovi, A. and Goldberg, Y. (2020) ‘Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?’, *Proceedings of ACL 2020*, pp. 4198–4205. https://doi.org/10.18653/v1/2020.acl-main.386.
