# Chapter 4

# Results

## 4.1 Introduction

This chapter reports the frozen final five-stage experiment. Every result is derived from the canonical Stage 1--5 artifacts; no model was called during final analysis. Results follow the pipeline: preflight, recommendation evaluation, explanation generation, extraction and verification, then paired claim-level contrasts. A claim described as unsupported is unsupported by the supplied source under the frozen protocol; it is not thereby false in the world. Citation syntax is also kept separate from citation entailment.

The explanation intervention is narrow and controlled. No-RAG and Rule-RAG explain the same locked recommendation for the same case and generator. Both receive common context A. Rule-RAG alone receives the exact five-rule reranking trace B. The experiment therefore measures the effect of making stored decision evidence available during generation, rather than comparing explanations of different recommendations.

## 4.2 Final preflight and frozen design

Stage 1 passed the final preflight gate. Dataset and split inputs, embeddings, candidate-pool construction, prompts, schemas, and output hashes were frozen before confirmatory work. The final knowledge base contains 200 curated fashion rules, exactly 40 each for bags, bottoms, outerwear, shoes, and tops. The audit confirmed unique identifiers, complete provenance, and no exact or threshold-defined near duplicates.

Validation-only grids fixed the operational point: image/text fusion of 0.40/0.60, CLIP/evidence reranking of 0.75/0.25, and five retrieved rules per candidate. These checks are not pooled with confirmatory results. Crucially, the five rules saved during reranking are the same records supplied as Rule-RAG evidence. No separate explanation-time retrieval occurred.

## 4.3 Recommendation results

The recommendation evaluation contained 1,000 held-out cases, 200 per category, drawn from 734 underlying query outfits. Confidence intervals use 5,000 percentile bootstrap replicates clustered by query outfit. The task remains controlled same-category candidate-pool ranking, with held-out outfit co-occurrence as an offline relevance proxy.

| Method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM text | 4.8% | 12.0% | 17.7% | 8.4% | 10.3% | 10.2% |
| CLIP image | 3.2% | 12.6% | 22.0% | 7.9% | 10.9% | 10.0% |
| CLIP text | 3.5% | 13.8% | 21.2% | 8.6% | 11.0% | 10.1% |
| Fused CLIP | 4.5% | 14.3% | 23.1% | 9.4% | 12.2% | 11.4% |
| Evidence rerank | 3.8% | 13.2% | 22.5% | 8.5% | 11.4% | 10.6% |

Fused CLIP was the strongest tested conventional retrieval pathway. Evidence reranking did not improve these aggregate relevance measures, so it must not be presented as an accuracy improvement. It did, however, change the top-ranked recommendation in 26.5% of cases, with mean top-five overlap of 3.859, mean top-one evidence-score gain of 0.1473, and mean pre-to-post rank shift of 0.566. One hundred and forty-eight of the 200 rules occurred in at least one locked trace. The symbolic component therefore materially participated in the decisions it later helped explain.

## 4.4 Explanation completion and pairing

Five hundred evidence-eligible locked recommendations, balanced at 100 cases per category, formed the explanation study. Gemma 4 12B, Llama 3.1 8B Instruct, and Ministral 3 14B Instruct generated both conditions, giving 3,000 attempted cells. Stage 2 accepted 2,969 explanations. The 31 terminal failures were all Llama Rule-RAG responses exceeding the shared 75-word limit after permitted retries; none was replaced or silently repaired.

Final inference therefore uses generator-specific complete pairs: Gemma 474, Llama 438, and Ministral 456. For the overall estimate, generator-level within-case differences are averaged before resampling, leaving 498 underlying cases with at least one complete generator pair. This prevents multiple generator outputs for a case being treated as independent observations and avoids raw unequal condition totals.

## 4.5 Extraction and verification completion

Qwen 3.5 9B extracted atomic claims from the 2,969 accepted explanations. Stage 3 accepted 2,965 extraction records containing 17,710 claims; four terminal extraction failures were retained. Phi-4 14B then verified claims against A, the exact trace, the full-KB candidate packet, and observed citations. Stage 4 accepted 2,861 verification records covering 16,804 claims. The 104 terminal verification failures remain in the canonical missingness record.

The verifier preserves separate fields for `trace_support`, `full_kb_support`, `common_reference_support`, and `citation_entailment`. Of the verified claims, 2,058 were trace-supported and 2,095 were supported by the full KB packet. Common-reference support was 961 supported, 13 not supported, and 15,830 not applicable. Citation entailment was 1,820 entails, 5,502 does not entail, and 9,482 not applicable.

A deterministic correction addressed 163 logically impossible labels where trace support was supported but full-KB support was not supported. For every affected claim, the trace rule was confirmed to occur in its full-KB packet. Only the full-KB label was changed to supported; no model was rerun and no other field was altered. The final invariant is explicit: trace support implies full-KB support.

## 4.6 Paired explanation results

Primary contrasts use 5,000 paired percentile bootstrap replicates clustered by underlying case, with Holm adjustment across four prespecified overall metrics. Table 4.2 presents Rule-RAG minus No-RAG. Positive values favour Rule-RAG except for UIFR, where lower is preferable.

For metric \(m\), the reported effect is the average within-case difference after first averaging the available generator-specific complete-pair differences:

\[
\widehat{\Delta}_m=
\frac{1}{|Q|}\sum_{q\in Q}
\left(\frac{1}{|G_q|}\sum_{g\in G_q}
\bigl[m(E^{\mathrm{RuleRAG}}_{qg})-m(E^{\mathrm{NoRAG}}_{qg})\bigr]\right).
\tag{4.1}
\]

Here, \(Q\) is the set of underlying cases with at least one complete pair and \(G_q\) is the available generator set for case \(q\). The expression clarifies why three generator outputs for one case are not treated as three independent cases.

| Metric | Paired cases | Difference | 95% CI | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| Reranking-trace claim support rate | 498 | +21.02 pp | +19.72 to +22.37 pp | 0.0016 |
| Full-KB claim support rate | 498 | +21.40 pp | +20.11 to +22.71 pp | 0.0016 |
| Unsupported Item-Fact Rate | 53 | +0.63 pp | -5.03 to +5.66 pp | 0.9042 |
| Trace-supported claims per 100 words | 498 | +1.60 | +1.48 to +1.72 | 0.0016 |

Rule-RAG substantially increased claims supported by both the exact trace and the wider final KB packet. The increase of 1.60 trace-supported claims per 100 words shows that the result is not merely an effect of output volume. For No-RAG, agreement with hidden B is post-hoc alignment; for Rule-RAG, support is evidence-grounded because B was available in the prompt.

UIFR is inconclusive. It applies only when both paired explanations contain the relevant common-reference item-fact claim type, leaving 53 pairs. The interval includes both small benefit and small harm, so it cannot justify a corpus-wide claim that Rule-RAG reduces item-fact overreach.

## 4.7 Robustness, citations, and research questions

The primary support effects were positive for every generator. Trace-support differences were +25.09 percentage points for Gemma, +26.74 for Llama, and +11.35 for Ministral; full-KB differences were +25.22, +27.26, and +11.77 points. All corresponding confidence intervals excluded zero. All five categories also showed positive trace and full-KB support differences, with intervals excluding zero. This establishes directional robustness within the tested roster and categories, not universal generalisation.

Citation markers improve inspectability but are not treated as support. A claim can carry a malformed, irrelevant, or non-entailing rule identifier. The verified citation-entailment field is therefore essential: the experiment demonstrates source provenance and stronger source-grounded claim rates, but not that citation syntax alone validates a claim.

RQ1 is supported: trace exposure increased trace support by 21.02 points and full-KB support by 21.40 points. RQ2 is inconclusive because UIFR has sparse complete-pair eligibility. RQ3 is answered conservatively: citation syntax does not demonstrate citation entailment. RQ4 is supported for the primary support outcomes, whose direction held across all generators and categories; it is limited by failure asymmetry and sparse UIFR eligibility.

## 4.8 Chapter summary

The final run establishes an auditable chain from reranking to explanation. The 200-rule KB changed a meaningful share of rankings without improving fused-CLIP relevance, and its exact five-rule trace was supplied unchanged to Rule-RAG. In complete-pair, case-clustered analysis, Rule-RAG produced substantially more trace-supported and full-KB-supported claims, as well as more trace-supported claims per one hundred words. The results do not establish general recommendation superiority, human preference, full-catalogue performance, or reduced UIFR. Chapter 5 interprets this bounded but robust grounding result alongside its limitations.

## 4.9 Interpretation of recommendation effectiveness

The recommendation results are best read as a trade-off rather than a ranking win. Text-only MiniLM obtained the highest HR@1, whereas fused CLIP produced the best aggregate top-five, top-ten, NDCG, and reciprocal-rank values among the evaluated methods. This pattern is plausible in a controlled outfit-completion setting: a single exact hit at rank one and broader ranking quality are related but not identical properties. It also cautions against reducing the comparison to one preferred metric.

The evidence reranker preserved much of the fused model's ranking performance but was lower on every headline aggregate metric in Table 4.1. Its HR@10 was 22.5% rather than 23.1%, NDCG@10 was 11.4% rather than 12.2%, and MRR was 10.6% rather than 11.4%. The bootstrap intervals quantify uncertainty around each method, but this study was not designed as a non-inferiority trial with a predeclared equivalence margin. The correct conclusion is consequently not that the two systems are equivalent. It is that the final evidence does not demonstrate an accuracy advantage for reranking, while it does demonstrate that the rule component materially changes selected recommendations.

This distinction matters for the explanation study. A system can expose a more inspectable evidence record by weighting a symbolic score, yet that record may lead it away from the latent compatibility signal captured by the multimodal encoder. Conversely, a system can optimise an offline relevance proxy without retaining a human-readable account of why an item rose above alternatives. The present results do not resolve that design tension. They make it observable: conventional relevance and evidence traceability are separate evaluation objectives.

## 4.10 Trace participation and coverage

The 148 rules observed in at least one locked trace indicate broad participation by the final KB. The remaining rules were not necessarily irrelevant or defective; they may have been inapplicable to the sampled cases, represented less common styling relations, or ranked below the five retained rules. Nevertheless, reporting coverage avoids a misleading picture in which a 200-rule resource appears fully exercised simply because it was available to retrieval.

The stored trace contains more than rule text. It preserves the rule identifiers, retrieval similarity, reliability weight, bonus, ordering, and weighted contribution used by the evidence score. This permits a direct audit of the chain from candidate scoring to the Rule-RAG prompt. The trace is therefore stronger than a post-hoc list of thematically relevant rules. It does not, however, capture every feature of the CLIP score, and it should not be described as a complete causal explanation of all neural computations. It is an exact account of the symbolic evidence component of the hybrid reranker.

This architectural boundary also clarifies why the trace and full-KB outcomes are both reported. Trace support is the stricter outcome: it ties a claim to rule evidence that contributed to the selected candidate's reranking. Full-KB support permits relevant rules in the record's larger candidate packet, including rules that did not appear among the selected five. The small difference between the support totals does not eliminate the conceptual distinction. A thesis claim about decision evidence should rely primarily on trace support, while full-KB support provides a broader knowledge-grounding view.

## 4.11 Missingness and failure accounting

The pipeline retained failure records at every stage. Of 3,000 generation attempts, 2,969 explanations were accepted. Of these, 2,965 were successfully extracted and 2,861 completed verification. This yields a final claim-verification coverage of 16,804 claims from 2,861 explanation records. Failures were not converted to zero claims, favourable labels, or synthetic replacements. Such handling would create a false appearance of complete coverage and could bias a comparison if a condition were more difficult to parse or verify.

The generation failures were not balanced: all 31 occurred in Llama Rule-RAG. This is a protocol-compliance limitation rather than evidence that Rule-RAG fails semantically. Still, it matters because a complete-pair estimate describes retained paired outputs, not the behaviour of cells that failed the output contract. The generator-specific pairing policy addresses the immediate statistical issue: it prevents a Rule-RAG total of 469 being compared with a No-RAG total of 500 as if the observations were paired. It does not turn missing outputs into evidence of success.

Verification failure was smaller but present in both conditions. Gemma had 22 No-RAG and three Rule-RAG terminal verification failures; Llama had 18 and 16; Ministral had 22 and 23. These counts show why missingness must be reported by generator and condition rather than as one aggregate percentage. The final results should be interpreted with the associated denominator table, particularly for secondary outcomes whose eligibility is already restricted.

## 4.12 Statistical interpretation

The overall explanation analysis uses the underlying case as the resampling unit because outputs from different generators share the same locked recommendation and evidence. The analysis first computes each available generator-specific paired difference, then averages available generator effects within case, and finally bootstraps cases. This procedure respects both repeated measurement and unequal complete-pair availability. It is more conservative than treating the 1,368 complete case--generator pairs as independent rows.

The reported confidence intervals are percentile intervals from 5,000 paired bootstrap replicates. Holm correction is applied across the four prespecified overall explanation outcomes. The three positive support outcomes have the smallest attainable two-sided bootstrap p-value in this design, 0.0004 before adjustment and 0.0016 after adjustment. This does not make the effects universally valid; it means that, conditional on the frozen cases, models, and evaluator, the observed paired support differences are precise and unlikely to have arisen from resampling variation alone.

The UIFR result illustrates why effect size, eligibility, and precision must be read together. Its point estimate is close to zero and its interval is much wider than those of the primary support metrics. The limiting factor is not merely a lack of arithmetic power; it is that a common-reference item-fact outcome is substantively defined only for a small subset of paired explanations. Reporting it as a null result with its denominator is more informative than either omitting it or extrapolating it to every output.

## 4.13 Qualitative reading of the quantitative result

The quantitative contrast supports a recognisable qualitative mechanism. A No-RAG explanation can offer fluent advice about balance, formality, or coordination without having access to the rules that helped rerank the item. Such language may overlap with general fashion knowledge and occasionally agree with the hidden trace, but the agreement is post-hoc. A Rule-RAG explanation has access to the actual rule identifiers and texts, making it more likely to state a relation that the trace can support.

That mechanism does not licence indiscriminate copying of rules into prose. A rule can be conditional, generic, or only partly applicable to the catalogue record. The desirable explanation uses the rule as a bounded relational rationale and avoids turning a generic prescription into an asserted product fact. The final verification schema detects this distinction through the separate trace, full-KB, common-reference, and citation fields. It is precisely this source separation, rather than generic stylistic fluency, that makes the experiment auditable.

Canonical qualitative records are retained with their prompt packets, locked recommendation, trace, raw explanation, extracted claims, and verification fields. They should be used in the thesis to illustrate both a successful trace-grounded explanation and a failure mode such as an unentailed citation or an unsupported instance-level detail. They are explanatory examples, not extra evidence selected to replace the corpus-level statistics.

## 4.14 Validity boundaries

Several boundaries govern the meaning of these findings. The rule base is curated and finite; it is not a complete representation of fashion expertise or verified product metadata. Candidate pools are sampled rather than catalogue-wide. Outfit co-occurrence supplies a useful held-out relevance signal, but it does not encode all legitimate combinations or personal preference. The five categories omit many product types and do not represent every cultural, seasonal, or commercial styling context.

The explanation analysis is restricted to recommendations for which an evidence trace was available. This is necessary for the trace intervention but prevents generalisation to cases where rule retrieval yields no usable evidence. Images were used by the retrieval model but were not converted into textual evidence. Consequently, a visually true colour, pattern, or silhouette claim may still be unsupported by the closed-world explanation packet. This conservative design protects provenance but limits the descriptive richness of a permissible explanation.

Finally, claim extraction and verification are automated. Qwen and Phi have distinct roles, which reduces direct same-model confirmation, but neither substitutes for a human annotation study. Bootstrap intervals reflect variation over cases, not uncertainty in the extractor or verifier. The results are therefore strongest as a reproducible systems comparison under a specified evaluator contract. They are not proof of universal factuality, user trust, or complete faithfulness to the neural portion of the ranker.

## 4.15 Practical implications of the result

The findings suggest a practical design principle for evidence-aware recommenders. Evidence should be created as part of the decision process and carried forward as typed data, rather than retrieved only after a recommendation has been selected. In the present system, this means that a later explanation can name and use rules whose identifiers, contributions, and ordering are already known. The same pattern could support developer audit tools, user-facing source panels, or post-deployment monitoring without claiming that a natural-language response is self-verifying.

The result also argues for cautious language around catalogue properties. A styling rule can justify a relational recommendation, such as seeking compatible formality or colour coordination, but it cannot establish an unstated physical property of an item. Systems should distinguish product metadata, visual evidence, and generic style knowledge in their prompts and interfaces. Citation displays are useful when they allow a reader to inspect the underlying rule, but their value depends on preserving the claim--source relation rather than merely adding bracketed identifiers.

Finally, the findings show why evidence-grounding evaluation should be multi-objective. A deployment team could legitimately prioritise relevance, traceability, factual restraint, user satisfaction, latency, or coverage differently. The present experiment does not provide one scalar score that settles these choices. It supplies a reproducible way to reveal their separation: conventional retrieval effectiveness was strongest for fused CLIP, whereas trace-grounded explanation claims were strongest when the reranking trace was visible to the generator.

## 4.16 Reproducibility and audit trail

The final analysis is reproducible from frozen records rather than from regenerated language. Stage manifests bind the input and output hashes for each stage, including the final recommendation, explanation, extraction, verification, and analysis tables. The release record identifies the final verification SHA-256 as `0f554e58be51c0529c59814f3c5de379ec66c02afbb8fa2c5e48249a32ae9b3e`. The final implementation also passed `ruff check .` and 55 automated tests. These checks do not validate the substantive fashion rules or the models' semantic judgements, but they make record substitution, schema drift, and broken joins detectable.

The joins between stages were exact. Three thousand explanation records were attempted; 2,969 accepted Stage-2 outputs entered extraction; 2,965 accepted Stage-3 records entered verification; and 2,861 accepted Stage-4 records provided canonical claim labels. Claim identifiers were preserved from extraction into verification. The final analysis reads the Stage-4 schema directly, including trace support, full-KB support, common-reference support, and citation entailment. It does not use the legacy outcome names or intermediate experimental tables.

One provenance amendment is reported rather than concealed. The Stage-1 manifest preserves the original prompt-configuration hash and records that the final Stage-4 verifier contract used an authorised corrected configuration. The release is bound to the actual final prompt and configuration hashes. This was an interface and consistency correction, not a performance-oriented regeneration: no generator, extractor, or verifier call was repeated. The distinction is material because transparent provenance is preferable to pretending that a later detected contract defect never occurred.

## 4.17 Results in relation to the experimental claim

Taken together, the results support a specific systems claim. When the exact expert-rule trace that participated in reranking is passed to an explanation generator, the generated explanation contains more claims that the frozen verifier judges supported by that trace and by the corresponding full-KB packet. The effect is large in absolute terms, survives multiplicity adjustment, and is stable over the tested generators and target categories. The result is not based on a post-hoc retrieval set, a changed recommendation, or raw unpaired condition totals.

The claim is intentionally narrower than several tempting alternatives. It does not show that the rule base makes recommendation more accurate, that all Rule-RAG prose is correct, that cited rules always entail the surrounding text, or that users prefer these explanations. Nor does it establish a complete explanation of CLIP's latent reasoning. Instead, it establishes an auditable connection between a symbolic component of ranking and an explanation condition, with claim-level evidence of stronger grounding under the stated evaluator.

This narrowness is a strength for interpretation. The recommendation findings prevent an unsupported relevance claim; the UIFR result prevents an unsupported product-factuality claim; citation-entailment records prevent decorative source markers being counted as proof; and missingness reporting prevents terminal failures disappearing from the analysis. The remaining positive result is therefore less broad than a generic assertion that Rule-RAG is “better,” but it is more reproducible and more defensible.

The result is also practically interpretable. A 21-percentage-point support difference is not a small formatting effect around an already saturated outcome; it represents a substantial change in the share of assessed claims that can be tied to recorded rule evidence. The parallel full-KB result provides a useful cross-check: the effect is not created solely by a narrow trace label, but persists when the verifier considers the record's wider final KB packet. At the same time, the small gap between these outcomes should not be used to collapse them. The trace estimate remains the more direct test of alignment with evidence used in reranking.

The study's controls make a simpler alternative explanation less likely. Rule-RAG did not receive a different recommendation, a different query, or a newly retrieved set of rules. It received the stored decision trace for the recommendation that both conditions were required to explain. Generator-specific complete pairing then compared like with like, and case-clustered resampling acknowledged that the three generators share a case context. These choices do not remove every prompt effect, but they make the central evidence intervention unusually explicit and auditable.

The final conclusion should therefore be read as comparative rather than absolute. Within the frozen experiment, trace access changed the evidential character of generated explanations. Outside that boundary, performance may depend on rule coverage, catalogue metadata, candidate-pool composition, decoding behaviour, and the reliability of automated verification. The release package preserves enough detail for those dependencies to be examined rather than hidden behind a single headline metric.

This is the appropriate evidential standard for the thesis: a transparent, bounded systems result whose assumptions, failures, limitations, and reproducibility record remain available for independent scrutiny by future reviewers, examiners, and subsequent independent replication studies worldwide.

## References

[1] Järvelin, K. and Kekäläinen, J. (2002) ‘Cumulated gain-based evaluation of IR techniques’, *ACM Transactions on Information Systems*, 20(4), pp. 422–446. https://doi.org/10.1145/582415.582418.

[2] Efron, B. and Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*. New York: Chapman & Hall/CRC.

[3] Holm, S. (1979) ‘A simple sequentially rejective multiple test procedure’, *Scandinavian Journal of Statistics*, 6(2), pp. 65–70.

[4] Jacovi, A. and Goldberg, Y. (2020) ‘Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?’, *Proceedings of ACL 2020*, pp. 4198–4205. https://doi.org/10.18653/v1/2020.acl-main.386.
