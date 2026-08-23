# Chapter 4

# Results

## 4.1 Frozen result set

This chapter reports only the final frozen canonical outputs. The recommendation evaluation has 1,000 held-out cases (200 per category). The explanation study began with 3,000 planned cells. Stage 9 retained 2,987 accepted explanations and recorded 13 terminal generation failures. Stage 10 extracted 17,396 claims. Stage 11 verified 17,389 claims and recorded one terminal verification record. The paired availability rule yields 1,488 usable No-RAG/Rule-RAG explanation pairs for the support and density comparisons.

## 4.2 Recommendation results

The V3 evidence component materially participated in ranking: the top-ranked item changed in 47.6% of the 1,000 locked cases, while 99.7% of top-five lists changed. Mean evidence gain at rank one was 0.0142 and the mean absolute rank shift in the top-ten union was 5.06 positions. Full held-out recommendation effectiveness and paired contrasts are reported in `artifacts/tables/table_02_recommendation_results.csv` and `table_03_recommendation_contrasts.csv`. These recommendation results establish the decision context; the primary outcome below is claim support in explanations of the resulting locked items.

## 4.3 Primary and secondary explanation outcomes

Table 4.1 gives the frozen paired estimates. Rates are reported as percentages except supported claims per 100 words. Confidence intervals are paired 5,000-replicate bootstrap intervals. Rule-RAG has substantially greater exact-trace and full-KB support, lower UIFR on its much smaller jointly eligible population, and a higher density of exact-trace-supported claims.

| Metric | Paired n | No-RAG | Rule-RAG | Difference (Rule-RAG − No-RAG) | 95% CI |
|---|---:|---:|---:|---:|---:|
| Exact-Trace Claim Support Rate | 1,488 | 4.89% | 50.22% | +45.34 pp | +43.44 to +47.21 pp |
| Full-KB Claim Support Rate | 1,488 | 5.33% | 50.46% | +45.13 pp | +43.26 to +46.99 pp |
| UIFR (lower is better) | 65 | 41.41% | 26.79% | −14.62 pp | −27.31 to −2.44 pp |
| Exact-Trace Supported Claims per 100 Words | 1,488 | 0.50 | 4.81 | +4.31 | +4.09 to +4.53 |

## 4.4 Exact-Trace and Full-KB support

Rule-RAG’s exact-trace support rate was 50.22%, compared with 4.89% for No-RAG. Full-KB support was similarly 50.46% versus 5.33%. The close Rule-RAG rates indicate that most verified rule support came from the supplied exact trace rather than a different applicable rule found in the wider V3 KB. No-RAG’s small full-KB rate is post-hoc consistency with hidden expert knowledge; it is not trace grounding.

The result is present for each generator and category. Exact-trace support for Rule-RAG ranges from 38.75% for Ministral to 57.48% for Llama, compared with 4.09%–6.02% for No-RAG. By category, Rule-RAG ranges from 41.93% for shoes to 60.59% for bottoms, while all No-RAG category rates remain below 11.05%. Claim-type and trace-size tables are retained in the canonical Stage 12 artifacts.

## 4.5 Unsupported Item-Fact Rate

UIFR is calculated only for literal, common-reference-eligible facts and only on the 65 paired cases where both conditions contain eligible claims. Rule-RAG’s UIFR is 26.79% against 41.41% for No-RAG. This is an evidence-scope result, not a claim that the unsupported statements are false. The restricted denominator is important: many generated claims are styling relations or rationales that are deliberately ineligible for a literal item-fact metric.

## 4.6 Grounded-information density

Exact-trace supported claims per 100 words is 4.81 for Rule-RAG and 0.50 for No-RAG. This indicates that the support advantage is not explained merely by longer text. Both conditions used the same 45–75 word acceptance contract, although realised length remains a behavioural characteristic of the generators.

## 4.7 Robustness by generator, category, claim type, and trace size

The generator breakdown shows the exact-trace support pattern for Gemma (5.84% No-RAG; 52.80% Rule-RAG), Llama (6.02%; 57.48%), and Ministral (4.09%; 38.75%). Rule-RAG exceeds No-RAG in every category and every reported claim-type group. Trace-size results remain favourable for one through four applicable rules: Rule-RAG support is 46.50%, 53.51%, 44.50%, and 41.46%, respectively, compared with 4.98%, 5.67%, 6.03%, and 2.86% for No-RAG. Small three- and four-rule strata require caution.

## 4.8 Trace utilisation and citation diagnostics

Rule-RAG trace utilisation averages 90.30% across 1,496 available Rule-RAG explanations: on average, explanations reflect most supplied exact-trace rules. Citation syntax/ID validity is deterministic. Of 8,912 citation-bearing claim occurrences, all are canonical valid K-series occurrences in the frozen output. Citation entailment is stricter: 2,882 of those occurrences (32.34%) have a cited rule that entails the associated claim. Citation validity therefore cannot be inferred from syntax alone.

## 4.9 Qualitative records

The thesis-ready qualitative examples in `reports/stage12_qualitative_examples_expanded.md` use real, unedited frozen records. They include strong and weak examples for exact-trace support, full-KB support, UIFR, grounded-information density, trace utilisation, and citation entailment, together with generator/category robustness examples. Selection is deterministic and retains counterexamples rather than selecting only successes.

## 4.10 Figures and canonical tables

The support-rate figure displays Exact-Trace and Full-KB support as percentages on a shared scale. UIFR is plotted separately and marked lower-is-better. Grounded-information density has its own scale. The supporting breakdowns, trace-utilisation data, citation diagnostics, and qualitative examples are available in the Stage 12 canonical tables and report.
