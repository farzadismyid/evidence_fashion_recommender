# Evidence participation analysis

This analysis uses frozen candidate evidence scores, cached embeddings, the frozen 3,000-case schedule, and the validation-selected 0.75 CLIP / 0.25 evidence reranker. It did not call an LLM, Ollama, judge, or external API, and did not modify frozen ranking outputs.

## Definitions and threshold status

No frozen binary applicability/support threshold exists. `CandidateEvidenceScorer.score` selects the top five rules within the target category and combines their continuous similarities; it does not determine whether a rule supports a candidate. Thus **rule-backed Coverage@K is not confirmed and is reported as N/A**, not as the presence of a retrieved rule. The scorer-selected rule count is a structural count of five score-contributing rules, not an entailment/applicability count.

Validation-only, outcome-blind score sensitivity thresholds (not rule-backed coverage): validation_q25=0.580563, validation_q50=0.603026, validation_q75=0.627691.

A retrieved rule is merely returned by retrieval; a query-relevant rule concerns the query; a candidate-applicable rule would need a binary applicability rule (absent); a score-contributing rule is one of the frozen top-five terms in the continuous scorer. Higher evidence score indicates stronger support under that frozen system, not objective recommendation correctness.

## Cohort results

Historical 300: top-1 changed in 150/300 (50.0%); mean evidence score@10 was 0.6102. New 2,700: 1314/2,700 (48.7%); score@10 0.6041. Expanded 3,000: 1,464/3,000 (48.8%); score@10 0.6047.

For 3,000 cases, reranking increased mean evidence score from 0.5946 to 0.6079 at 1, 0.5951 to 0.6056 at 5, and 0.5956 to 0.6047 at 10. Among changed top-1 results, 100.0% promoted a higher-score candidate (mean difference 0.0271); top-five/top-ten overlaps were 65.7%/71.5%, with mean absolute rank shift 8.09.

## Accuracy–evidence interpretation

The paired cluster bootstrap finds positive continuous evidence-score shifts (see CSV) while the frozen expanded recommendation report found HR@10 close but inconclusive and NDCG@10/MRR significantly lower for reranking. This supports an accuracy–evidence participation trade-off under the frozen scorer, not an accuracy improvement or rule-entailment claim.

## Reproducibility and limitations

`scripts/analyze_evidence_participation.py` reconstructs only deterministic ranks from cached embeddings and saved evidence scores. Candidate-level rule identities/applicability labels were not saved in the frozen expanded output, so no candidate-applicable supporting-rule count or binary coverage can be recovered without a new retrieval pass; such a pass was intentionally not performed. The requested expanded handoff file was absent at analysis start, so it was not altered.
