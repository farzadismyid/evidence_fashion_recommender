# Chapter 5

# Discussion and Conclusion

## 5.1 Conclusions

The completed project demonstrates a bounded form of evidence-grounded fashion explanation. A V3 expert-rule KB contributed to reranking and preserved an exact, antecedent-applicable trace for the locked recommendation. When the same trace was supplied to Rule-RAG generation, exact-trace claim support increased from 4.89% to 50.22% and full-KB support from 5.33% to 50.46% across 1,488 paired available explanations. Grounded claims per 100 words increased from 0.50 to 4.81.

The evidence boundary also matters for concrete product statements. On the limited common-reference-eligible paired population, UIFR fell from 41.41% to 26.79%. This means fewer literal item-fact claims were unsupported by the supplied case reference; it does not establish that unsupported claims are false in the world. The same distinction applies to No-RAG’s small post-hoc KB support rate: a hidden rule that happens to fit an explanation is not evidence that the generator used it.

The recommendation component is evidence-participating rather than evidence-proven superior. V3 reranking changed the top result in 47.6% of 1,000 cases and changed nearly every top-five list. The final thesis does not equate this rank movement with universal recommendation improvement; recommendation effectiveness is reported separately in the frozen Stage 6 tables.

## 5.2 Interpretation

Rule-RAG’s advantage is consistent across the three tested generators, five target categories, reported claim types, and trace sizes. The exact-trace and full-KB rates are nearly identical for Rule-RAG, which supports the interpretation that available trace evidence, rather than unrelated KB retrieval, accounts for most observed support. Mean trace utilisation of 90.30% further indicates that Rule-RAG explanations usually reflect the supplied evidence.

Citation syntax was reliable in the frozen records: 8,912 citation-bearing claim occurrences were canonical K-series references. Semantic citation reliability was much weaker: only 32.34% of those cited occurrences had a cited rule that entailed the associated claim. The study therefore supports citation syntax as an auditability mechanism, not citation presence as proof of grounded reasoning.

## 5.3 Limitations

The study is limited to a fixed offline dataset, five target categories, and three local open-weight generators. Accessories outside bags are intentionally excluded because explicit expert-rule coverage was insufficient for the final controlled design. The V3 rule base is finite and expert-curated; its coverage and wording define the claims the evaluation can support.

The final corpus retains thirteen terminal generation failures and one terminal verification record. The analysis uses paired available cases and records these failures rather than fabricating replacements. The human calibration stage strengthens checks on extraction and verification contracts, but the final evaluation still relies on Qwen claim extraction and Phi verification. It does not replace a broad human study, product-ground-truth audit, or external entailment benchmark.

UIFR has a small paired denominator (65) because the eligibility policy deliberately excludes subjective styling, suitability, and KB-derived rationale claims. This conservatism clarifies the literal-fact construct but limits its generality. Citations likewise remain a weak semantic signal despite their clean syntax.

## 5.4 Future work

Future work should obtain expert and user annotations for claim support, usefulness, and factual product properties; extend the rule base only where authoritative evidence supports new antecedent-consequent families; and study whether stronger citation attachment or constrained generation improves semantic citation entailment. A larger category taxonomy should be introduced only after coverage is explicit enough to preserve the current applicability discipline. Online interaction, user preferences, and temporal catalogue updates are outside this frozen offline release.

## 5.5 Final statement

Within the released experimental boundary, exposing a stored exact V3 rule trace yields substantially more trace-supported explanatory content than giving the same generator only case context. The project’s contribution is not that every generated explanation is true or that a citation guarantees faithfulness. It is a reproducible, paired framework that records which evidence entered a recommendation, evaluates whether generated claims are entailed by that evidence, and preserves both positive findings and remaining failures for scrutiny.
