# Chapter 2

# Literature Review and Conceptual Framework

## 2.1 Multimodal recommendation and explanation

Fashion compatibility is contextual: an item can complement an outfit without being visually similar to it. Multimodal retrieval therefore combines catalogue text with visual representations, while rule-based knowledge provides an inspectable source of domain constraints. The present work uses these components for different purposes: neural representations rank candidates, and explicit expert rules provide a bounded evidence layer for reranking and explanation.

Natural-language rationales should not be assumed faithful merely because they are plausible. A generator can introduce unobserved materials, colours, comfort properties, or styling relationships. Retrieval augmentation can make relevant context available, but it still requires claim-level checking to establish whether the stated evidence entails the complete proposition.

## 2.2 Evidence boundary

The experiment distinguishes three evidence sources. Common reference A contains literal case information: the user request and supplied query and locked-item fields. Exact trace B contains only the V3 rules that passed the deterministic antecedent-applicability gate and contributed to the locked recommendation. The full V3 KB is used only as candidate expert knowledge; applicability is assessed separately before full-KB entailment.

This distinction prevents category analogy and closed-world inference. A rule supports a claim only when its antecedent is established and its consequent directly entails the claim. Concrete common-reference facts can establish literal item/context properties, but they cannot establish a general styling principle, suitability, colour harmony, or a KB-derived rationale.

## 2.3 Final evaluation constructs

The primary construct is Exact-Trace Claim Support Rate: supported extracted claims divided by all extracted claims, evaluated against the exact V3 trace provided to Rule-RAG. Full-KB Claim Support Rate uses any applicable V3 rule. Unsupported Item-Fact Rate (UIFR) is the share of common-reference-eligible concrete factual claims that are not supported; lower is better. Exact-Trace Supported Claims per 100 Words measures grounded information density.

Citation occurrence syntax is deterministic. `[K###]` is canonical; grouped, malformed, duplicate-within-bracket, and unknown references are invalid. Citation absence is not an invalid citation. Semantic citation entailment is assessed only for valid occurrences. These measures do not establish factual truth outside the supplied evidence.

## 2.4 Experimental interpretation

For No-RAG, a claim that happens to agree with an applicable V3 rule is post-hoc KB consistency, not evidence use. For Rule-RAG, exact-trace support and trace utilisation provide an auditable indication that supplied decision evidence is reflected in the explanation. Neither automated verification nor citation presence establishes human preference. This conceptual separation governs the methodology and conclusions that follow.
