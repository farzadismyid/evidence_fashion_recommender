# Chapter 1

# Introduction

## 1.1 Problem and scope

Fashion recommendation must identify a complementary item and explain why it fits an outfit. Fluent language alone does not show which evidence influenced a recommendation. This thesis therefore evaluates an evidence-constrained multimodal recommender in which a recommendation is locked before explanation generation and an exact symbolic rule trace is retained from reranking.

The final controlled experiment uses five target categories: bags, bottoms, outerwear, shoes, and tops. Accessories outside bags are excluded because sufficiently explicit expert-rule coverage was not available to support a controlled evidence evaluation. The scope is an offline experiment on held-out Polyvore cases; it is not a user study or a claim of world-factual product knowledge.

## 1.2 Research objective

The system combines CLIP image and text retrieval with a V3 expert-rule knowledge base. Rules whose antecedents are established by the case can contribute to reranking and form the exact trace for the locked recommendation. For each explanation case, No-RAG receives the same case context as Rule-RAG, while Rule-RAG additionally receives only that exact trace. The recommendation identity, generator, and case are held constant within the pair.

The final study asks:

- **RQ1:** How does expert-rule reranking affect held-out recommendation effectiveness and rank order?
- **RQ2:** Does giving the generator the exact stored rule trace increase support for its extracted claims?
- **RQ3:** Does trace access reduce unsupported concrete item-fact assertions while increasing grounded explanatory information?
- **RQ4:** Are observed explanation effects robust across the tested generators, target categories, claim types, and trace sizes?

## 1.3 Contributions

This work contributes a reproducible evidence-constrained pipeline with a locked-recommendation contrast, strict antecedent-and-consequent rule applicability, canonical K-series citation handling, and frozen artifacts for generation, extraction, verification, and analysis. The final explanation evaluation separates exact-trace support, broader full-KB support, common-reference item-fact support, trace utilisation, and citation diagnostics rather than collapsing them into a generic quality score.

## 1.4 Thesis organisation

Chapter 2 positions the study and defines its evidence boundary. Chapter 3 specifies the frozen data, V3 knowledge base, recommendation, generation, extraction, verification, calibration, and statistical procedures. Chapter 4 reports the final frozen recommendation and explanation results. Chapter 5 gives conclusions, limitations, and future work.
