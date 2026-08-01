# Methodology preserved from the original notebook

The modular implementation preserves the original research stages:

1. Parse Polyvore item and outfit identifiers.
2. Map fine-grained product categories to target recommendation groups.
3. Compare MiniLM text retrieval with CLIP image, text, and fused retrieval.
4. Treat same-outfit items in the requested category as compatibility positives.
5. Retrieve external styling rules from the fashion knowledge base.
6. Rerank a larger CLIP candidate pool using a light evidence score.
7. Compare No-RAG, Item-RAG, Rule-RAG, and Hybrid-RAG explanations.
8. Enforce candidate-locked and leakage-safe prompts.
9. Evaluate ranking, evidence coverage, citation behaviour, unsupported claims,
   explanation quality, and paired statistical differences.

## Systematic metric families

- Recommendation: Precision@K, Recall@K, HitRate@K, NDCG@K, reciprocal rank, and positive rank.
- RAG retrieval: evidence coverage, rule count, unique rules, high-reliability rules,
  category/input compatibility, Precision@K, Recall@K over applicable KB rules, HitRate@K,
  NDCG@K, and reciprocal rank.
- Faithfulness: citation presence/correctness/precision, evidence overlap,
  evidence-aware unsupported claims, and occasion drift.
- Explanation safety: candidate substitution, prompt leakage, and explanation length.
- Independent quality evaluation: faithfulness, usefulness, specificity,
  style appropriateness, and grounding safety from Qwen3 judging Llama generations.
- Inference: paired bootstrap confidence intervals, Holm correction, and
  Benjamini-Hochberg FDR correction.

Human evaluation is excluded from the current systematic experiment and remains future work.

The archived notebook remains the source of truth for the first historical run. Changes to
the modular methodology must be represented by new named configuration files rather than
silent edits to a reported experiment.
