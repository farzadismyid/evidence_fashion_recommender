# Evidence-Constrained Multimodal Fashion Recommendation

This repository implements the final clean experiment specified in [proposal.md](proposal.md).
It evaluates controlled sampled-pool recommendation across `tops`, `bottoms`, `shoes`,
`outerwear`, and `bags`, then tests whether explanations conditioned on the exact expert-rule
trace used for reranking are more evidentially grounded.

The sole active knowledge base is `data/kb/fashion_rules.csv`: 200 frozen rules, 40 for each
target category. The confirmatory operating point is fixed at 0.40 image / 0.60 CLIP text and
0.75 CLIP / 0.25 rule evidence with up to five applicable rules. Validation grids are descriptive
sensitivity analyses only.

The run is governed by five approval-gated stages:

1. preflight, clean reset, and final freeze;
2. final recommendations and 3,000 fresh explanations;
3. fresh atomic-claim extraction;
4. fresh claim verification;
5. final analysis and release closure.

Canonical runtime outputs are under `.runtime/current/`; compact reproducibility materials are
published only after release in `artifacts/release/`. Earlier development outputs are not active
results.
