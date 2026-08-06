# Blinded 360-claim researcher-audit protocol

Purpose: assess agreement between independent human labels and the frozen automatic verifier without revealing variant, generator, automatic label, verifier reason, scores, or model metadata.

Population: complete post-recovery atomic-claim verification rows. Seed: 42. The deterministic design samples 360 distinct explanations: 90 per grounding variant and 30 per generator family within each variant. It first secures one occurrence of every available automatic label, including contradicted and not-verifiable, then fills quotas while capping sampled claims at two per paper case. The sealed key records automatic label, provenance, nominal inclusion probability, and inverse-probability weight. Nominal probabilities are the pre-label-priority within variant/generator explanation draw probability times one-over-claims-per-explanation; label-priority selection makes the weights design documentation rather than a claim of exact inclusion probabilities under every constraint.

Researchers must annotate only `blinded_360_claims.csv`, retain `human_label` and `human_notes`, and not open the key until annotation is complete. The scoring script is intentionally not run during preparation.
