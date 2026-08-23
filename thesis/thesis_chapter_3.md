# Chapter 3

# Methodology

## 3.1 Design

The final experiment is a deterministic offline, paired comparison. The held-out recommendation evaluation contains 1,000 locked cases, balanced at 200 cases per target category. A separate explanation cohort contains 500 cases, balanced at 100 cases each for bags, bottoms, outerwear, shoes, and tops. Both conditions explain the same locked item for the same case and generator; their only evidence difference is access to the exact rule trace.

All datasets, configuration files, prompt registry, model digests, V3 KB, output JSONL files, and manifests are hash-recorded. The final results use only the current frozen runtime artifacts identified in the Stage 13 release manifest.

## 3.2 Data, taxonomy, and recommendation

The study uses the pinned Polyvore catalogue split by outfit. The five-category taxonomy is bags, bottoms, outerwear, shoes, and tops. Other accessories were excluded before the final controlled experiment because the project could not establish sufficiently explicit, defensible expert-rule coverage for them.

Candidates are retrieved using normalised CLIP text and image representations. The frozen fusion combines image and text scores, then an evidence score from V3 rules is applied for reranking. Each V3 rule has structured antecedent fields and externally registered evidence. The reusable applicability gate permits a rule into a trace only if its antecedent is established by case context; a consequent is not inferred merely from a similar category or from absence of a listed alternative.

## 3.3 V3 knowledge base and exact traces

V3 is the production knowledge base (`data/kb/fashion_rules_v3.csv`). Full-KB retrieval returns candidate rules only. Before a claim can be judged supported, the verifier assesses antecedent applicability and whether the stated consequent directly entails the complete claim. Exact traces contain only antecedent-applicable V3 rules that influenced reranking. This separates decision evidence from broader available expert knowledge.

Each Rule-RAG prompt receives the exact stored trace and is instructed to cite only canonical, separate references such as `[K025] [K099]`. Citation occurrence syntax is validated in deterministic code. No-RAG receives identical case context but no rule text or citations.

## 3.4 Explanation generation

The finalized matrix crosses 500 cases, two conditions, and three locally served generators: Gemma 4 12B, Llama 3.1 8B Instruct, and Ministral 3 14B Instruct. The shared completion contract is 45–75 words, with a target near 65 words and normally two to three sentences. The locked recommendation must be preserved exactly. Bounded retries, schema checks, citation checks, raw-response retention, and terminal-failure recording apply equally to both conditions.

Stage 9 froze 2,987 accepted explanations from 3,000 cells; thirteen terminal generation failures remain recorded and are not replaced. Analyses use only available paired records. This prevents post-hoc text editing or silent substitution.

## 3.5 Claim extraction and verification

Qwen 3.5 9B extracts atomic claims from each accepted explanation under the frozen Stage 5 claim-extraction contract. Claims have consecutive identifiers in textual order; extraction retains raw model output, validates the schema, and uses bounded deterministic retries. Stage 10 produced 17,396 claims from 2,987 explanations with no terminal extraction failures.

Phi 4 14B verifies each claim. It must return exactly one ordered row for every supplied claim, with a verdict, evidence fields, and reason. Missing, empty, or duplicate rows fail closed after bounded retry. Four verdicts have a shared contract:

- **Supported:** the supplied source directly entails the complete claim.
- **Contradicted:** the source directly entails an incompatible proposition.
- **Unsupported:** relevant source material is in scope but does not entail the claim.
- **Not verifiable:** the supplied packet cannot establish the claim.

Verification is source-specific: full-KB entailment, exact-trace entailment, and common-reference item-fact support are not derived from one generic support judgement. Common-reference eligibility is deterministic and limited to literal item/context facts. Styling relations, suitability, comfort, formality, colour harmony, and KB-derived rationales are excluded unless literally supplied. Stage 11 froze 17,389 verified claims; one terminal verification record containing seven claims remains recorded.

## 3.6 Calibration

Stage 5 used a disjoint human calibration packet. Human gold was retained independently from sealed Qwen and Phi outputs. Calibration compared proposition-level extraction coverage and precision while keeping atomisation disagreement separate. Claim alignment resolves exact matches first, then deterministic entity- and polarity-aware proposition similarity with one-to-one matching. Human annotation QA and alignment are calibration-only; they do not alter final test outputs.

## 3.7 Statistical analysis

Stage 12 evaluates only frozen Stages 9–11 records. Exact-Trace Claim Support Rate is the primary metric. Full-KB Claim Support Rate, UIFR, and exact-trace supported claims per 100 words are secondary. Rule-RAG trace utilisation and citation diagnostics are auditability diagnostics. No-RAG versus Rule-RAG comparisons use paired available cases, 5,000 paired bootstrap replicates for 95% confidence intervals, paired tests, and Holm correction across primary contrasts. `not_supported` means not substantiated by the evaluated source, not factually false.

## 3.8 Reproducibility boundary

Canonical outputs are the Stage 9 explanations, Stage 10 extractions, Stage 11 verifications, and Stage 12 tables/figures. Their SHA-256 values appear in `artifacts/manifests/stage13_release_manifest.json`. Qualitative examples are selected from these records without rewriting. This design supports replication of the stored computation, while not claiming human evaluation, causal access to every neural representation, or factual verification outside supplied evidence.
