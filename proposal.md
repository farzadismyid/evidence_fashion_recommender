# Final Clean Run Proposal and Build Specification

## Evidence-Constrained Multimodal Fashion Recommendation with Expert-Rule-Grounded Explanations

### Status

This document is the **authoritative specification for the second and final clean experimental run**.

It supersedes all earlier active proposal files, interim run specifications, old KB counts, previous recommendation/explanation result sets, post-hoc trace experiments, and old Stage 9–13 result artifacts. Earlier work is development history only.

The final experiment described here is the project. The goal is to run it once from a clean state so that:

- the final knowledge base is fixed before recommendation evaluation;
- the exact rules used in reranking are the exact rules supplied to Rule-RAG explanations;
- all 3,000 explanations are generated fresh;
- all claims are extracted and verified fresh;
- statistics use the correct case-level clustering;
- only one canonical final result set remains active;
- the thesis is updated from the final run only; and
- a complete journal-paper draft is written from the final frozen results.

After the final run is frozen, **no further LLM generation, extraction, verification, judging, KB redesign, prompt tuning, or experimental rerun is planned** unless a genuine software/data-corruption bug invalidates the run.

---

# 1. Final project title

**Evidence-Constrained Multimodal Fashion Recommendation with Expert-Rule-Grounded Explanations**

---

# 2. Final research problem

Multimodal fashion recommenders can retrieve visually and semantically compatible products, but generated explanations may be generic, weakly grounded, or disconnected from the explicit knowledge used by the recommendation system.

This project studies whether supplying the **same expert-rule evidence that contributes to recommendation reranking** makes generated explanations more grounded, auditable, and restrained than explanations generated without that rule evidence.

The study evaluates two connected but distinct components:

1. **Recommendation**
   - text-only retrieval;
   - image-only retrieval;
   - multimodal fused retrieval;
   - expert-rule-aware reranking.

2. **Explanation**
   - No-RAG explanation using common case context only;
   - Rule-RAG explanation using the same common case context plus the exact stored expert-rule trace that contributed to reranking.

The system does **not** claim access to an LLM's hidden internal reasoning. The supported claim is narrower:

> Rule-RAG explanations are conditioned on explicit, auditable expert-rule evidence that also contributed to the reranking score of the locked recommendation.

---

# 3. Research questions

The research questions are framed around **magnitude, specificity, trade-offs, and robustness**, rather than the trivial question of whether giving an LLM extra text can change its output.

**RQ1 — Multimodal recommendation**  
How do image-only, text-only, and validation-selected multimodal fusion compare on controlled fashion recommendation ranking, and what genuinely multimodal fusion setting provides the strongest validation performance?

**RQ2 — Recommendation/evidence trade-off**  
How does expert-rule-aware reranking change recommendation utility and expert-evidence participation, and what validation-selected operating point best balances these objectives?

**RQ3 — Reranking-trace grounding**  
To what extent does conditioning explanation generation on the **exact expert-rule trace that contributed to reranking** increase claim-level support relative to the same recommendation explained without that trace?

**RQ4 — Trace specificity versus general KB agreement**  
Is any grounding improvement specific to the actual reranking trace, or does it mainly reflect broader agreement with applicable rules elsewhere in the expert KB?

**RQ5 — Factual restraint**  
Does reranking-trace conditioning reduce unsupported concrete item-specific factual assertions under the same common factual reference?

**RQ6 — Robustness**  
How stable are the explanation-grounding effects across generator models, target categories, claim types, and trace sizes?

---

# 4. Final experimental principles

- Final target categories: `tops`, `bottoms`, `shoes`, `outerwear`, `bags`.
- Recommendation evaluation: **1,000 deterministic test cases**, 200 per category.
- Explanation evaluation: **500 deterministic evidence-eligible cases**, 100 per category.
- Every explanation case uses the top-1 recommendation produced by the **final expert-rule-aware reranker**.
- No-RAG and Rule-RAG use the same case, same locked recommendation, same common context, same generator, same generation settings, and same length/style contract.
- The only experimental intervention is access to the final stored expert-rule trace.
- Rule-RAG receives the **exact stored rule-scoring trace used by reranking**.
- No second rule retrieval is allowed after recommendation locking.
- Prompts must not name the experimental condition.
- No image caption, image-derived attribute, object detector, or visual-to-text model may add textual evidence to explanations.
- All final explanations, extractions, verifications, and final metrics are regenerated fresh. No previous outputs may be reused.
- Models run sequentially by complete batch and are unloaded between model families.
- Test data must not tune prompts, KB content, fusion weights, reranking weights, or metric definitions.
- Failed cells are retained transparently and are not silently replaced.
- Final No-RAG vs Rule-RAG statistics use paired available cases.
- Bootstrap resampling is clustered by **underlying case/query outfit**, not case-generator row.
- `not_supported` means not substantiated by the evaluated evidence, not objectively false.
- No DeepSeek or other optional judge is part of the final confirmatory run.
- Earlier developmental calibration exercises are not acceptance gates for the final run and are not headline experimental results.

---

# 5. Dataset and final taxonomy

## 5.1 Dataset

Primary dataset:

```text
Marqo/polyvore
```

Pinned provenance:

- revision: `8c782ee447faf2d2a0402ac883cf07d3b3f43e1c`;
- configuration: `default`;
- split: `data`;
- acquisition date: `2026-08-06`;
- licence metadata: `apache-2.0`;
- raw fields: `image`, `category`, `text`, `item_ID`.

The clean run must reproduce the current five-category processed dataset and leakage-resolved split logic before the final experiment begins.

Current expected processed counts from the implemented project are:

- 47,872 eligible items;
- 19,094 eligible outfits;
- 13,365 development outfits;
- 2,864 validation outfits;
- 2,865 test outfits.

If clean preparation does not reproduce the expected counts, stop before the final experiment and investigate the discrepancy.

## 5.2 Five-category taxonomy

Internal categories must be named exactly:

```text
tops
bottoms
shoes
outerwear
bags
```

The exact raw-category mapping already implemented in the current project is authoritative and must be configuration-driven and hash-bound.

Bag allowlist:

- Bags;
- Handbags;
- Shoulder Bags;
- Tote Bags;
- Clutches;
- Messenger Bags;
- Men's Bags;
- Men's Messenger Bags.

Exclude backpacks, briefcases, luggage, jewellery, eyewear, watches, belts, headwear, and all other non-bag accessories.

The thesis and paper must never call the evaluated `bags` category “accessories”. Accessories outside bags are outside the final experiment because sufficiently explicit controlled KB coverage was not retained for them.

## 5.3 Split and leakage requirements

Verify:

- deterministic splits;
- zero outfit overlap across development, validation, and test;
- zero prohibited positive/query leakage;
- no query item in its own negative pool;
- no item from the query outfit used as a negative;
- exact-image duplicate handling according to the leakage-resolved implementation;
- deterministic sampling and tie-breaking.

---

# 6. Final knowledge base

## 6.1 Canonical KB

The **only active final KB** is already complete and ready to use:

```text
data/kb/fashion_rules.csv
```

It contains exactly:

```text
200 rules
40 tops
40 bottoms
40 shoes
40 outerwear
40 bags
```

This file is authoritative for the final clean run.

Do not rebuild, reduce, expand, rewrite, rename, or replace the KB during the final experiment unless a genuine file-corruption or schema-integrity bug is discovered.

Do not use `v1`, `v2`, `v3`, `expanded`, `fallback`, or similar version labels in the final active filename or final thesis/paper terminology.

Original rule IDs must be preserved for provenance.

## 6.2 Final KB validation only

Stage 1 must **validate, not redesign**, the completed KB.

Required checks:

- exactly 200 rows;
- exactly 40 rules per target category;
- unique non-empty rule IDs;
- no exact duplicate rules;
- near-duplicate audit;
- required source/provenance fields present;
- valid target-category values;
- input→target coverage summary;
- source/provenance distribution;
- final file SHA-256.

The final KB audit is a release-integrity check only. It must not alter rule content based on recommendation or explanation results.

## 6.3 Rule weighting

All final rules participate in retrieval with **equal scoring weight**.

Source reliability/provenance may remain as metadata for auditability, but it must not cause one rule to receive a larger retrieval weight than another in the final experiment.

The candidate-specific rule representation uses:

- query broad category;
- query raw dataset text;
- user request;
- candidate broad category;
- candidate raw dataset text;
- target category.

Use the current local semantic rule-retrieval model:

```text
qwen3-embedding:0.6b
```

with its pinned digest/configuration.

## 6.4 Applicability before scoring

A rule may enter a candidate trace only if its antecedent is compatible with the actual case.

The shared retrieval/scoring function must:

1. apply target-category restrictions;
2. apply antecedent/query/candidate applicability gates;
3. score only retained applicable rules;
4. choose up to the configured top-k applicable rules;
5. compute the candidate evidence score from those exact rules;
6. store those exact rules in the candidate trace.

Do not pad a trace with irrelevant rules merely to reach the configured maximum.

If fewer than the configured maximum number of applicable rules exist, use the genuine smaller trace.

If no rule is applicable:

```text
candidate evidence score = 0
trace = []
```

## 6.5 Candidate evidence score

For the final confirmatory run, the default maximum rule count is:

```text
rule_top_k = 5
```

For selected-rule semantic scores:

```text
evidence_score =
    0.7 × max(selected_rule_scores)
    +
    0.3 × mean(selected_rule_scores)
```

If only one rule is present, both max and mean refer to that rule.

The exact implementation must reproduce the score from the stored trace.

All scoring constants and rule-count settings must be read from configuration rather than hard-coded.

---

# 7. Multimodal recommendation system

## 7.1 Image and text pathway

Use the frozen CLIP implementation already used by the project for image and CLIP-text encoding.

The **default and confirmatory operating point** is fixed as:

```text
image_weight = 0.40
text_weight  = 0.60

fused_query = normalize(
    0.40 × image_embedding
    +
    0.60 × text_embedding
)
```

The final 1,000-case confirmatory recommendation run uses this `0.40 / 0.60` setting.

For methodological enrichment and sensitivity analysis, also run one **validation-only image/text fusion grid**:

```text
image_weight = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
text_weight  = 1 - image_weight
```

The grid is **diagnostic only**. It does not select or replace the default operating point.

Report validation HR@10, NDCG@10, and MRR for every grid point and clearly highlight the fixed `0.40 / 0.60` default.

Even if another validation grid point performs better, the final confirmatory test run still uses:

```text
0.40 image / 0.60 text
```

All default weights, grid values, metrics to record, and seeds must be stored in `configs/experiment.yaml` and must be changeable there for future experiments. They must not be duplicated as hidden source-code constants.

Images are used for recommendation retrieval/ranking only. No image-derived text enters explanation generation or verification.

## 7.2 Controlled candidate pool

For each recommendation case construct a deterministic controlled pool containing:

- all valid same-outfit positives in the target category;
- up to 99 deterministic negatives from other outfits;
- no query item;
- no negative from the query outfit;
- deterministic ordering and tie-breaking.

Describe this as **controlled sampled-pool recommendation ranking**, not full-catalogue retrieval, personalization, or direct user utility.

## 7.3 Recommendation baselines

Evaluate:

1. MiniLM text baseline;
2. CLIP image baseline;
3. CLIP text baseline;
4. fused CLIP baseline;
5. expert-rule-aware reranking.

Use the exact currently pinned model implementations and record immutable model identifiers/digests.

## 7.4 Evidence-aware reranking

Normalize CLIP compatibility and expert-rule evidence scores within each candidate pool.

The **default and confirmatory operating point** is fixed as:

```text
clip_weight     = 0.75
evidence_weight = 0.25
rule_top_k      = 5

final_score =
    0.75 × normalized_CLIP_score
    +
    0.25 × normalized_evidence_score
```

The final 1,000-case confirmatory reranking run uses exactly these defaults.

For methodological enrichment and sensitivity analysis, also run one **validation-only reranking grid**:

```text
evidence_weight = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
clip_weight     = 1 - evidence_weight
rule_top_k      = [1, 3, 5]
```

For every validation configuration record:

- HR@10;
- NDCG@10;
- MRR;
- mean top-1 evidence score;
- mean evidence-score gain;
- changed-top-1 rate;
- rule-trace participation.

Construct and save the validation Pareto frontier as a **sensitivity/ablation artifact only**.

Do not use the grid or Pareto frontier to choose the final test configuration.

Clearly highlight the fixed confirmatory setting:

```text
0.75 CLIP / 0.25 evidence / top-k 5
```

Even if another validation configuration performs better on one or more objectives, the final confirmatory test run still uses the fixed defaults above.

All default weights, grid values, rule counts, metrics, thresholds, and plotting/analysis settings must be stored in `configs/experiment.yaml` and remain changeable there for future experiments. They must not be duplicated as hidden source-code constants.

Store for every candidate:

- candidate ID;
- original CLIP score;
- normalized CLIP score;
- raw evidence score;
- normalized evidence score;
- final score;
- exact stored rule trace;
- pre-rerank rank;
- post-rerank rank.

For every contributing rule preserve:

- rule ID;
- exact rule text;
- semantic score;
- applicability result;
- retrieval rank;
- contribution/order.

## 7.5 Critical trace invariant

Mandatory invariant:

> The rule packet passed to the Rule-RAG generator must be the exact stored rule packet that contributed to the locked candidate's evidence score during the final reranking run.

No second retrieval. No post-hoc reconstruction. No replacement after recommendation locking.

A hash link must prove:

```text
reranking trace
=
stored locked-candidate trace
=
Rule-RAG B evidence
```

---

# 8. Recommendation evaluation

Use:

```text
1,000 test cases
200 tops
200 bottoms
200 shoes
200 outerwear
200 bags
```

Report:

- HR@1;
- HR@5;
- HR@10;
- NDCG@1;
- NDCG@5;
- NDCG@10;
- MRR;
- micro average;
- category results;
- macro average;
- 95% confidence intervals;
- paired contrasts, especially reranking vs fused CLIP.

Do not claim that expert reranking improves recommendation accuracy unless final metrics support it.

---

# 9. Evidence-participation diagnostics

Report:

- top-1 recommendation change rate;
- top-5 overlap/change;
- evidence-score gain;
- pre/post rank shift;
- mean rank shift;
- trace-size distribution;
- rule frequency;
- percentage of the 200-rule KB used at least once;
- rule use by category;
- packet prevalence;
- slot-level share;
- packet duplication;
- Shannon entropy where useful;
- within-category overlap;
- candidate-specific variation;
- generic-rule flags.

These are participation/diversity diagnostics, not proof that every rule is objectively correct.

---

# 10. Final explanation cohort

After final reranking, select:

```text
500 cases total
100 tops
100 bottoms
100 shoes
100 outerwear
100 bags
```

A case is eligible only when its **final locked top-1 recommendation has at least one genuine contributing rule in its stored reranking trace**.

Within each category:

- sample deterministically with a fixed seed/hash policy;
- do not prefer multi-rule traces;
- do not prefer higher evidence scores;
- do not prefer specific rules;
- do not inspect explanation outcomes.

Before generation verify:

- exactly 500 unique case IDs;
- exactly 100 per category;
- at least one stored reranking rule per case;
- same locked recommendation for both conditions;
- same common context A;
- exact stored B trace;
- trace-size distribution reported;
- no second retrieval.

---

# 11. Explanation experiment

## 11.1 Common context A

Both conditions receive:

- user request;
- minimal query-item identity from approved dataset text/category;
- minimal locked-recommendation identity from approved dataset text/category.

A is task context, not expert evidence.

## 11.2 Expert evidence B

Rule-RAG additionally receives:

> the exact stored expert-rule trace that contributed to the final locked recommendation's reranking score.

## 11.3 Conditions

### No-RAG

```text
A only
```

### Rule-RAG

```text
A + exact stored B
```

The only intended experimental difference is B.

## 11.4 Final generation contract

Both conditions use the same contract:

- concise professional fashion-stylist explanation;
- normally 2–3 sentences;
- minimum 45 words;
- maximum 75 words;
- target approximately 65 words;
- explain why the recommendation works;
- do not pad merely to reach length;
- do not invent concrete item attributes or case-specific facts that were not supplied;
- preserve the exact locked recommendation.

No-RAG may use general fashion knowledge.

Rule-RAG uses the supplied expert rules as additional evidence and must:

- use only applicable supplied rules;
- cite an exact rule ID only when it supports the associated statement;
- use canonical separate citations such as `[K025] [K099]`;
- never invent rule IDs;
- never cite a merely related rule as support.

Condition names must not appear in prompts.

## 11.5 Frozen base prompts

No-RAG base instruction:

```text
You are a professional fashion stylist. Write one concise, helpful explanation for the supplied recommendation. You may use general fashion knowledge, but do not invent concrete item attributes or case-specific facts that were not supplied.
```

Rule-RAG uses the same base instruction plus:

```text
You are also given expert styling rules. Use only applicable supplied rules as additional evidence and cite an exact [K###] rule ID only when that rule directly supports the associated statement. Do not invent rule IDs or concrete item facts.
```

The shared user template enforces the identical 45–75-word, 2–3-sentence contract.

## 11.6 Generator roster

Use all three approved local generators:

1. `gemma4:12b`
2. `llama3.1:8b-instruct-q8_0`
3. `ministral-3:14b-instruct-2512-q4_K_M`

Record exact tag, immutable digest, parameter size, quantization, context length, temperature, top-p, top-k, seed, token ceiling, timeout, and inference-server version.

Use current deterministic decoding, including `temperature = 0` and seed 42 where supported.

## 11.7 Full fresh matrix

Generate all **3,000 explanations fresh**:

```text
500 × 2 × 3 = 3,000
```

No previous explanation may be reused.

Sequential batches:

1. Gemma — all 1,000 cells;
2. unload;
3. Llama — all 1,000 cells;
4. unload;
5. Ministral — all 1,000 cells;
6. unload.

Use the existing bounded retry policy. Do not use unlimited/manual repair loops. After the retry limit, preserve a transparent terminal failure.

Freeze generation before claim extraction.

---

# 12. Atomic claim extraction

Use:

```text
Qwen 3.5 9B
qwen3.5:9b
```

Qwen processes every accepted final explanation from the clean run.

The extractor determines **what claims exist**, not whether they are supported.

Each claim should:

- represent one independently verifiable proposition where practical;
- preserve entities, polarity, negation, and meaning;
- include explicit item attributes and substantive styling relations;
- include occasion/formality/suitability assertions when stated;
- avoid inferred claims not explicitly present;
- avoid duplicates;
- use stable case/model/condition/claim IDs.

Allowed types:

- body_fit;
- colour;
- comfort;
- formality;
- item_type;
- material;
- occasion;
- season;
- styling_relation;
- trend;
- visual_match;
- other.

Requirements:

- no arbitrary claim cap;
- complete explanation preserved;
- no evidence shown to the extractor;
- no support verdict during extraction;
- exact claim order preserved;
- malformed output rejected;
- bounded retries;
- terminal failures retained.

NER is not a substitute because it identifies entities, not proposition-level assertions.

---

# 13. Claim verification

Use:

```text
Phi-4 14B
phi4:14b
```

Phi verifies every successfully extracted final claim against the final 200-rule `fashion_rules.csv`.

## 13.1 Reranking-trace support

Verdict:

```text
supported
not_supported
```

`supported` requires at least one rule in the **exact stored reranking trace** whose antecedent applies and whose consequent directly entails the complete claim.

Do not use analogical category substitution, unsupported extrapolation, keyword overlap alone, or negative inference from omitted alternatives.

## 13.2 Full-KB support

Verdict:

```text
supported
not_supported
```

Check whether any applicable final-KB rule directly supports the claim using the current frozen full-KB candidate-retrieval + applicability + entailment procedure.

Retrieval similarity alone is never support.

## 13.3 Common-reference factual support

Only literal item/context facts are eligible, including explicitly supplied item identity/type, colour, material, pattern, fit, silhouette, or construction/detail.

Styling judgments such as “balanced”, “elegant”, “sophisticated”, “works well”, or “suitable for dinner” are N/A unless explicitly supplied as facts.

Output:

```text
eligible factual claim -> supported / not_supported
ineligible styling claim -> N/A
```

Unit-test the deterministic boundary with literal examples such as one-shoulder, black, faux leather, crossbody, and cotton.

## 13.4 Citation validation

Citation syntax and rule-ID existence are checked deterministically.

For valid cited rules Phi evaluates:

```text
entails
does_not_entail
```

Citation validity and citation entailment remain separate.

## 13.5 Verification integrity

Require:

- every claim ID preserved exactly;
- no missing/duplicate IDs;
- no unknown rule IDs;
- exact evidence packet hashes;
- bounded retries;
- invalid structured output rejected;
- terminal failures retained;
- raw verifier attempts preserved in release data.

---

# 14. Final explanation metrics

Metric definitions are frozen before final results are calculated.

## 14.1 Primary — Reranking-Trace Claim Support Rate

```text
supported claims against exact stored reranking trace
------------------------------------------------------
all extracted substantive claims
```

Compare No-RAG and Rule-RAG against the same stored case trace.

For No-RAG this is post-hoc alignment with the reranking trace. For Rule-RAG it measures grounding to evidence actually supplied during generation.

## 14.2 Secondary — Full-KB Claim Support Rate

```text
claims supported by at least one applicable final-KB rule
---------------------------------------------------------
all extracted substantive claims
```

The same full KB is used for both conditions.

## 14.3 Secondary — Unsupported Item-Fact Rate (UIFR)

```text
not_supported eligible concrete factual claims
------------------------------------------------
all common-reference-eligible factual claims
```

Lower is better. Report the eligible sample size prominently.

## 14.4 Secondary — Reranking-Trace Supported Claims per 100 Words

```text
number of exact-trace-supported claims
--------------------------------------- × 100
explanation word count
```

## 14.5 Robustness

Report main explanation metrics by:

- generator;
- target category;
- claim type;
- trace size.

The overall paired result remains primary.

## 14.6 Rule-RAG trace utilization

Rule-RAG-only diagnostic:

```text
trace rules meaningfully represented by supported claims
---------------------------------------------------------
rules in stored reranking trace
```

Do not use this as a headline No-RAG comparison.

## 14.7 Citation diagnostics

Rule-RAG-only:

- canonical syntax;
- existing rule ID;
- citation entailment.

Do not use citation presence as a cross-condition headline metric.

---

# 15. Statistical protocol

## 15.1 Recommendation

Use the underlying query outfit/case as the resampling unit.

- 5,000 bootstrap replicates;
- fixed seed;
- 95% percentile CIs;
- paired contrasts;
- Holm correction for the predefined primary recommendation comparison family.

## 15.2 Explanation

The underlying case is the clustering unit. Case-generator rows are not independent bootstrap units.

Use:

- paired complete-case No-RAG vs Rule-RAG analysis;
- case-clustered paired bootstrap;
- 5,000 replicates;
- fixed seed;
- 95% CIs;
- absolute percentage-point differences for rates;
- suitable paired significance tests;
- Holm correction for the predefined primary family;
- overall, generator-level, and category-level reporting;
- sample size for every metric.

## 15.3 Failure reporting

Report terminal failures by condition, generator, and stage.

If failure rates differ by condition, report the difference and include a simple conservative sensitivity/worst-case bound for headline paired metrics.

Do not replace failed records with hand-written outputs.

---

# 16. Real qualitative examples

Use only real frozen records from the final clean run. Do not invent or rewrite examples.

For every main metric/diagnostic include both a strong and weak example where available.

Each example includes:

- case ID;
- generator;
- condition;
- target category;
- trace size;
- explanation;
- extracted claim(s);
- relevant stored reranking rule(s) or common-reference facts;
- verifier outcome;
- metric value/outcome;
- one-sentence interpretation.

Required example families:

1. Reranking-Trace Support — strong + weak, preferably paired.
2. Full-KB Support — supported + not-supported.
3. UIFR — grounded factual claim + unsupported factual claim.
4. Supported Claims per 100 Words — high + low density.
5. Robustness — representative Gemma/Llama/Ministral and category examples.
6. Trace Utilization — high + partial/low Rule-RAG use.
7. Citation Diagnostics — entails + does-not-entail if present.

Selection must be deterministic. Prefer predeclared median/quantile examples rather than only extreme successes.

---

# 17. Publication-quality tables and figures

All final visuals must come from final canonical machine-readable outputs.

Required recommendation artifacts:

- dataset/category count table;
- final 200-rule KB audit table;
- full recommendation metric table;
- category-level recommendation table;
- paired-contrast/CI plot;
- reranking rank/evidence diagnostics;
- changed-top-1 summary;
- deterministic recommendation examples using actual images.

Required explanation artifacts:

1. Claim support rates: Reranking-Trace and Full-KB, No-RAG vs Rule-RAG, percentage scale, 95% CI, paired n.
2. UIFR as a separate figure, percentage scale, “lower is better”, eligible paired n shown.
3. Supported claims per 100 words as a separate figure with its own scale.
4. Robustness tables/figures by generator, category, claim type, trace size.
5. Rule-RAG trace-utilization and citation diagnostics.
6. Real qualitative example tables/panels for every metric.

Do not place metrics with incompatible units on the same y-axis.

Visual requirements:

- readable thesis-size fonts;
- colour-blind-safe palette;
- consistent condition/category styling;
- visible uncertainty intervals;
- SVG/PDF for vector plots;
- 300-DPI PNG where raster is required;
- proper tables rather than screenshots;
- captions, source-data references, configuration hashes, and provenance.

---

# 18. Repository and canonical-output policy

## 18.1 Final active repository

Use the current clean repository as the final implementation. The old development/archive repository remains untouched and is not an active result source.

## 18.2 Only active KB

Only:

```text
data/kb/fashion_rules.csv
```

may be treated as active.

No active file/report may claim the final KB has 120, 126, 209, or another count. Final count is exactly **200**.

## 18.3 Clean second-run reset

Before Stage 1:

- remove all previous **live** recommendation results;
- remove all previous **live** explanation results;
- remove all previous **live** extraction results;
- remove all previous **live** verification results;
- remove old Stage-12 tables/figures/examples/manifests;
- remove the V3 full-pool counterfactual from active runtime destinations;
- remove stale final-result text derived from previous runs;
- remove duplicate current/live result sets.

Do not delete source code required by the final system, relevant tests, pinned model caches, raw dataset cache, Git history, or the separate historical development repository.

Previous results may remain recoverable through Git history/external archive, but after reset **the live repository exposes only the second clean run as canonical**.

## 18.4 Canonical runtime

Use one active destination per stage:

```text
.runtime/current/data
.runtime/current/embeddings
.runtime/current/recommendations
.runtime/current/explanations
.runtime/current/extraction
.runtime/current/verification
.runtime/current/final_analysis
```

Partial work uses temporary paths and becomes canonical only after successful stage completion.

## 18.5 Reproducibility release package

Create:

```text
artifacts/release/
```

with compact canonical files required to recompute final tables without rerunning LLMs:

- final 1,000 recommendation case inputs/rankings;
- final 500 explanation case inputs;
- exact reranking traces;
- accepted explanations;
- extraction outputs;
- verification outputs;
- final configuration/hashes;
- final analysis source tables.

Do not include model weights, raw dataset cache, or large embedding caches.

---

# 19. Configuration and provenance

All experiment-facing values live in:

```text
configs/experiment.yaml
configs/models.yaml
configs/prompts.yaml
```

**Every variable that can change an experimental result must be configuration-driven.** The code must read these values from configuration even when this proposal fixes a confirmatory default. Changing the config after Stage 1 freeze defines a different future experiment and must produce a new configuration hash. This includes fixed operating points, candidate grids, thresholds, tolerances, selection objectives, tie-breaking rules, seeds, model settings, prompts, retry policies, validators, and output controls. No experiment-critical magic number may exist only inside source code.

Record at minimum:

- dataset revision and category mapping;
- split/case-selection seeds;
- recommendation and explanation case counts;
- candidate-pool size and negative-sampling rules;
- fixed default image/text fusion weights and the validation sensitivity grid;
- fixed default CLIP/evidence reranking weights and the validation sensitivity grid;
- fixed default `rule_top_k=5` and the validation rule-top-k sensitivity values;
- validation sensitivity metrics and Pareto-frontier analysis settings;
- KB path/hash;
- prompt templates/hashes;
- model IDs/digests;
- generation parameters;
- retry/terminal-failure policy;
- verification labels and evidence boundaries;
- bootstrap settings and multiplicity correction;
- canonical output paths.

Every stage manifest contains:

- timestamp;
- Git commit;
- resolved configuration hash;
- input/output hashes;
- model IDs/digests;
- row counts;
- failure counts;
- seeds;
- command;
- environment summary.

---

# 20. Testing and release quality

Required current-pipeline tests:

## Data

- deterministic split;
- expected processed counts;
- zero split leakage;
- five-category mapping;
- bag allowlist;
- candidate-pool validity.

## KB

- exactly 200 rows;
- exactly 40/category;
- unique IDs;
- no exact duplicates;
- near-duplicate audit;
- provenance present;
- final hash locked.

## Rule retrieval/reranking

- applicability before scoring;
- up-to-five behavior;
- no padding;
- deterministic semantic ranking;
- evidence-score reproduction;
- exact stored trace reproduction;
- no second retrieval;
- trace hash identity between reranking and Rule-RAG B.

## Explanation inputs

- same A across conditions;
- same locked recommendation;
- B only in Rule-RAG;
- no condition-label leakage;
- no image-derived text;
- 45–75-word validator;
- locked-item preservation.

## Extraction

- schema validity;
- stable claim IDs;
- no duplicate IDs;
- no support judgement.

## Verification

- binary trace support;
- binary full-KB support;
- deterministic common-reference eligibility;
- citation syntax validation;
- invalid rule-ID rejection;
- evidence-packet hash preservation.

## Statistics

- paired complete-case logic;
- case-clustered bootstrap;
- denominator/N/A handling;
- Holm correction.

## Release

- final JSONLs available from release package;
- final tables recomputable without LLM calls;
- no current manifest references obsolete KB/results;
- no obsolete 120/126/209 final-KB claim remains active.

Run:

```text
ruff check .
complete unit test suite
complete integration/release suite
```

No central current-pipeline test should remain skipped merely because an artifact was historically archived.

If Docker is not part of the final reproducibility route, remove the obsolete/nonfunctional Dockerfile rather than leaving a broken pathway.

---

# 21. Non-goals

Do not expand into:

- full-catalogue production serving;
- user accounts or marketplace infrastructure;
- reinforcement learning;
- end-to-end model training;
- image captioning for explanation evidence;
- fashion-attribute extraction/object detection;
- automatic fashion-rule generation;
- causal interpretation of hidden LLM reasoning;
- a new human-subject evaluation;
- a new LLM judge experiment;
- a new embedding benchmark;
- further KB expansion beyond the frozen 200 rules.

The priority is one rigorous, reproducible final experiment.

---

# 22. Final five-stage run

# 22. Final five-stage run

## Stage 1 — Preflight, clean reset, and final freeze

**No generator, extractor, or verifier calls are allowed in this stage.**

1. Reset previous live experimental outputs so the second clean run is the only active result set.

2. Validate the already completed `data/kb/fashion_rules.csv`:
   - exactly 200 rules;
   - exactly 40 rules per category;
   - unique rule IDs;
   - valid schema and provenance fields;
   - no exact duplicates;
   - freeze and record the existing file hash;
   - do not modify rule content.

3. Regenerate or validate the five-category processed dataset and leakage-resolved splits.

4. Validate the expected current dataset and split counts.

5. Regenerate or validate all required recommendation embeddings against the pinned model, dataset, and configuration hashes.

6. Rebuild the semantic embeddings for the final 200-rule KB.

7. Run the two final **validation-only sensitivity grids**:
   - image/text fusion grid from Section 7.1;
   - evidence-weight/rule-top-k grid from Section 7.4.

   Save every tested configuration, validation metric, and the descriptive Pareto frontier.

   These grids are sensitivity/ablation analyses only and **do not select or replace the final confirmatory operating point**.

8. Freeze the confirmatory defaults:
   - CLIP model;
   - Qwen rule-embedding model;
   - image/text fusion = `0.40 / 0.60`;
   - CLIP/evidence reranking = `0.75 / 0.25`;
   - `rule_top_k = 5`;
   - random seeds;
   - controlled candidate pool = positive(s) + up to 99 deterministic negatives.

   All values must be read from configuration files rather than hard-coded. They may be changed only for a future experiment, not after the final run begins.

9. Freeze the explanation prompts and the identical 45–75-word generation contract for both conditions.

10. Freeze the Qwen claim-extraction prompt/schema and Phi claim-verification prompt/schema.

11. Perform a **small automated extractor/verifier integrity check** before the final LLM run:
    - retain the previous calibration only as developmental validation evidence;
    - do not create a new human-labelled calibration set;
    - do not reopen prompt tuning or threshold tuning;
    - confirm the pinned Qwen and Phi model tags/digests;
    - confirm the frozen extraction and verification prompt hashes;
    - confirm structured-output schemas still validate;
    - confirm claim IDs survive extraction → verification unchanged;
    - confirm the final 200-rule IDs are accepted by the verifier pipeline;
    - confirm the exact reranking-trace evidence packet can be passed to Phi without schema or rule-ID errors;
    - confirm canonical `[K###]` citation parsing and invalid-ID rejection still work;
    - confirm common-reference factual eligibility logic still handles simple literal cases correctly;
    - use only synthetic/unit-test records or previously retained development examples for this check;
    - do not use final test outputs and do not perform any new model calibration.

12. Remove obsolete active requirements and references:
    - old 120/126/209 final-KB counts;
    - old post-hoc V3-trace wording;
    - old four-way verifier requirement;
    - old DeepSeek requirement;
    - old trace-size-biased sampling;
    - old case-generator bootstrap unit.

13. Add or update final exact-trace invariant tests proving:
    - the stored reranking trace is exactly the trace used to calculate the candidate evidence score;
    - the same stored trace is passed to Rule-RAG;
    - no second rule retrieval occurs after recommendation locking.

14. Run the complete preflight test suite and lint checks.

15. Produce a preflight manifest proving that the following are frozen and internally consistent:
    - final 200-rule KB and hash;
    - processed dataset and split hashes;
    - recommendation embeddings;
    - validation sensitivity-grid outputs;
    - fixed confirmatory defaults;
    - model identities/digests;
    - prompt hashes;
    - configuration hashes;
    - extractor/verifier integrity-check result;
    - exact-trace invariant tests;
    - clean canonical runtime state.

### Stage 1 pass gate

Do not start Stage 2 unless:

- the final KB has exactly 200 rules and 40 per category;
- dataset and leakage checks pass;
- at least 100 evidence-eligible explanation cases per category are feasible;
- both sensitivity grids are complete;
- the fixed defaults remain `0.40/0.60`, `0.75/0.25`, and `rule_top_k=5`;
- the automated Qwen/Phi integrity check passes;
- exact trace storage and trace reuse are unit-tested;
- no second retrieval path exists;
- current tests protect the final pipeline;
- all experiment-facing settings are frozen and configuration-driven.

Stop and report any genuine blocker before spending final generator, extractor, or verifier calls.

---

## Stage 2 — Final recommendations and 3,000 fresh explanations

### Part A — Recommendation run

1. Run final recommendation baselines on the frozen 1,000 cases:
   - MiniLM text;
   - CLIP image;
   - CLIP text;
   - fused CLIP.
2. Run final 200-rule expert-aware reranking on the same 1,000 cases.
3. Store exact candidate traces during reranking.
4. Calculate recommendation metrics and evidence-participation diagnostics.
5. Freeze final 1,000-case recommendation results.
6. Select final 500 explanation cases:
   - exactly 100/category;
   - final top-1 has at least one contributing rule;
   - deterministic;
   - no trace-size preference.
7. Freeze case IDs, A context, locked recommendation, exact B trace, and input hashes.
8. Verify hash identity between reranking trace and Rule-RAG B.

### Part B — Fresh explanation run

Generate the full matrix:

```text
500 × 2 × 3 = 3,000
```

No previous explanation may be reused.

Sequentially run:

1. Gemma complete batch;
2. Llama complete batch;
3. Ministral complete batch.

Use bounded retries only.

Report attempted/accepted cells, terminal failures by model/condition, word-count distributions, prompt/model hashes, trace hashes, and output hashes.

Freeze Stage 2 before Stage 3.

---

## Stage 3 — Fresh atomic-claim extraction

Run Qwen 3.5 9B over every accepted final Stage-2 explanation.

No prior extraction may be reused.

Requirements:

- evidence-independent atomic extraction;
- stable IDs;
- no claim cap;
- no support judgement;
- structured validation;
- bounded retries;
- duplicate checks;
- terminal failures retained.

Report explanations processed, successes/failures, total claims, mean/median claims per explanation, generator/condition breakdown, retries, duplicates, and hashes.

Freeze Stage 3.

---

## Stage 4 — Fresh claim verification

Run Phi-4 14B over every final Stage-3 claim.

No prior verification may be reused.

For each claim record:

1. reranking-trace support — `supported / not_supported`;
2. full-200-rule-KB support — `supported / not_supported`;
3. common-reference factual support — `supported / not_supported / N/A`;
4. citation entailment — `entails / does_not_entail / N/A`.

Requirements:

- exact final trace only;
- final `fashion_rules.csv` only;
- no obsolete post-hoc packet;
- exact claim-ID preservation;
- exact evidence-packet hashes;
- deterministic citation syntax validation;
- bounded retries;
- terminal failures retained.

Report records/claims verified, failures, support totals, common-reference eligible n, citation counts, retries, and hashes.

Freeze Stage 4.

**After Stage 4, no more generator/extractor/verifier model calls are allowed.**

---

## Stage 5 — Final results, metrics, visual outputs, and release closure

Use only frozen Stage 1–4 outputs.

### Validation sensitivity and ablation results

Preserve and report the Stage-1 validation-only sensitivity analyses:

- complete image/text fusion grid;
- fixed confirmatory fusion point `0.40/0.60` clearly highlighted;
- complete evidence-weight/rule-top-k grid;
- reranking Pareto frontier as a descriptive trade-off analysis;
- fixed confirmatory reranking point `0.75/0.25, top-k=5` clearly highlighted.

These are methodological/sensitivity results only. They do not select or replace the fixed confirmatory defaults and must never be recomputed using the final test set.

### Recommendation results

Calculate/freeze:

- HR@1/5/10;
- NDCG@1/5/10;
- MRR;
- category breakdowns;
- paired contrasts;
- case-clustered 95% CIs;
- evidence participation/rank-change diagnostics.

### Explanation results

Primary:

- Reranking-Trace Claim Support Rate.

Secondary:

- Full-KB Claim Support Rate;
- UIFR;
- Reranking-Trace Supported Claims per 100 Words.

Diagnostics:

- generator;
- category;
- claim type;
- trace size;
- Rule-RAG trace utilization;
- citation validity/entailment.

Statistics:

- paired complete-case comparisons;
- case-clustered bootstrap;
- 5,000 replicates;
- 95% CIs;
- paired tests;
- Holm correction;
- terminal-failure reporting;
- conservative failure sensitivity where appropriate.

### Qualitative evidence

Generate strong/weak real examples for every metric and diagnostic according to Section 16.

### Release closure

- generate final publication-quality tables/figures;
- copy required canonical JSONLs/source tables to `artifacts/release/`;
- regenerate final manifests and figure/table registry;
- remove obsolete live result references;
- verify no previous-run numbers remain in active reports;
- run full unit/integration tests;
- run `ruff check .`;
- review complete Git diff;
- trace every final number to a canonical source table;
- create one final release manifest with all hashes.

Then mark:

```text
EXPERIMENTAL PROJECT: CLOSED
```

No further model experiment is required.

---

# 23. After the five-stage run — thesis update

Only after Stage 5 is frozen, update the thesis from the final clean run.

## Chapter 1

Align problem, motivation, objectives, research questions, contributions, and terminology. Remove claims based on old KB counts or experiments.

## Chapter 2

Align literature review with:

- multimodal fashion recommendation;
- explainable recommendation;
- expert knowledge / knowledge-based recommendation;
- RAG/evidence-grounded generation;
- factuality versus evidence faithfulness;
- claim-level evaluation.

## Chapter 3

Describe exactly:

- five categories;
- final 200-rule KB, 40/category;
- dataset/splits;
- controlled candidate pool;
- CLIP multimodal retrieval;
- expert-rule scoring;
- evidence-aware reranking;
- exact trace invariant;
- 1,000 recommendation cases;
- 500 explanation cases;
- 3 generators / 3,000 cells;
- Qwen extraction;
- Phi binary verification;
- final metrics;
- case-clustered statistics;
- reproducibility protocol.

## Chapter 4

Replace every previous recommendation/explanation number, table, figure, and qualitative example with the final clean-run outputs only.

## Chapter 5

Interpret only final results and include limitations such as:

- one fashion dataset/domain;
- controlled sampled candidate pool rather than full-catalog retrieval;
- explanation study restricted to evidence-eligible recommendations;
- automated claim extraction/verification;
- no image-derived textual explanation evidence;
- citation entailment separate from citation syntax;
- smaller eligible sample for common-reference factual metrics where applicable.

## Thesis audit

Before thesis freeze:

- every number traces to a final table/manifest;
- every figure traces to canonical source data;
- KB always reported as 200 rules;
- category always reported as `bags`, not `accessories`;
- no obsolete V1/V2/V3 result is presented as current;
- reranking trace and explanation evidence are described consistently;
- DOCX chapters are rebuilt from canonical sources.

---

# 24. After thesis update — full journal-paper draft

After the thesis is internally consistent, write a complete standalone journal manuscript from the final frozen experiment.

First submission target:

**Journal of Intelligent Information Systems (Springer).**

Suggested structure:

1. Title
2. Abstract
3. Keywords
4. Introduction
5. Related Work
6. Proposed Method
7. Experimental Setup
8. Recommendation Results
9. Explanation-Grounding Results
10. Qualitative Analysis
11. Discussion
12. Limitations
13. Conclusion
14. References

The paper should emphasize:

- the exact same expert-rule trace contributes to reranking and is supplied to Rule-RAG;
- controlled No-RAG vs Rule-RAG ablation;
- claim-level evidence verification;
- recommendation/evidence trade-off;
- robustness across three generators and five categories;
- auditable real examples.

Do not claim:

- hidden-chain-of-thought faithfulness;
- universal fashion correctness;
- full-catalog production recommendation performance;
- human preference superiority unless directly evaluated;
- that `not_supported` means false.

Use only final clean-run results and do not trigger a new model experiment. Prepare the complete manuscript draft but do not submit automatically.

---

# 25. Final acceptance criteria

## KB

- `data/kb/fashion_rules.csv` exists.
- Exactly 200 rules.
- Exactly 40/category.
- Final hash recorded.
- No other active KB used.

## Recommendation

- Validation-only image/text fusion sensitivity grid completed before test evaluation.
- Validation-only evidence-weight/rule-top-k sensitivity grid completed before test evaluation.
- Complete grid results and descriptive Pareto frontier preserved.
- Final test uses fixed `0.40/0.60` fusion and fixed `0.75/0.25, top-k=5` reranking regardless of grid winners.
- 1,000 final cases, 200/category.
- Main pool uses up to 99 negatives.
- All final baselines run.
- Expert reranking uses the final 200-rule KB.
- Exact contributing trace stored.
- Final metrics reproducible.

## Explanation

- 500 final evidence-eligible cases, 100/category.
- No trace-size-biased selection.
- Same case/recommendation/context across conditions.
- Rule-RAG B is exactly the reranking trace.
- 3,000 fresh generation cells attempted.
- No old explanation reused.

## Extraction

- Qwen processes accepted final explanations.
- Stable atomic claims.
- No support judgement during extraction.
- Failures transparent.

## Verification

- Phi verifies final claims.
- Reranking-trace support binary.
- Full-KB support binary.
- Common factual support eligibility-aware.
- Citation entailment separate.
- Failures transparent.

## Statistics

- Paired analysis.
- Case-clustered bootstrap.
- 5,000 replicates.
- 95% CIs.
- Sample sizes visible.
- Multiple-comparison handling documented.

## Reproducibility

- One canonical final result set.
- Essential JSONLs in `artifacts/release/`.
- Config/model/prompt/data/KB hashes recorded.
- Final tables recomputable without LLM calls.
- Full tests/lint executed.
- No current-pipeline tests hidden behind obsolete archive skips.

## Thesis and paper

- Thesis uses final clean results only.
- Old counts/results removed from active writing.
- Thesis DOCX rebuilt.
- Full journal manuscript drafted from the frozen final run.
- No further experiment required.

---

# 26. Instructions to Codex

Follow this document **stage by stage**.

Do not jump ahead.

At the end of each of the five stages, stop and report:

- files changed;
- commands run;
- tests run;
- exact row/case/output counts;
- failures;
- artifacts/manifests created;
- hashes;
- whether the stage passed;
- whether any genuine blocking bug remains.

Do not continue to the next stage until researcher approval.

The second clean run replaces all previous active experimental results.

Do not preserve compatibility with obsolete result counts, obsolete KB sizes, obsolete post-hoc trace logic, obsolete evaluator labels, or obsolete metrics.

The final implemented experiment defined here is authoritative.
