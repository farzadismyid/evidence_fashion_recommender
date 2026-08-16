# Project Proposal and Build Specification

## Evidence-Constrained Multimodal Fashion Recommendation with Faithful, Source-Grounded Explanations

### Purpose of this document

This document is the authoritative build specification for a **clean final implementation** of the project. It is intended for a coding agent such as Codex or Claude Code.

The current development repository contains useful experiments, frozen artifacts, audits, and historical implementations, but it has grown too large and difficult to understand. The final system must therefore be implemented in a **new clean repository** with a small, stable, easily navigable structure.

The old repository must remain unchanged as an archive and provenance source. Do not delete it. Do not silently copy everything from it. Only transfer components that are required by this specification and that pass validation.

### Authoritative scope decision

The recommendation system remains multimodal: query images are encoded by CLIP for retrieval
and ranking. The explanation experiment does **not** perform image captioning, visual-attribute
extraction, fashion-attribute classification, object detection, or any other conversion of
images into textual evidence.

Explanation evidence is restricted to:

- A: the user request/query text, minimal query-item name, and minimal locked-recommendation name;
- B: the exact stored expert-rule scoring trace used during reranking.

No image-derived description may be added to A or B. Image-grounded textual explanation is
deferred to future work and must be reported as a limitation rather than implemented in the
main study.

This decision is final for the clean implementation. Exploratory Florence-2 and
Marqo-FashionSigLIP trials are not part of the confirmatory pipeline. The calibrated Marqo
trial achieved high selective precision only by abstaining heavily: overall accepted coverage
was 37.8%, with 10.0% pattern coverage and 26.3% silhouette coverage. Broad accessory and
out-of-vocabulary handling would require a separate fashion-attribute-recognition study.
Do not reintroduce a visual extractor, its dependencies, or its pilot artifacts without
explicit researcher approval.

---

# 1. Project title

**Evidence-Constrained Multimodal Fashion Recommendation with Faithful, Source-Grounded Explanations**

---

# 2. Research problem

Most multimodal fashion recommenders retrieve visually or semantically compatible products but provide explanations that are generic, weakly supported, or disconnected from the actual evidence used by the system.

This project studies whether explanations become more evidence-grounded when they are conditioned on retrieved expert-authored fashion rules.

The project must separately evaluate:

- recommendation quality for different types based on just the user text query, just the user image query, and then fused embedding of both user text and image;
- evidence participation in reranking as an ablation study;
- explanation quality if we can find any reliable metric for it;
- atomic-claim support;
- unsupported-claim rate;
- the difference between surface quality and evidence grounding.

The system must not claim that an explanation is causally faithful to the model’s internal reasoning. The supported claim is narrower:

> The explanation is grounded in explicitly supplied and auditable evidence.

Use terms such as:

- evidence-grounded explanation;
- atomic-claim support;
- unsupported claim rate;
- evidence-constrained generation;
- source-grounded explanation.

---

# 3. Core research contribution

The central experiment is a controlled two-condition evidence ablation using A and B.

For the same query, user request, and locked recommendation, generate two explanations:

| Condition | Information supplied |
|---|---|
| No-RAG | A only |
| Rule-RAG | A + B |

Where:

## A — common task context

A is fixed and contains only:

- the user request or query text;
- the minimal query-item name, category label, or `text` value from the dataset;
- the minimal identity/name of the locked recommendation, using its category label or `text` value from the dataset.


Example:

```text
User request:
Recommend shoes that work with this outfit.

Query item:
Day dress.

Locked recommendation:
Ankle boots.
```


## B  — exact expert-rule scoring trace

B contains the exact rule IDs and rule texts that contributed to the locked candidate’s evidence score during reranking.

B must not be produced by a second independent retrieval call.

The same stored trace must drive:

1. candidate evidence scoring;
2. recommendation reranking;
3. explanation evidence.

Example:

```json
{
  "candidate_id": "28371",
  "evidence_score": 0.614,
  "rules": [
    {
      "rule_id": "R021",
      "rule_text": "For a midi dress in daytime smart-casual styling, ankle boots or polished flats can maintain an appropriate level of formality.",
      "similarity": 0.69,
      "reliability_weight": 1.0,
      "weighted_score": 0.69
    }
  ]
}
```

The trace must preserve:

- rule ID;
- rule text;
- semantic similarity;
- reliability weight;
- any category or query-group bonus;
- final weighted contribution;
- retrieval rank;
- any filtering decision.

---

# 4. End-to-end system

## 4.1 User input and study boundary

The deployed recommendation system accepts:

- a query-item image;
- a natural-language user request;
- a minimal query-item name or category label;
- a requested target recommendation category when necessary.

Example:

```text
Image: user-uploaded dress photo
Query item name: day dress
Request: Recommend shoes for a smart-casual evening outfit.
Target category: shoes
```

For this experiment and during controlled evaluation, the query image and minimal item identities are taken from the pinned Polyvore case.

The controlled explanation ablation is therefore dataset-bound:

- recommendation retrieval follows the real multimodal input path;
- A uses the request and minimal item names;
- B uses the stored expert-rule scoring trace.

For a real user-uploaded image, the minimal query-item name or category required by A must be user-supplied. The demo must expose this limitation rather than inventing item information or running an unapproved image-to-text model.

## 4.2 Image pathway and explanation-evidence boundary

Use one image pathway only:

### CLIP embedding for recommendation retrieval

Use CLIP or an equivalent frozen multimodal encoder:

```text
Query image → image embedding
Query text/request → text embedding
Image embedding + text embedding → fused query embedding (select the image/text fusion weights using a validation-only grid search or another documented, defensible validation procedure)
```

The raw embedding is used only for retrieval and ranking. It must not be passed to a text-only
explanation LLM as human-readable evidence.

The image remains important to the multimodal recommender and may be displayed in qualitative
figures. It is not a textual evidence source in the main explanation experiment.
Image-grounded textual explanations are future work and a documented limitation.

## 4.3 Candidate recommendation retrieval

Use fused multimodal retrieval.

Existing validated design:

- image encoder: CLIP;
- text encoder: CLIP text tower for multimodal retrieval;
- candidate corpus: fashion catalogue items;
- category-aware candidate filtering;
- normalized image and text embeddings;
- validation-selected image/text fusion.

Current validated operating point:

```text
fused_query = normalize(0.40 × image_embedding + 0.60 × text_embedding)
```

This weight must remain configurable and must be selected on validation data rather than test data.

The system should return a ranked candidate list, normally top 5 for user-facing output.

For controlled research evaluation, use a deterministic candidate pool containing:

- all same-outfit positives in the target category;
- up to 999 deterministic negatives from other outfits;
- no query item;
- no item from the query outfit among negatives;
- deterministic candidate sampling and ordering.

The maximum number of negatives must be stored in configs/experiment.yaml and must not be hard-coded in the implementation.

The implementation may additionally support validation-only sensitivity analysis with smaller candidate pools, such as 100 and 500 items, but the main controlled evaluation should use the configured 1,000-item pool.

Clearly describe this setup as sampled controlled-pool ranking, not full-catalogue retrieval.
## 4.4 Expert-rule retrieval

The fashion-rule knowledge base is a distinct external corpus. The approved file is `data/kb/fashion_rules.csv` (KB v3).

Current asset:

- approximately 126 unique rules;
- five target categories:
  - accessories;
  - bottoms;
  - outerwear;
  - shoes;
  - tops;
- source provenance fields;
- reliability labels;
- scenario, style, occasion, season, colour, formality, fit, and category fields.

The rule retriever must construct a candidate-specific representation from:

- query item category;
- query raw dataset text;
- user request;
- candidate category;
- candidate raw dataset text;
- target category.

Use semantic retrieval with category restrictions and reliability weighting.

Current scoring basis to preserve unless validation justifies a change:

```text
weighted_rule_score =
    semantic_similarity
    × reliability_weight
    + optional query-group bonus
```

Reliability weights:

```text
high   = 1.00
medium = 0.85
low    = 0.65
```

Candidate evidence score:

```text
0.7 × maximum weighted selected-rule score
+
0.3 × mean weighted selected-rule score
```

Default top-k:

```text
candidate_top_k = 5
```

However, the implementation must correct the previous inconsistency:

> Filtering, top-k selection, scoring, stored trace, and explanation evidence must all use one shared function and one shared result object.

For accessories and any other category-specific filter:

- apply the filter before final top-k selection;
- compute the score from the final retained rules;
- store those exact rules;
- pass those exact rules to the explanation generator.

No second retrieval call is allowed after reranking.

## 4.5 Evidence-aware reranking

Combine the fused CLIP compatibility score and candidate evidence score.

Existing validated form:

```text
final_score =
    0.75 × normalized_CLIP_score
    +
    0.25 × normalized_evidence_score
```

Both scores are min-max normalized within the controlled candidate pool.

Requirements:

- store the original CLIP score;
- store the normalized CLIP score;
- store the raw evidence score;
- store the normalized evidence score;
- store the final score;
- store the exact evidence trace;
- store the pre-rerank and post-rerank rank;
- preserve deterministic tie-breaking.

The 0.75/0.25 weight must be configurable and selected on validation data only.

Do not describe a retrieved rule as objectively applicable merely because it has a high semantic score. The score is evidence participation, not proof of correctness.

## 4.5.1 Validation-only Pareto-frontier selection

Evidence-aware reranking must use genuine multi-objective Pareto-frontier analysis rather
than selecting one manually preferred weight.

For every configured reranking setting, evaluate on the validation split:

### Recommendation objectives

- HR@10;
- NDCG@10;
- MRR.

### Evidence-participation objectives

- mean evidence score of the top-ranked candidate;
- mean evidence-score gain after reranking;
- proportion of cases whose top-ranked candidate changes;
- top-k rule-trace participation.

A configuration is Pareto dominated when another configuration performs at least as well
on every selected objective and strictly better on at least one.

The system must:

1. evaluate every configured reranking weight and rule-count setting;
2. calculate the non-dominated Pareto frontier;
3. save all frontier and dominated points;
4. plot recommendation quality against evidence participation;
5. select the operating point using a rule fixed before test evaluation.

The preferred selection rule is either:

- the Pareto knee point; or
- the configuration with the greatest evidence participation while keeping validation
  NDCG@10 within a configured relative-loss tolerance from the best recommendation-only
  configuration.

The selection rule, tolerance, metrics, and tie-breaking procedure must be stored in the
configuration.

The test split must never be used to select the operating point. Test results are calculated
once after the selected configuration is frozen.

A higher rule score demonstrates participation under the rule-scoring function; it does not
prove objective rule correctness or applicability.

Use `0.75 × normalized_CLIP_score + 0.25 × normalized_evidence_score` as the initial
reference operating point. The validation-only Pareto analysis must report the recommended
weight and its trade-offs. Any change from the reference setting requires researcher approval
after reviewing the validation report and before the final test configuration is frozen.

## 4.6 Locked recommendation for explanation evaluation

The user-facing recommender may return five items.

The explanation experiment should lock exactly one recommendation per evaluation case so that both evidence conditions explain the same item.

Preferred choice:

```text
locked recommendation = top-ranked item after final evidence-aware reranking
```

Store:

- case ID;
- query item ID;
- request;
- target category;
- locked candidate ID;
- locked candidate minimal identity;
- candidate rule trace B;
- pre-rerank and post-rerank rank;
- all score components.

Using one locked item is intentional. It prevents recommendation identity from confounding the explanation comparison.

---

# 5. Explanation generation

The full confirmatory explanation experiment must use **500 locked evaluation cases**. Each locked case is generated under both explanation conditions for every approved generator.

## 5.1 Two conditions

For every locked case:

### No-RAG

```text
A only
```

### Rule-RAG

```text
A + B
```

## 5.2 Prompt regimes

Both conditions receive the same user-facing task message containing A. However,
grounding instructions are applied only to the rule evidence-assisted condition.

### Free No-RAG baseline

No-RAG receives:

- A;
- the ordinary user request to explain the recommendation.

It receives no evidence-grounding system prompt, no citation requirement, no instruction to
restrict claims to supplied evidence, and no instructed word-count bound.

No-RAG is therefore a free-generation LLM baseline.

A sufficiently high configurable generation-token safety ceiling must still exist to prevent
runaway inference. Use the same ceiling across both conditions wherever model context
limits permit, and record any model-specific exception in the configuration and run manifest.
This ceiling must not be presented to the model as a desired explanation length.

### Evidence-assisted condition

Rule-RAG receives the grounding system prompt.

The grounding system prompt must:

- use only supplied evidence for factual and styling claims;
- avoid inventing attributes or facts absent from A and B;
- cite an exact rule ID only when that rule entails the claim;
- avoid citing merely related rules;

The condition name must never appear in the prompt.

Because No-RAG is intentionally unconstrained while RAG condition is evidence-constrained,
explanation length and claim count must be reported by condition. A secondary length-matched
sensitivity analysis should be provided if length differences materially affect comparisons.

No-RAG must not falsely be described as having no information: it still receives A. The primary comparison should emphasize whether Rule-RAG increases rule-supported claims and reduces unsupported claims.

## 5.3 Generator design

Use a configurable roster of local instruction-tuned LLMs.

The previous experiment used:

- Llama 3.2;
- Mistral;
- Gemma 3 12B.

The final repository must not rely on unpinned tags. Record:

- exact model tag;
- full immutable digest when available;
- parameter size;
- context-window setting;
- temperature;
- top-p;
- top-k;
- seed if supported;
- token limit;
- timeout;
- inference server version.

Recommended decoding:

```text
temperature = 0
```

All generation outputs must be immutable after the generation stage and hash-bound before evaluation.

---
## 5.4 Validation-only explanation-variable optimisation

Before the full explanation experiment, optimise the evidence-assisted explanation design
using validation data only. Preserve the controlled ablation: A is fixed and identical across
both conditions, while B-related settings apply only to the evidence-assisted condition.

No-RAG remains the free-generation baseline. It is not given evidence-grounding instructions,
a citation requirement, or an instructed word-count target.

### Evidence-assisted configuration search

Searchable RAG variables must be configuration-driven and may include:

- maximum requested explanation length;
- number of expert rules supplied;
- rule-evidence formatting template;
- ordering of supplied rules;
- rule-citation requirement:
  - required;
  - optional;
- inclusion or exclusion of rule scores;
- inclusion or exclusion of reliability labels;
- concise versus detailed grounding-system-prompt variants.

Suggested initial values include:

- requested RAG word limits: 45, 60, and 75;
- rule counts: 1, 3, and 5;
- rule presentation order where more than one rule is supplied;
- citations: required or optional.

The complete Cartesian product may be too expensive. The implementation may therefore use:

- staged grid search;
- fractional factorial design;
- successive halving;
- or another documented resource-efficient search method.

Selection must use validation cases only and consider multiple objectives:

- atomic-claim support rate;
- unsupported-claim rate;
- contradiction rate;
- citation-entailment accuracy;
- general explanation quality;
- clarity;
- specificity;
- explanation length;
- number of atomic claims;
- malformed-output rate;
- generation latency.

Save every tested configuration and result. Select a non-dominated or predefined
validation-optimal setting, freeze it, and use it unchanged for the test experiment. Do not
choose settings by inspecting final test results.

After selection:

1. confirm and freeze the fixed A representation;
2. freeze the RAG prompt/evidence configuration;
3. run the 50-case final pilot using those selected settings;
4. obtain researcher approval before beginning the full explanation experiment.

# 6. Atomic-claim extraction

The extractor operates on the complete generated explanation.

It must extract every independent fashion or styling proposition.

Examples of atomic claims:

- “The boots are black.”
- “The block heel makes the outfit more formal.”
- “The neutral shoes reduce colour competition.”
- “The bag complements the dress.”
- “The material is leather.”

Do not use named-entity recognition as a substitute. NER identifies entities, not proposition-level claims. but mention this in methodlogy why we didnt use NER.

Preferred claim schema:

```json
{
  "claims": [
    {
      "claim_id": "C1",
      "claim_text": "The boots are black.",
      "claim_type": "colour"
    }
  ]
}
```

Allowed claim types may include:

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
- preserve complete original explanation;
- do not assess truth during extraction;
- reject malformed or empty structured output;
- record retries and failures;
- never treat extraction failure as zero unsupported claims.

---

# 7. Claim verification

The previous mutually exclusive source-label design must be replaced.

A claim can be supported by multiple evidence sources. The final verifier must separate:

1. support status;
2. support sources.

## 7.1 Support status

Allowed values:

```text
supported
unsupported
contradicted
not_verifiable
```

Definitions:

### supported

At least one supplied source semantically entails the claim.

### unsupported

Relevant evidence is available, but it does not entail the claim.

### contradicted

At least one supplied source affirmatively conflicts with the claim.

### not_verifiable

The supplied evidence is insufficient, ambiguous, or unable to settle the claim.

Absence of support is not automatically contradiction.

## 7.2 Support sources

Allowed values:

```text
query_or_locked_item
rule_evidence
```

A claim may have multiple support sources.

Example:

```json
{
  "claim_id": "C2",
  "support_status": "supported",
  "support_sources": [
    "query_or_locked_item",
    "rule_evidence"
  ],
  "supporting_rule_ids": ["R021"],
  "citation_entails_claim": true,
  "brief_reason": "The locked-item identity in A identifies a block heel, and R021 connects this footwear type with the requested formality."
}
```

Requirements:

- verify against one common union packet for both variants;
- include query context, request, locked item identity, and B;
- separately record what evidence was shown during generation;
- do not remove evaluation evidence based on generation condition;
- require exact claim-ID coverage;
- reject duplicate, missing, or unknown IDs;
- validate rule IDs;
- require rule entailment for rule support;
- record N/A rather than converting failed verification to unsupported.

The verifier is an entailment-style LLM classifier, not:

- word overlap;
- cosine thresholding;
- keyword matching.

---

# 8. General explanation judging

Judge the full explanation separately from atomic-claim verification.

Recommended dimensions:

- input consistency;
- general quality using a defensible evaluation method;
- clarity;
- specificity;
- hallucination control;
- evidence-use correctness.

Use anchored integer scales, for example 1–5, with higher always better.

Primary judging should be cross-model:

- judge family different from generator family.

Keep word-count compliance separate from quality scoring.

---

# 9. Recommendation evaluation

The main confirmatory recommendation evaluation must use **1,000 deterministic evaluation cases**, sampled according to the configured split and case-selection rules.

Evaluate at least:

- MiniLM text baseline;
- CLIP image baseline;
- CLIP text baseline;
- fused CLIP baseline;
- evidence-aware reranking.

The two text baselines are intentionally distinct:

- **MiniLM text baseline:** a sentence-transformer baseline for general text-to-text semantic
  similarity;
- **CLIP text baseline:** the CLIP text tower, whose representations are aligned with visual
  concepts through image-text pretraining.

Keeping both reveals whether text-only recommendation performance is driven mainly by general
language semantics or by a visually aligned embedding space.

Metrics:

- HR@1;
- HR@5;
- HR@10;
- NDCG@1;
- NDCG@5;
- NDCG@10;
- MRR.

Report:

- micro average;
- category-level results;
- category macro average;
- historical pilot subset if retained;
- full confirmatory cohort;
- confidence intervals;
- paired contrasts.

Bootstrap unit:

```text
query outfit
```

All cases belonging to a query outfit must remain together within a bootstrap replicate.

Recommended:

- 5,000 bootstrap replicates;
- fixed seed;
- 95% percentile intervals;
- paired tests;
- clearly defined multiple-comparison correction family.

Do not describe evidence reranking as improving recommendation accuracy unless the metrics support it.

A defensible possible outcome is:

> Evidence-aware reranking materially changes recommendation order and raises the candidate evidence score while remaining competitive in top-k hit rate, although rank-sensitive metrics may decline slightly.

---

# 10. Evidence-participation and rule-retrieval diagnostics

The final system must explicitly audit whether rules actually participate and whether retrieval is overly generic.

Required outputs:

- percentage of top-1 recommendations changed by reranking;
- percentage of top-5 recommendations changed by reranking;
- evidence-score change at top 1, 5, and 10;
- pre/post ranking overlap;
- mean rank shift;
- exact rule-trace preservation;
- rule frequency;
- percentage of KB rules ever used;
- top rules overall and by category;
- packet-level prevalence;
- slot-level share;
- Shannon entropy;
- within-category Jaccard overlap;
- between-category Jaccard overlap;
- number of distinct five-rule packets;
- percentage of identical packets;
- candidate-level variation;
- query-level variation;
- generic-rule flags.

Important distinction:

```text
packet prevalence ≠ retrieval-slot share
```

For example, a rule appearing in 180 of 900 packets has 20% packet prevalence, even though it occupies only 4% of 4,500 retrieval slots.

Repeated rules are not automatically invalid. The audit must distinguish:

- legitimate category anchor rules;
- overly generic rules;
- candidate-specific rules;
- category-only retrieval behaviour.

A healthy packet may contain:

```text
one broad category anchor
+
several query/candidate-specific rules
```

A weak packet is one where the same five generic rules occur for nearly every candidate in a category.

---
# 11. Publication-ready figures, tables, and qualitative examples

The project must generate publication-ready visual and tabular artifacts throughout the
pipeline. Producing metrics alone is insufficient.

Every major experiment must preserve the underlying machine-readable data and create the
figures, tables, and qualitative examples required for the thesis and research paper.

## 11.1 General requirements

For every experiment, determine whether the result is best communicated as:

- a table;
- a plot or curve;
- a qualitative image panel;
- a worked example;
- or a combination of these.

Do not create decorative figures. Every artifact must communicate a research question,
methodological decision, result, trade-off, error pattern, or representative example.

All figure and table generation must be reproducible from saved machine-readable outputs.
Do not manually construct final numbers or edit plotted values.

Each artifact must have:

- a stable figure or table ID;
- a descriptive title;
- axis labels and units where applicable;
- sample size;
- uncertainty intervals where applicable;
- the experiment configuration or configuration hash;
- the source data path;
- the script or function that generated it;
- a draft publication-ready caption;
- accessible legends and readable text;
- no misleading truncated axes;
- no unsupported causal interpretation.

Charts must be exportable in vector formats where possible:

- PDF or SVG for plots, diagrams, and curves;
- high-resolution PNG for image-based qualitative panels;
- CSV or Parquet for the underlying table data;
- Markdown or LaTeX-compatible tables for writing.

## 11.2 Required methodology figures

At minimum, create:

1. **End-to-end system architecture**
   - user image and request;
   - CLIP image/text encoding;
   - fused multimodal retrieval;
   - expert-rule retrieval;
   - evidence-aware reranking;
   - locked recommendation;
   - A/B explanation conditions;
   - claim extraction;
   - verification;
   - judging;
   - final evaluation.

2. **A/B evidence-ablation diagram**
   - No-RAG = A;
   - Rule-RAG = A + B;

3. **Evidence-trace diagram**
   - retrieved rules;
   - rule scores;
   - candidate evidence score;
   - reranked recommendation;
   - exact same stored rules passed to the explanation generator.

4. **Dataset-processing and split diagram**
   - raw items and outfits;
   - preprocessing;
   - broad-category mapping;
   - development, validation, and test splits;
   - controlled candidate-pool construction.

## 11.3 Required dataset and KB tables/figures

Create:

- item and outfit count table;
- category-frequency table and plot;
- outfit-size distribution;
- broad-category distribution;
- usable-case counts;
- train/validation/test split counts;
- candidate-pool statistics;
- KB rule counts by target category;
- KB source-type distribution;
- reliability-label distribution;
- source coverage table;
- rule-retrieval frequency plot;
- within-category and between-category rule-overlap plots;
- generic-rule and repeated-packet diagnostics.

## 11.4 Required recommendation figures and tables

Create:

- full recommendation metric table for every baseline;
- category-level recommendation table;
- confidence-interval or forest plot for paired model contrasts;
- HR@K and NDCG@K curves across K where useful;
- pre-reranking versus post-reranking rank-distribution plot;
- evidence-score gain plot;
- top-k overlap plot;
- proportion of changed top-1 recommendations;
- category-specific reranking effects;
- recommendation-quality versus evidence-participation Pareto frontier;
- selected Pareto operating point clearly highlighted;
- dominated and non-dominated configurations preserved in the source table.

## 11.5 Required hyperparameter and grid-search artifacts

Every validation search must save:

- every tested configuration;
- all validation objectives;
- the selected configuration;
- the selection rule;
- dominated and non-dominated points;
- failures and excluded configurations.

Create suitable visualizations such as:

- heatmaps for two-dimensional parameter combinations;
- line plots for single-variable effects;
- parallel-coordinate plots for larger searches;
- Pareto-frontier plots for multi-objective searches;
- tables of the top non-dominated settings;
- sensitivity curves for evidence weight;
- sensitivity curves for rule count;
- effects of word-count instruction;
- effects of rule-evidence formatting and ordering;
- effects of citation requirements;
- interaction plots where two variables materially interact.

Do not select settings by visual inspection of test results. All search figures must be based
on validation data only.

## 11.6 Required qualitative recommendation examples

Create publication-ready panels showing the actual fashion images.

Each selected case should display:

- query image;
- user request;
- top five recommendations before evidence reranking;
- top five recommendations after evidence reranking;
- candidate rank and score changes;
- selected candidate;
- concise evidence trace;
- explanation where appropriate.

Include a fixed, documented selection strategy rather than cherry-picking examples.

At minimum, include:

- representative successful cases;
- cases where evidence reranking improves the relevant item’s position;
- cases where reranking reduces recommendation quality;
- cases with high rule participation;
- cases with weak or generic rule retrieval;
- examples from all five target categories.

Example selection must be deterministic, such as:

- median-performing cases;
- largest positive rank changes;
- largest negative rank changes;
- predefined random cases using a fixed seed.

## 11.7 Required explanation-ablation examples

For the same query and locked recommendation, create side-by-side examples of:

- No-RAG explanation;
- Rule-RAG explanation;

Each panel must show:

- query image and fixed A context;
- user request;
- locked recommendation image and minimal identity;
- B expert-rule evidence;
- the two explanations;
- rule citations where present;
- explanation word count;
- number of extracted claims;
- support and unsupported-claim summaries.

Use the same case for both conditions so that evidence condition is the only intended
difference.

Include:

- representative cases;
- cases where expert rules improve grounding;
- failure cases;
- cases where a fluent explanation remains weakly grounded.

## 11.8 Required claim-extraction and verification examples

Create worked examples showing:

1. original explanation;
2. extracted atomic claims;
3. claim types;
4. verifier support status;
5. all identified support sources;
6. supporting rule IDs;
7. verifier reason;
8. whether a cited rule entails the claim.

Include examples of:

- supported claims;
- unsupported claims;
- contradicted claims;
- not-verifiable claims;
- claims supported by multiple sources;
- malformed or ambiguous claims;
- representative extraction and verification edge cases identified in saved outputs.

Also create:

- claim-count distribution;
- support-rate table by condition and generator;
- unsupported-claim-rate plot;
- support-source frequency plot;
- contradiction and not-verifiable counts;
- confidence-interval plots for condition contrasts.

## 11.9 Required general-judge artifacts

Create:

- score table by condition, generator, judge, and dimension;
- cross-model primary results;
- all-judge sensitivity results;
- score-distribution plots;
- confidence-interval plots;
- generator-by-judge matrix;
- surface-quality versus claim-grounding comparison;
- examples where judges assign high general quality but verification finds weak support;
- examples where grounded explanations receive only moderate stylistic-quality scores.

For worked judge examples, show:

- explanation;
- displayed evidence;
- dimension scores;
- brief judge reason;
- whether the judgment is cross-model or self-family.

Do not reveal hidden experimental condition names to the judge during evaluation, but they may
be added later when constructing the research figure.

## 11.10 Tables versus figures

Use tables when exact values and comparisons are important.

Use figures when communicating:

- trends;
- distributions;
- uncertainty;
- trade-offs;
- sensitivity;
- ranking changes;
- category variation;
- error patterns.

Use qualitative image panels when the reader must inspect:

- visual compatibility;
- recommendation changes;
- evidence relevance;
- explanation differences;
- claim-level reasoning.

Important results may have both a table and a figure, but avoid duplicating information
without a clear reason.

## 11.11 Artifact organisation

Use only the following approved subdirectories under `artifacts/`:

```text
artifacts/
├── figures/
├── tables/
├── examples/
└── manifests/
```

Do not create separate folders for every experiment or every figure.

Use stable names such as:

```text
fig_01_system_architecture.pdf
fig_02_pareto_frontier.pdf
fig_03_recommendation_examples.png
fig_04_explanation_ablation_examples.png
fig_05_claim_support_by_condition.pdf

table_01_dataset_statistics.csv
table_02_recommendation_results.csv
table_03_claim_grounding_results.csv
table_04_judge_results.csv
```

The human-readable interpretation and intended paper/thesis placement should be added to:

- `reports/methodology.md`;
- `reports/final_results.md`.

Do not create a new Markdown report for each plot.

## 11.12 Figure and table registry

Maintain one registry in `artifacts/manifests/figure_table_registry.csv` containing:

- artifact ID;
- artifact type;
- title;
- research question;
- source data;
- generation function or script;
- configuration hash;
- output path;
- caption;
- intended thesis chapter;
- intended paper section;
- status;
- notes.

The final analysis stage must verify that every major reported result has an associated table,
figure, or justified textual presentation.

The `figures`, `tables`, `examples`, and `manifests` directories are controlled subfolders
under `artifacts/`; they are not additional top-level directories.

---

# 12. Dataset and assets

Primary dataset:

```text
Marqo/polyvore
```

Pinned and validated provenance:

- revision: `8c782ee447faf2d2a0402ac883cf07d3b3f43e1c`;
- configuration: `default`;
- split: `data`;
- acquisition date: `2026-08-06`;
- licence metadata: `apache-2.0`;
- dataset fingerprint: `9c97dc763773e2a2`;
- cache fingerprint:
  `cc1ddeafe547ef65395df0959c7cd0bd3a717c9288ee9f759b0ccfa9ad777b08`;
- raw schema: `image`, `category`, `text`, `item_ID`.

Validated Stage 2 counts:

- 94,096 items;
- 21,587 outfits;
- 66,749 target-category items;
- 15,267 development outfits;
- 3,147 validation outfits;
- 3,173 test outfits;
- zero item/outfit overlap across splits;
- all five target categories support deterministic controlled pools; the updated main evaluation setting is up to 999 negatives and must be validated before the confirmatory run.

The regenerated target-item count is authoritative for the pinned revision. Historical
approximate counts must not be forced into the clean implementation.

Requirements:

- preserve the pinned revision and acquisition metadata;
- record raw and processed counts;
- record all filtering;
- store the broad-category mapping and hash;
- preserve deterministic split logic;
- confirm zero query/positive item and outfit overlap across splits;
- audit near-duplicate image/product leakage where feasible;
- use only the dataset’s `category` and `text` fields for the minimal item identities in A;
- never create textual explanation evidence by analysing the `image` field.

Do not commit the full dataset or Hugging Face cache to Git.

---

# 13. Knowledge base requirements

The approved knowledge base is KB v3 only:

- repository path: `data/kb/fashion_rules.csv`;
- rows: 126;
- unique non-empty rule IDs: 126;
- SHA-256:
  `ad19fc788769ebd5fec65ee8aa6b62e4cfc8fbf1f67725392b754a327c2dced3`.

The KB must remain auditable.

Required fields:

- rule_id;
- rule_text;
- input_category;
- recommended_category;
- occasion;
- style;
- colour context;
- season;
- formality;
- body/fit context where relevant;
- evidence keywords;
- source type;
- source title;
- author;
- year;
- URL or identifier;
- reliability label;
- reliability rationale;
- evidence basis;
- scope;
- limitations;
- manual verification status;
- notes.

Required audits:

- duplicate IDs;
- exact duplicates;
- near-duplicate rules;
- unsupported claims;
- weak or unverifiable sources;
- category imbalance;
- source imbalance;
- conflict detection;
- manual-verification completeness;
- licensing and redistribution status.

Do not describe the KB as comprehensive fashion knowledge or validated ground truth.

---

# 14. Clean repository requirement

Build a new repository.

Suggested names:

```text
evidence_fashion_recommender_dev
```

for the existing archive, and:

```text
evidence_fashion_recommender
```

for the clean final implementation.

The old repository must remain untouched except for read-only reference.

The clean repository must use exactly the following top-level structure unless explicit approval is obtained.

```text
evidence_fashion_recommender/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── configs/
│   ├── experiment.yaml
│   └── models.yaml
│
├── data/
│   ├── README.md
│   └── kb/
│       └── fashion_rules.csv
│
├── src/
│   └── evidence_fashion/
│       ├── data.py
│       ├── retrieval.py
│       ├── rule_retrieval.py
│       ├── reranking.py
│       ├── explanation.py
│       └── evaluation/
│           ├── recommendation.py
│           ├── claim_extraction.py
│           ├── claim_verification.py
│           ├── judging.py
│           └── statistics.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_embeddings.py
│   ├── run_recommendation_eval.py
│   ├── run_explanation_eval.py
│   └── run_final_analysis.py
│
├── notebooks/
│   └── final_pipeline_demo.ipynb
│
├── tests/
│
├── reports/
│   ├── methodology.md
│   └── final_results.md
│
└── artifacts/
    ├── README.md
    ├── figures/
    ├── tables/
    ├── examples/
    └── manifests/
```

## 14.1 Structural rules

- Do not create additional top-level folders without approval.
- Do not create a new report file for every small check.
- Append related audit results to the appropriate stable report.
- Keep machine-readable outputs under `artifacts/`.
- Keep only two main human-readable reports:
  - `reports/methodology.md`;
  - `reports/final_results.md`.
- Keep temporary files outside the repository or under ignored runtime directories.
- Do not commit:
  - model caches;
  - raw datasets;
  - embeddings;
  - large checkpoints;
  - generated images;
  - full LLM cache files.
- Store hashes and paths in manifests.
- Every module must have one clear responsibility.
- Avoid duplicated utility modules.
- Avoid versioned filenames such as:
  - `final_v2_new_fixed.py`;
  - `analysis_latest_final2.py`.
- Use Git history instead of filename proliferation.
- Delete superseded code only within the new clean repository after tests confirm it is unused.
- Never delete provenance from the old development repository.

---

# 15. Configuration
## 15.1 Configuration completeness requirement

Every experiment-facing decision, tunable parameter, threshold, weight, model setting,
sampling choice, prompt option, and output-control value must be stored in
`configs/experiment.yaml` or `configs/models.yaml`.

No experiment-critical magic numbers or hidden defaults may appear in source code.

This requirement applies to:

- dataset revision and preprocessing thresholds;
- split ratios and random seeds;
- category mappings and enabled categories;
- candidate-pool size and negative-sampling rules;
- embedding model IDs;
- image/text fusion weights;
- rule-retrieval top-k;
- reliability weights and bonuses;
- accessory and category-filter settings;
- CLIP/evidence reranking weights;
- Pareto-search ranges and selection criteria;
- explanation condition definitions;
- prompt-template selection;
- word-count settings;
- rule-evidence count;
- approved Polyvore fields used for A item identities and duplicate-suppression policy;
- A-context formatting;
- citation requirements;
- evidence ordering;
- generation parameters;
- retry and repair policies;
- extraction and verification labels;
- judge dimensions and score anchors;
- bootstrap replicates, confidence levels, and multiplicity corrections;
- manual-audit size, quotas, seed, and case caps.

Internal programming variables do not need to be configuration entries. The requirement
applies to anything whose value could alter an experiment, result, comparison, or reported
method.

Every run manifest must store the fully resolved configuration and its SHA-256 hash.
Changing any experiment-facing setting must produce a different configuration hash.

`configs/experiment.yaml` should define:

- recommendation evaluation case count;
- explanation evaluation case count;
- random seeds;
- dataset revision;
- split ratios;
- target categories;
- controlled-pool size;
- embedding model IDs;
- image/text fusion weights;
- rule top-k;
- rule reliability weights;
- score fusion weights;
- explanation cases;
- explanation conditions;
- generator roster;
- extractor/verifier model;
- judge roster;
- retry policy;
- bootstrap settings;
- output paths.

Indicative configuration entries include:

```yaml
recommendation_evaluation:
  case_count: 1000

candidate_pool:
  max_negatives: 999

explanations:
  case_count: 500
  conditions: [no_rag, rule_rag]

reranking:
  reference_clip_weight: 0.75
  reference_evidence_weight: 0.25

reranking_search:
  evidence_weights: [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
  rule_top_k: [1, 3, 5]
  selection_method: pareto_knee
  maximum_ndcg10_relative_loss: 0.02

explanation_evidence:
  common_A_fields: [user_request, query_item_minimal_name, locked_item_minimal_name]
  B_source: exact_stored_rule_scoring_trace
  forbid_image_derived_text: true

explanation_search:
  rag_word_limits: [45, 60, 75]
  rule_counts: [1, 3, 5]
  citation_modes: [required, optional]
```

`configs/models.yaml` should define:

- exact model tags;
- immutable digests;
- context length;
- temperature;
- top-p;
- top-k;
- seed;
- token limits;
- timeout;
- device;
- precision.

No experiment-critical parameter should be hidden inside source code.

---

# 16. Artifact and manifest policy

Each major stage must create one manifest.

Required manifests:

- data preparation manifest;
- embedding manifest;
- recommendation-evaluation manifest;
- evidence-trace manifest;
- explanation-generation manifest;
- claim-extraction manifest;
- verification manifest;
- judging manifest;
- final-analysis manifest;
- manual-audit manifest.

Each manifest should include:

- timestamp;
- Git commit;
- configuration hash;
- input artifact hashes;
- output artifact hashes;
- model IDs and digests;
- row counts;
- failure counts;
- seed;
- environment summary;
- command executed.

Successful frozen outputs must never be overwritten in place.

Use a clearly named run ID and immutable directories under ignored storage, while keeping only compact manifests in Git.

---

# 17. Reproducible command sequence

The README must make the complete workflow understandable from one page.

Expected sequence:

```bash
uv sync --extra cuda

uv run python scripts/prepare_data.py
uv run python scripts/build_embeddings.py
uv run python scripts/run_recommendation_eval.py
uv run python scripts/run_explanation_eval.py
uv run python scripts/run_final_analysis.py
```

The scripts must support:

```bash
--config configs/experiment.yaml
--dry-run
--resume
--validate-only
```

The explanation script should support a pilot mode:

```bash
uv run python scripts/run_explanation_eval.py \
  --config configs/experiment.yaml \
  --pilot-cases 50
```

Do not launch the full explanation experiment until the pilot is approved.

---

# 18. Required pilot before full run

Run 50 cases first.

The pilot must display and save:

- query image;
- fixed A context;
- user request;
- top candidates before reranking;
- exact candidate rule-evidence traces;
- reranked candidates;
- locked recommendation;
- exact A and B blocks;
- both rendered prompts;
- generated explanations;
- extracted claims;
- verification outputs;
- general judge outputs.

Manual pilot checks:

1. A, B do not overlap improperly.
2. No prompt contains a condition label.
3. B exactly matches the scoring trace.
4. Rules vary appropriately with candidate/query context.
5. No image-derived text or unsupported item information is introduced.
6. The verifier can assign multiple support sources.
7. All schemas validate.
8. No hidden test leakage is present.
9. Outputs are understandable without opening many folders.

Only after these checks pass may the 500-case full explanation experiment begin.

---

# 19. Testing requirements

Minimum tests:

## Data

- deterministic split;
- no outfit overlap;
- query exclusion;
- negative-pool validity;
- category mapping;
- processed-count checks.

## Retrieval

- normalized embeddings;
- deterministic ranking;
- category filtering;
- fusion calculation;
- tie-breaking.

## Rule retrieval

- category restriction;
- filter-before-top-k;
- reliability weighting;
- exact score reproduction from stored trace;
- identical B trace between reranking and explanation;
- no second retrieval.

## Prompt construction

- A/B non-overlap;
- no condition-label leakage;
- no hidden metadata;
- no image-derived text enters A or B;
- exact rule citations;
- stable rendered prompt hash.

## Structured LLM outputs

- extraction schema;
- verification schema;
- multi-source support;
- missing/duplicate claim rejection;
- invalid rule-ID rejection;
- retry and N/A handling.

## Statistics

- clustered bootstrap;
- paired comparison;
- multiplicity correction;
- denominator and N/A handling.

Run Ruff and the complete test suite before every local commit.

---

# 20. Migration from the development repository

The coding agent must first produce a migration inventory.

For each candidate file from the old repository, classify it as:

- transfer;
- rewrite;
- archive only;
- obsolete.

Transfer only:

- validated category mapping;
- deterministic split logic;
- tested embedding wrappers;
- ranking metrics;
- tested bootstrap/statistics functions;
- reusable schemas;
- reproducibility metadata.
 if even 1% you think they should reproduce dont worry and run again since a clean project is very important.
Rewrite:

- explanation prompt construction;
- A-context construction;
- rule-score/retrieval trace;
- verifier schema;
- deterministic explanation-metric tooling;
- clean CLI scripts;
- manifest handling.

Archive only:

- old frozen explanation runs;
- old mutually exclusive verifier outputs;
- previous recovery scripts;
- historical audit reports;
- duplicated one-off analysis scripts.

The new repository must not inherit the old directory sprawl.

---

# 21. Final deliverables

The completed clean project must provide:

1. a working real-input recommendation demo with user-supplied query-item name;
2. deterministic dataset preparation;
3. multimodal recommendation retrieval;
4. candidate-specific rule retrieval;
5. evidence-aware reranking;
6. exact evidence traces;
7. two-condition explanation generation;
8. atomic-claim extraction;
9. multi-source claim verification;
10. cross-model general judging;
11. recommendation evaluation;
12. explanation-grounding evaluation;
13. rule-diversity analysis;
14. manual-audit preparation and scoring;
15. compact methodology report;
16. compact final-results report;
17. reproducibility manifests;
18. tests;
19. one final notebook demo;
20. a simple README explaining the entire system.

---

# 22. Acceptance criteria

The project is complete only when all of the following are true.

## Architecture

- The end-to-end pipeline is implemented.
- The clean repository uses the approved structure.
- No unnecessary top-level folders exist.
- The README explains where every important file belongs.

## A/B validity

- A contains common context only.
- B is the exact scoring trace.
- No condition label appears in any prompt.
- No evidence block is duplicated.

## Recommendation

- Retrieval and reranking are deterministic.
- Metrics are reproduced from stored rankings.
- Evidence participation is measurable.
- Recommendation claims are appropriately qualified.

## Explanation

- 2 conditions differ only by intended evidence.
- Explanations are immutable after generation.
- Rule citations are exact and auditable.
- No image-derived caption or attribute enters A or B.
- Factual item claims must be supported by A, and/or B.

## Evaluation

- Every extracted claim is verified or explicitly N/A.
- Support status and support source are separate.
- Multiple support sources are allowed.
- Cross-model judging is reported.
- Automated evaluation limitations are reported explicitly.
- Confidence intervals use the correct clustering unit.

## Reproducibility

- Dataset revision is pinned.
- Model digests are recorded.
- Configurations are hashed.
- Every major artifact is hash-bound.
- The full workflow can be reproduced from the README and manifests.

## Publication readiness

- Every major quantitative result has a reproducible source table.
- Every important trend, trade-off, ablation, or sensitivity analysis has an appropriate
  publication-ready visualization.
- Recommendation examples include the original images.
- The two explanation conditions are shown side by side for identical locked cases.
- Worked claim-extraction, verification, and judge examples are preserved.
- Example selection follows a fixed documented strategy and is not cherry-picked.
- Every figure and table has a caption, source-data reference, configuration hash, and
  generation provenance.
- All artifacts required for the paper and thesis can be regenerated from saved outputs.
---

# 23. Non-goals

Do not expand the project into:

- a production e-commerce platform;
- a user-account system;
- a web marketplace;
- a full-catalogue scalable serving system;
- causal interpretability research;
- automatic fashion-rule authoring;
- reinforcement learning;
- end-to-end model training;
- image captioning for explanation evidence;
- visual-attribute or fashion-attribute extraction;
- image-grounded textual explanation in the main study;
- a large front-end application.

The priority is a rigorous, reproducible research implementation.

---

# 24. Instructions to the coding agent

1. Do not start by copying the old repository wholesale.
2. Create the clean structure first.
3. Produce a migration inventory before moving code.
4. Stop and report whenever the old implementation conflicts with this specification.
5. Do not silently preserve known flaws for compatibility.
6. Do not invent dataset fields or derive textual explanation evidence from images.
7. Do not create additional top-level directories.
8. Do not create dozens of one-off reports.
9. Do not run the full LLM experiment before pilot approval.
10. Do not overwrite historical frozen artifacts.
11. Commit in small, understandable stages.
12. Do not push unless explicitly instructed.
13. At the end of every stage, report:
    - files changed;
    - tests run;
    - artifacts created;
    - decisions requiring researcher approval;
    - the next recommended step.

---

# 25. Recommended build stages

## Stage 1 — clean skeleton and migration inventory

- create repository structure;
- create configs;
- create README;
- inventory old code;
- no model calls.

## Stage 2 — data and deterministic evaluation cases

- port dataset adapter;
- pin revision;
- implement splits;
- implement candidate pools;
- validate counts and leakage.

## Stage 3 — multimodal embeddings and retrieval interfaces

- implement and validate MiniLM text embeddings;
- implement and validate CLIP image and text embeddings;
- implement fused CLIP query embeddings;
- cache embeddings outside Git and record compact manifests;
- record the no-image-derived-text explanation decision concisely in `reports/methodology.md`;
- run only the approved embedding validation required by this stage.

## Stage 4 — unified evidence trace and reranking

- implement one rule retrieval/scoring function;
- store exact trace;
- ensure B is identical to score contributors;
- run rule-diversity diagnostics.

## Stage 5 — explanation optimisation, prompt construction, and 50-case pilot

- keep A fixed and run only the validation-only RAG-variable search;
- select and freeze the RAG prompt configuration;
- build and validate non-overlapping A/B blocks;
- render both prompts;
- remove condition leakage;
- run the 50-case pilot using the selected configuration;
- perform manual inspection and obtain researcher approval.

## Stage 6 — recommendation evaluation

- run the 1,000-case confirmatory recommendation evaluation;
- run baselines;
- run evidence reranking;
- calculate metrics and clustered statistics;
- freeze locked cases.

## Stage 7 — full explanation experiment

- generate two conditions for 500 locked cases across approved generators;
- hash outputs;
- no later regeneration except explicit failed-key recovery.

## Stage 8 — extraction, verification, judging, and final statistics

- extract atomic claims;
- verify with multi-source schema;
- run cross-model judges;
- preserve N/A records.
- calculate grounding and study-specific metrics;
- calculate paired contrasts and sensitivity analyses.

## Stage 10 — final cleanup and release review

- remove unused code from the clean repository;
- confirm no directory sprawl;
- finalize README;
- finalize methodology and results reports;
- verify manifests and hashes;
- run complete test suite.

---

# 26. Final research framing

The final study should answer:

1. Does expert-rule evidence improve atomic-claim support compared with no external evidence?
2. How does evidence-aware reranking affect recommendation quality and rule participation?
3. Do general explanation-quality scores agree with evidence-grounding scores?
4. Are the explanation findings directionally stable across generators, categories, aggregation
   levels, and the predeclared length sensitivity?

The intended contribution is not merely that RAG helps. It is:

> Explicit expert-rule evidence may improve support for relational and styling claims compared with a no-RAG baseline, while recommendation quality and explanation grounding may involve measurable trade-offs.

This claim must be accepted only if the corrected experiment supports it.
