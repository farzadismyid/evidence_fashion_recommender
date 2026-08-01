# Final Evaluation Fix Protocol

## Purpose

This protocol defines the final high-value fixes before the next major thesis/paper evaluation run.

The goal is not to fix every possible limitation. The goal is to strengthen the claims that matter most:

1. Multimodal retrieval improves recommendation ranking compared with text-only retrieval/image only retrieval.
2. Evidence-aware reranking creates an accuracy/explainability trade-off.
3. Expert-rule grounding improves explanation auditability and hallucination control.
4. No-RAG may produce fluent explanations, but it is not externally auditable.
5. Rule-RAG and Hybrid-RAG must be evaluated with clear rule-grounding and hallucination metrics, not vague faithfulness.

The reviewer audit is supporting context. This protocol is the source of truth for implementation priorities.

---

## Required conceptual fixes

### 1. Rename old faithfulness

The old judge dimension `faithfulness_to_available_information` should no longer be treated as the main faithfulness metric.

Report it as:

```text
contextual_faithfulness
```

For No-RAG, describe it as:

```text
input-grounded consistency
```

Do not compare No-RAG, Item-RAG, Rule-RAG, and Hybrid-RAG using one vague faithfulness construct.

### 2. Add final explanation-evaluation constructs

Report these constructs separately.

#### Input consistency

Does the explanation match the query item, user request, and locked recommended item?

Applies to all variants.

#### Rule-grounded faithfulness

Are styling claims supported by retrieved expert fashion rules?

Applies mainly to Rule-RAG and Hybrid-RAG.

For No-RAG and Item-RAG, report:

```text
N/A: no expert rule evidence supplied
```

#### Hallucination / unsupported fashion claims

Does the explanation introduce unsupported claims about colour, material, formality, occasion, season, comfort, trend, body fit, item type, or styling relation?

Applies to all variants, but support source must be labelled.

#### Evidence misuse

Does the explanation use the wrong rule, wrong category, wrong item type, or evidence that does not support the claim?

Applies especially to Rule-RAG and Hybrid-RAG.

#### General explanation quality

Is the explanation clear, specific, concise, and useful-looking?

This is a surface-quality/general-quality metric, not an external-grounding metric.

---

## Required judge redesign

### 1. Separate claim extraction from claim verification

Do not extract claims and verify them inside the same vague general judge response.

Use two steps.

#### Step A: Atomic claim extraction

Extract all atomic fashion/styling claims from each explanation.

Do not cap at three claims.

Useful claim types:

```text
item_type
colour
material
occasion
formality
season
comfort
trend
body_fit
styling_relation
visual_match
other
```

#### Step B: Claim verification / entailment

For each extracted claim, judge support against the allowed reference packet.

Labels:

```text
supported_by_query_or_locked_item
supported_by_item_evidence
supported_by_rule_evidence
unsupported
contradicted
not_verifiable
```

For Rule-RAG and Hybrid-RAG, rule-supported claims are central.

For No-RAG, external rule support should be N/A, not zero and not failure.

### 2. Remove perfect score for empty claim lists

If no claims are extracted, do not assign claim support = 1.0.

Use:

```text
claim_extraction_failed = true
claim_support = N/A
claim_count = 0
```

Report claim extraction coverage.

### 3. Use anchored rubrics

The judge prompt must include dimension-specific anchors.

Example anchors:

```text
input_consistency:
1 = contradicts query/request/recommended item
3 = mostly consistent but vague or partially unsupported
5 = fully consistent with query/request/recommended item

rule_grounded_faithfulness:
N/A = no rule evidence supplied
1 = most styling claims unsupported by rules
3 = some claims supported, some unsupported
5 = all substantive styling claims supported by retrieved rules

hallucination_risk:
1 = many unsupported or invented fashion claims
3 = some unsupported claims
5 = no unsupported fashion claims found

evidence_misuse:
1 = uses wrong rule/item/category or misapplies evidence
3 = minor evidence-use issues
5 = no evidence misuse found

general_quality:
1 = unclear or unhelpful
3 = understandable but generic
5 = clear, specific, concise, useful-looking
```

### 4. Use cross-model-only judging as primary

If a model is used as a generator, its own judgments of its outputs should not be primary.

Primary analysis:

```text
cross-model-only judge results
```

Self-judge rows may be kept only as sensitivity analysis.

Remove false prompt wording such as:

```text
The generator is a different model
```

unless it is guaranteed true for that row.

---

## Required recommendation-evaluation fixes

### 1. Add clean modality comparison

Run recommendation/ranking evaluation on identical cases for:

```text
MiniLM text-only
CLIP image-only
CLIP text-only
CLIP fused image+text
```

This reduces the MiniLM-vs-CLIP confound.

Report:

```text
HR@1
HR@5
HR@10
NDCG@5
NDCG@10
MRR or mean positive rank
```

Use the same candidate sets and splits across methods.

### 2. Add fusion-weight validation curve

Current fused CLIP uses fixed image/text weights. Validate them.

Suggested grid:

```text
image/text:
1.00 / 0.00
0.90 / 0.10
0.80 / 0.20
0.70 / 0.30
0.60 / 0.40
0.50 / 0.50
0.40 / 0.60
0.30 / 0.70
0.20 / 0.80
0.10 / 0.90
0.00 / 1.00
```

Selection rule:

```text
Primary: validation NDCG@10
Tie-breaker 1: validation HR@10
Tie-breaker 2: validation MRR / mean positive rank
Tie-breaker 3: simpler or more balanced setting
```

Freeze the selected fusion weight before test evaluation.

### 3. Evidence-reranking weight

Keep the existing validation-selected CLIP/evidence reranking approach.

Ensure the final held-out command reads the frozen selected artifact instead of hard-coding values.

Report the trade-off honestly:

```text
Evidence reranking is not claimed to uniformly improve accuracy. It is evaluated as an explainability/auditability-oriented reranking mechanism.
```

---

## Required Hybrid-RAG validation fixes

Hybrid-RAG means:

```text
locked recommended item + item/catalogue evidence + expert rule evidence
```

Current Hybrid-RAG tuning is useful but incomplete because item evidence count was fixed and the previous search was not full-factorial.

Add a compact validation-only Hybrid-RAG grid.

Suggested grid:

```text
word_budget: 35, 55, 75
rule_count: 3, 5
item_count: 0, 2, 5
evidence_order: rules_first, item_first
```

Total:

```text
3 × 2 × 3 × 2 = 36 configurations
```

If runtime is too high, use a smaller grid but explain the reduction.

### Hybrid-RAG selection rule

Do not use arbitrary weighted composite scores as the main selector.

Use priority-based selection:

```text
1. Minimise hallucinated fashion-claim rate.
2. Maximise rule-supported styling-claim rate.
3. Minimise evidence misuse / candidate substitution.
4. Maximise evidence overlap, reported separately for rule and item evidence.
5. Require acceptable general clarity.
6. Prefer shorter valid explanations when differences are practically tied.
```

Define practical tie threshold before selection, for example:

```text
Metrics within 1 percentage point are treated as practically tied.
```

The final selected Hybrid-RAG configuration must be frozen before test evaluation.

---

## Required statistical fixes

### 1. Outfit-clustered paired bootstrap

Do not treat every case as fully independent if multiple cases come from the same outfit.

Primary confidence intervals should use outfit-clustered paired bootstrap.

Unit of resampling:

```text
outfit_id
```

not individual row/judgment.

Preserve pairing across variants/generators/judges.

Report:

```text
mean difference
95% CI
effect size where appropriate
number of cases
number of unique outfits
```

Do not present 10,800 judgments as 10,800 independent samples.

### 2. Primary comparison families

Predefine a small set of primary comparisons.

Recommended primary explanation comparisons:

```text
Rule-RAG vs No-RAG
Hybrid-RAG vs No-RAG
Rule-RAG vs Item-RAG
Hybrid-RAG vs Rule-RAG
```

Recommended primary recommendation comparisons:

```text
Fused CLIP vs MiniLM text-only
Fused CLIP vs CLIP image-only
Fused CLIP vs CLIP text-only
Evidence-reranked fused CLIP vs pure fused CLIP
```

Correct multiple comparisons within these families.

Everything else should be labelled exploratory.

---

## Required metric reporting changes

### General-quality table

Report:

```text
general_quality
clarity
specificity if retained
contextual_faithfulness / input consistency
```

This table may include No-RAG.

### External-grounding table

Report:

```text
rule-grounded faithfulness
rule-supported claim rate
unsupported fashion-claim rate
evidence misuse rate
citation-to-claim support
rule evidence overlap
item evidence overlap
claim extraction coverage
```

For No-RAG:

```text
external rule grounding = N/A
rule evidence overlap = N/A
citation-to-claim support = N/A
```

Do not treat structural zero evidence as a failed observed grounding score.

### Citation metrics

Citation correctness should not be vacuously 1.0 for variants without citations.

Report citation metrics only when citations are required/present.

Add citation-to-claim entailment if feasible:

```text
Does the cited rule actually support the nearby claim?
```

---

## Counterfactual false-match metric

The previous counterfactual false-match metric should not be used as a main result if it is tautological by construction.

Either:

```text
1. Remove it from the main paper claims.
```

or:

```text
2. Replace it with a true counterfactual retrieval test.
```

This is not a required central fix unless the paper claims rule selectivity.

---

## Reproducibility freeze

Before the final test run, the repository must be clean and frozen.

Required:

```text
git status must be clean
commit final protocol/config changes
create a run tag
save resolved config
save selected hyperparameters
save split/case schedule hashes
save KB hash
save prompt hashes
save model names/revisions/digests
save runtime/environment metadata
save dependency lock hash
save final command list
```

Final outputs should include a manifest proving:

```text
source commit
dirty = false
config hash
case schedule hash
KB hash
prompt hash
model/runtime info
output hashes
```

---

## What should remain limitations

Do not try to fix these unless explicitly decided later:

```text
No human evaluation of trust/usefulness/preference.
Single Polyvore-style dataset.
Same-outfit co-occurrence is proxy relevance.
Fashion rules are curated styling evidence, not universal fashion truth.
Full-catalogue retrieval may remain future work unless added.
Hard-negative evaluation may remain future work unless added.
Hybrid-RAG is validation-selected, not globally optimal.
Model judges are proxy evaluators with modest agreement.
```

---