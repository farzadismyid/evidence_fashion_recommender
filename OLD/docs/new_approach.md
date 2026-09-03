# New Approach

## Purpose

This document defines the clean, reproducible fresh-start protocol for the evidence-constrained fashion recommendation experiment. It replaces the former multi-accessory experiment with five evaluated categories while retaining the established 1,000-case recommendation design.

The five evaluated categories are:

- tops;
- bottoms;
- shoes;
- outerwear; and
- bags.

Recommendation evaluation remains 1,000 cases, with 200 cases per category. The explanation experiment uses a deterministic 500-case subset, with 100 locked recommendations per category. Existing recommendation metrics remain unchanged. Final explanation metrics will be selected only after claim-extraction and claim-verification quality has been reviewed.

## Scientific principles

- Prompts must be neutral and must not direct any evaluator to favour Rule-RAG.
- Prompt development must use a calibration set that is separate from the final test cases.
- Prompts, schemas, model identities and evidence boundaries must be visible and version-controlled.
- Once frozen, changing a prompt invalidates that role's outputs and all dependent stages.
- No-RAG and Rule-RAG must use the same recommendation and common context.
- Models must run sequentially by complete batch, with outputs saved and the model unloaded before the next model is loaded.
- The live repository must contain one canonical output per stage, without V1/V2-style duplicates.
- Archived previous work must remain outside the live experimental outputs and outside Git tracking.

## Stage 1 — Freeze the five-category taxonomy

Replace the multi-accessory taxonomy with tops, bottoms, shoes, outerwear and bags.

The bag allowlist is:

- Bags;
- Handbags;
- Shoulder Bags;
- Tote Bags;
- Clutches;
- Messenger Bags;
- Men's Bags; and
- Men's Messenger Bags.

Backpacks, briefcases, luggage and every non-bag accessory are excluded. The internal category must be named `bags`, not `accessories`, so prompts, manifests, reports and thesis text cannot accidentally imply that eyewear, jewellery or other accessories were evaluated.

The recommendation design remains 1,000 cases, with 200 cases per category. HR@K, NDCG@K, MRR and the established paired recommendation analyses remain unchanged.

Stage 1 data preparation must also produce the deterministic audit-only 200-case bag sample used
by Stage 2. Stage 7 later reproduces these cases from the frozen configuration; it does not select
replacement cases after the KB audit.

## Stage 2 — Audit and freeze bag rules

Audit the currently identified bag-explicit rules for actual applicability rather than relying on keyword matching alone.

The audit must:

- retain only rules whose rule text genuinely supports recommending a bag;
- exclude generic accessory-only evidence from bag retrieval;
- assess coverage by query category, occasion, gender context and formality;
- verify that each bag case can retrieve applicable evidence rather than merely five technically eligible rules;
- measure rule frequency, concentration, packet duplication and pairwise overlap; and
- identify unsupported contexts before the experimental protocol is frozen.

The minimum Stage 2 pass gates are: every bag case has at least one directly supported applicable
rule; no generic accessory-only rule enters a bag packet; every retained online rule records a
source locator and access date; and every unsupported context is either covered before result
inspection or explicitly excluded. If a later prompt requires exactly five rules, every included
case must have five applicable rules; otherwise the prompt must state "up to five" before it is
frozen. Rule concentration and packet-overlap thresholds must be approved alongside the expanded
KB, before recommendation results are inspected.

If important contexts lack adequate support, properly sourced rules must be added before the experiment, or those contexts must be explicitly excluded and reported as limitations. Rules must not be added after inspecting final condition results.

## Stage 3 — Centralize and freeze all LLM prompts

Create one canonical configuration file, `configs/prompts.yaml`, containing:

- the No-RAG explanation system prompt and user template;
- the Rule-RAG explanation system prompt and user template;
- the claim-extraction system prompt and user template;
- the claim-verification system prompt and user template;
- the optional blind-judge system prompt and user template; and
- structured-output retry and repair instructions.

Each role must expose its system prompt, user template, permitted evidence, prohibited inference, output schema, token limit, temperature, seed and retry behaviour. Every completed manifest must record the exact rendered prompts and SHA-256 hashes.

Prompt wording must be professional, neutral, closed-world where required, free from condition leakage and explicit about uncertainty. The same filename is overwritten before freezing when calibration changes are approved; Git provides revision history, so duplicate prompt files are unnecessary.

## Stage 4 — Enforce sequential model batches

The model order is:

1. Gemma 4 12B generates its complete explanation batch, saves it and unloads.
2. Llama 3.1 8B Q8 generates its complete explanation batch, saves it and unloads.
3. Ministral 3 14B generates its complete explanation batch, saves it and unloads.
4. Qwen 3.5 9B extracts claims from all completed explanations, saves them and unloads.
5. Phi-4 14B verifies all extracted claims, saves them and unloads.
6. The pipeline stops for inspection.
7. DeepSeek-R1 14B is used as an optional blind judge only after a separate approval decision.

Models must not run concurrently or switch on a record-by-record basis.

## Stage 5 — Calibrate extraction and verification

Create a small human-annotated calibration set that is disjoint from the final 500 cases. It must cover compound claims, explicit attributes, styling and functional inferences, unsupported but plausible statements, partial entailment, contradiction, negation, invalid citations, bag examples and outputs from both conditions.

Qwen extraction and Phi verification must be tested against these annotations. Freeze their prompts only after claim coverage, atomization, duplicate control, claim-ID preservation, verdict/reason consistency and structured-output reliability are acceptable.

The calibration annotations, decision rules and pass criteria must be retained as methodological artifacts. Calibration examples must never be included in headline test results.

## Stage 6 — Establish canonical runtime destinations

Use one active destination for each stage, for example:

```text
.runtime/current/data
.runtime/current/embeddings
.runtime/current/recommendations
.runtime/current/explanations
.runtime/current/extraction
.runtime/current/verification
.runtime/current/judging
```

Run identifiers and hashes belong inside manifests rather than duplicate directory names. Each stage writes resumable progress to a temporary working location and atomically replaces its canonical completed output. Failed or partial attempts must not appear as completed results.

The existing archive remains ignored and immutable. Live `reports/` and `artifacts/` are populated only from the current approved run.

## Stage 7 — Regenerate data, embeddings and recommendation results

Rerun the following because the taxonomy and rule eligibility have changed:

1. data preparation and category audit;
2. split and leakage validation;
3. text and image embeddings for the revised eligible dataset;
4. validation-stage fusion and reranking checks;
5. applicable recommendation-stage tuning and sensitivity checks;
6. the full 1,000-case recommendation evaluation; and
7. recommendation diagnostics, statistical tables and manifests.

The pinned raw Polyvore dataset, dependency downloads and model caches may be reused. Previous processed data, embeddings and experimental results may not be mixed with the new run.

## Stage 8 — Select and freeze 500 explanation cases

Select a deterministic subset of the new recommendations containing exactly:

- 100 tops;
- 100 bottoms;
- 100 shoes;
- 100 outerwear; and
- 100 bags.

Before freezing, verify that there are no duplicate case IDs, every bag passes the allowlist, every Rule-RAG case has a valid stored evidence trace, and both conditions use the identical recommendation and common context.

## Stage 9 — Generate all explanations fresh

Generate 3,000 explanations:

- 500 cases;
- two conditions per case; and
- three generator models.

Record model and prompt hashes, rendered prompts, outputs, evidence visibility, word counts, latency, token counts, refusals, retries and terminal failures. No explanation from the previous experiment may be reused.

## Stage 10 — Extract all claims

Qwen 3.5 9B extracts claims from all **2,985 accepted explanations** using one frozen extraction prompt and identical treatment across No-RAG / Rule-RAG and Gemma / Llama / Mistral outputs.

Extraction must be evidence-independent: Qwen identifies what the explanation asserts, not whether the claims are correct or supported.

Each extracted claim should:
- represent one independently verifiable proposition where practical;
- preserve the original meaning, entities, polarity and negation;
- retain explicit item attributes, styling relations, suitability claims and other substantive assertions;
- avoid adding inferred claims that are not stated in the explanation;
- use stable case/model/condition/claim IDs.

After extraction, inspect only structural and reliability diagnostics:
- structured-output and terminal-failure rates;
- claim counts by model and condition;
- duplicate rate;
- obvious atomization/split-merge issues;
- claim-ID preservation;
- model/condition balance.

Do not judge support during extraction and do not perform another human annotation exercise.

Freeze Stage 10 outputs once the extraction batch is structurally valid and sufficiently reliable under the Stage 5 calibration findings.


## Stage 11 — Verify all extracted claims

Phi-4 14B verifies every Stage 10 claim.

Verify each claim separately against:

1. **Exact trace**
   - supported / not_supported
   - Support requires an applicable V3 trace rule that directly entails the complete claim.

2. **Full KB**
   - supported / not_supported
   - Check against any applicable V3 rule in the full KB.

3. **Common-reference facts**
   - Only literal item/context facts are eligible.
   - eligible → supported / not_supported
   - styling or subjective claims → N/A

4. **Citations**
   - Syntax and rule IDs checked deterministically.
   - Phi checks valid citations as entails / does_not_entail.

`not_supported` means only “not substantiated by this evidence,” not false.

After verification, check:
- structured-output failures;
- claim-ID preservation;
- false-positive support;
- common-reference eligibility;
- citation entailment;
- consistency with Stage 5 calibration.

Freeze Stage 11 before calculating headline explanation metrics.

Only after extraction and verification are complete should the optional DeepSeek blind judging stage be considered. If used, it must remain a separate evaluation source with randomized recorded A/B order, concealed condition identities, citation/condition markers removed, and neutral presentation-quality dimensions only.

## Stage 12 — Explanation evaluation and visual outputs

Use only the frozen Stage 9–11 canonical outputs. No new model runs or additional evaluation models are required.

### Primary metric
1. **Exact-Trace Claim Support Rate**
   - supported claims / all extracted claims
   - Measures how strongly each explanation aligns with the exact V3 rules that influenced the recommendation.
   - Compare No-RAG vs Rule-RAG on paired available cases.

### Secondary metrics
2. **Full-KB Claim Support Rate**
   - supported claims / all extracted claims using any applicable V3 rule.
   - Measures broader consistency with the expert KB.

3. **Unsupported Item-Fact Rate (UIFR)**
   - not-supported factual claims / all eligible factual claims.
   - Lower is better.
   - Use only common-reference-eligible concrete item/context facts.

4. **Exact-Trace Supported Claims per 100 Words**
   - Measures grounded explanatory information while controlling for response length.

### Diagnostic / robustness analyses
5. Report the main metrics by:
   - generator;
   - target category;
   - claim type;
   - trace size (1/2/3/4 rules).

6. **Rule-RAG Trace Utilization**
   - Measure how much of the supplied exact trace is actually reflected in the Rule-RAG explanation.
   - Treat as a Rule-RAG-only diagnostic, not a headline No-RAG comparison.

7. **Citation diagnostics**
   - citation syntax/ID validity;
   - citation entailment.
   - Treat as Rule-RAG-only auditability diagnostics.

### Statistical analysis
For fair No-RAG vs Rule-RAG comparisons:
- use paired available cases only;
- report condition means/rates and absolute percentage-point differences;
- report 95% paired bootstrap confidence intervals using 5,000 replicates;
- test paired condition differences;
- apply Holm correction across primary comparisons;
- report results overall and by generator.

Do not use structurally unfair metrics as headline evidence, and do not interpret `not_supported` as factually false.

### Outputs
Generate publication-quality:
- main results tables;
- generator/category breakdown tables;
- confidence-interval figures;
- trace-size robustness plots;
- citation/trace-utilization diagnostic tables;
- a small set of representative paired qualitative examples.

Use readable final-size fonts, colour-blind-safe colours, consistent condition/category styling, visible confidence intervals, vector SVG/PDF where possible, and 300-DPI PNG copies where needed.

Automated checks must confirm figure dimensions, resolution, font sizes and canonical-output provenance.

## Stage 13 — Final thesis update and project release audit

Use only the frozen canonical outputs from Stages 1–12. Do not rerun models, regenerate explanations, change metrics, tune prompts, rebuild the KB, or introduce new experiments.

### Thesis update

1. Revise Chapter 3 to reflect the final:
   - five-category taxonomy;
   - V3 expert-rule KB;
   - recommendation and reranking pipeline;
   - explanation conditions;
   - model roster;
   - prompt registry;
   - claim extraction and verification protocol;
   - calibration procedure;
   - statistical analysis.

2. Replace Chapter 4 results, tables and figures with the final frozen recommendation and explanation results.

3. Include the Stage 12 explanation evaluation:
   - Exact-Trace Claim Support Rate;
   - Full-KB Claim Support Rate;
   - Unsupported Item-Fact Rate;
   - Exact-Trace Supported Claims per 100 Words;
   - robustness analyses;
   - trace-utilization diagnostics;
   - citation diagnostics;
   - real qualitative examples.

4. Revise Chapter 5 conclusions and limitations so all claims match the final results.

5. Align Chapters 1 and 2 with the final terminology, research questions and scope.

6. State clearly that accessories outside bags were excluded because sufficiently explicit expert-rule coverage was not available for the final controlled experiment.

7. Rebuild all DOCX chapters from the canonical Markdown sources.

8. Ensure every reported numerical result can be traced to a current canonical table and manifest.

### Final release audit

Before closing the project:

- run all unit and integration tests;
- verify the frozen 1,000 recommendation cases and 500 explanation cases and category balance;
- verify final Stage 9, 10, 11 and 12 counts and terminal failures;
- verify model, dataset, KB, prompt and configuration hashes;
- verify all tables and figures are generated from canonical frozen outputs;
- verify qualitative examples come from real frozen records and are unedited;
- confirm no archive/legacy outputs are treated as current results;
- confirm no duplicate live result sets exist;
- confirm no obsolete metrics or previous evaluation results remain in the thesis;
- review the complete Git diff;
- create the final release manifest;
- commit and push only after all checks pass.

### Final project status

After Stage 13 passes, freeze the repository as the final experimental release.

No additional model runs, evaluator changes, KB revisions, metric changes or experimental reruns are required.

Any remaining work after Stage 13 is limited to thesis/paper writing, presentation, formatting, figure/table placement, references and language refinement.