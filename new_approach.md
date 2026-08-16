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

## Stage 2 — Audit and freeze bag rules

Audit the currently identified bag-explicit rules for actual applicability rather than relying on keyword matching alone.

The audit must:

- retain only rules whose rule text genuinely supports recommending a bag;
- exclude generic accessory-only evidence from bag retrieval;
- assess coverage by query category, occasion, gender context and formality;
- verify that each bag case can retrieve applicable evidence rather than merely five technically eligible rules;
- measure rule frequency, concentration, packet duplication and pairwise overlap; and
- identify unsupported contexts before the experimental protocol is frozen.

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

Qwen 3.5 9B extracts claims from all 3,000 explanations as one complete model batch.

After extraction, inspect:

- structured-output and terminal failure rates;
- claim counts and condition balance;
- missed, duplicated and improperly compound claims;
- claim-type consistency;
- claim-ID integrity; and
- a stratified manual review sample.

No-RAG and Rule-RAG must receive identical extraction treatment. The extractor decides what claims exist, not whether they are supported.

## Stage 11 — Verify claims and decide on optional judging

Phi-4 14B verifies every extracted claim against the frozen evidence protocol as one complete model batch.

After verification, inspect verdict/reason consistency, false-positive entailment, unsupported versus not-verifiable distinctions, citation handling, evidence-source attribution and agreement with the calibration annotations.

Do not calculate headline explanation results at this point. First decide whether extraction and verification are sufficiently reliable. Only then decide whether DeepSeek judging adds useful independent information. If judging is approved, use randomized and recorded A/B position, conceal condition identities, validate score/preference consistency and treat judge results as a separate evidence source.

## Stage 12 — Select explanation metrics and rebuild visual outputs

After extractor and verifier approval, decide which explanation metrics are primary, secondary, diagnostic or removed. Metrics that create structurally unfair No-RAG versus Rule-RAG comparisons must not be used as headline results.

Generate all reports, tables, figures and qualitative examples only from approved canonical outputs. Use a shared publication-quality visual style with:

- readable fonts at final thesis insertion size;
- vector SVG or PDF wherever possible;
- 300-DPI PNG copies where raster output is required;
- a colour-blind-safe palette;
- consistent condition and category colours;
- clearly visible uncertainty intervals;
- larger legends, labels, ticks and annotations;
- proper tables rather than table screenshots; and
- readable image panels and captions.

Automated visual checks must confirm dimensions, resolution and minimum configured font sizes.

## Stage 13 — Revise the thesis and perform the release audit

Update the thesis only from approved outputs:

1. revise Chapter 3 for the taxonomy, models, prompt registry, evidence protocol and sequential MoA workflow;
2. replace all Chapter 4 results, tables and figures;
3. revise Chapter 5 interpretations, conclusions and limitations;
4. align Chapters 1 and 2 with the final terminology and research scope;
5. state explicitly that accessories outside bags were excluded because they lacked sufficiently explicit KB coverage;
6. rebuild all DOCX chapters from the canonical Markdown sources; and
7. trace every reported number to a current table and manifest.

Before release, run unit and integration tests; verify the 1,000/500 case counts and category balance; verify model, data and prompt hashes; confirm that archive files are not tracked; confirm there are no duplicate live results; review the complete Git diff; and push only after the new approach is internally reproducible.

## Remaining-issue coverage

All previously identified non-metric issues are included in this approach:

| Previously identified issue | Addressed in |
|---|---|
| Real-data calibration of extractor and verifier | Stage 5 |
| Bag-rule coverage across query and gender contexts | Stage 2 |
| Rule concentration with a small bag-rule pool | Stage 2 |
| Fair and explicit verifier evidence boundaries | Stages 3, 5 and 11 |
| Semantic validation beyond valid JSON | Stages 5, 10 and 11 |
| Prompt visibility and freezing | Stage 3 |
| Retry and terminal-failure transparency | Stages 3, 6 and 9–11 |
| One canonical output lifecycle | Stage 6 |
| Optional judge decision and calibration | Stage 11 |
| Thesis-scale visual accessibility | Stage 12 |
| Thesis consistency and result provenance | Stage 13 |

These issues do not require a second independent plan. They require implementation and explicit pass/fail checks within the named stages. The only deliberately deferred design decision is the final set of explanation metrics, which is handled after extraction and verification inspection in Stage 12. Recommendation metrics are retained.

## Start gate

No full experimental run begins until Stages 1–5 are implemented and their tests and calibration checks pass. This prevents an expensive run from being invalidated by taxonomy, KB, prompt or evaluator defects discovered afterward.
