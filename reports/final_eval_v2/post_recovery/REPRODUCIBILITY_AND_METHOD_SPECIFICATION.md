# Reproducibility and method specification

Status: post-recovery final-evaluation specification. This document is a code-and-frozen-artifact description, not a new run. Facts not evidenced by those materials are marked **requires researcher confirmation**.

## Scope and immutable record

The final explanation evaluation contains 300 held-out cases, four grounding variants, and three generators: 3,600 preserved explanations and 10,800 generator--judge assignments. The post-recovery analysis is bound to the Stage 3 explanation SHA-256 `6fbae305fa00051d771201c72a61d1f38cc7de3f834c40dc28cf42e296040e45`; Stage 4D reported zero canonical-hash changes among originally successful rows. The compact hash inventory is `artifact_inventory.csv`; `analysis_manifest.json` records seed 42, 5,000 resamples, 95% confidence level, one extraction N/A, 84 verification N/As, and four judgment N/As.

## Knowledge base and evidence scoring

### Contents and provenance

The configured KB is `data/kb/fashion_rules_v3.csv` (v3). It has 126 rows and 126 unique `rule_id` values. It has 39 distinct source titles. Source types/counts are fashion magazine 54, retailer style guide 31, academic paper 18, museum/institution 9, book 7, fashion editorial 5, and reputable blog 2. Reliability labels are high (75) and medium (51); there are no low-labelled rules. Recommended-category counts are accessories 26, shoes 26, bottoms 25, outerwear 25, and tops 24.

Fields include `rule_id`, `rule_text`, `input_category`, `recommended_category`, scenario/style/colour/occasion/season tags, gender/body-fit/formality context, `evidence_keywords`, source type/title/author/year/URL, reliability and rationale, evidence basis, scope, limitations, manual-verification status, and notes. Retrieval text concatenates, where present, `rule_text`, `input_category`, `recommended_category`, `occasion`, `style`, and `source_title` (`evidence.py:19-37`).

The repository checks duplicate IDs (`kb_audit.py`) and stores a `manual_verification_status` field, but it does not provide a complete, frozen audit trail for source discovery/selection, rule authoring, cleaning, near-duplicate removal, conflict resolution, or manual review decisions. Those processes and the interpretation of each source's licence/permission **require researcher confirmation**. Source coverage is limited to the 39 recorded sources and five target categories; it should not be represented as comprehensive fashion knowledge or as independently validated ground truth.

### Candidate-specific retrieval and score

The evidence representation is built from query category/group/text, candidate category/group/text, user request, and target category (`evaluation/evidence_ranking.py:15-31`). The text embedding model is `sentence-transformers/all-MiniLM-L6-v2`; category-restricted cosine similarity selects `candidate_top_k=5` rules. Each selected similarity is multiplied by a source-reliability weight (high 1.00, medium 0.90, low 0.75 if present) and receives +0.05 when the query group appears in the rule's `input_category`. The candidate score is

`0.7 × max(weighted selected scores) + 0.3 × mean(weighted selected scores)`.

Accessory retrieval additionally applies the frozen lexical candidate-type compatibility filter when available. This is candidate-specific retrieval, not a claim-entailment test. There is **no frozen binary applicability or support threshold**: a retrieved rule is not automatically query-relevant, candidate-applicable, or supporting. The evidence-participation report therefore treats continuous scores as primary and labels score-threshold analyses as validation-derived sensitivity analyses, not rule-backed coverage.

## Data, preprocessing, and splits

The configured dataset is Hugging Face `Marqo/polyvore`, split `data`, with revision `null` (un-pinned); the exact upstream dataset revision, licence, download date, and raw item/outfit counts **require researcher confirmation**. The adapter uses item ID, outfit ID parsed before the final underscore, category, text, and image fields. Configured query category is dresses; target broad categories are accessories, bottoms, outerwear, shoes, and tops.

Preprocessing uses the default broad-category mapping and enables removal of suspicious text/image mismatches and low-information images (image standard-deviation threshold 8.0). Exact mapping rules and final processed row/outfit counts are in local derived parquet inputs rather than committed compact reports and **require researcher confirmation** for an external release. A usable evaluation case requires a query item and at least one same-outfit item in the requested target category. The query is the held-out query-item metadata plus its request; a positive/relevant target is every other same-outfit item in the target category. The query item is excluded.

Splits are deterministic by `query_outfit_id`: SHA-256-derived uniform value with seed 42 assigns 60% development, 20% validation, 20% test (`evaluation/splits.py:9-42`). Thus all cases from an outfit share a partition. The frozen expanded protocol reports zero outfit overlap and zero query-item/positive-item overlap between split pairs. Shared global-corpus candidate IDs (16,233 validation/test IDs) are expected corpus reuse, not positive/query leakage; near-duplicate product/image leakage beyond IDs remains a reviewer limitation.

The historical evaluation is a balanced 300-case test schedule (60/category). The outcome-blind expanded cohort retains it and adds SHA-256-ranked eligible test keys to 3,000: accessories 973, bottoms 497, outerwear 291, shoes 679, tops 560. Its key hash is `622a83499197633ce46406bc6fe066e0771e3f5b103b52906e34f1348ff733a1`; it has 1,829 query-outfit clusters. All 34,130 eligible test cases were not used because new query embedding/evidence-score work was not already cached; selection did not inspect recommendation outcomes.

## Models and structured LLM evaluation

The frozen generator roster is Ollama `llama3.2` (digest `a80c4f17acd5`), `mistral` (`6577803aa9a0`), and `gemma3:12b` (`f4031aab637d`). Judges are `qwen3:8b` (`500a1f067a9f`), `mistral`, and `gemma3:12b`; all configured temperatures are 0.0, `think=false`, and generation maximum is 240 tokens while judging maximum is 400. Timeouts are 180 seconds except `gemma3:12b` at 240 seconds. Parameter sizes are only explicit in the tag `gemma3:12b`; the other model parameter sizes, Ollama context-window settings, server build, and whether the displayed digest prefixes are full immutable digests **require researcher confirmation**.

The record identifies the three generation models and three judges, but does not separately freeze distinct extractor/verifier model assignments in a compact manifest; their exact role-to-model mapping **requires researcher confirmation** from the ignored checkpoints/run manifests. Prompts and JSON schemas are implemented in `claim_evaluation.py`: extraction returns every atomic claim (`claim_id`, text, type) without a cap; verification returns one label, rule IDs, entailment flag, and reason for every extracted claim. Allowed labels are query/locked-item supported, item-evidence supported, rule-evidence supported, unsupported, contradicted, and not-verifiable. Malformed structured output receives two retries and then one syntax/JSON repair attempt; unresolved records remain N/A. Stage 4D applies targeted recovery only to explicit N/A/failed keys, with one retry and local JSON repair where possible; it did not regenerate Stage 3 explanations.

General quality is judged on input consistency, general quality, clarity, specificity, hallucination risk, and evidence misuse. Cross-model judgments (judge family different from generator family) are primary; all judges including self-family judgments are sensitivity analysis. The 35-word constraint is calculated separately (`word_count`, `over_35_words`) and never causes text replacement.

## Recommendation protocol

The fixed target corpus has 67,524 items: accessories 29,463, shoes 12,623, tops 10,892, bottoms 9,209, outerwear 5,158. Each controlled pool contains all same-outfit positives plus up to 99 deterministic sampled negatives from other outfits, excluding the query item and query outfit; it is therefore **sampled controlled-pool ranking, not full-catalogue retrieval**. Historical test pools average 100.21 items.

Text baseline embeddings use normalized `sentence-transformers/all-MiniLM-L6-v2`; image/text embeddings use normalized `openai/clip-vit-base-patch32`. Exact embedding dimensionalities **require researcher confirmation** from frozen arrays/model manifests. Fused CLIP uses the validation-selected `image_weight=0.40` and `text_weight=0.60` recorded in `validation/fusion_tuning/selected_fusion.json` and `fuse_embeddings`. Evidence reranking min--max normalizes pool-local CLIP and evidence scores and sorts `0.75 × normalized CLIP + 0.25 × normalized evidence`. This operating point was selected on validation by `evidence_in_loop_pareto_v2`; 1.00/0.00 is the accuracy-oriented reference.

Target embeddings are cached; the historical 300 query embeddings and candidate evidence scores were cached, while the added 2,700 had deterministic query embeddings/evidence scores materialized once and hash-bound. Ranking metrics are HR@1/5/10 (any positive in top k), NDCG@1/5/10 (discounted binary relevance normalized by ideal DCG), and MRR (reciprocal first-positive rank), as implemented in `evaluation/ranking.py`. Reports provide micro, category, unweighted category-macro, historical-300, new-2,700, and combined-3,000 results.

Paired recommendation comparisons resample query outfits with all their cases retained: 5,000 replicates, seed 42, 95% percentile CIs. Two-sided bootstrap p-values use the smaller tail; zero finite estimates are displayed as `< 1/5000`. The configuration lists Holm and Benjamini--Hochberg methods, but the exact adjustment family applied to each published table **requires researcher confirmation** from the corresponding statistics artifact. Evidence participation uses the same cluster bootstrap and continuous score results; it cannot recover candidate-applicable rule counts because those labels/identities were not saved for expanded recommendations.

## Explanation-evaluation protocol

For each of 300 cases, the locked recommendation is explained by `no_rag`, `item_rag`, `rule_rag`, and `hybrid_rag` using each of three generators: 3,600 explanations. `no_rag` sees neither retrieved item nor rule evidence; `item_rag` sees item evidence; `rule_rag` sees rule evidence; `hybrid_rag` sees both. Query context, request, and locked recommendation remain available under the frozen prompt construction.

Claim extraction is a separate LLM call; verification is a separate entailment-style LLM call. The union-pool audit confirms all four variants were verified against the identical full reference pool: query metadata, user request, locked item metadata, retrieved item evidence, and retrieved rule evidence. Two flags record only what evidence was shown during generation. Across all 900 case--generator groups, all five common source fields matched across variants; this is not inferred from summary results.

Extraction/verification N/A records are excluded from applicable claim-rate denominators and reported separately; N/A is never converted to unsupported or support. The post-recovery clustered bootstrap resamples 300 test-case clusters, retaining all generator/variant/claim records, with 5,000 replicates, seed 42, and 95% percentile intervals. The final claim bootstrap report contains estimates, paired contrasts, and p-values.

## Environment, release, and exact reproduction inputs

The repository requires Python >=3.11,<3.13, `uv`, and the `uv.lock` environment; the CUDA extra pins torch/torchaudio/torchvision 2.11.0. The operating-system version, GPU model/driver, CUDA runtime used for the frozen run, resolved package versions, and exact hardware throughput **require researcher confirmation** from the uncommitted run environment/manifests. The project licence is MIT (`LICENSE`); third-party dataset/model/KB-source licences and redistribution permissions **require researcher confirmation**.

Core commands and order are in `docs/reproducibility.md`: validate/doctor; validation fusion and reranking selection; final freeze/generation/extraction/verification/judging; pre-recovery analysis; targeted Stage 4D recovery; one post-recovery analysis. Required artifacts are `configs/final_eval_v2.yaml`, `uv.lock`, processed items/targets, KB CSV, target/query embedding caches, validation selection JSON/CSVs, Stage 3 explanation table, Stage 4 checkpoints/tables, and the listed manifests/inventories. Generated datasets, embeddings, model caches, and large checkpoints are intentionally ignored by Git and must be restored at their manifest paths.

## Result-to-artifact map

| Reported result or protocol claim | Frozen source artifact | Reproduction/implementation file |
|---|---|---|
| Final claim support and N/A rates | `POST_RECOVERY_CLAIM_BOOTSTRAP.md`, `claim_support_summary.csv` | `evaluation/post_recovery_claim_bootstrap.py`, `scripts/run_post_recovery_claim_bootstrap.py` |
| Cross-model and all-judge quality | `general_quality_primary_cross_model.csv`, `general_quality_sensitivity_all_judges.csv` | `evaluation/final_judging.py`, `evaluation/stage45_v2.py` |
| 3,600 explanation integrity and Stage 4D outcomes | `STAGE4D_HANDOFF.md`, `artifact_inventory.csv`, Stage4D completion manifest | `evaluation/stage4d_v2.py` |
| Common union verification packet | `UNION_VERIFICATION_POOL_AUDIT.md` | `evaluation/claim_evaluation.py`, `evaluation/stage45_v2.py` |
| Historical/expanded recommendation metrics | `EXPANDED_RECOMMENDATION_AND_EVIDENCE_ANALYSIS.md`, expanded `run_manifest.json` | `scripts/run_expanded_recommendation_eval.py` |
| Expanded cohort construction/leakage | `FROZEN_EXPANDED_PROTOCOL.md`, `frozen_protocol.json` | `evaluation/controlled.py`, `evaluation/splits.py`, `evaluation/ranking.py` |
| Evidence participation and limitations | `EVIDENCE_PARTICIPATION_ANALYSIS.md`, `evidence_participation/completion_manifest.json` | `scripts/analyze_evidence_participation.py`, `evaluation/evidence_ranking.py` |

## Researcher confirmation required

- Pin and disclose the Polyvore dataset revision, licence, acquisition date, raw/processed counts, and exact broad-category mapping.
- Supply a curation log for source selection, authoring, de-duplication, conflicts, manual audits, and all third-party permissions.
- Record full model digests, parameter sizes, role assignments for extractor/verifier, context windows, Ollama/server versions, and hardware/CUDA details.
- Publish resolved package/environment manifests and the frozen model/embedding-array dimensions.
- State the exact multiple-comparison adjustment family for each released comparison table and whether all source materials may be redistributed.

## Exact LLM evaluation assignments, prompts, and schemas

### Role-to-model assignments

These assignments are recovered from the post-recovery checkpoints and CLI wiring, rather than inferred from the generic configuration.

| Role/stage | Model ID recorded in checkpoint | Model/tag and size | Settings recorded in model ID/config | Evidence |
|---|---|---|---|---|
| Stage 4A atomic-claim extractor | `qwen3:8b@500a1f067a9f:temperature=0.0:max_tokens=400:think=False` | Ollama `qwen3:8b` (8B stated by tag), digest prefix `500a1f067a9f` | temperature 0.0; max tokens 400; `think=false`; timeout 180 s | all `extraction_checkpoint.jsonl` records; `cli.py:1359-1373`; `robustness.yaml` first `judges` entry |
| Stage 4B verifier | same qwen3:8b model ID | same | same | all `verification_checkpoint.jsonl` records; `cli.py:1383-1400` |
| Stage 4C general judges | qwen3:8b above; `mistral@6577803aa9a0:temperature=0.0:max_tokens=400:think=False`; `gemma3:12b@f4031aab637d:temperature=0.0:max_tokens=400:think=False` | Ollama qwen3:8b (8B), Mistral (size not recorded), Gemma 3 12B; longest available digest prefixes shown | temperature 0.0; max tokens 400; `think=false`; timeouts 180, 180, and 240 s respectively | 10,800 judgment checkpoint records; `cli.py:1408-1422`; `robustness.yaml` |
| Stage 4D extractor/verifier recovery | same qwen3:8b role assignment | same | same configured generator object; recovery itself permits one retry | `cli.py:1449-1472`; `stage4d_v2.py:198-271`; recovery completion manifest |
| Stage 4D general-judging recovery | judge selected by saved `judge_model`, among the same three | same | same configured judge objects; recovery permits one retry | `stage4d_v2.py:292-315`; recovery audit/checkpoint |

There is no separate extractor/verifier ensemble: one qwen3:8b model is used for each role. The general-quality design is a three-judge ensemble: every generated explanation receives one assignment from each listed judge, yielding 3,600 x 3 = 10,800 assignments. The original explanation generators are `llama3.2@a80c4f17acd5:temperature=0.0:max_tokens=240:think=False`, `mistral@6577803aa9a0:temperature=0.0:max_tokens=240:think=False`, and `gemma3:12b@f4031aab637d:temperature=0.0:max_tokens=240:think=False` (`robustness.yaml`; frozen explanation table).

The repository records no Ollama `seed`, context-window/`num_ctx`, top-p, top-k, repeat penalty, or other sampling options for these calls; those settings are **requires researcher confirmation**. No separate system-message field is passed: each routine constructs one runtime prompt string and sends it to `cached_generate`. The displayed digest values are the longest values stored by the repository (12 hexadecimal characters); full immutable image digests are **requires researcher confirmation**.

### Claim extraction method

This is LLM-based atomic-claim extraction, not named-entity recognition. An atomic claim is an individual fashion/styling proposition in the explanation; the model is told to extract every such claim, not assess truth, not omit claims, and impose no claim-count cap. Types are `body_fit`, `colour`, `comfort`, `formality`, `item_type`, `material`, `occasion`, `other`, `season`, `styling_relation`, `trend`, and `visual_match`.

System prompt: **none (single concatenated prompt).** User/runtime template (`claim_evaluation.py:77-96`; SHA-256 of the parameterized template with an empty explanation: `7553868751c2cc2ec1e821084833e3f1bd43dc217e7f5b0536dbab9f421b3e9d`):

```text
Extract every atomic fashion or styling claim from the explanation.
Do not assess whether claims are true. Do not omit claims and do not cap their number.
Allowed claim types: body_fit, colour, comfort, formality, item_type, material, occasion, other, season, styling_relation, trend, visual_match.

Explanation:
{generated_explanation}

Return one JSON object only:
{"claims":[{"claim_id":"C1","claim":"...","claim_type":"styling_relation"}]}
```

The required output schema is `{"claims": [claim...]}`, where each claim has a non-empty string text, a supplied or generated `claim_id` (default sequential `C1`, `C2`, ...), and a claim type. JSON is parsed directly or from the first braced object; `claims` must be a list, empty/malformed responses fail, and invalid types are normalized to `other`. The original Stage 4A policy makes up to three initial attempts (two retries) on JSON decode failure, then one same-model JSON-syntax-only repair. A repair must preserve claims/types rather than reassess them. Empty or still-invalid extraction becomes explicit extraction failure/N/A, never zero claims with perfect support (`stage45_v2.py:141-224, 436-576`). Stage 4D first attempts local syntax recovery of the saved malformed response, then makes at most two calls (initial plus one retry) with the same extraction prompt/model (`stage4d_v2.py:38-110, 198-231`).

Representative rendered frozen prompt (example only; `V2_TEST_0000_accessories`, `no_rag`, llama3.2 generation):

```text
Extract every atomic fashion or styling claim from the explanation.
Do not assess whether claims are true. Do not omit claims and do not cap their number.
Allowed claim types: body_fit, colour, comfort, formality, item_type, material, occasion, other, season, styling_relation, trend, visual_match.

Explanation:
The Clutches | shein sheinside black rivet floral clutch bag complements the open toe floral block heels by adding a touch of edgy sophistication with its metallic accents, grounding the overall feminine look.

Return one JSON object only:
{"claims":[{"claim_id":"C1","claim":"...","claim_type":"styling_relation"}]}
```

### Claim verification method

Verification is LLM entailment-style classification, not word matching or cosine-similarity thresholding. System prompt: **none (single concatenated prompt).** The implementation is `claim_evaluation.py:170-222`; it serializes the full `ReferencePacket` with JSON indentation and expects:

```text
Verify every extracted claim using only the structured reference packet.
Generation-evidence flags identify what the generator actually saw; other packet fields are
evaluation references only. Assign exactly one support label per claim.
Allowed labels: contradicted, not_verifiable, supported_by_item_evidence,
supported_by_query_or_locked_item, supported_by_rule_evidence, unsupported.
For rule support, include supporting_rule_ids. If a cited rule is present, state whether it
entails the claim in citation_entails_claim (true, false, or null).

Structured reference packet:
{JSON object below}

Claims:
{JSON list below}

Return one JSON object only:
{"verifications":[{"claim_id":"C1","support_label":"unsupported",
"supporting_rule_ids":[],"citation_entails_claim":null,"brief_reason":"..."}]}
```

The parameterized template SHA-256 (sentinel claims and a sentinel `ReferencePacket`, computed without a model call) is `0d582cb50721665a8afbf009ce915082111aeb97f33181380e6f75085a82d9e4`; runtime prompt fingerprints are not separately stored. The full union packet for every variant has `query_item_metadata`, `user_request`, `locked_recommended_item_metadata`, `retrieved_item_evidence`, `retrieved_rule_evidence`, `item_evidence_shown_to_generator`, and `rule_evidence_shown_to_generator`. The latter two flags distinguish generation availability only: item flag true for item/hybrid RAG, rule flag true for rule/hybrid RAG. They never remove either retrieved-evidence field from verification. The union-pool audit confirms common fields matched in all 900 case--generator groups across variants.

The schema requires `verifications` to be a list containing exactly one record for every extracted claim ID, with no duplicate/unknown IDs and an allowed label. Each record carries rule-ID list, `citation_entails_claim` (true/false/null), and a string reason. `parse_claim_verifications` rejects a missing/extra ID or invalid label. Claim extraction failure skips verification and becomes N/A; irreparable verification is N/A rather than unsupported. Stage 4B retries twice, attempts a JSON-only repair only after JSON decode failures, and otherwise retains failure (`stage45_v2.py:225-317, 620-751`). Stage 4D uses local parse/repair first and at most one retry with the same qwen model/prompt.

Representative rendered frozen verification prompt (the same case and extracted claims; dynamic packet text is exactly the frozen record used to obtain checkpoint packet hash `02b08bcf6607cc977a1d4746f46216ec2fe97011a425c4eb8a6893695a0826aa`):

```text
Verify every extracted claim using only the structured reference packet.
Generation-evidence flags identify what the generator actually saw; other packet fields are evaluation references only. Assign exactly one support label per claim.
Allowed labels: contradicted, not_verifiable, supported_by_item_evidence, supported_by_query_or_locked_item, supported_by_rule_evidence, unsupported.
For rule support, include supporting_rule_ids. If a cited rule is present, state whether it entails the claim in citation_entails_claim (true, false, or null).

Structured reference packet:
{"query_item_metadata":"open toe floral block heels black multi","user_request":"recommend accessories that complete this outfit","locked_recommended_item_metadata":"Clutches | shein sheinside black rivet floral clutch bag","retrieved_item_evidence":"Clutches | shein sheinside black rivet floral clutch bag","retrieved_rule_evidence":"[frozen retrieved-rule text; preserved in explanations.csv rule_evidence_text]","item_evidence_shown_to_generator":false,"rule_evidence_shown_to_generator":false}

Claims:
[{"claim_id":"C1","claim":"complements the open toe floral block heels","claim_type":"styling_relation"},{"claim_id":"C2","claim":"adding a touch of edgy sophistication","claim_type":"trend"},{"claim_id":"C3","claim":"metallic accents","claim_type":"visual_match"},{"claim_id":"C4","claim":"grounding the overall feminine look","claim_type":"styling_relation"}]

Return one JSON object only:
{"verifications":[{"claim_id":"C1","support_label":"unsupported","supporting_rule_ids":[],"citation_entails_claim":null,"brief_reason":"..."}]}
```

The bracketed rule-text notation above identifies the exact frozen field and avoids duplicating a long source passage in this specification; it is not a substituted runtime value. The code, frozen row, and saved packet hash are the authoritative complete rendering provenance.

### General-quality judging

System prompt: **none (single concatenated prompt).** `anchored_general_judge_prompt` (`final_judging.py:29-59`; parameterized template SHA-256 with empty row `e65724e8b3d93c5cef32ff35858a8524efcf865a7e6021845e688928c053363b`) gives all judges the following anchored rubric. Every dimension is integer 1--5 and **higher is better**, including `hallucination_risk` (5 means no unsupported claims) and `evidence_misuse` (5 means no misuse).

```text
Evaluate this explanation using the anchored dimensions below.
Do not infer unavailable attributes and do not identify the experimental variant.

input_consistency: 1=contradicts query/request/locked item; 3=mostly consistent but vague or partially unsupported; 5=fully consistent with query/request/locked item.
general_quality: 1=unclear or unhelpful; 3=understandable but generic; 5=clear, concise and useful-looking.
clarity: 1=hard to understand; 3=mostly understandable; 5=unambiguous and concise.
specificity: 1=generic; 3=some relevant detail; 5=specific to the supplied items/request.
hallucination_risk: 1=many unsupported/invented fashion claims; 3=some unsupported claims; 5=no unsupported fashion claims found.
evidence_misuse: 1=wrong rule/item/category or serious misapplication; 3=minor evidence-use issues; 5=no evidence misuse found. Use 5 when no external evidence was supplied and none is claimed or misused; this is not an external-grounding score.

Query item: {query_text}
User request: {user_request}
Locked recommended item: {recommended_text}
Generation evidence actually supplied:
{generation_evidence_text or 'No external generation evidence.'}
Explanation:
{generated_explanation}

Return one JSON object with integer keys input_consistency, general_quality, clarity, specificity, hallucination_risk, evidence_misuse, plus brief_reason.
```

`_parse_scores` requires every named key, converts to integer, and rejects values outside 1--5. The judge assignment matrix is the Cartesian product of generation model (llama3.2, Mistral, Gemma 3 12B) and judge model (qwen3:8b, Mistral, Gemma 3 12B): all nine cells are retained. A row is primary when normalized model families differ; same-family Mistral--Mistral and Gemma--Gemma assignments are excluded from cross-model primary summaries but retained, together with all rows, in all-judge sensitivity summaries (`final_judging.py:14-27, 88-134`).

Stage 4C tries a judge response up to three times, then one JSON-syntax-only repair only when the last error is JSON decoding; other errors become N/A. Stage 4D first tries local repair and then at most one retry with the saved judge model. The frozen outcome has four unresolved judgment N/As (`stage45_v2.py:321-412, 779-878`; Stage4D completion manifest).

Representative rendered frozen judge prompt (example only; same no-RAG case, qwen3:8b assignment):

```text
Evaluate this explanation using the anchored dimensions below.
Do not infer unavailable attributes and do not identify the experimental variant.

input_consistency: 1=contradicts query/request/locked item; 3=mostly consistent but vague or partially unsupported; 5=fully consistent with query/request/locked item.
general_quality: 1=unclear or unhelpful; 3=understandable but generic; 5=clear, concise and useful-looking.
clarity: 1=hard to understand; 3=mostly understandable; 5=unambiguous and concise.
specificity: 1=generic; 3=some relevant detail; 5=specific to the supplied items/request.
hallucination_risk: 1=many unsupported/invented fashion claims; 3=some unsupported claims; 5=no unsupported fashion claims found.
evidence_misuse: 1=wrong rule/item/category or serious misapplication; 3=minor evidence-use issues; 5=no evidence misuse found. Use 5 when no external evidence was supplied and none is claimed or misused; this is not an external-grounding score.

Query item: open toe floral block heels black multi
User request: recommend accessories that complete this outfit
Locked recommended item: Clutches | shein sheinside black rivet floral clutch bag
Generation evidence actually supplied:
No external generation evidence.
Explanation:
The Clutches | shein sheinside black rivet floral clutch bag complements the open toe floral block heels by adding a touch of edgy sophistication with its metallic accents, grounding the overall feminine look.

Return one JSON object with integer keys input_consistency, general_quality, clarity, specificity, hallucination_risk, evidence_misuse, plus brief_reason.
```

Original Stage 4 and Stage 4D use the same extraction, verification, and judging prompt constructors, schemas, qwen role assignment, and judge roster. The material difference is recovery control flow: Stage 4 has two retries then one model JSON repair; Stage 4D first tries deterministic local parse/closure repair and only then makes at most one retry, targeting explicit N/A keys. It preserves successful rows byte-for-byte at record level and recorded 384 recovery LLM calls, 17 local verification repairs, and 252 local judgment repairs.

## Reviewer-facing limitations

- This is one Polyvore-style dataset with sampled same-category controlled pools; it does not establish full-catalogue retrieval performance or external generalization.
- Shared corpus IDs are not positive/query leakage, but near-duplicate products/images across partitions were not fully audited.
- KB coverage/provenance and manual curation are only partially documented; retrieved/scoring rules do not prove applicability or correctness.
- LLM claim extraction, verification, and judging remain model-mediated measurements with residual N/A rows; cross-model judging mitigates but does not remove correlated-model bias.
- The expanded recommendation result supports stronger frozen-system evidence participation alongside lower rank-sensitive metrics; it is not evidence that reranking improves accuracy or objective fashion correctness.
