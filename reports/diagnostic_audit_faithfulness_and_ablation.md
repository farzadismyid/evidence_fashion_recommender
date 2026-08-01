# Diagnostic audit: faithfulness and ablations

Scope: static audit of existing code, configs, reports, logs, and artifacts only. No model, retrieval, embedding, indexing, judging, or robustness workflow was rerun. The confirmatory artifacts are the frozen test outputs under `outputs/robustness/final_study/`; hybrid prompt and reranker selection used validation data.

## 1. Judge prompts

### Final robustness explanation judge

- **Path/function:** `src/evidence_fashion_recommender/evaluation/robustness.py`, `_robustness_judge_prompt`; executed by `judge_robustness_study`.
- **Exact scored dimensions:** `faithfulness_to_available_information`, `usefulness_to_user`, `specificity`, `style_appropriateness`, `grounding_safety` (integer 1–5: very poor, poor, mixed, strong, excellent).
- **Exact claim labels:** `supported`, `unsupported`, `not_verifiable`; at most three atomic claims, each under 12 words. `_parse_judge` in `evaluation/study.py` normalizes malformed labels and records compliance.
- **Context:** query text, user request, locked recommended-item metadata, variant-specific evidence, and explanation. It is not “all available context” globally; it is only the context exposed for that variant.

Exact final prompt template:

```text
Independently evaluate this fashion explanation using only the supplied
information. The generator is a different model. Do not identify the experimental method.
Assess each dimension separately before choosing its score. Do not default all dimensions
to the same value.

Query: {query_text}
Request: {user_request}
Locked recommended item: {recommended_text}
Available evidence:
{evidence}
Explanation:
{generated_explanation}

Use integer scores: 1=very poor, 2=poor, 3=mixed, 4=strong, 5=excellent.
Required score keys: faithfulness_to_available_information, usefulness_to_user,
specificity, style_appropriateness, grounding_safety.
Also return a claims array with at most three atomic claims. Each claim object must have
the keys claim and support; support must be supported, unsupported, or not_verifiable.
Keep each claim under 12 words and brief_reason under 20 words.
Return one compact JSON object only, with the five score keys, claims, and brief_reason.
```

For No-RAG, `evidence_for_variant` returns empty text and the fallback inserted as “Available evidence” is the query, request, and locked item again. Thus No-RAG receives **no retrieved item or rule evidence**, but it is judged for consistency with its inputs.

### Validation/earlier overall judge

- **Path/function:** `src/evidence_fashion_recommender/evaluation/study.py`, `build_judge_prompt`; executed by `judge_explanations`, including hybrid validation selection.
- **Dimensions:** the same five 1–5 dimensions. It requests no claim array.
- **Context:** query category/description, request, locked item category/description, and variant evidence. Its exact No-RAG fallback says: “No external evidence was provided. Judge consistency using only the query, request, and locked recommended item.”

### Claim-level support judging

There is **no separate claim-support prompt, judge, or invocation in the final study**. `judge_robustness_study` asks the same overall judge, in the same response and against the same context, to extract up to three claims and assign the three labels above. `claim_support_rate = supported / claim_count`; if the returned claim list is empty, the code assigns `1.0`. Therefore claim support is judge-extracted, capped at three claims, context-dependent, and not an exhaustive external-evidence entailment score.

## 2. Context by variant

The generator prompt is built by `generation.py::build_explanation_prompt`; judge evidence is selected by `evaluation/study.py::evidence_for_variant`. The stored final cases contain five catalogue lines and five rule lines per case; the selected Hybrid prompt exposes both, with the catalogue section first.

| variant | generation context | judge context | claim-support context | item evidence? | rule evidence? | citations? |
|---|---|---|---|---:|---:|---|
| no_rag | Query, request, locked item, safety/length instructions; no retrieved section | Query, request, locked item; fallback repeats these as available evidence | Same combined-judge context | No | No | Not required; normally absent |
| item_rag | Common inputs + retrieved catalogue context (5 item lines) | Common inputs + `item_evidence_text` | Same | Yes | No | Not required; normally absent |
| rule_rag | Common inputs + retrieved expert rules (5 in final cases) | Common inputs + `rule_evidence_text` | Same | No | Yes | At least one supplied rule ID required |
| hybrid_rag | Common inputs + 5 catalogue lines then 5 rules under selected spec | Common inputs + `hybrid_evidence_text` | Same | Yes | Yes | At least one supplied rule ID required |

No-RAG neither receives nor is judged against external retrieved evidence. Query text, request, and locked-item metadata are still grounding inputs, so it is not context-free.

## 3. What “faithfulness” currently means

The prompt defines only “faithfulness_to_available_information” and supplies no dimension-specific rubric. Its reference set changes by variant:

- No-RAG: **input/context consistency** with query, request, and locked item.
- Item-RAG: input consistency plus **item-metadata grounding**.
- Rule-RAG: input consistency plus **rule grounding**.
- Hybrid-RAG: input consistency plus both external evidence types.

Consequently it is not a common external-evidence-faithfulness metric across all four variants. The defensible umbrella name is **contextual faithfulness** (or **input-grounded consistency** for No-RAG). External item/rule grounding should be reported separately and No-RAG should not be ranked on that construct.

## 4. Why No-RAG can win the overall judge score

`JUDGE_DIMENSIONS` contains the five dimensions above. Both `judge_robustness_study` and `judge_explanations` calculate a simple unweighted arithmetic mean: each dimension has weight **0.20**. Claim support, label compliance, citations, evidence overlap, and deterministic unsupported claims are not included.

Thus usefulness, specificity, and style alone contribute 60%; adding the loosely specified faithfulness dimension makes 80%. “Grounding safety” is only a label in the prompt—no external-evidence rubric or mandatory penalty is defined—and contributes 20%. Absence of external evidence is not penalized. No-RAG can therefore score highly by being fluent, useful, specific, stylistically appropriate, and consistent with the input, while avoiding the extra constraints and possible conflicts introduced by retrieved evidence. The stored all-model summary (`outputs/robustness/final_report/cross_model_variant_summary.csv`) indeed gives No-RAG the highest contextual-faithfulness and overall proxy scores, but this is not an external-grounding victory.

## 5. Hybrid-RAG validation selection

- **Final name:** `hybrid_w55_r5_candidate_first`.
- **Budget/count/order:** 55 words; 5 rule-evidence entries; 5 item-evidence lines (fixed by the case artifact, not a tuned factor); candidate/catalogue evidence before rules.
- **Selection partition:** validation only; 300 generated explanations per candidate.
- **Objective:** `automatic_selection_score + overall_judge_score / 5`. The automatic term is `evidence_overlap + 0.25*citation_presence - 0.25*unsupported_claims - 0.50*substitution_rate - 0.25*length_violation_rate`.
- **Winning validation result:** automatic `0.682218`, judge overall `4.288/5`, combined `1.539818`.
- **Evidence:** `outputs/robustness/hybrid_ablations/validation_selection.csv`; frozen record `outputs/robustness/final_study/selected_hybrid_spec.json`.

Tuned by a one-factor grid (`evaluation/robustness.py::one_factor_hybrid_specs`): word limits 55/75/100, rule counts 1/2/3/5, and candidate-first/rules-first order. The grid is centred on manually fixed 75 words, 3 rules, candidate-first, with the prior 55-word/5-rule candidate-first configuration explicitly added. Fixed rather than tuned: one generator and one judge for validation selection, temperature/token settings, five catalogue entries, prompt wording, candidate locking, metric weights, and the one-factor (not full-factorial) design. An eight-row ablation table **does exist** at the path above.

## 6. Ablation inventory

| ablation | exists? | code path | result path | partition | safe to cite? |
|---|---|---|---|---|---|
| Text-only vs CLIP image-only vs CLIP text-only vs fused multimodal | Partial/archived only | `archive/notebook_original.ipynb`; modular config exposes modes in `config.py`, but controlled modular evaluation compares only text baseline and fused CLIP | Results are embedded only in the archived notebook; no standalone modular four-way result artifact found | Unclear diagnostic sample | **No** as confirmatory accuracy evidence; cite only as exploratory diagnostic |
| Pure CLIP vs evidence reranking | Yes | `cli.py::command_tune_reranking` and held-out ranking path; scoring in controlled evaluation | `outputs/robustness/heldout_ranking/test_summary.csv` | Frozen test | **Yes**, with both HR@10 gain and NDCG@10 decline reported |
| Reranking-weight ablation | Yes | `cli.py::command_tune_reranking` | `outputs/robustness/reranking_tuning/validation_summary.csv`, `selected_weight.json` | Validation | **Yes** for model selection; final effectiveness must use held-out test |
| No-RAG vs Item-RAG vs Rule-RAG vs Hybrid-RAG | Yes | `generation.py`, `evaluation/robustness.py`, `evaluation/study.py` | `outputs/robustness/final_study/{automatic_summary,judge_results,judge_summary,statistical_tests}.csv` | Frozen test | **Yes**, subject to the construct caveat in this audit |
| Multi-generator / multi-judge explanation ablation | Crossed robustness study exists, not a clean causal ablation | `evaluation/robustness.py::generate_robustness_study`, `judge_robustness_study`, aggregation/reporting code | `outputs/robustness/final_study/{cross_model_judge_summary,judge_summary,judge_ensemble_summary,judge_agreement}.csv` | Frozen test | **Yes** as model-sensitivity/robustness evidence; **no** as proof of human quality or a single “best” generator/judge |

The ordinary `outputs/final/recommendation_metrics.csv` contains text baseline, fused CLIP, and evidence-reranked rows, not the requested four-way modality ablation.

## 7. Metric classification

| metric | primary classification | audit qualification |
|---|---|---|
| Overall judge score | General judge quality | Equal-weight mixture of five constructs; not a grounding score |
| Judge faithfulness | Input consistency | Variant-conditioned contextual faithfulness; becomes item/rule grounding only when that evidence is supplied |
| Judge usefulness | General judge quality | Model-judge proxy, not human usefulness |
| Grounding safety | Unclear | Name only; prompt supplies no operational definition or mandatory evidence penalty |
| Claim support | Input consistency | Same combined judge/context; external grounding only for RAG rows; empty claim list scores 1.0 |
| Label compliance | Unclear | Output-schema adherence, not explanation quality or grounding |
| Unsupported claims | Input consistency | Lexicon terms absent from variant evidence; for RAG it is evidence-relative, while No-RAG is tested against empty external evidence |
| Evidence overlap | External evidence grounding | Lexical overlap with the evidence exposed to the variant; decomposing item vs rule overlap would be more interpretable |
| Citation correctness | Rule-grounding | Checks cited rule IDs are a subset of retrieved IDs; vacuously 1 when no citations occur |
| Rule retrieval P@K / HR@K / MRR | Rule-grounding | KB-proxy or multi-judge-consensus relevance, depending on the result table |
| Counterfactual false-match rate | Rule-grounding | Tests category-incompatible rule matches under counterfactual targets |
| Substitution detector | Input consistency | Checks whether an explanation substitutes an item type outside query + locked-item types |

## 8. Final interpretation

- **Can we claim No-RAG is more faithful than Rule-RAG?** No—not for external evidence. Only that it scored higher on the current variant-conditioned **contextual-faithfulness proxy** in the aggregate artifact.
- **Can we claim No-RAG is better overall?** Only narrowly: it had the highest equal-weight model-judge composite in this frozen systematic study. Do not generalize this to objective system quality, external grounding, or human preference.
- **Can we claim Rule-RAG is stronger for external fashion-rule grounding?** Yes, cautiously, from rule-conditioned deterministic, citation, retrieval, and claim-support evidence; state the exact metric and proxy nature rather than treating the overall judge as proof.
- **Can we claim Hybrid-RAG is strongest for evidence overlap?** Yes. It has the highest pooled evidence-overlap result in the stored test outputs; note that it has more/either-type evidence available and the metric is lexical.
- **Can we claim human usefulness/trust?** No. No human study was run; human review is explicitly future work.
- **No-RAG in external-evidence comparisons:** use **N/A**, with No-RAG retained as the baseline for general-quality and input-consistency comparisons. Do not use zero as if it were observed failed grounding: its zero overlap is structural because no evidence was supplied. A separate “external-evidence availability/coverage = 0” may be shown descriptively.

## Recommended next actions

### No rerun needed

- Rename judge faithfulness to “contextual faithfulness”; label No-RAG “input-grounded consistency.”
- Separate general-quality, external-grounding, rule-grounding, and item-grounding tables; mark No-RAG external grounding N/A.
- Disclose the equal 0.20 judge weights, unspecified grounding-safety rubric, combined claim/overall judge, empty-claim behavior, and model-judge status.

### Recompute existing outputs

- From existing CSVs, report grounding metrics only over eligible RAG variants and add confidence intervals/effect sizes.
- Split pooled evidence overlap into item-evidence and rule-evidence overlap where the stored evidence fields permit it; report citation precision only where citations are required.
- Produce sensitivity composites from existing judge columns (general quality versus grounding-oriented columns) without replacing the prespecified primary result.

### Rerun judging only

- For a fair external-faithfulness comparison, judge every already-generated explanation against the same external reference packet, with explicit dimension rubrics and an evidence-absence outcome.
- Use a separate exhaustive claim-entailment prompt/judge; do not default empty claim extraction to perfect support.

### Rerun generation + judging

- Needed for a controlled evidence-content/order study, a full-factorial Hybrid prompt ablation, item-evidence-count tuning, or a clean generator ablation under newly standardized prompts.
- Needed to make the four-way text/image/CLIP-text/fused retrieval comparison feed controlled downstream explanations.

### Rerun full robustness

- Only needed if retrieval/index construction, embedding modalities, KB content, reranker architecture, data splits, or robustness protocol changes. Human usefulness/trust evaluation remains a separate future-work study, not something model-judge reruns can establish.
