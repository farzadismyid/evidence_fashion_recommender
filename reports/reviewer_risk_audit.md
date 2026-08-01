# Reviewer risk audit

Scope: static reviewer-style audit of the current source, configuration, documentation,
reports, and frozen robustness artifacts. No model, retrieval, generation, judging, or
other expensive workflow was run. “Requires rerun” distinguishes no rerun, analysis-only
recomputation from stored outputs, judge-only rerun, generation-plus-judge rerun, and full
retrieval/robustness rerun.

## Executive assessment

The repository supports a careful **systematic, non-human comparison** on an outfit-disjoint
test partition. It supports claims that the four prompting conditions behave differently,
that Rule-RAG/Hybrid-RAG reduce the project's deterministic lexical unsupported-term count,
that Hybrid-RAG has the greatest lexical evidence overlap, and that a validation-selected
0.90/0.10 reranker trades slightly higher test HitRate@10 for slightly lower NDCG@10 than
pure CLIP. It does **not** yet support strong claims of factual faithfulness, hallucination
reduction, human usefulness, general recommendation superiority, or optimal multimodal
fusion. The principal threats are construct validity, weak/unequal baselines, model-judge
dependence and low agreement, case-level inference despite outfit clustering, incomplete
ablations, and incomplete provenance from a dirty working tree.

The minimum defensible path is an **analysis-only correction plus a judge-only rerun**.
A stronger paper needs a limited **generation-plus-judge rerun** for fair evidence budgets
and the missing item-count baseline. A full 3,600-explanation robustness rerun is not
necessary unless retrieval/fusion/reranking changes are made part of the final claims.

## 1. Main paper claims currently supported

1. **Severity: low. Affected claim:** the final study is a systematic, non-human robustness
   comparison with 3 generators, 4 variants, 300 cases, and 3 judges. **Evidence:**
   `outputs/robustness/final_study/explanations.csv`, `judge_results.csv`, and
   `FINAL_STUDY_MANIFEST.json` contain 3,600 explanations and 10,800 judgments; error tables
   are empty. **Recommended fix:** state “model-based systematic evaluation,” never human
   preference or user benefit. **Rerun:** no.

2. **Severity: low. Affected claim:** development, validation, and test query outfits are
   disjoint. **Evidence:** `evaluation/splits.py`; the frozen schedules contain zero outfit
   overlap. **Recommended fix:** report both case and unique-outfit counts and publish the
   split manifest beside final outputs. **Rerun:** no; analysis/reporting only.

3. **Severity: low. Affected claim:** the selected reranker produces a test-set trade-off,
   not uniform improvement. **Evidence:** `outputs/robustness/final_report/before_after_ranking.csv`
   and `FINAL_ROBUSTNESS_REPORT.md`: HR@10 0.260 to 0.270, NDCG@10 0.1377 to 0.1348.
   **Recommended fix:** preserve the trade-off wording and add uncertainty. **Rerun:** no;
   recomputation from per-case ranking outputs if retained.

4. **Severity: medium. Affected claim:** Rule-RAG and Hybrid-RAG reduce the deterministic
   unsupported-term metric and Hybrid-RAG has the highest lexical evidence overlap.
   **Evidence:** `final_study/automatic_evaluation.csv` and `automatic_summary.csv`.
   **Recommended fix:** name the measures exactly (“lexicon-based unsupported-term count”
   and “lexical overlap”), not hallucination or factual faithfulness. **Rerun:** no.

5. **Severity: medium. Affected claim:** model sensitivity is substantial and judge
   agreement is modest. **Evidence:** `judge_summary.csv`, `cross_model_judge_summary.csv`,
   and `judge_agreement.csv` (reported mean Spearman 0.209). **Recommended fix:** make this
   a principal result rather than a footnote. **Rerun:** no.

## 2. Main paper claims currently weak or unsafe

1. **Severity: high. Affected claim:** RAG “improves faithfulness” or “reduces
   hallucination.” **Path:** `evaluation/study.py:59-65,136-171`,
   `evaluation/robustness.py:133-200`. The reference evidence changes by variant; the
   deterministic detector is a small term lexicon; and model claim extraction is capped at
   three claims. **Recommended fix:** restrict claims to the named proxies, or rerun a common-
   reference, exhaustive claim-entailment evaluation. **Rerun:** judge-only for the latter.

2. **Severity: high. Affected claim:** No-RAG is “more faithful” or “better overall.”
   **Path:** `final_report/cross_model_variant_summary.csv`. No-RAG is judged against only
   query/request/locked-item context, whereas RAG variants are judged against additional
   evidence and constraints. **Recommended fix:** call this contextual consistency/general
   model-judge quality; mark external grounding N/A for No-RAG. **Rerun:** no for wording;
   judge-only for a fair common-reference comparison.

3. **Severity: high. Affected claim:** evidence reranking improves recommendation quality.
   It improves only test HR@10 and worsens NDCG@10; no inferential comparison is reported.
   **Path:** `FINAL_ROBUSTNESS_REPORT.md`. **Recommended fix:** describe a descriptive
   recall/rank-position trade-off and add paired, outfit-clustered uncertainty. **Rerun:**
   analysis-only if per-case results are available.

4. **Severity: high. Affected claim:** the counterfactual false-match rate validates rule
   selectivity. **Path:** `evaluation/verification.py::counterfactual_category_test`.
   Retrieval is already restricted to the actual target category, while the test merely
   asks whether returned rules' category equals a rotated target; zero is therefore close
   to guaranteed by construction. **Recommended fix:** actually rerun rule retrieval under
   counterfactual queries/categories and judge applicability. **Rerun:** retrieval/evaluation
   rerun, not explanation generation.

5. **Severity: medium. Affected claim:** findings generalize to fashion recommendation.
   The task uses same-outfit co-occurrence as relevance, dresses/other selected query groups,
   five target categories, and sampled same-category negatives from one Polyvore dataset.
   **Path:** `evaluation/controlled.py`, `evaluation/ranking.py`, `configs/robustness.yaml`.
   **Recommended fix:** delimit the claim to Polyvore same-outfit retrieval. **Rerun:** no;
   broader datasets/user evaluation would be new work.

## 3. Metric-definition risks

1. **Severity: high. Affected claim:** “unsupported claims” measures factual hallucinations.
   **Path:** `evaluation/study.py::_unsupported_terms`. It counts occurrences from a fixed
   vocabulary absent verbatim from evidence, missing paraphrases and most factual claims and
   penalizing potentially valid input-supported statements. **Recommended fix:** rename it,
   publish the vocabulary, and use atomic claim entailment as the primary hallucination
   metric. **Rerun:** judge-only for entailment.

2. **Severity: high. Affected claim:** claim-support rate is reliable. **Path:**
   `evaluation/robustness.py:154-200`. Judges extract at most three claims, extraction and
   verification occur in one response, and zero extracted claims score 1.0. **Recommended
   fix:** separate exhaustive claim extraction from verification; score empty extraction as
   missing/error, not perfect support; report coverage and claims per explanation. **Rerun:**
   judge-only using stored explanations.

3. **Severity: high. Affected claim:** “citation correctness = 1” demonstrates grounding.
   **Path:** `evaluation/study.py:162-166`. Correctness is subset membership only; no-citation
   No-RAG/Item-RAG rows receive vacuous correctness and precision of 1. It does not test
   whether cited rules support the associated prose. **Recommended fix:** condition citation
   precision on citation presence and add citation-to-claim entailment. **Rerun:** analysis-only
   for conditional rates; judge-only for entailment.

4. **Severity: high. Affected claim:** evidence overlap is comparable across variants.
   The available evidence volume and type differ structurally; No-RAG must score zero and
   Hybrid receives both evidence sets. **Path:** `evaluation/study.py:59-65,171`.
   **Recommended fix:** report item- and rule-overlap separately, normalize for evidence
   opportunity, and mark No-RAG external overlap N/A. **Rerun:** mostly analysis-only from
   stored evidence fields.

5. **Severity: medium. Affected claim:** overall judge score is a validated primary outcome.
   It is an unweighted mean of five ordinal dimensions, including nearly constant grounding
   safety, without psychometric validation. **Path:** `evaluation/robustness.py:183-185`.
   **Recommended fix:** predeclare primary dimensions, report them separately, and provide
   sensitivity analyses for alternative composites. **Rerun:** analysis-only.

6. **Severity: medium. Affected claim:** Precision@K is directly interpretable across cases.
   Cases can contain multiple same-outfit positives, while candidate-set size is approximately
   100 and precision divides by fixed K. **Path:** `evaluation/ranking.py`. **Recommended fix:**
   report the positive-count distribution and macro results stratified by number of positives.
   **Rerun:** analysis-only.

7. **Severity: medium. Affected claim:** “evidence coverage = 1” means useful evidence was
   retrieved. It only means at least one rule ID exists. **Path:** `evaluation/study.py:400-405`.
   **Recommended fix:** call it retrieval non-emptiness and foreground judged relevance.
   **Rerun:** no.

## 4. Faithfulness vs hallucination risks

1. **Severity: high. Affected claim:** a common faithfulness construct is compared across
   all variants. No-RAG uses input consistency; Item-RAG uses catalogue metadata; Rule/Hybrid
   use rules. **Path:** `evaluation/study.py::evidence_for_variant` and judge prompt.
   **Recommended fix:** separate input consistency, item grounding, rule grounding, and
   external-evidence faithfulness. **Rerun:** no for re-labelling; judge-only for common-reference
   scores.

2. **Severity: high. Affected claim:** the final study includes independent claim-level
   verification. Although `evaluation/verification.py::verify_claims` exists, the final CLI
   obtains claims inside the same general judge response; no separate verification artifact
   appears in `FINAL_STUDY_MANIFEST.json`. **Recommended fix:** correct the report wording or
   run the separate verifier with an exhaustive rubric. **Rerun:** judge-only.

3. **Severity: medium. Affected claim:** “not_verifiable” is conservatively handled without
   bias. Noncanonical labels are normalized, overall compliance is only about 0.76, and
   malformed descriptive responses may be systematically model/variant dependent.
   **Path:** `evaluation/study.py::_parse_judge`, `final_report/judge_summary.csv`.
   **Recommended fix:** report normalization rules and sensitivity restricted to compliant
   responses. **Rerun:** analysis-only.

## 5. No-RAG comparison risks

1. **Severity: high. Affected claim:** No-RAG is a matched control. It has less context,
   no citation obligation, and no risk of contradicting retrieved evidence, yet shares the
   same concise prompt and locked recommendation. **Path:** `generation.py`.
   **Recommended fix:** treat it as an input-only baseline, not a comparable external-grounding
   system; include common-reference judging. **Rerun:** judge-only.

2. **Severity: medium. Affected claim:** zero evidence overlap/zero citations indicate poor
   No-RAG grounding. These are structural outcomes because no evidence is supplied.
   **Recommended fix:** use N/A, not zero, in comparative grounding tables. **Rerun:** no.

3. **Severity: medium. Affected claim:** No-RAG's overall-score advantage is substantive.
   Differences are small, judge dependent, and the composite rewards fluency/style while
   evidence-constrained variants carry extra obligations. **Recommended fix:** report effect
   sizes, clustered intervals, and dimension-level results without winner language.
   **Rerun:** analysis-only.

## 6. Item-RAG baseline risks

1. **Severity: high. Affected claim:** Item-RAG is a strong item-grounded baseline. Its
   evidence is five descriptions of other retrieved candidates, excluding the recommended
   item, rather than factual metadata or visual attributes of the locked recommendation.
   **Path:** `evaluation/study_cases.py`. **Recommended fix:** define the intended baseline;
   preferably give it verified metadata/captions for the query and locked item. **Rerun:**
   generation-plus-judge, and retrieval/captioning if new evidence must be built.

2. **Severity: high. Affected claim:** Item-RAG and Rule-RAG differ only by evidence source.
   Rule-RAG must cite an ID; Item-RAG has no attribution mechanism, while its context may
   describe alternative products that the prompt orders the model to ignore. **Path:**
   `generation.py`, `study_cases.py`. **Recommended fix:** match evidence counts/tokens and
   attribution requirements, or explicitly describe the comparison as non-isolated.
   **Rerun:** generation-plus-judge for a causal comparison.

3. **Severity: medium. Affected claim:** five item lines are an appropriate evidence budget.
   Item count was fixed and never validated. **Path:** `study_cases.py`; hybrid audit outputs.
   **Recommended fix:** add item-count ablation and report token counts. **Rerun:** limited
   validation generation-plus-judge.

## 7. Rule-RAG / Hybrid-RAG interpretation risks

1. **Severity: high. Affected claim:** lower unsupported-term counts prove better factual
   rule grounding. Rule texts contain more of the detector vocabulary and give lexical
   overlap more opportunities, mechanically favoring Rule/Hybrid. **Recommended fix:** use
   claim-to-rule entailment and evidence-opportunity-adjusted analyses. **Rerun:** judge-only.

2. **Severity: medium. Affected claim:** citations are semantically correct. Current checks
   validate only that IDs were retrieved. **Recommended fix:** evaluate whether each cited
   rule entails the adjacent claim and whether important claims are cited. **Rerun:** judge-only
   or human annotation.

3. **Severity: medium. Affected claim:** Hybrid's higher overlap means better grounding.
   Hybrid has strictly more evidence and the selected configuration uses five rules plus
   five item lines. **Recommended fix:** compare under matched evidence-token budgets and
   decompose overlap. **Rerun:** generation-plus-judge for a causal claim.

4. **Severity: medium. Affected claim:** KB evidence is externally authoritative. The KB is
   curated, mixes books, editorials and retailer guides, contains medium-reliability sources,
   and its authoring/verification protocol and inter-annotator agreement are not established
   in the audited paper materials. **Path:** `data/kb/fashion_rules_v3.csv`, `kb_audit.py`.
   **Recommended fix:** publish provenance, inclusion criteria, verification status, source
   coverage, and limitations. **Rerun:** no unless KB content changes.

## 8. Recommendation retrieval ablation risks

1. **Severity: high. Affected claim:** multimodality itself causes the gain over MiniLM.
   The modular controlled results do not include CLIP image-only and CLIP text-only arms;
   MiniLM versus fused CLIP confounds modality, model family, query template, and embedding
   space. **Path:** `evaluation/controlled.py`; `reports/modular_baseline_results.md`.
   **Recommended fix:** add CLIP image-only, CLIP text-only, and fused CLIP on identical cases.
   **Rerun:** retrieval/ranking only; no explanations required.

2. **Severity: high. Affected claim:** performance represents corpus retrieval. Evaluation
   ranks same-outfit positives among 99 randomly sampled same-category negatives, not the
   full catalogue, and lacks hard-negative sensitivity. **Path:** `evaluation/ranking.py`.
   **Recommended fix:** state “sampled candidate-set ranking”; add full-corpus or repeated
   negative-sampling/hard-negative evaluation. **Rerun:** retrieval/ranking.

3. **Severity: medium. Affected claim:** same-outfit membership equals compatibility.
   Co-occurrence is an implicit proxy with false negatives (compatible items in other
   outfits) and merchandising/stylist bias. **Recommended fix:** frame as benchmark relevance
   and add human compatibility assessment as future work. **Rerun:** not essential.

## 9. CLIP image/text fusion-weight risks

1. **Severity: high. Affected claim:** 0.60 image / 0.40 text is optimal or validated. It is
   inherited from `configs/default.yaml` and no modular fusion-weight selection artifact or
   runner exists. **Path:** `models/multimodal.py`, `configs/default.yaml`.
   **Recommended fix:** call it a fixed heuristic or run validation-only fusion ablation.
   **Rerun:** retrieval/ranking only.

2. **Severity: medium. Affected claim:** weighted embedding fusion has a clear linear meaning.
   Image/text embeddings are individually combined then normalized, so downstream cosine
   effects are nonlinear in the declared weights. **Path:** `models/multimodal.py::fuse`.
   **Recommended fix:** describe the exact operation and test score-level versus embedding-
   level fusion if theoretically important. **Rerun:** retrieval/ranking.

3. **Severity: medium. Affected claim:** text contributes genuine item information rather
   than target/request leakage. The CLIP query text explicitly includes request and target
   category, while the query image encodes only the source item. **Path:**
   `evaluation/controlled.py::build_clip_query_text`. **Recommended fix:** add query-template
   ablations and explain the asymmetry. **Rerun:** retrieval/ranking.

## 10. Evidence-reranking weight-selection risks

1. **Severity: high. Affected claim:** 0.90/0.10 is robustly optimal. Validation NDCG@10 is
   0.13873 versus 0.13840 for pure CLIP, a tiny difference selected without uncertainty;
   the grid is coarse and excludes CLIP weights below 0.65. **Path:**
   `reranking_tuning/validation_summary.csv`. **Recommended fix:** report the flatness and a
   one-standard-error/practical-equivalence rule; include confidence intervals. **Rerun:**
   analysis-only for existing grid; retrieval/ranking for a broader grid.

2. **Severity: medium. Affected claim:** only one hyperparameter was selected. Evidence
   scoring also contains fixed, unvalidated reliability weights, a +0.05 input bonus, and a
   0.7 max/0.3 mean aggregation. **Path:** `evaluation/evidence_ranking.py`.
   **Recommended fix:** disclose these as fixed heuristics and ablate them if claiming the
   reranker design is generally effective. **Rerun:** retrieval/ranking.

3. **Severity: medium. Affected claim:** score mixing is stable. Per-candidate-set min-max
   normalization makes weights query- and pool-dependent; constant score vectors become all
   ones. **Path:** `reranking.py`. **Recommended fix:** document this and test rank/standardized
   normalization sensitivity. **Rerun:** retrieval/ranking.

4. **Severity: medium. Affected claim:** test comparison uses the recorded selected artifact.
   The held-out command hard-codes `[0.9, 1.0]` rather than reading `selected_weight.json`.
   **Path:** `cli.py:1034-1036`. **Recommended fix:** load and verify the frozen selection
   artifact and fingerprint it. **Rerun:** no if values are verified identical; code fix before
   future reruns.

## 11. Hybrid-RAG word budget / rule count / item count / order risks

1. **Severity: high. Affected claim:** Hybrid variables were comprehensively tuned. The
   eight configurations form a one-factor grid around 75 words/3 rules/candidate-first plus
   an explicitly injected legacy 55-word/5-rule setting, not the 3×4×2 factorial grid.
   **Path:** `evaluation/robustness.py::one_factor_hybrid_specs` and
   `hybrid_ablations/validation_selection.csv`. **Recommended fix:** describe it as economical
   one-factor screening, not optimization; use a factorial/sensible subset if interactions
   matter. **Rerun:** limited validation generation-plus-judge.

2. **Severity: high. Affected claim:** Hybrid evidence quantity was tuned fairly. Rule count
   varies, but item count remains fixed at five and evidence token totals are not matched.
   **Recommended fix:** add item counts and matched total-token budgets. **Rerun:** limited
   validation generation-plus-judge.

3. **Severity: medium. Affected claim:** selected prompt is generator/judge robust. Selection
   uses only the first generator (Llama 3.2) and first judge (Qwen3 8B). **Path:**
   `cli.py:923-948`. **Recommended fix:** call it single-pair selection or test ranking
   stability across models using a small validation subset. **Rerun:** generation-plus-judge.

4. **Severity: medium. Affected claim:** selection objective is principled/pre-registered.
   Its weights are hand-coded, mixes bounded and count metrics, includes lexical overlap and
   a judge composite, and no sensitivity analysis is shown. **Path:**
   `evaluation/robustness.py:248-255`, `cli.py:946-950`. **Recommended fix:** justify weights,
   publish component ranges, and perform rank-sensitivity analysis. **Rerun:** analysis-only.

5. **Severity: medium. Affected claim:** order effects are evaluated consistently. The
   generation prompt can be candidate-first, but the stored `hybrid_evidence_text` shown to
   judges is rules-first regardless of selected generation order. **Path:** `study_cases.py`
   and `evaluation/study.py::evidence_for_variant`. **Recommended fix:** clarify that judges
   assess evidence content, not prompt-order fidelity; preserve exact generation evidence
   ordering if order is a substantive claim. **Rerun:** judge-only or generation-plus-judge.

## 12. LLM-judge risks and judge-agreement risks

1. **Severity: high. Affected claim:** quality rankings are model-independent. Mean pairwise
   Spearman is 0.209, some dimension pairs are near zero, and mean absolute differences are
   material. **Path:** `final_report/judge_agreement.csv`. **Recommended fix:** avoid a single
   winner; show per-judge effects and disagreement intervals. **Rerun:** no.

2. **Severity: high. Affected claim:** judges are independent. Mistral and Gemma act as both
   generators and judges; the prompt falsely says “The generator is a different model” even
   for self-judge rows. Self-judges are excluded only from one cross-model summary, not the
   primary ensemble/statistical tables. **Path:** `evaluation/robustness.py:138-140,187-190`,
   `cli.py:1136-1170`. **Recommended fix:** make cross-model-only analysis primary and remove
   the false instruction; use a judge family not used for generation if possible. **Rerun:**
   analysis-only for cross-model tables; judge-only for genuinely independent judges.

3. **Severity: high. Affected claim:** judge scores operationalize the named dimensions.
   Only endpoint labels (very poor to excellent) are supplied; there are no dimension-specific
   anchors, grounding-safety is nearly constant, and order/calibration effects are untested.
   **Recommended fix:** add explicit rubrics, randomized/blinded presentation, calibration
   items, and score-distribution checks. **Rerun:** judge-only.

4. **Severity: medium. Affected claim:** judges are blind to variant. The prompt includes
   variant-specific evidence structure, rule IDs, and citation patterns, making condition
   inference easy despite not naming the method. **Recommended fix:** acknowledge partial,
   not full, blinding; standardize formatting where possible. **Rerun:** judge-only for a
   redesigned blind protocol.

5. **Severity: medium. Affected claim:** deterministic temperature zero guarantees exact
   reproducibility. Local inference can still vary by Ollama/runtime/hardware; only shortened
   displayed digests and environment metadata are preserved. **Recommended fix:** archive
   full model manifests, Ollama version, parameters, prompt hashes, and raw outputs.
   **Rerun:** no for current outputs.

## 13. Statistical-testing and multiple-comparison risks

1. **Severity: high. Affected claim:** confidence intervals and p-values account for sampling
   dependence. Test data contain 300 cases but only 201 unique outfits; bootstrap resamples
   cases independently rather than outfits. Multiple target cases can share a query outfit.
   **Path:** `evaluation/statistics.py`, frozen schedules. **Recommended fix:** cluster bootstrap
   by outfit (and preserve paired variants/generators/judges). **Rerun:** analysis-only.

2. **Severity: high. Affected claim:** multiplicity is fully controlled. Holm/BH correction is
   applied separately inside each generator×judge block across 48 comparisons, while the final
   artifact contains 432 tests; no global or clearly predeclared family is justified.
   **Path:** `cli.py:1156-1170`, `statistical_tests.csv`. **Recommended fix:** predeclare a small
   primary family and correct within defensible families; label all else exploratory.
   **Rerun:** analysis-only.

3. **Severity: high. Affected claim:** bootstrap p-values are valid hypothesis tests. The
   implementation computes the fraction of ordinary bootstrap means crossing zero rather
   than a null-centered/bootstrap-t or permutation distribution; zeros can also result from
   finite Monte Carlo resolution. **Path:** `evaluation/statistics.py::paired_bootstrap`.
   **Recommended fix:** emphasize confidence intervals or use paired randomization/permutation
   tests and report minimum attainable p-values. **Rerun:** analysis-only.

4. **Severity: medium. Affected claim:** 10,800 judgments provide n=10,800 independent
   evidence. The real sampling unit is at most the outfit/case; generator and judge repeats
   are crossed repeated measures. **Recommended fix:** use hierarchical/mixed-effects or
   cluster-bootstrap analysis and never present judgment count as independent sample size.
   **Rerun:** analysis-only.

5. **Severity: medium. Affected claim:** statistically significant means practically
   important. Many score differences are a few hundredths on a 1–5 scale. **Recommended fix:**
   report standardized/ordinal effect sizes and smallest effects of interest. **Rerun:**
   analysis-only.

## 14. Data leakage / split risks

1. **Severity: medium. Affected claim:** the split eliminates all leakage. It eliminates query-
   outfit overlap, but target catalogue items, embedding/index construction, KB development,
   and possible duplicate/near-duplicate products across outfits are not partition-audited.
   **Path:** `evaluation/splits.py`, `cli.py`. **Recommended fix:** state exactly what is split;
   audit item IDs, image/text near duplicates, and provenance across partitions. **Rerun:**
   analysis/data audit; full rerun only if duplicates are removed.

2. **Severity: medium. Affected claim:** validation/test schedules are independent samples.
   Cases are drawn from an oversampled deterministic prefix/pool built before hashing, and
   the same seed is reused within each category. This is reproducible but not demonstrated
   representative of the full dataset. **Path:** `cli.py:833-862`, `balanced_sample`.
   **Recommended fix:** publish pool counts/coverage and compare category/item distributions.
   **Rerun:** analysis-only.

3. **Severity: medium. Affected claim:** no researcher leakage occurred through KB/prompt
   iteration. The protocol states selection discipline, but no chronology or immutable
   preregistration proves when KB v3, metric weights, prompt wording, and grids were frozen.
   **Recommended fix:** provide commit/timestamp provenance and label post hoc design choices.
   **Rerun:** no.

## 15. Reproducibility risks

1. **Severity: high. Affected claim:** final results map to an exact source state. The frozen
   environment records Git commit `fdf769...` with `dirty: true`; dirty changes are not
   captured by a patch in the final manifest. **Path:** `final_report/environment_doctor.json`.
   **Recommended fix:** archive the exact source diff/config alongside outputs and rerun future
   finals only from a clean tagged commit. **Rerun:** no if the historical diff can be recovered;
   otherwise exact reproduction is impossible and must be disclosed.

2. **Severity: high. Affected claim:** final artifacts are self-contained. Final study/report
   manifests hash output files but do not include resolved config, source diff, schedules,
   input case hash, KB hash, dependency lock hash, or cache provenance. **Recommended fix:**
   extend manifests and deposit all inputs. **Rerun:** no for packaging existing artifacts.

3. **Severity: medium. Affected claim:** reported results can be regenerated from README.
   The convenience script runs environment sync, tests, schedule/case rebuilding, tuning and
   the entire study, but does not explicitly bind the held-out stage to `selected_weight.json`.
   **Path:** `scripts/reproduce_robustness.ps1`, `cli.py`. **Recommended fix:** document staged
   commands and immutable input/output hashes. **Rerun:** no.

4. **Severity: medium. Affected claim:** model pinning is complete. Model revisions are pinned
   for Hugging Face and digest prefixes for Ollama, but tokenizer/runtime/Ollama version and
   decoding implementation are not all in the final report. **Recommended fix:** archive full
   manifests and container/environment lock. **Rerun:** no.

5. **Severity: low. Affected claim:** repository state is clean and complete. Current Git
   status reports a deleted `requirements.txt`; even if unrelated, reviewers need a clean
   release snapshot. **Recommended fix:** resolve repository hygiene before release without
   altering frozen artifacts. **Rerun:** no.

## 16. Tables/figures that should be added before paper submission

1. **Severity: high. Affected claim:** recommendation gains are attributable to multimodal
   fusion. Add one table with MiniLM, CLIP image-only, CLIP text-only, fused CLIP, and reranked
   fused CLIP on identical validation/test cases, with outfit-clustered CIs. **Path:** current
   gap in `final_report/before_after_ranking.csv`. **Rerun:** retrieval/ranking.

2. **Severity: high. Affected claim:** explanation quality/grounding trade-off is robust.
   Add a per-variant table separating general quality, input consistency, item grounding,
   rule grounding, citation entailment, claim coverage, and hallucination rate; No-RAG
   external grounding should be N/A. **Rerun:** partly analysis-only, partly judge-only.

3. **Severity: high. Affected claim:** uncertainty is valid. Add paired effect sizes and
   outfit-clustered CIs for primary endpoints; show generator×judge forest plots and explicitly
   distinguish self versus cross-model judgments. **Rerun:** analysis-only.

4. **Severity: medium. Affected claim:** Hybrid selection is transparent. Add the full eight-row
   validation table, component scores, uncertainty, evidence token counts, and a diagram of
   fixed versus tuned variables. **Rerun:** analysis-only.

5. **Severity: medium. Affected claim:** rule retrieval is credible. Add category-stratified
   consensus P@K/HitRate/MRR, relevance prevalence, agreement, KB source/reliability distribution,
   and genuine counterfactual retrieval results. **Rerun:** analysis-only except genuine
   counterfactual retrieval.

6. **Severity: medium. Affected claim:** judge robustness is adequate. Add score histograms,
   ceiling rates (especially grounding safety), pairwise agreement heatmaps, compliance/empty-
   claim rates, and sensitivity excluding self-judges/noncompliant responses. **Rerun:**
   analysis-only.

## 17. Minimum fixes required before final rerun

1. **Severity: high. Affected claim:** factual faithfulness/hallucination. Freeze explicit
   constructs and rubrics; separate exhaustive atomic claim extraction from common-reference
   entailment; eliminate perfect scores for empty claim lists. **Rerun:** judge-only.

2. **Severity: high. Affected claim:** inferential validity. Replace case bootstrap with an
   outfit-clustered paired procedure, define primary outcomes/comparison families, and use a
   valid paired test. **Rerun:** analysis-only.

3. **Severity: high. Affected claim:** independent judging. Make cross-model-only outcomes
   primary, correct the false “different model” prompt statement, add anchored rubrics, and
   report self-judge sensitivity. **Rerun:** judge-only for revised prompts.

4. **Severity: high. Affected claim:** reproducibility. Freeze a clean commit, resolved config,
   input/schedule/KB hashes, complete model/runtime manifest, and selection artifacts before
   any new final evaluation. **Rerun:** no by itself.

5. **Severity: high. Affected claim:** baseline fairness. At minimum relabel Item-RAG accurately,
   match evidence budgets in reporting, and stop treating structural zero/N/A metrics as
   comparative outcomes. **Rerun:** no for honest wording; generation-plus-judge for a stronger
   Item-RAG baseline.

6. **Severity: medium. Affected claim:** counterfactual specificity. Remove the current
   tautological zero-rate claim or implement a true counterfactual retrieval test. **Rerun:**
   retrieval/evaluation only.

## 18. Optional fixes that improve paper strength but are not essential

1. **Severity: medium. Affected claim:** optimal Hybrid configuration. Run a compact factorial
   validation design including item count and matched token budgets across two generator/judge
   pairs. **Rerun:** limited generation-plus-judge.

2. **Severity: medium. Affected claim:** optimal fusion/reranking. Run dense validation curves
   for fusion and reranking, with stability/error bars rather than winner-only reporting.
   **Rerun:** retrieval/ranking only.

3. **Severity: medium. Affected claim:** user usefulness and trust. Add blinded human evaluation
   with rater training, inter-rater reliability, and power analysis. **Rerun:** new human study,
   not model robustness.

4. **Severity: low. Affected claim:** generality. Add another fashion dataset or temporally/
   stylistically distinct catalogue. **Rerun:** new full retrieval evaluation.

5. **Severity: low. Affected claim:** explanation robustness. Test prompt paraphrases and
   decoding seeds/temperatures. **Rerun:** generation-plus-judge.

## 19. Things that should be framed as limitations rather than fixed now

1. **Severity: medium. Affected claim:** human usefulness/preference/trust. No human evaluation
   exists. **Recommended fix:** explicitly leave these unanswered. **Rerun:** no.

2. **Severity: medium. Affected claim:** universal fashion correctness. Fashion rules are
   contextual, culturally contingent, and sourced from mixed editorial/retail/book material.
   **Recommended fix:** frame the KB as curated styling evidence, not objective fashion laws.
   **Rerun:** no.

3. **Severity: medium. Affected claim:** general recommendation compatibility. Same-outfit
   co-occurrence and sampled negatives are proxy labels. **Recommended fix:** bound conclusions
   to the benchmark. **Rerun:** no.

4. **Severity: low. Affected claim:** robustness across all LLMs. Three local model families
   improve sensitivity analysis but do not establish universal model independence.
   **Recommended fix:** state model scope. **Rerun:** no.

5. **Severity: low. Affected claim:** causal isolation of RAG. Prompt/context/citation burdens
   differ between variants. **Recommended fix:** describe the study as system-variant comparison
   unless matched interventions are added. **Rerun:** no.

## 20. Final recommendation

**Recommended category: judge-only rerun, preceded by analysis-only corrections.**

Before submission, recompute inference with outfit clustering, define primary comparison
families, make cross-model-only results primary, condition citation metrics correctly, and
reframe structural/lexical proxies. Then rerun judging over the already generated 3,600
explanations using: (1) explicit anchored rubrics; (2) a common external reference packet;
(3) separate exhaustive claim extraction and entailment; (4) no perfect empty-claim default;
and (5) judges that are not also generators where feasible. This addresses the highest-risk
faithfulness, hallucination, and judge-validity claims without regenerating explanations.

If the paper intends to claim fair causal advantages of Item-RAG/Rule-RAG/Hybrid-RAG, add a
**limited generation-plus-judge rerun** with a meaningful locked-item evidence baseline,
matched evidence-token budgets, and item-count variation. If it intends to claim optimized
multimodal retrieval or reranking superiority, add retrieval-only fusion/full-corpus/hard-
negative studies. Reserve a **full robustness rerun** for changes to retrieval outputs, KB
content, reranker architecture, split membership, or the final generation prompts—not for
the analysis and judge corrections alone.

