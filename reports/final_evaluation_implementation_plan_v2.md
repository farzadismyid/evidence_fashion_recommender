# Final Evaluation v2 implementation plan

Status: planning only. No implementation, configuration change, retrieval run, generation,
or judging is authorized by this document. Implementation begins only after review and
approval.

## Scope and governing decisions

The v2 evaluation will use these roots consistently:

```text
outputs/final_eval_v2/
reports/final_eval_v2/
```

Historical outputs under `outputs/robustness/` remain immutable. The implementation will
not introduce `outputs/final_evaluation/`.

The workflow is divided into independently resumable stages. Stage 1 ends with a mandatory
decision gate that determines whether old explanations can be judged as legacy generations
or whether all four variants must be regenerated under frozen v2 settings.

## Staged implementation

### Stage 0A — Implement, test, and run development-only smoke checks

Implement configuration, modality evaluation, selection artifacts, Hybrid grid support,
claim extraction/verification, anchored general-quality judging, clustered statistics,
reporting, and provenance checks. Exercise code with unit tests and development-only smoke
fixtures. Do not use validation or test results for implementation debugging, and do not
perform the final reproducibility freeze in Stage 0A.

**Rerun category:** no rerun for implementation and tests; development-only smoke checks.

### Stage 1 — Retrieval validation and frozen test evaluation

On identical candidate sets and partitions:

1. Evaluate MiniLM text-only, CLIP image-only, CLIP text-only, and fused CLIP.
2. Select the fusion weight on validation using NDCG@10, then HR@10, then MRR/positive
   rank, then the declared balanced-setting tie-breaker.
3. Preserve the existing evidence-reranking validation method, but make test evaluation
   load the frozen selected artifact instead of hard-coding 0.90/0.10.
4. Evaluate the selected fused retriever and selected reranker once on test.
5. Materialize v2 locked recommendations and their retrieved item/rule evidence packets.

**Rerun category:** retrieval-only run.

#### Mandatory Stage 1 decision gate

Compare v2 results with the old locked cases used by
`outputs/robustness/final_study/explanations.csv` at the same case IDs. Produce a versioned,
machine-readable comparison containing at least:

- locked recommended item ID equality and changed-item rate;
- recommendation-rank changes;
- retrieved rule-ID set/order equality and changed-rule-packet rate;
- retrieved item-evidence ID/text equality and changed-item-packet rate;
- hashes of complete old and v2 generation reference packets;
- category-stratified and overall change rates.

“Material change” must be frozen before opening the comparison. The conservative default is:

- any changed locked recommendation is material for that case;
- any changed evidence packet is material for evidence-grounding claims;
- any nontrivial aggregate change, or a change capable of altering a primary comparison,
  triggers clean regeneration.

Decision outcomes:

**Gate A — Material changes found:** regenerate all four variants for all final test cases
under frozen v2 recommendations, evidence packets, prompts, and evidence-budget policy.
Partial regeneration of only Hybrid rows is not the preferred paper design.

**Gate B — No material changes:** existing explanations may be reused only as
`legacy-generation/v2-judging`. Reports and manifests must state that generation came from
the legacy frozen prompt/recommendation packets and that only judging/statistics/reporting
were redesigned. They must not be presented as clean v2 generations.

If prompt wording, evidence-budget policy, or final locked recommendations change for any
substantive comparison, default to Gate A and regenerate all four variants.

### Stage 2 — Hybrid validation and selection

Hybrid validation must consume the **validation evidence packets produced by the selected
Stage 1 fusion and reranking configuration**. The selected Stage 1 artifact hashes and the
validation packet hash must be inputs to every Hybrid prompt/cache fingerprint and selection
record. Old v1 evidence packets must not be used for v2 Hybrid selection. If they are retained
for diagnostic comparison, label the run `legacy_v1_packets_only`; it is ineligible to select
the final v2 Hybrid configuration.

The full target grid is:

```text
word_budget: 35, 55, 75
rule_count: 3, 5
item_count: 0, 2, 5
evidence_order: rules_first, item_first
total: 36 configurations
```

Configurations with `item_count=0` are labelled **rule-only candidates within the Hybrid
validation search**. They are not eligible to be named final Hybrid-RAG because final
Hybrid-RAG requires item/catalogue evidence plus expert-rule evidence. They may inform a
Rule-RAG-style budget comparison or show that item evidence is unhelpful.

Selection uses the frozen priority order:

1. minimise hallucinated fashion-claim rate;
2. maximise rule-supported styling-claim rate;
3. minimise evidence misuse/candidate substitution;
4. compare rule and item evidence overlap separately;
5. require acceptable general clarity;
6. prefer shorter valid explanations within a one-percentage-point practical tie.

No arbitrary weighted composite is the primary selector.

#### Optional staged runtime plan

The default economical route is:

1. Run all 36 configurations on a frozen, category-balanced validation subset.
2. Select 4–6 finalists using only the priority rule and practical-tie threshold.
3. Run those finalists on all validation cases.
4. Add a second generator/judge model pair only when finalists remain practically tied or
   their ordering appears model-dependent.
5. Freeze the selected eligible Hybrid configuration before test generation.

The subset size, balancing rule, finalist count, and tie-handling procedure must be frozen
before the screening run. Screening results cannot be treated as final validation estimates.

The alternative is the full direct run: 36 configurations × 300 validation cases = 10,800
generations for one generator, before judge calls. A second full generator doubles generation
calls; one judge adds 10,800 calls. This cost must be stated explicitly in logs and reports.

**Rerun category:** validation generation+judge run.

### Final freeze stage — after Stage 1 and Stage 2 selections

The immutable final freeze occurs only after validation has selected fusion, reranking, and
Hybrid settings. Before any final test explanation generation or v2 judging, freeze:

- clean source commit and run tag;
- resolved v2 configuration and its hash;
- development/validation/test schedule and case hashes;
- KB and dependency-lock hashes;
- model names, revisions/digests, runtime versions, and decoding settings;
- prompt templates and prompt-schema hashes;
- primary endpoints and comparison families;
- selected fusion artifact and validation-input hash;
- selected reranking artifact and validation-input hash;
- selected eligible Hybrid artifact, Stage 1 validation-packet hash, and Hybrid validation
  output hash;
- Stage 1 decision-gate definition and comparison artifact;
- exact final command list.

The freeze command must fail if any selection is missing, references the wrong partition,
uses legacy-only Hybrid packets, or Git is dirty.

**Rerun category:** no rerun; provenance/freeze operation after validation selections.

### Stage 3 — Final explanation generation or legacy reuse

Apply the Stage 1 gate:

- **Gate A:** generate No-RAG, Item-RAG, Rule-RAG, and Hybrid-RAG for every frozen test case
  and every selected generator using the same v2 locked recommendations and variant-specific
  evidence policy. This is a full final explanation rerun.
- **Gate B:** copy/reference immutable legacy explanations without rewriting their content,
  label every row `generation_protocol=legacy_v1` and
  `evaluation_protocol=v2`, and retain old prompt/reference hashes.

Even under Gate B, any newly generated row must not be silently mixed with legacy rows in a
primary comparison. Mixed-protocol results belong only in sensitivity analyses.

**Rerun category:** full final rerun under Gate A; no generation rerun under Gate B.

### Stage 4 — v2 claim extraction and verification

Claim extraction and verification are separate cached stages.

#### Atomic claim extraction

Extract all atomic styling/fashion claims without a three-claim cap. Record claim text,
claim type, extraction model, raw response, parser status, and explanation hash. Supported
types include item type, colour, material, occasion, formality, season, comfort, trend,
body fit, styling relation, visual match, and other.

An empty or failed extraction records:

```text
claim_extraction_failed = true
claim_count = 0
claim_support = N/A
```

It never receives perfect support.

#### Structured evaluation reference packet

Every verification row uses an explicit, versioned packet with separately identified fields:

```text
query item metadata
user request
locked recommended item metadata
retrieved item evidence, when available
retrieved rule evidence, when available
```

The packet records which evidence was available to the generator and which evidence is being
supplied only as an evaluation reference. Reports must distinguish:

- **generation evidence:** context actually shown when producing the explanation;
- **evaluation reference evidence:** context supplied to the verifier for consistent scoring.

Each claim receives exactly one primary support label:

```text
supported_by_query_or_locked_item
supported_by_item_evidence
supported_by_rule_evidence
unsupported
contradicted
not_verifiable
```

Rule/Hybrid reports also record supporting rule IDs and citation-to-claim entailment when a
citation is present. No-RAG and Item-RAG receive external rule-grounding N/A rather than zero
because expert rules were not generation evidence.

**Rerun category:** judge-only rerun over existing explanations under Gate B; part of the
full final rerun pipeline under Gate A.

### Stage 5 — Anchored general-quality and input-consistency judging

Run a separate judge prompt with dimension-specific anchors for input consistency,
general quality, clarity, specificity if retained, hallucination risk, evidence misuse, and
rule-grounded faithfulness where applicable. Rename the historical
`faithfulness_to_available_information` construct to `contextual_faithfulness`; describe it
as input-grounded consistency for No-RAG.

Primary cross-model-only analysis is defined exactly as:

```text
exclude a judgment when judge_model_family == generation_model_family
```

Model-family normalization must be explicit and tested—for example, digest, tag, or parameter
suffixes must not cause the same base family to be treated as different. The judge prompt
must not claim that the generator is different unless the row satisfies this rule.

All-judge results, including self-family judgments, are retained only as sensitivity analysis
and clearly separated from primary tables and inference.

**Rerun category:** judge-only rerun under Gate B; part of the full final rerun pipeline under
Gate A.

### Stage 6 — Statistics, reporting, and final provenance

Use outfit-clustered paired bootstrap as the primary interval method, preserving pairing
across variants, generators, and judges. Report case count and unique outfit count. Do not
treat the number of judgments as the independent sample size.

Primary explanation comparisons:

```text
Rule-RAG vs No-RAG
Hybrid-RAG vs No-RAG
Rule-RAG vs Item-RAG
Hybrid-RAG vs Rule-RAG
```

Primary recommendation comparisons:

```text
Fused CLIP vs MiniLM text-only
Fused CLIP vs CLIP image-only
Fused CLIP vs CLIP text-only
Evidence-reranked fused CLIP vs pure fused CLIP
```

Correct multiple comparisons within these two predefined families. Everything else is
exploratory. Produce separate general-quality and external-grounding tables; do not use
structural zeroes for unavailable grounding or citation metrics.

**Rerun category:** analysis-only recomputation after Stage 1 and Stages 4–5 outputs exist.

## 1. Files likely to change

| Area | Likely files | Planned change | Required run |
|---|---|---|---|
| Configuration schema | `src/evidence_fashion_recommender/config.py` | Fusion grid, Hybrid item counts, staged screening, tie threshold, primary families, model-family rules | No rerun |
| Versioned configuration | new `configs/final_eval_v2.yaml` | Freeze v2 grids, prompts, models, outputs, and selection rules | No rerun |
| CLI | `src/evidence_fashion_recommender/cli.py` | Add staged commands, artifact-bound test evaluation, decision gate, and freeze checks | No rerun |
| Modality/fusion evaluation | `evaluation/controlled.py`; possibly new `evaluation/modality.py` | Four modality arms and fusion validation on identical candidates | Retrieval-only run |
| Multimodal embeddings | `models/multimodal.py` | Expose reusable separate image/text vectors and weight-specific fusion | Retrieval-only run |
| Reranking | `evaluation/tuning.py`, `reranking.py` | Read frozen selected artifacts; remove test hard-coding | Retrieval-only run |
| Recommendation/evidence comparison | new `evaluation/protocol_gate.py` | Compare v2 locked items/evidence packets with legacy packets | Analysis-only recomputation after retrieval |
| Hybrid grid | `evaluation/robustness.py`, `generation.py`, `study_cases.py` | Full four-variable spec, item slicing, exact order, rule-only labels | Validation generation+judge run |
| Claim extraction | new `evaluation/claim_extraction.py` | Exhaustive typed extraction and failure handling | Judge-only rerun |
| Claim verification | `evaluation/verification.py` or new `evaluation/claim_verification.py` | Structured packets, source labels, contradiction and citation entailment | Judge-only rerun |
| General judging | `evaluation/robustness.py` or new `evaluation/judging.py` | Anchored rubrics and model-family exclusion | Judge-only rerun |
| Metrics | `evaluation/study.py`, `evaluation/explanations.py` | N/A semantics, separate overlaps, renamed contextual metric | Analysis-only and judge-only |
| Statistics | `evaluation/statistics.py` | Outfit-clustered pairing and primary multiplicity families | Analysis-only recomputation |
| Reporting | `robustness_reporting.py`, `reporting.py` | v2 tables, legacy/v2 labels, primary/sensitivity separation | Analysis-only recomputation |
| Provenance | `reproducibility.py`, `artifacts.py` | Clean freeze and complete input/output manifests | No rerun |
| Workflow | new `scripts/reproduce_final_eval_v2.ps1` | Resumable stages and explicit gate handling | Stage-dependent |
| Documentation | `README.md`, `docs/methodology.md`, `docs/reproducibility.md`, new v2 command documentation | Use only `outputs/final_eval_v2/` and `reports/final_eval_v2/` | No rerun |
| Tests | files under `tests/` | Split guards, family matching, N/A metrics, grid labels, clustered bootstrap, gate, manifests | No model rerun |

Historical configs, reports, and `outputs/robustness/` artifacts will not be overwritten.

## 2. New scripts and commands

Proposed commands:

```text
tune-clip-fusion
evaluate-final-retrieval-v2
compare-locked-artifacts-v2
run-hybrid-validation-v2
freeze-final-eval-v2
run-final-explanations-v2
extract-claims-v2
verify-claims-v2
judge-general-quality-v2
analyze-final-eval-v2
build-final-report-v2
```

Proposed orchestrator:

```powershell
.\scripts\reproduce_final_eval_v2.ps1 -Stage validation
.\scripts\reproduce_final_eval_v2.ps1 -Stage retrieval
.\scripts\reproduce_final_eval_v2.ps1 -Stage gate
.\scripts\reproduce_final_eval_v2.ps1 -Stage explanations
.\scripts\reproduce_final_eval_v2.ps1 -Stage judging
.\scripts\reproduce_final_eval_v2.ps1 -Stage analysis
.\scripts\reproduce_final_eval_v2.ps1 -Stage report
```

Each stage must accept explicit input/selection paths and refuse partition-incompatible
artifacts.

## 3. New output directories

```text
outputs/final_eval_v2/
├── validation/
│   ├── fusion_tuning/
│   ├── reranking_tuning/
│   └── hybrid_tuning/
│       ├── screening/
│       └── finalists/
├── retrieval/
│   ├── validation/
│   └── test/
├── decision_gate/
├── freeze/
├── test/
│   ├── locked_cases/
│   ├── explanations/
│   ├── claim_extraction/
│   ├── claim_verification/
│   └── general_quality_judging/
├── analysis/
│   ├── primary/
│   ├── exploratory/
│   └── sensitivity/
└── manifests/

reports/final_eval_v2/
├── tables/
├── figures/
├── FINAL_EVALUATION_REPORT.md
└── FINAL_REPORT_MANIFEST.json
```

No v2 command will default to `outputs/final_evaluation/`.

## 4. Validation/test separation

1. Preserve outfit-level partitions and frozen schedule hashes.
2. Use development only for code/prompt smoke tests.
3. Select fusion, reranking, and Hybrid settings only on validation.
4. Freeze primary endpoints, comparison families, tie rules, prompts, and artifacts before
   test evaluation.
5. Require selection artifact hashes in every test command.
6. Reject validation inputs in test commands and test inputs in tuning commands.
7. Run the Stage 1 comparison gate without using test outcome quality to retune settings.
8. Treat the gate as a protocol-consistency decision, not model selection.
9. Do not regenerate or rejudge selectively based on favorable test outcomes.
10. Label all post-freeze unplanned analyses exploratory.

## 5. Expected runtime category

| Stage | Expected runtime | Calls/work | Required run |
|---|---|---|---|
| Code, tests, freeze tooling | Short | Local only | No rerun |
| Fusion and modality evaluation | Moderate | Cached embeddings and ranking | Retrieval-only run |
| Reranking validation/test | Moderate | Candidate/evidence scoring | Retrieval-only run |
| Decision gate | Short–moderate | Artifact comparison | Analysis-only recomputation |
| Hybrid staged screening | High | 36 × selected subset generations plus judging | Validation generation+judge run |
| Hybrid full direct grid | Very high | 10,800 generations and 10,800 judgments per model pair | Validation generation+judge run |
| Hybrid finalists | High | 4–6 × 300 generations and judgments | Validation generation+judge run |
| Legacy explanation reuse | None for generation | Existing 3,600 rows | No generation rerun |
| Clean v2 generation | Very high | 3,600 explanations for three generators | Full final rerun |
| Claim extraction | High | One call per explanation per extractor | Judge-only rerun |
| Claim verification | High | One packet call per explanation/verifier | Judge-only rerun |
| General-quality judging | Very high | Cross-model judgments; up to 10,800 calls | Judge-only rerun |
| Clustered analysis/reporting | Short–moderate | Stored tables and bootstrap | Analysis-only recomputation |

## 6. Cache and resume strategy

- Use versioned namespaces ending in `_v2`; never collide with robustness v1 caches.
- Reuse image, text, target, and KB embeddings only when complete fingerprints match.
- Build fusion weights from cached separate CLIP image/text vectors; do not re-encode per weight.
- Cache candidate sets separately so all retrieval arms use identical candidates.
- Fingerprint LLM work by full model identity, runtime settings, exact prompt, prompt schema,
  case/explanation hash, and structured reference-packet hash.
- Checkpoint one raw response per case/model/stage and assemble final tables only after
  cardinality and schema validation.
- Store parsing status separately; failed/empty extraction is resumable and never a successful
  support result.
- Include all four Hybrid variables and screening/finalist phase in validation cache keys.
- Keep validation and test namespaces distinct.
- Save exact old/v2 packet hashes at the Stage 1 gate.
- Make every stage idempotent and provide `--resume` plus a read-only completeness check.
- Do not mix legacy and v2 generated rows in primary outputs without explicit protocol columns.

## 7. Risks

- **Decision-gate ambiguity:** “material” can be manipulated after seeing changes. Freeze the
  threshold and conservative rules before comparison.
- **Hybrid screening bias:** a small subset may mis-rank configurations. Use balanced sampling,
  retain 4–6 finalists, and reserve full validation for selection.
- **Rule-only naming error:** `item_count=0` could be reported as Hybrid. Enforce an explicit
  `candidate_type=rule_only` field and final-Hybrid eligibility check.
- **Runtime explosion:** direct 36×300 generation plus multi-model judging is expensive. Use
  staged screening unless resources justify the direct run, and report call counts.
- **Selection overfitting:** 36 configurations remain vulnerable to validation overfitting.
  Use priority selection, practical ties, and limited second-pair sensitivity only for finalists.
- **Reference-packet confusion:** common evaluation evidence may be mistaken for generation
  evidence. Store and report both separately on every claim.
- **Self-family leakage:** model aliases/tags could evade exclusion. Centralize and test family
  normalization.
- **Malformed claims:** schema failures could bias support rates. Report extraction coverage,
  raw outputs, retries, and N/A failures.
- **Mixed generation protocols:** reusing legacy explanations can weaken clean v2 claims. Label
  them prominently and prefer full four-variant regeneration after substantive changes.
- **Statistical dependence:** cases, generators, and judges are crossed and outfit-clustered.
  Preserve pairing and avoid treating judgment count as sample size.
- **Test-driven iteration:** the gate must not become an excuse to retune after viewing test
  quality. It compares protocol artifacts only.
- **Reproducibility failure:** dirty source or missing hashes invalidate the freeze. Commands
  must fail closed.

## 8. Change-to-rerun classification

| Change | Required category |
|---|---|
| Implement schemas, commands, tests, manifests, and documentation | No rerun |
| Rename old faithfulness and apply N/A reporting | Analysis-only recomputation |
| Recompute conditional citation and separate overlap tables | Analysis-only recomputation |
| Add outfit-clustered bootstrap and primary comparison corrections | Analysis-only recomputation |
| Compare old and v2 locked recommendation/evidence packets | Analysis-only recomputation after retrieval |
| Add four-way modality comparison | Retrieval-only run |
| Add fusion validation and selected test evaluation | Retrieval-only run |
| Bind held-out reranking to its frozen artifact | Retrieval-only run |
| Remove the old counterfactual claim | No rerun |
| Replace it with real counterfactual retrieval | Retrieval-only run |
| Run 36-config Hybrid screening/finalists | Validation generation+judge run |
| Add `item_count`, exact order, and rule-only eligibility | Validation generation+judge run |
| Extract all atomic claims from legacy explanations | Judge-only rerun |
| Verify claims against structured packets | Judge-only rerun |
| Add anchored general-quality judging | Judge-only rerun |
| Make cross-model-only results primary from existing compatible judgments | Analysis-only recomputation |
| Obtain redesigned cross-model judgments | Judge-only rerun |
| Gate B: reuse unchanged old explanations as legacy-generation/v2-judging | Judge-only rerun |
| Gate A: changed locked items/evidence, prompts, or evidence-budget policy | Full final rerun of all four variants |
| Change only Hybrid validation search but final generation prompts remain identical | Validation generation+judge run; final rerun determined by gate |
| Change final prompt/evidence policy for any substantive variant comparison | Full final rerun of all four variants |
| Change KB, split membership, recommendations, or final reference construction materially | Full final rerun |
| Build final v2 tables, figures, and manifests | Analysis-only recomputation |

## Approval gates before coding and execution

1. Approve this implementation plan.
2. Approve exact new CLI names and file boundaries before implementation.
3. Approve the Stage 1 material-change definition before retrieval comparison.
4. Approve Hybrid screening subset size and finalist policy before generation.
5. Approve judge families, anchored prompts, and reference-packet schema before judging.
6. Approve the clean freeze manifest, created only after Stage 1 and Stage 2 validation
   selections, before any final test generation or judging.
