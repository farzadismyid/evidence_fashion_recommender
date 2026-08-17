# Evidence-Constrained Multimodal Fashion Recommendation

This is the clean research implementation specified by
`Evidence_Fashion_Recommender_Project_Proposal.md`. It studies sampled controlled-pool fashion
ranking and evidence-grounded explanations while keeping the recommendation image pathway
strictly separate from textual explanation evidence.

## Current scope

The repository is being rebuilt under `new_approach.md`; previous experimental outputs are not
current evidence for the five-category run. Stages 1 and 2 introduce exact category cleaning, a
configuration-driven five-category bag taxonomy, deterministic Polyvore cases, validated MiniLM/CLIP
retrieval, preserved validation sensitivities, a researcher-selected approximately 100-candidate
main Stage 4 run, auditable five-rule reranking, fresh `rag_c3` validation, and a disjoint 50-case
pilot with a separate length-matched sensitivity, the 1,000-case confirmatory recommendation
evaluation, the corrected 3,000-output shared-cap explanation corpus, atomic-claim extraction,
cross-model claim verification, paired automated judging, and zero-call study-specific explanation metrics.
Those metrics distinguish decision-trace alignment, conservative unsupported item attributes, and
citation integrity while retaining visible-evidence grounding as secondary. All reported
explanation-evaluation results come from the saved Stage 8 system outputs; no human or external
audit result is claimed.

Current gate status: Stage 1 is frozen in
`artifacts/manifests/stage1_taxonomy_freeze_manifest.json`. Stage 2 is frozen in
`artifacts/manifests/stage2_kb_freeze_manifest.json`: the pre-experiment 200-case bag audit
supports all cases, maximum rule prevalence is 28%, and 104 unique non-empty packets satisfy the
frozen diversity gates. The audit includes explicitly observed text from the other items in each
outfit as permitted context. See
`reports/stage2_bag_case_applicability_audit.json` and
`reports/five_category_kb_audit.md`. The repository is ready to proceed to Stage 3.

The dataset is pinned to `Marqo/polyvore` revision
`8c782ee447faf2d2a0402ac883cf07d3b3f43e1c`. Runtime datasets, images, model caches, and
embedding arrays are ignored. Compact manifests under `artifacts/manifests/` bind runs to the
resolved configuration, inputs, outputs, models, row counts, environment, seed, and command.
Exact duplicate image hashes are grouped at outfit-component level before evaluation splits are
frozen. Deterministic minimal-change reassignment plus singleton rebalancing preserves the
configured outfit quotas while preventing exact-image hashes from crossing splits.
Raw data is never modified: 377 observed category values are mapped by exact configured keys,
with unlisted and review categories excluded from prepared data. Bags use the exact eight-value
allowlist in the configuration; backpacks, briefcases, luggage and non-bag accessories are
excluded. Dataset counts and the category audit must be regenerated before Stage 1 is frozen.

## Repository map

- `configs/experiment.yaml`: every experiment-facing dataset, split, sampling, fusion,
  evaluation, and output decision.
- `configs/models.yaml`: immutable embedding revisions and later approved LLM settings.
- `configs/fashionclip_baseline.yaml`: pinned FashionCLIP 2.0 additive baseline.
- `data/kb/fashion_rules.csv`: canonical 100-rule, five-category, citation-audited KB.
- `data/kb/legacy_rule_audit.csv`: row-level disposition and provenance for all 126 legacy rules.
- `data/kb/kb_source_registry.csv`: unique-page source validation and rule concentration registry.
- `data/kb/kb_rule_similarity_audit.csv`: reviewed near-duplicate rule pairs.
- `reports/five_category_kb_audit.md`: coverage matrix, unsupported contexts, and pass/fail gates.
- `src/evidence_fashion/data.py`: pinned dataset adaptation, exact taxonomy, splits, and
  deterministic broad-category candidate pools.
- `src/evidence_fashion/retrieval.py`: MiniLM/CLIP encoding, normalized fusion, scoring, and
  deterministic ranking.
- `scripts/prepare_data.py` and `scripts/build_embeddings.py`: Stage 2 and Stage 3 commands.
- `reports/methodology.md`: stable methods, migration decisions, validations, and limitations.
- `reports/final_results.md`: stable results report populated only by approved later stages.
- `artifacts/`: compact manifests plus later figures, tables, and examples.
- `tests/`: deterministic unit, contract, and validation tests.

## Environment and Stage 2–5 commands

Python 3.11 is supported. Install and validate with:

```bash
uv sync --extra cuda --extra dev
uv run --extra cuda --extra dev ruff check .
uv run --extra cuda --extra dev pytest
```

Prepare the pinned data and validation cases without downloading or writing in dry-run mode:

```bash
uv run python scripts/prepare_data.py --config configs/experiment.yaml --dry-run
```

Run Stage 2 preparation, then the approved small Stage 3 embedding validation:

```bash
uv run --extra cuda --extra dev python scripts/prepare_data.py --config configs/experiment.yaml --validate-only
uv run --extra cuda --extra dev python scripts/build_embeddings.py --config configs/experiment.yaml --validate-only
```

Both scripts also accept `--resume`; successful hash-bound outputs are reused rather than
overwritten. The runtime root defaults to ignored `.runtime/` and can be redirected with
`--runtime-root` to storage outside the repository.

Run validation-only recommendation evaluation, then explanation optimisation and the pilot:

```bash
uv run --extra cuda --extra dev python scripts/run_recommendation_eval.py --config configs/experiment.yaml --validate-only
uv run --extra cuda --extra dev python scripts/run_explanation_eval.py --config configs/experiment.yaml --optimize-only
uv run --extra cuda --extra dev python scripts/run_explanation_eval.py --config configs/experiment.yaml --pilot-cases 50
```

Do not run the full 500-case explanation experiment before the required pilot and researcher
approval.

Run the frozen 1,000-case Stage 6 recommendation evaluation with:

```bash
uv run --extra cuda --extra dev python scripts/run_stage6_recommendation_eval.py --config configs/experiment.yaml
```

Run the additive FashionCLIP baseline on the locked cases with:

```bash
uv run --extra cuda --extra dev python scripts/run_stage6b_fashionclip_baseline.py --config configs/fashionclip_baseline.yaml
```

Run or resume the frozen Stage 7 generation-only experiment with:

```bash
uv run --extra cuda --extra dev python scripts/run_stage7_explanation_generation.py --config configs/experiment.yaml
uv run --extra cuda --extra dev python scripts/run_stage7_explanation_generation.py --config configs/experiment.yaml --resume
```

Stage 7 does not extract, verify, or judge claims; those operations belong to Stage 8.

Run or resume Stage 8 assessment with:

```bash
uv run --extra cuda --extra dev python scripts/run_stage8_explanation_assessment.py --config configs/experiment.yaml
uv run --extra cuda --extra dev python scripts/run_stage8_explanation_assessment.py --config configs/experiment.yaml --resume
```

The post-completion Stage 8 verification revision is a deterministic aggregation of saved records
and makes no model calls:

```bash
uv run --extra dev python scripts/revise_stage8_verification_metrics.py
```

It reports visible-evidence grounding as a secondary analysis in the final study-specific layer,
preserves common-reference A+B support as post-hoc alignment, audits refusal handling, and
stratifies results by verifier structural normalization.

Derive the additive study-specific explanation metrics with:

```bash
uv run --extra dev python scripts/derive_stage8_study_metrics.py
```

Run the deterministic Stage 10 cleanup/release integrity review with:

```bash
uv run --extra dev python scripts/run_stage10_release_review.py
```

The command verifies the completed Stage 1–8 manifests and canonical artifacts, confirms that no
human or external evaluation-audit footprint remains, and writes the current release-readiness table and Stage 10
manifest. It makes no model calls.

## Research boundary

CLIP consumes images only for retrieval and ranking. Explanation evidence is restricted to A
(the request plus frozen query/recommended item identities, categories, and item text) and, in the
later evidence condition, B
(the exact stored scoring trace). The system does not caption images or infer textual visual
attributes. Candidate ranking is reported as sampled controlled-pool ranking, not full-catalogue
retrieval.
