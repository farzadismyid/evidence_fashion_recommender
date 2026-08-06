# Frozen expanded recommendation-evaluation protocol

Status: frozen before expanded recommendation evaluation. No recommendation outcome from the proposed expanded cohort was inspected or used in this decision.

## Decision

The frozen cohort is **3,000 leakage-free held-out test cases**. This is hierarchy option 2.

The all-eligible option is not selected even though 34,130 independent held-out cases exist: target embeddings are cached, but query embeddings and candidate evidence scores exist only for the current 300-case test and must be produced for any newly added case. The all-eligible run would require 3.42 million candidate evidence-score computations and 34,130 query embeddings, which is an unbounded large evaluation job relative to the available cached work. A 3,000-case evaluation is sufficiently large, leakage-free, and a bounded tenfold expansion of the completed test. This choice depends only on eligibility, split protection, cache coverage, and planning cost—not on recommendation metrics.

## Eligibility and existing schedules

Eligibility was recomputed read-only from `data/processed/items_clean.parquet` and `data/processed/target_items_clean.parquet` with `build_evaluation_cases` (`src/evidence_fashion_recommender/evaluation/controlled.py:25-70`) using the configured five target categories and seed 42. The repository split rule (`evaluation/splits.py:19-42`) deterministically assigns by `query_outfit_id`, with 60% development, 20% validation, and 20% test.

| Target category | All eligible | Validation eligible | Held-out test eligible | Existing validation | Existing test | Frozen expanded test |
|---|---:|---:|---:|---:|---:|---:|
| accessories | 54,827 | 11,123 | 11,072 | 60 | 60 | 973 |
| bottoms | 28,196 | 5,673 | 5,651 | 60 | 60 | 497 |
| outerwear | 16,741 | 3,555 | 3,311 | 60 | 60 | 291 |
| shoes | 37,564 | 7,449 | 7,731 | 60 | 60 | 679 |
| tops | 31,457 | 6,398 | 6,365 | 60 | 60 | 560 |
| **Total** | **168,785** | **34,198** | **34,130** | **300** | **300** | **3,000** |

The existing 300-case test is a subset of the full eligible test partition: all 300 `(query_item_id, target_category)` keys are present in the recomputed test population. It remains a documented subset of the frozen 3,000 cohort.

## Frozen cohort construction

- Seed: `42`.
- Start with all existing rows from `outputs/robustness/schedules/test_schedule.csv` (300 total; 60 per category).
- For each category, add the lowest SHA-256-ranked remaining eligible test rows until its pre-declared allocation above is met. Rank key: `SHA256("42|{query_item_id}|{target_category}")`, ascending; tie-breaker: `query_item_id`, ascending.
- Candidate identity is `(query_item_id, target_category)`; no recommendation outcomes, ranks, scores, labels beyond eligibility, or metric tables participate in selection.
- The canonical JSON serialization of the sorted 3,000 selected identity keys has SHA-256 `622a83499197633ce46406bc6fe066e0771e3f5b103b52906e34f1348ff733a1`. The corresponding existing-300 identity-key hash is `fee5cfaedb7bdc2f83b8fa778579221750ac8195735153b9dac0dbb94a205549`.
- The resulting cohort has 1,829 unique query outfits. Repeated cases from an outfit are retained as one outfit cluster for uncertainty estimation; all reported uncertainty must resample query outfits, not individual cases.

The category allocations are proportional to the natural held-out test distribution (largest-remainder rounding to 3,000), while retaining the existing balanced 300-case subset. The primary report must provide both micro-averaged metrics and category-macro metrics (unweighted mean of the five category metrics), with category-level counts and metrics.

## Leakage and duplication audit

| Check | Result | Interpretation |
|---|---:|---|
| Outfit overlap, development/validation | 0 | Disjoint by deterministic outfit assignment. |
| Outfit overlap, development/test | 0 | Disjoint by deterministic outfit assignment. |
| Outfit overlap, validation/test | 0 | Disjoint by deterministic outfit assignment. |
| Query-item overlap across every split pair | 0 | No query example crosses partitions. |
| Positive-item overlap across every split pair | 0 | Positives remain within their outfit partition. |
| Existing validation/test candidate-item overlap | 16,233 unique items | Expected shared global retrieval corpus, not positive/query leakage. |

`build_controlled_candidate_set` (`evaluation/ranking.py:28-53`) always removes the query outfit from the negative pool and removes the query item. The expanded evaluation must use the same fixed target corpus and per-case construction: all same-outfit target-category items are positives; 99 deterministic negatives are drawn only from other outfits. This prevents an own-outfit negative or a query item from entering a candidate set. The global candidate corpus is shared across splits, so raw candidate-ID overlap alone is not evidence of leakage; it must not be used for tuning or to fit parameters after validation selection.

## Tuning restriction

The current validation schedule contains 300 cases and is the only schedule passed to the documented fusion/reranking selection path. `command_build_robustness_schedules` constructs the partitions, and the CLI’s tuning command defaults to `outputs/robustness/schedules/validation_schedule.csv` (`src/evidence_fashion_recommender/cli.py:194-210`); final-evaluation source code also requires `selected_on == "validation"` (`evaluation/v2_sources.py:174-180`). No repository artifact inspected indicates that additional held-out test cases were used for tuning.

No expanded-cohort rows may be used to select fusion weights, evidence weights, prompts, rules, candidate settings, or any other operating point. The existing validation-selected fusion and reranking artifacts remain fixed.

## Candidate pools and cache state

The fixed target corpus contains 67,524 target items: 29,463 accessories, 12,623 shoes, 10,892 tops, 9,209 bottoms, and 5,158 outerwear. Each controlled case has all same-outfit positives plus up to 99 negatives, so the observed current test mean is 100.21 candidates/case (30,064 rows / 300 cases). By category, current means are accessories 100.85, bottoms 100.02, outerwear 100.00, shoes 100.12, and tops 100.08.

Target MiniLM/CLIP image/CLIP text embeddings are cached and reusable (381 MiB combined). Existing 300-case query embeddings are cached (1.61 MiB combined), as are their 30,064 candidate evidence scores in the candidate table. Neither query embeddings nor candidate evidence scores are cached for the additional 2,700 selected cases; they must be materialized once, saved, hash-bound, and reused for all expanded metrics. `evaluation/materialization.py:55-68` fingerprints query-embedding caches, and `v2_sources.py:96-161` writes candidate evidence scores into a provenance-bound candidate table.

## Planning estimates

These are storage and runtime planning estimates, not observed evaluation outcomes. Storage scales from the existing 300-case source candidate table (5.998 MB) and query-embedding arrays (1.690 MB); the reusable 381 MiB target cache is excluded.

| Cohort | Candidate evidence-score computations (approximately) | New query embeddings | Incremental candidate + query storage | Conservative runtime budget |
|---|---:|---:|---:|---:|
| 1,000 cases | 100,000 | 1,000 | ~26 MB | 0.5–1.5 hours |
| 3,000 cases | 300,000 | 3,000 | ~77 MB | 1.5–4.5 hours |
| All 34,130 eligible test cases | 3.42 million | 34,130 | ~875 MB | 17–51 hours |

The runtime envelope is a linear operational budget from the completed pipeline’s 100-candidate-per-case design, not a historical wall-clock measurement (the committed manifests do not record stage durations). It includes query embedding, candidate evidence embedding/scoring, selected-packet construction, and ranking; it excludes any explanation generation or LLM evaluation, which are out of scope.

## Freeze integrity

The JSON companion `outputs/recommendation_eval_expanded/frozen_protocol.json` binds the selected cohort identity hash, seed, allocation, source hashes, cache conditions, and this Markdown document hash. Any expanded evaluation must reject a mismatching schedule, corpus, candidate table, embedding cache, or selection artifact rather than silently regenerate or substitute inputs.

Source hashes: `items_clean.parquet` `a6fbbba1eeb5650432cbd9d82d8aa792256b75d990cda1a13a6e3857c2a30237`; `target_items_clean.parquet` `569dcbe7dc9dc54163bc73969d1ff7355ea89065a23d52e883f6760a06552965`; existing test schedule `4e5dea27b82bd0e40d4a0aa347b62a934ab90da271f78aad3a4fa96f2f531102`; existing validation schedule `38fee97ccf7197a803d43b673ea14c8f5e88b3ff2e3c6949f26157936ebadc90`; existing development schedule `44eab7b0154d7f370e379bbe292f9d852de0d5480d67ab23133d936414a1d00f`; existing test candidate table `b12f4c87827a37590a9de7b4d0937365975be6bf156fffd8fa7cd7618811f3a5`; final-v2 config `f5bbef6fa2912d6ea6028c1fcd3e6fc8ab2a409168a334ba0454af4a3148ca8`.
