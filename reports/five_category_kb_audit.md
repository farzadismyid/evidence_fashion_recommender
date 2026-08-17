# Five-category knowledge-base audit

Audit date: 2026-08-16  
Scope: tops, bottoms, shoes, outerwear, bags  
Experimental condition results inspected: **no**

## Freeze status

- **Stage 1: FROZEN.** The manifest is
  `artifacts/manifests/stage1_taxonomy_freeze_manifest.json`. It binds the pinned dataset,
  five-category taxonomy, exact split counts, category audit, 1,000 balanced evaluation cases,
  and the deterministic 200-case bag audit sample. Split overlap and cross-split exact-image
  leakage are both zero.
- **Stage 2: FROZEN.** The pre-experiment audit at
  `reports/stage2_bag_case_applicability_audit.json` found directly applicable rules for all 200
  bag cases. Coverage is 45/45 bottoms queries, 34/34 outerwear queries, 65/65 shoe queries and
  56/56 top queries. The immutable freeze manifest is
  `artifacts/manifests/stage2_kb_freeze_manifest.json`.

The final pre-result thresholds are maximum rule prevalence 30%, duplicate-packet participation
70%, and at least 100 unique non-empty packets. The final audit passes at 28%, 67.5%, and 104
respectively, while retaining 200/200 case coverage. No experimental condition results were
inspected.

## Outcome

The legacy audit accounts for **126 / 126** rules. No legacy rule text was carried forward
verbatim. The canonical replacement contains 100 narrowly scoped rules: exactly 20 for each of
tops, bottoms, shoes, outerwear and bags. It cites 43 distinct source pages and retains
source identity, URL, locator, access date, evidence summary, scope, limitations, reliability,
source-validation status, version, and predecessor IDs in every row. Every active source URL was
reachable and its cited passage was checked for direct support on 2026-08-16; no retained rule is
backed only by a broken, unidentified, or method-only reference.

Legacy dispositions are: 23 outside the five-category taxonomy, 25 based on the retired mixed
accessory ontology, 16 with citation overreach, and 62 reassessed for conservative rewriting or
replacement. The row-level record is `data/kb/legacy_rule_audit.csv`; the compact decision
manifest is `data/kb/legacy_kb_audit.yaml`. The original 126-row file remains immutable in the
archive at the path bound by that manifest and its SHA-256 digest.

## Schema and evidence policy

The canonical schema separates recommendation target, permitted query categories, hard query
terms, candidate terms, required contexts, scenario metadata, source metadata, evidence summary,
scope, and limitations. `audit_status=retain` means that the cited passage supports the narrow
wording stored in `rule_text`; it does not mean the prescription is universally or causally true.

High reliability denotes direct prescriptive support from an established fashion editorial or
etiquette guide. Medium reliability denotes a direct stylist/retailer prescription or a clearly
documented editorial outfit direction with narrower generalisability. Low-reliability and
method-only citations are excluded. The final set has 51 high- and 49 medium-reliability rules.

No individual source page supplies more than 7 rules (7% of the KB), and the source registry
contains 43 pages from 7 named editorial organisations. Publication-family concentration remains
a declared limitation: 74 rules come from Vogue/GQ brands. This is defensible as a transparent
curated editorial KB, but it must not be described as a consensus of independent fashion
authorities. Future expansion should prioritize independent publishers rather than add further
Vogue/GQ rules.

## Static coverage matrix

Counts are rules permitted for a query-category/target-category pair before hard term and context
checks. Same-category recommendations are intentionally excluded.

| Query category | Bags | Bottoms | Outerwear | Shoes | Tops |
|---|---:|---:|---:|---:|---:|
| Bags | 0 | 3 | 3 | 3 | 3 |
| Bottoms | 12 | 0 | 8 | 9 | 8 |
| Outerwear | 13 | 3 | 0 | 5 | 4 |
| Shoes | 17 | 7 | 5 | 0 | 5 |
| Tops | 13 | 7 | 6 | 7 | 0 |

The machine-readable matrix is `data/kb/coverage_matrix.csv`. The unique-page provenance audit is
`data/kb/kb_source_registry.csv`. The normalized-text similarity audit is
`data/kb/kb_rule_similarity_audit.csv`. Sixty-three pairs cross the conservative 0.72 wording-
similarity review threshold; all are retained with an explicit distinct-target or distinct-context
disposition, and no exact normalized rule-text duplicate exists.

## Unsupported-context report

The retriever now fails closed for all five targets. A rule is unavailable when its target does
not match, its audit status is not retained, its query category is not declared, required context
is absent, or its declared query/candidate terms do not occur in permitted case text. Query-term
declarations use AND-of-OR logic, so every stated condition must be present. This means:

- same-category requests are unsupported by design;
- generic cases lacking the garment or scenario language required by every eligible rule are
  unsupported rather than filled with a weak generic rule;
- deterministic text from the other items in the same outfit is now included as permitted context;
- no inference is made from images, protected attributes, body shape, weather, or unrecorded
  occasion context;
- the matrix is structural coverage rather than empirical case support; the frozen Stage 1 sample
  demonstrates that 0 of 200 bag cases are currently unsupported;
- editorial prescriptions encode situated style guidance, not objective aesthetic truth.

No unsupported context was patched by reading prior rankings, metrics, generations, judgments,
or any other output: experimental condition results were not inspected.

## Pass/fail gates

| Gate | Result | Criterion |
|---|---|---|
| Legacy accounting | PASS | 126 unique legacy IDs classified once |
| Frozen taxonomy | PASS | only tops, bottoms, shoes, outerwear and bags |
| Canonical integrity | PASS | 100 unique IDs; 20 per target; required provenance fields non-empty |
| Citation restraint | PASS | no low-reliability or method-only rules; limitations explicit |
| Source validity | PASS | 100/100 direct HTTPS citations with locators and access dates |
| Source-page concentration | PASS | maximum 7/100 rules from any one page |
| Duplicate review | PASS | no exact normalized duplicates; all 63 flagged pairs dispositioned |
| Static cross-category coverage | PASS | every one of 20 valid directed category pairs has >=3 rules |
| Fail-closed applicability | PASS | audit, category, context and term filters run before top-k |
| Stage 1 taxonomy/data freeze | PASS | frozen manifest binds 1,000 cases and 200 bag audit cases |
| Empirical bag-case coverage | PASS | 200/200 supported; zero unsupported cases |
| Bag-rule prevalence | PASS | maximum 28%, below the pre-result 30% threshold |
| Bag packet diversity | PASS | 104 unique packets; 67.5% duplicate-packet participation, below 70% |
| Stage 2 KB freeze | PASS | immutable manifest binds the KB, audits, thresholds and Stage 1 freeze |

The preliminary `bag_rule_audit.csv` and `stage2_bag_rule_audit.md` are retained only as historical
Stage 2 artifacts. They are superseded by this five-category audit and are no longer loaded by the
evaluation configuration.
