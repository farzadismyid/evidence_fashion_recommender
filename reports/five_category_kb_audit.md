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
- **Stage 2: NOT FROZEN — hard gate failed.** The pre-experiment audit at
  `reports/stage2_bag_case_applicability_audit.json` found directly applicable rules for 9 of 200
  bag cases (4.5%). Coverage was 0/45 bottoms queries, 0/34 outerwear queries, 6/65 shoe queries and
  3/56 top queries. The other 191 cases contain generic product identity text without the
  occasion, formality or multi-item context required by the defensible bag rules.

Freezing Stage 2 in this state would contradict the minimum gate in `new_approach.md`. The failure
cannot be repaired by weakening term filters or treating a merely category-eligible rule as
applicable. A design decision is required: enrich the permitted case context with other
non-target outfit identities, build and source a substantially broader bag rule set for the raw
query subcategories, or revise/exclude the unsupported bag-case population before Stage 2 is
frozen. No later stage should begin until that decision is implemented and the audit passes.

## Outcome

The legacy audit accounts for **126 / 126** rules. No legacy rule text was carried forward
verbatim. The canonical replacement contains 75 narrowly scoped rules: exactly 15 for each of
tops, bottoms, shoes, outerwear and bags. It cites 36 distinct source pages and retains
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
method-only citations are excluded. The final set has 59 high- and 16 medium-reliability rules.

No individual source page supplies more than 7 rules (9.3% of the KB), and the source registry
contains 36 pages from 8 named editorial organisations. Publication-family concentration remains
a declared limitation: 68 rules come from Vogue/GQ brands. This is defensible as a transparent
curated editorial KB, but it must not be described as a consensus of independent fashion
authorities. Future expansion should prioritize independent publishers rather than add further
Vogue/GQ rules.

## Static coverage matrix

Counts are rules permitted for a query-category/target-category pair before hard term and context
checks. Same-category recommendations are intentionally excluded.

| Query category | Bags | Bottoms | Outerwear | Shoes | Tops |
|---|---:|---:|---:|---:|---:|
| Bags | 0 | 3 | 3 | 3 | 3 |
| Bottoms | 14 | 0 | 6 | 6 | 4 |
| Outerwear | 15 | 3 | 0 | 5 | 4 |
| Shoes | 13 | 4 | 3 | 0 | 4 |
| Tops | 13 | 5 | 5 | 5 | 0 |

The machine-readable matrix is `data/kb/coverage_matrix.csv`. The unique-page provenance audit is
`data/kb/kb_source_registry.csv`. The normalized-text similarity audit is
`data/kb/kb_rule_similarity_audit.csv`: no exact rule-text duplicates were found; 11 pairs above
the conservative 0.72 review threshold were retained only because they address different targets,
query origins or explicit contexts.

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
  demonstrates that 191 of 200 bag cases are currently unsupported;
- editorial prescriptions encode situated style guidance, not objective aesthetic truth.

No unsupported context was patched by reading prior rankings, metrics, generations, judgments,
or any other output: experimental condition results were not inspected.

## Pass/fail gates

| Gate | Result | Criterion |
|---|---|---|
| Legacy accounting | PASS | 126 unique legacy IDs classified once |
| Frozen taxonomy | PASS | only tops, bottoms, shoes, outerwear and bags |
| Canonical integrity | PASS | 75 unique IDs; 15 per target; required provenance fields non-empty |
| Citation restraint | PASS | no low-reliability or method-only rules; limitations explicit |
| Source validity | PASS | 75/75 reachable direct HTTPS citations with locators and access dates |
| Source-page concentration | PASS | maximum 7/75 rules from any one page |
| Duplicate review | PASS | no exact normalized duplicates; all 11 flagged pairs dispositioned |
| Static cross-category coverage | PASS | every one of 20 valid directed category pairs has >=3 rules |
| Fail-closed applicability | PASS | audit, category, context and term filters run before top-k |
| Stage 1 taxonomy/data freeze | PASS | frozen manifest binds 1,000 cases and 200 bag audit cases |
| Empirical bag-case coverage | FAIL | 2/200 supported; Stage 2 is not frozen |

The preliminary `bag_rule_audit.csv` and `stage2_bag_rule_audit.md` are retained only as historical
Stage 2 artifacts. They are superseded by this five-category audit and are no longer loaded by the
evaluation configuration.
