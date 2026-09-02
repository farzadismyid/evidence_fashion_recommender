# Data

The project uses the pinned `Marqo/polyvore` revision recorded in
`configs/experiment.yaml`. Raw rows, decoded images, Hugging Face caches, prepared Parquet
files, and embeddings are runtime data and must not be committed.

Use the paths configured in `configs/experiment.yaml` when preparing local data; the repository
does not include a dataset download or model cache.

`kb/fashion_rules.csv` is the canonical 200-rule KB with 40 rules each for bags, bottoms,
outerwear, shoes, and tops.
Every row carries applicability fields, a citation locator, access date, evidence summary, scope,
limitations, reliability, audit status and predecessor IDs. Retrieval applies the audit,
category, context and explicit term gates before top-k scoring for every target.

`kb/legacy_kb_audit.yaml` binds the archived 126-row predecessor by path and SHA-256 and records
the four decision classes. `kb/legacy_rule_audit.csv` expands that audit to one row per legacy
rule, preserving its original text and citation. `kb/coverage_matrix.csv` is the machine-readable
static five-category coverage matrix. `kb/kb_source_registry.csv` records one row per validated
source page, while `kb/kb_rule_similarity_audit.csv` records every normalized rule-text pair above
the review threshold. These artifacts were built without inspecting experimental
condition results. The earlier bag-only audit is historical and no longer loaded by the pipeline.

Only the dataset's `category` and `text` fields may provide minimal item identities for later
explanation context. The image field is used only by the CLIP recommendation pathway and may
not be converted into textual explanation evidence.
