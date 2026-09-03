# Data

The project uses the pinned `Marqo/polyvore` revision recorded in
`configs/experiment.yaml`. Raw rows, decoded images, Hugging Face caches, prepared Parquet
files, and embeddings are runtime data and must not be committed.

Use the paths configured in `configs/experiment.yaml` when preparing local data; the repository
does not include a dataset download or model cache.

`kb/fashion_rules.csv` is the canonical 200-rule KB with 40 rules each for bags, bottoms,
outerwear, shoes, and tops. Every row carries applicability fields, a citation locator, access
date, evidence summary, scope, limitations, reliability, audit status, and predecessor IDs.
Retrieval applies category, context, and explicit-term gates before top-k scoring for every target.

The remaining KB files retain final-run provenance: the five-category coverage matrix, source
registry, and duplicate-similarity audit. Superseded KB variants and historical audits are in
`OLD/data/kb/`.

Only the dataset's `category` and `text` fields may provide minimal item identities for later
explanation context. The image field is used only by the CLIP recommendation pathway and may
not be converted into textual explanation evidence.
