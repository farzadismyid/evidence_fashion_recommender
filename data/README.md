# Data

The project uses the pinned `Marqo/polyvore` revision recorded in
`configs/experiment.yaml`. Raw rows, decoded images, Hugging Face caches, prepared Parquet
files, and embeddings are runtime data and must not be committed.

`kb/fashion_rules.csv` is the approved 126-row KB v3 asset. Its authoritative SHA-256 is
`ad19fc788769ebd5fec65ee8aa6b62e4cfc8fbf1f67725392b754a327c2dced3`.

Only the dataset's `category` and `text` fields may provide minimal item identities for later
explanation context. The image field is used only by the CLIP recommendation pathway and may
not be converted into textual explanation evidence.

