# Final Evaluation v2 Stage 3 Real-Run Handoff

## Outcome

Stage 3 completed successfully under the immutable corrected v2 freeze. Gate A generated all four
variants for all 300 frozen test cases and all three selected generators.

```text
cases: 300
variants: 4
generators: 3
expected rows: 3,600
written rows: 3,600
generation errors: 0
runtime: 9,529.6 seconds (2 h 38 m 49.6 s)
```

No claim extraction, claim verification, judging, statistical analysis, or final reporting ran.

## Freeze binding

The freeze was created from clean source commit:

```text
b5c62220d58486ddbe1d4856c8142c23341f373f
```

It binds:

```text
reranking policy: evidence_in_loop_pareto_v2
CLIP / evidence weights: 0.75 / 0.25
validation Stage 1 packet hash: c561b2016ba8ad7b8d06aac7cb3dd8a56793e91adfa5f4f70186a448ae3188f9
test Stage 1 packet hash: 445dca5a3e513d6e425e110d32dd4323097a12544ec8a13a1af07977b93df6b3
Hybrid selection: hybrid_w35_r5_i2_item_first
decision gate: regenerate_all_variants
freeze manifest hash: 79e7c061b89c581e502338c6a1a4db52460d1b440ad0824a3d9656f36a538859
```

## Generation policy

All variants used the same declared 35-word budget. Rule-RAG and Hybrid-RAG used five rules;
Item-RAG and Hybrid-RAG used two item-evidence entries. Hybrid used item-first evidence ordering.
Every row records `generation_protocol=final_eval_v2`, `evaluation_protocol=v2`, the prompt
fingerprint, generator identity/digest/settings, and freeze-manifest hash.

Generators:

```text
llama3.2@a80c4f17acd5
mistral@6577803aa9a0
gemma3:12b@f4031aab637d
```

Each generator/variant cell contains exactly 300 rows.

## Length-compliance diagnostic

The shared policy was applied consistently, but local models did not always obey the 35-word
instruction. This is a generated-output property to retain and evaluate, not grounds for selective
regeneration after observing results.

| Generator | No-RAG | Item-RAG | Rule-RAG | Hybrid-RAG |
|---|---:|---:|---:|---:|
| llama3.2 | 78 | 71 | 112 | 120 |
| mistral | 157 | 122 | 236 | 220 |
| gemma3:12b | 26 | 18 | 79 | 73 |

Values are counts exceeding 35 words out of 300 rows per cell. Stage 4 should report this as a
protocol-compliance metric and should not truncate or rewrite explanations before evaluation.

## Outputs

```text
outputs/final_eval_v2/freeze/FINAL_FREEZE_MANIFEST.json
outputs/final_eval_v2/explanations/explanations.csv
outputs/final_eval_v2/explanations/generation_errors.csv
outputs/final_eval_v2/explanations/generation_manifest.json
reports/final_eval_v2/generation_summary.md
```

Generation artifact fingerprint:

```text
c757ad0a237fb4f7a50f8c15fc10b0380b030ea42d3cad13ba666022d80eee27
```

## Verification

Before the real run:

```text
full project suite: 66 passed
Ruff: all checks passed
freeze source state: clean
freeze corrected reranker binding: passed
freeze corrected Stage 1/2 binding: passed
```

After the run:

```text
row cardinality: 12 cells x 300 = 3,600, passed
generation errors: 0
variant set: No-RAG, Item-RAG, Rule-RAG, Hybrid-RAG
generator set: three frozen models
```

## Exact next step

Stage 4 should implement and run separate resumable `extract-claims-v2` and `verify-claims-v2`
commands against the immutable 3,600-row explanation artifact, followed by anchored general-quality
judging. Extraction must include all atomic claims without the historical three-claim cap, and no
Stage 3 explanation may be changed during evaluation.
