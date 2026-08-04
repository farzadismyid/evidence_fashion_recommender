# Stage 4C General Judging Handoff

Status: complete.

- Explanation-judge pairs represented: 10,800 / 10,800
- Successful judgments: 10,538
- Persistent judgment failures retained as N/A: 262
- Malformed JSON responses repaired: 76
- N/A rows with any score populated: 0
- Original explanations represented in length audit: 3,600 / 3,600

## Primary: cross-model-only judging

The primary table excludes self-family judgments and all N/A failures.

| Variant | n | N/A | Input consistency | General quality | Clarity | Specificity | Hallucination risk | Evidence misuse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_rag | 2,043 | 57 | 4.553 | 3.817 | 4.333 | 3.449 | 3.821 | 4.958 |
| item_rag | 1,923 | 177 | 4.729 | 3.980 | 4.321 | 3.647 | 3.708 | 4.845 |
| rule_rag | 2,085 | 15 | 4.535 | 3.928 | 4.274 | 3.658 | 3.800 | 4.299 |
| hybrid_rag | 2,089 | 11 | 4.641 | 3.924 | 4.294 | 3.645 | 3.829 | 4.379 |

## Sensitivity: all judges, including self-family

These values are diagnostics only and are not the primary result.

| Variant | n | N/A | Input consistency | General quality | Clarity | Specificity | Hallucination risk | Evidence misuse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_rag | 2,643 | 57 | 4.551 | 3.793 | 4.297 | 3.343 | 3.641 | 4.955 |
| item_rag | 2,523 | 177 | 4.730 | 3.889 | 4.272 | 3.498 | 3.541 | 4.844 |
| rule_rag | 2,683 | 17 | 4.594 | 3.958 | 4.305 | 3.619 | 3.669 | 4.206 |
| hybrid_rag | 2,689 | 11 | 4.704 | 3.941 | 4.314 | 3.583 | 3.675 | 4.312 |

Of the 262 N/A judgments, 260 were cross-model rows and 2 were self-family rows. By judge, Qwen
accounted for 259 failures and Gemma for 3; Mistral had none. Failed rows are recorded in
`failed_general_judgments.csv` with their raw malformed response when available, error, retry count,
and repair status.

## Length compliance (separate outcome)

Overall, 2,288 / 3,600 explanations complied with the 35-word limit (63.56%); 1,312 exceeded it.
Mean length was 34.56 words. Longer explanations were not truncated, rewritten, regenerated, or
silently rewarded through the quality scores.

| Generator | Variant | Mean words | Over 35 | Compliance |
|---|---|---:|---:|---:|
| Gemma | no_rag | 31.27 | 26 | 91.33% |
| Gemma | item_rag | 30.14 | 18 | 94.00% |
| Gemma | rule_rag | 32.74 | 79 | 73.67% |
| Gemma | hybrid_rag | 32.44 | 73 | 75.67% |
| Llama | no_rag | 33.67 | 78 | 74.00% |
| Llama | item_rag | 33.63 | 71 | 76.33% |
| Llama | rule_rag | 34.56 | 112 | 62.67% |
| Llama | hybrid_rag | 34.76 | 120 | 60.00% |
| Mistral | no_rag | 35.94 | 157 | 47.67% |
| Mistral | item_rag | 34.55 | 122 | 59.33% |
| Mistral | rule_rag | 41.17 | 236 | 21.33% |
| Mistral | hybrid_rag | 39.80 | 220 | 26.67% |

The final checkpoint contains 10,800 unique judgment keys. During execution, a duplicate worker was
detected and stopped; the checkpoint was normalized using deterministic first-write-wins before a
single-worker resume. The pre-normalization checkpoint is retained as an audit backup. The original
Stage 3 explanation CSV hash matches the Stage 4C input hash.

