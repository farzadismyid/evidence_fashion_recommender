# Stage 4D targeted recovery handoff

Status: complete. This is the **POST-RECOVERY FINAL ANALYSIS** bundle.

## Recovery outcome

The recovery operated only on explicit failed/N/A keys. It made 384 LLM calls: 276
initial calls and 108 retries, below the strict 1,092-call ceiling. Deterministic local
JSON repair avoided calls for 17 verification rows and 252 judgment rows.

| Sub-stage | Attempted | Recovered | Still N/A |
|---|---:|---:|---:|
| Claim extraction | 4 | 3 | 1 |
| Claim verification | 279 | 195 | 84 verification failures, plus 1 extraction N/A |
| General judgment | 262 | 258 | 4 |

All 3,600 explanations retain the original SHA-256
`6fbae305fa00051d771201c72a61d1f38cc7de3f834c40dc28cf42e296040e45`.
The deterministic merge reported zero canonical-hash changes among successful source
rows. Failed outputs were replaced only after the existing parsers validated their
schema, exact keys, claim IDs, labels, and score semantics. Unresolved outputs remain
N/A.

## Analysis rules

- The preserved pre-recovery bundle remains under `reports/final_eval_v2/pre_recovery`.
- This bundle was generated once from the new merged post-recovery tables only.
- Cross-model-only judging is primary; all-judge/self-judge results are sensitivity
  diagnostics.
- Length compliance is reported separately and no explanation was regenerated,
  truncated, rewritten, or normalized.

## Main post-recovery results

Verified-claim support from any permitted evidence is 56.72% for `no_rag`, 58.87% for
`item_rag`, 79.14% for `rule_rag`, and 81.81% for `hybrid_rag`. Unsupported-claim rates
are respectively 36.86%, 34.88%, 18.20%, and 16.10%.

Primary cross-model general-quality means are 3.849 (`no_rag`), 4.066 (`item_rag`),
3.933 (`rule_rag`), and 3.927 (`hybrid_rag`). These scores must be interpreted alongside
the separate length-compliance table and the remaining N/A coverage above.

## Audit and validation

The immutable-source hashes, per-key replacement logs, and completion manifest are in
`outputs/final_eval_v2/recovery/stage4d`. Merged evaluation tables are in
`outputs/final_eval_v2/post_recovery`. The report artifact inventory contains hashes and
row counts for every analysis input.

Validation requirements: 3,600 extraction rows, 3,600 verification rows, 10,800 judgment
rows, unique existing keys, unchanged successful-row canonical hashes, unchanged source
explanation hash, Ruff, and the full test suite.
