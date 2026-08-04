# Stage 4A Claim Extraction Handoff

Completed: 3600/3600

Atomic claims: 8295

Repaired malformed JSON responses: 0

Persistent malformed-JSON failures: 4

Successful empty extractions: 0

The four persistent failures are recorded in `failed_extractions.csv` after the
initial attempt, two same-row retries, and one repair-only attempt. They remain
explicit failure/N/A rows and must be excluded from support-rate denominators;
they are not interpreted as perfectly supported explanations.

The extraction checkpoint contains 3600 unique explanation keys, and the final
tables retain all 3600 explanations. The 8295 total counts only successfully
extracted atomic claims; the four failure/N/A placeholder rows in `claims.csv`
are not claims.

Original Stage 3 explanations were read only and remain unchanged. Claim verification and general judging have not run.

Claim verification can safely start next because coverage is complete and the
failure rows are explicitly represented as N/A. Verification must preserve that
N/A treatment rather than scoring those rows as supported.
