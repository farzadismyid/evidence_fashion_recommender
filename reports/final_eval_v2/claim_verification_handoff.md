# Stage 4B Claim Verification Handoff

Status: complete.

- Explanations represented: 3,600 / 3,600
- Successful atomic-claim verifications: 6,402
- Extraction-failure explanation rows retained as N/A: 4
- Verification-failure explanation rows retained as N/A: 276
- Malformed JSON responses repaired: 51
- N/A rows with support scores: 0

The four failed Stage 4A extraction rows did not call the verifier. They have blank support labels
and were not interpreted as unsupported, support=0, or support=1. Persistent Stage 4B row failures
were handled the same way after two retries and an available syntax-only repair attempt.

## Successful claim labels

| Variant | Claims | Item evidence | Query/locked item | Rule evidence | Unsupported | Contradicted | Not verifiable |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_rag | 1,571 | 171 | 78 | 573 | 651 | 3 | 95 |
| item_rag | 1,527 | 557 | 55 | 229 | 598 | 0 | 88 |
| rule_rag | 1,696 | 67 | 34 | 1,234 | 316 | 4 | 41 |
| hybrid_rag | 1,608 | 348 | 64 | 880 | 282 | 8 | 26 |
| **Total** | **6,402** | **1,143** | **231** | **2,916** | **1,847** | **15** | **250** |

## N/A explanation rows

| Variant | N/A rows |
|---|---:|
| no_rag | 82 |
| item_rag | 62 |
| rule_rag | 70 |
| hybrid_rag | 66 |
| **Total** | **280** |

The original Stage 3 explanation CSV hash matches the Stage 4B input hash. Explanations were read
only and were not truncated, rewritten, or regenerated.

