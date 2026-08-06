# Union verification-pool audit

## Finding

**A. CONFIRMED:** `no_rag`, `item_rag`, `rule_rag`, and `hybrid_rag` were each verified against the same complete *union* reference pool for the same `(paper_case_id, generation_model)`: query metadata, user request, locked recommended-item metadata, retrieved item evidence, and retrieved expert-rule evidence.

This is a code-and-saved-input finding, not an inference from the result summaries. The two generation-availability booleans intentionally differ by variant, so the complete serialized `ReferencePacket` fingerprint is not expected to be identical across variants. Those booleans are provenance metadata, not an omission from the verifier's reference pool.

## Code path inspected

| Concern | Direct implementation evidence |
|---|---|
| Packet schema | `src/evidence_fashion_recommender/evaluation/claim_evaluation.py:50-57` declares all five reference fields plus two generator-availability flags. |
| Packet construction | `build_reference_packet` in `claim_evaluation.py:64-74` copies `query_text`, `user_request`, `recommended_text`, `item_evidence_text`, and `rule_evidence_text` unconditionally. Only lines 72-73 branch on variant, and they set flags only. |
| Verifier prompt | `claim_verification_prompt` in `claim_evaluation.py:170-189` serializes the entire packet with `asdict(packet)` (line 172) and tells the verifier that generation-evidence flags say what the generator saw while all other fields are evaluation references (lines 174-176). |
| Primary Stage 4B invocation | `run_claim_verification_v2` in `stage45_v2.py:595-707` validates the immutable 3,600-row source (line 605); for every non-N/A extraction it builds the packet and prompt at lines 645-646, and writes `packet.fingerprint` at line 662. |
| Identity/key logic | `KEY_COLUMNS` and `_key` in `stage45_v2.py:31-35` use the exact identity tuple `(paper_case_id, grounding_variant, generation_model)`. `validate_explanations` at lines 113-123 requires 3,600 rows and unique keys. |
| Locked evidence source | `src/evidence_fashion_recommender/evaluation/v2_sources.py:258-269` writes `recommended_text`, `item_evidence_text`, `rule_evidence_text`, their packet/source metadata, and marks the protocol `final_eval_v2_selected`. |
| Stage 4D recovery | `stage4d_v2.py:242-286` reuses `build_reference_packet(row)` and `claim_verification_prompt(...)` for only targeted N/A verification recoveries (lines 259-269); successful source rows are retained unchanged (lines 244-250). Post-recovery verification tables are materialized at lines 144-151 and 337-341. |

No variant-specific branch removes `retrieved_item_evidence` or `retrieved_rule_evidence` in the verification path.

## Packet fields and variant treatment

Every verifier packet has these fields:

| Packet field | Source column | `no_rag` | `item_rag` | `rule_rag` | `hybrid_rag` |
|---|---|---:|---:|---:|---:|
| `query_item_metadata` | `query_text` | present | present | present | present |
| `user_request` | `user_request` | present | present | present | present |
| `locked_recommended_item_metadata` | `recommended_text` | present | present | present | present |
| `retrieved_item_evidence` | `item_evidence_text` | present | present | present | present |
| `retrieved_rule_evidence` | `rule_evidence_text` | present | present | present | present |
| `item_evidence_shown_to_generator` | variant rule | false | true | false | true |
| `rule_evidence_shown_to_generator` | variant rule | false | false | true | true |

Thus generation-available evidence was tracked separately from the complete verification reference pool. The flags are expressly included so verification can distinguish what a generator saw from evaluation-only references; they do not gate the two retrieved-evidence fields.

## Saved-input and checkpoint audit

The canonical post-recovery explanation input is `outputs/final_eval_v2/post_recovery/explanations/explanations.csv` (3,600 rows; SHA-256 `6fbae305fa00051d771201c72a61d1f38cc7de3f834c40dc28cf42e296040e45`, recorded in `artifact_inventory.csv`). It contains all five source columns used by `build_reference_packet` for every variant.

The canonical saved verification records are `outputs/final_eval_v2/post_recovery/claims/verification/verification_checkpoint.jsonl` (3,600 unique explanation keys) and `verified_claims.csv`. There are 3,515 complete and 85 N/A records (one extraction N/A plus 84 unresolved verification N/As). The checkpoint preserves a `reference_packet_hash` for 3,512 complete records; the three successfully recovered records inherited an N/A source record and Stage 4D did not add that field when replacing its verification payload (`stage4d_v2.py:272-283`). This is a saved-provenance limitation, not a variant-specific reference pool.

For all **900** complete `(paper_case_id, generation_model)` groups in the frozen 3,600-row input, I compared SHA-256 values of the five source fields across its four variants. Each group contained exactly the four variants; there were **0 field mismatches**. This is a stronger saved-input comparison than the five required examples.

## Deterministic examples

Selection rule: the first five lexical `paper_case_id` values, using the pinned `gemma3:12b@f4031aab637d:temperature=0.0:max_tokens=240:think=False` generator. Each row below represents the SHA-256 of the field value shared by all four variants within that case. Each selected checkpoint record was complete.

| Case | Query | Request | Locked item | Item evidence | Rule evidence |
|---|---|---|---|---|---|
| `V2_TEST_0000_accessories` | `6b0cc30739bd71dd1bef112e6d70f1fc039110a352fff68d5c4cbee729ffc7a4` | `35f6f731b4027e2b27d2d5a8b510e39cdb26e0b1f28e7db191497fd88e00b5b8` | `42bca2eb84be4e9e59edf27af984a6f1f70dcc6f3f2a29f8cd4147213868c326` | `42bca2eb84be4e9e59edf27af984a6f1f70dcc6f3f2a29f8cd4147213868c326` | `ff76b21f586fe8973099eaf94adf90d2b00b9bcfc47a8c870ddfa77ee6a45d47` |
| `V2_TEST_0001_accessories` | `8edb161ffa295289da7a598fe6f5f68ebcce8b1f43e4134ccc6865a26f1b6865` | `35f6f731b4027e2b27d2d5a8b510e39cdb26e0b1f28e7db191497fd88e00b5b8` | `08913b64d5e7bc9194cc0225ed4f018b93f88f1741bf316ff1d34339a6b77a7e` | `08913b64d5e7bc9194cc0225ed4f018b93f88f1741bf316ff1d34339a6b77a7e` | `9695e5d1ceb6fb9cc7486d1a8f9b91bc7eb26533214aec0c33327d99e2570616` |
| `V2_TEST_0002_accessories` | `f4eec4b47510598a5900af172a645e2cae8394dd26ef8bd18be027aab2e03795` | `35f6f731b4027e2b27d2d5a8b510e39cdb26e0b1f28e7db191497fd88e00b5b8` | `1b7b95da41d382524c1448f6503006ff9dad828a54ae43140e2cd580bdb58223` | `1b7b95da41d382524c1448f6503006ff9dad828a54ae43140e2cd580bdb58223` | `9fdac2ecede9ff943734c1628a1d92924450a7f762d859cf1fd11d1079c4f4c5` |
| `V2_TEST_0003_accessories` | `24b953f4020abc2c24768ce33deb0b1e5c58b87ecb4e2f2916d4ea1c0d0cc696` | `35f6f731b4027e2b27d2d5a8b510e39cdb26e0b1f28e7db191497fd88e00b5b8` | `181dfae7f18569a1eb43015ea5a96c578dfd005f874079f2473b3bdf44c920b6` | `181dfae7f18569a1eb43015ea5a96c578dfd005f874079f2473b3bdf44c920b6` | `b2354bb68be274fd2275877899c6f4471cb09cd2df3ac74a9d9b5907fccb24ed` |
| `V2_TEST_0004_accessories` | `113b5a4fc971b19772c0ecf9d9abefd13413b7e951bf23c006141ba7d02f14de` | `35f6f731b4027e2b27d2d5a8b510e39cdb26e0b1f28e7db191497fd88e00b5b8` | `6e0f63cb27ccb8beeb4d913fea22a94575996141d956bec4eb605c6f388fbecc` | `6e0f63cb27ccb8beeb4d913fea22a94575996141d956bec4eb605c6f388fbecc` | `7176d425db973b9e8991e2a9f9a8694544ff967cba7be8fe814751ad5a06ba10` |

For example, the four stored `reference_packet_hash` values for `V2_TEST_0000_accessories` are different: `no_rag` `02b08bcf6607cc977a1d4746f46216ec2fe97011a425c4eb8a6893695a0826aa`, `item_rag` `cd8f9f3c7d25843ef7cd4fa3c017952fa64f42e7224d5d80ff0ea6930511069f`, `rule_rag` `76a941dacf0eb1ce9fca60986874e74cbb6d38b30b9a323d082208cd4fbac6ba`, and `hybrid_rag` `ca090d80fdaf07c941040b71c45d50a11d20eb81e2483b2f6d14338ead294151`. This is explained exactly by the two deliberately variant-specific availability flags being included in `ReferencePacket.fingerprint` (`claim_evaluation.py:59-61`), while the five evidence/reference-field hashes above are identical.

## Limitations

- Prompt payloads were not separately serialized as files; the frozen explanation input plus the deterministic packet builder and saved `reference_packet_hash` values are the available direct provenance chain.
- N/A records have no verification prompt/hash because Stage 4B correctly skipped prompt creation when extraction failed or retained an unresolved failure. Their frozen input rows were still included in the all-900-group common-field comparison.
- This audit establishes equality of the reference *pool*, not equality of the extracted claims or model responses; those intentionally vary by explanation.

No verification, extraction, judging, generation, API call, or frozen-artifact modification was performed for this audit.
