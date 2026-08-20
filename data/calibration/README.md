# Stage 5 human calibration annotations

This directory is intentionally empty of completed annotations until a human annotator supplies
them. Never populate it with model-generated labels or final-test cases.

Use [`stage5_annotation_template.json`](stage5_annotation_template.json) as the record schema.
Create `stage5_annotations.jsonl` with one complete JSON object per condition. Each
`calibration_case_id` must appear once for `no_rag` and once for `rule_rag`; every record must
come from the validation split, so it is disjoint from the final test-set explanations.

The annotation decision rules are:

- Atomize independently checkable propositions, retaining textual order and `C1`, `C2`, … IDs.
- Label full-KB entailment, exact-trace entailment, and common-reference item-fact support
  independently. Do not fill one label by copying another.
- Treat the supplied full-KB rules as candidates: decide their antecedent applicability before
  using them for full-KB entailment. Rule IDs and citations use the canonical `K###` namespace.
- Keep citation validity separate: mark grouped citations (for example `[K025, K099]`) invalid.
- Use `not_verifiable` when evidence cannot settle a claim and `contradicted` only for affirmative
  conflict.
- Include the required coverage tags across the paired corpus, including both conditions and at
  least one bag example.

Run `python scripts/run_stage5_calibration.py --validate-only` before model calibration. It will
refuse incomplete, non-human, out-of-split, unpaired, or under-covered annotations.
