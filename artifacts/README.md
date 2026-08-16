# Artifacts

This directory contains compact, Git-tracked manifests and publication artifacts. Human and
external evaluation audits are outside the final study scope, and no such packet or result is
retained.
Large or mutable runtime outputs belong under the ignored runtime root configured in
`configs/experiment.yaml`. Successful outputs are content-hashed and are never overwritten.

`tables/table_stage8_grounding_revision.csv` is the single canonical derived table for the Stage 8
visible-evidence, claim-role, micro/macro/paired, category, generator, and normalization-sensitivity
analyses. The original `table_stage8_claim_verification.csv` remains unchanged as the completed
common-reference A+B baseline.

`tables/table_stage8_study_specific_metrics.csv` is the additive canonical table for DTA, UIAR,
unsupported attribute density, rule-citation precision/coverage, and the unchanged secondary
grounding/reference rates. Its 30-pair closest-length subset is sensitivity-only.

`tables/table_07_publication_readiness.csv` is a frozen Stage 6-era snapshot. The current authority
is `tables/table_stage10_release_readiness.csv`, bound by
`manifests/stage10_release_manifest.json`. Stage 10 performs deterministic cleanup and integrity
checks without rerunning Stages 1–8.
