# Stage 13 Release Audit

Status: **release_frozen_with_recorded_terminal_failures**.

## Canonical-output checks

- **Frozen 1,000 recommendation cases** — pass; expected `1000`, observed `1000`.
- **Frozen 500 explanation cases** — pass; expected `500`, observed `500`.
- **Five-way 100-case category balance** — pass; expected `True`, observed `True`.
- **Stage 9 matrix cells** — pass; expected `3000`, observed `3000`.
- **Stage 9 accepted explanations** — pass; expected `2987`, observed `2987`.
- **Stage 9 terminal failures** — pass; expected `13`, observed `13`.
- **Stage 10 extraction records** — pass; expected `2987`, observed `2987`.
- **Stage 10 extracted claims** — pass; expected `17396`, observed `17396`.
- **Stage 11 complete verification records** — pass; expected `2986`, observed `2986`.
- **Stage 11 verified claims** — pass; expected `17389`, observed `17389`.
- **Stage 11 terminal failures** — pass; expected `1`, observed `1`.
- **Qualitative rows use frozen explanation IDs** — pass; expected `15`, observed `15`.
- **Canonical live output paths are unique** — pass; expected `3`, observed `3`.
- **Rebuilt DOCX chapters** — pass; expected `5`, observed `5`.
- **Obsolete explanation metrics in thesis** — pass; expected `0`, observed `0`.

## Figure checks

- **stage12_support_rates** — pass; SVG=True, PNG=2400x1410 at 300x300 DPI.
- **stage12_uifr** — pass; SVG=True, PNG=2400x1410 at 300x300 DPI.
- **stage12_supported_per_100** — pass; SVG=True, PNG=2400x1410 at 300x300 DPI.

## Test checks

- **D:\Projects\evidence_fashion_recommender\.venv\Scripts\python.exe -m pytest -q** — pass; 86 passed, 16 skipped, 1 warning in 1.44s
- **D:\Projects\evidence_fashion_recommender\.venv\Scripts\python.exe -m ruff check src tests scripts/run_stage13_release_audit.py scripts/build_thesis_chapters.py** — pass; All checks passed!

## Recorded limitations

- Stage 9 retains 13 terminal generation failures; all accepted outputs remain frozen.
- Stage 11 retains one terminal verification record (seven claims); analyses use paired available records.
- Stage 12 metrics are frozen automated-evaluator measures, not human preference or world-factual truth.

## Canonical provenance

- `.runtime\current\explanations\stage9-v3-generation-b691865366b3\explanations.jsonl` — `30bccb5d398b4eff5cdd099315d2f2a49d82cec74274dba6644461798d16a101`
- `.runtime\current\extraction\stage10-claim-extraction-80c0a1d2df6c\extractions.jsonl` — `d92e64c77a8542749b39423ada9a385031824bcc6c9fa1c8dc58bc150ad8a9e0`
- `.runtime\current\verification\stage11-claim-verification-d92e64c77a85\verifications.jsonl` — `537e9b54e3b30e28bb114eb6ae3c715d513143430b4000014e9caa9549b65d4f`
- `data\kb\fashion_rules_v3.csv` — `5f2d3b7ecaf165aaff02e9447d005d4017ea6df0d3b52a425701cb77b86d3aef`
- `configs\experiment.yaml` — `532300dd6facd7c129a4165be729269993fb41cda2dd75da85aed4ca865fc85e`
- `configs\models.yaml` — `d008a3d7217c0d6df5c8e86fea6bbfff25208921774f10acd823e05a1205ceb1`
- `configs\prompts.yaml` — `deb2f8832633560fab4dcab175a38e381766b35e3b76bc9ff826f0cd9a24fae7`
- `artifacts\manifests\data_preparation_leakage_resolved_manifest.json` — `a68b4f5161eabc17bd0704c09745c160493a08a92c62007474b80b48d593be5e`
