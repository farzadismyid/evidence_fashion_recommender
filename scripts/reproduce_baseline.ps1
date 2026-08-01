param(
    [ValidateSet("cpu", "cuda")]
    [string]$Profile = "cuda",
    [switch]$RegenerateEmbeddings
)

$ErrorActionPreference = "Stop"

uv sync --frozen --extra dev --extra $Profile
uv run efr --config configs/paper_baseline.yaml doctor
uv run efr --config configs/paper_baseline.yaml audit-kb
uv run pytest

if ($RegenerateEmbeddings) {
    uv run efr --config configs/paper_baseline.yaml --set cache.policy=refresh build-embeddings
}
else {
    uv run efr --config configs/paper_baseline.yaml build-embeddings
}

uv run efr --config configs/paper_baseline.yaml build-indexes
uv run efr --config configs/paper_baseline.yaml evaluate-ranking
$BaselineRun = Get-ChildItem outputs/runs -Directory |
    Where-Object Name -Like "paper-baseline-v3_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv run efr --config configs/paper_improved.yaml evaluate-ranking
$ImprovedRun = Get-ChildItem outputs/runs -Directory |
    Where-Object Name -Like "paper-improved-light-rerank_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv run efr --config configs/paper_improved.yaml build-study-cases `
    --output outputs/modular_study_cases.csv
uv run efr --config configs/paper_baseline.yaml run-explanation-study `
    --input outputs/modular_study_cases.csv
$StudyRun = Get-ChildItem outputs/runs -Directory |
    Where-Object Name -Like "paper-baseline-v3_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

uv run efr build-final-report `
    --baseline-run $BaselineRun.FullName `
    --improved-run $ImprovedRun.FullName `
    --study-run $StudyRun.FullName `
    --output outputs/final
