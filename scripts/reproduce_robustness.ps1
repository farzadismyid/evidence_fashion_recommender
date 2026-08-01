param(
    [ValidateSet("cpu", "cuda")]
    [string]$Profile = "cuda",
    [switch]$SkipEnvironmentSync
)

$ErrorActionPreference = "Stop"

if (-not $SkipEnvironmentSync) {
    uv sync --frozen --extra dev --extra $Profile
}

uv run efr --config configs/robustness.yaml doctor
uv run pytest

if (-not (Test-Path outputs/robustness/before_baseline)) {
    uv run efr freeze-baseline
}

uv run efr --config configs/robustness.yaml build-robustness-schedules

foreach ($Split in @("development", "validation", "test")) {
    uv run efr --config configs/robustness.yaml build-study-cases `
        --schedule "outputs/robustness/schedules/${Split}_schedule.csv" `
        --output "outputs/robustness/${Split}_cases.csv"
}

uv run efr --config configs/robustness.yaml tune-reranking
uv run efr --config configs/robustness.yaml evaluate-heldout-ranking
uv run efr --config configs/robustness.yaml run-hybrid-ablations
uv run efr --config configs/robustness.yaml run-robustness-study
