"""Case-clustered bootstrap analysis for post-recovery claim verification."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

VARIANTS = ("no_rag", "item_rag", "rule_rag", "hybrid_rag")
SUPPORT_LABELS = {
    "supported_by_rule_evidence",
    "supported_by_item_evidence",
    "supported_by_query_or_locked_item",
}
METRICS = {
    "any_permitted_evidence_support": lambda frame: frame["support_label"].isin(SUPPORT_LABELS),
    "generation_available_evidence_support": lambda frame: (
        (
            (frame["grounding_variant"].isin(["item_rag", "hybrid_rag"]))
            & (frame["support_label"] == "supported_by_item_evidence")
        )
        | (
            (frame["grounding_variant"].isin(["rule_rag", "hybrid_rag"]))
            & (frame["support_label"] == "supported_by_rule_evidence")
        )
    ),
    "rule_support": lambda frame: frame["support_label"] == "supported_by_rule_evidence",
    "item_support": lambda frame: frame["support_label"] == "supported_by_item_evidence",
    "query_locked_item_support": lambda frame: (
        frame["support_label"] == "supported_by_query_or_locked_item"
    ),
    "unsupported": lambda frame: frame["support_label"] == "unsupported",
    "contradicted": lambda frame: frame["support_label"] == "contradicted",
    "not_verifiable": lambda frame: frame["support_label"] == "not_verifiable",
}
NA_COVERAGE = "claim_evaluation_na_coverage"
PAIRED_COMPARISONS = (
    ("item_rag", "no_rag"),
    ("rule_rag", "no_rag"),
    ("hybrid_rag", "no_rag"),
    ("rule_rag", "item_rag"),
    ("hybrid_rag", "item_rag"),
    ("hybrid_rag", "rule_rag"),
)


def _percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.nanquantile(values, [alpha / 2, 1.0 - alpha / 2])
    return float(lower), float(upper)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def build_case_clusters(verified_claims: pd.DataFrame, checkpoint: pd.DataFrame) -> pd.DataFrame:
    """Aggregate source rows to test-case/variant clusters without changing denominators."""

    identity = ["paper_case_id", "grounding_variant", "generation_model"]
    _require_columns(
        verified_claims,
        [*identity, "verification_status", "support_label"],
        "verified claims",
    )
    _require_columns(checkpoint, [*identity, "verification_status"], "verification checkpoint")
    if checkpoint.duplicated(identity).any():
        raise ValueError("Verification checkpoint explanation identities must be unique.")

    complete = verified_claims[verified_claims["verification_status"] == "complete"].copy()
    claim_rows = complete.groupby(["paper_case_id", "grounding_variant"], as_index=False).size()
    claim_rows = claim_rows.rename(columns={"size": "claim_denominator"})
    for metric, predicate in METRICS.items():
        counts = (
            complete.assign(_count=predicate(complete).astype(int))
            .groupby(["paper_case_id", "grounding_variant"], as_index=False)["_count"]
            .sum()
        )
        claim_rows = claim_rows.merge(
            counts.rename(columns={"_count": f"{metric}_numerator"}),
            on=["paper_case_id", "grounding_variant"],
            how="outer",
        )

    explanation_counts = (
        checkpoint.groupby(["paper_case_id", "grounding_variant"], as_index=False)
        .size()
        .rename(columns={"size": "explanation_denominator"})
    )
    na_counts = (
        checkpoint.assign(_na=(checkpoint["verification_status"] == "N/A").astype(int))
        .groupby(["paper_case_id", "grounding_variant"], as_index=False)["_na"]
        .sum()
    )
    clusters = explanation_counts.merge(
        na_counts.rename(columns={"_na": f"{NA_COVERAGE}_numerator"}),
        on=["paper_case_id", "grounding_variant"],
        validate="one_to_one",
    ).merge(claim_rows, on=["paper_case_id", "grounding_variant"], how="left")
    clusters["claim_denominator"] = clusters["claim_denominator"].fillna(0).astype(int)
    for metric in METRICS:
        clusters[f"{metric}_numerator"] = clusters[f"{metric}_numerator"].fillna(0).astype(int)
    return clusters


def _ordered_cluster_arrays(
    clusters: pd.DataFrame,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cases = sorted(clusters["paper_case_id"].unique())
    expected = pd.MultiIndex.from_product(
        [cases, VARIANTS], names=["paper_case_id", "grounding_variant"]
    )
    indexed = clusters.set_index(["paper_case_id", "grounding_variant"]).reindex(expected)
    if indexed["explanation_denominator"].isna().any():
        raise ValueError("Every sampled test case must contain all four variants.")
    if (indexed["explanation_denominator"] <= 0).any():
        raise ValueError("Every case/variant cluster must contain at least one explanation.")
    numerators: dict[str, np.ndarray] = {}
    denominators: dict[str, np.ndarray] = {}
    for metric in METRICS:
        numerators[metric] = (
            indexed[f"{metric}_numerator"].to_numpy().reshape(len(cases), len(VARIANTS))
        )
        denominators[metric] = (
            indexed["claim_denominator"].to_numpy().reshape(len(cases), len(VARIANTS))
        )
    numerators[NA_COVERAGE] = (
        indexed[f"{NA_COVERAGE}_numerator"].to_numpy().reshape(len(cases), len(VARIANTS))
    )
    denominators[NA_COVERAGE] = (
        indexed["explanation_denominator"].to_numpy().reshape(len(cases), len(VARIANTS))
    )
    return cases, numerators, denominators


def run_case_clustered_claim_bootstrap(
    verified_claims: pd.DataFrame,
    checkpoint: pd.DataFrame,
    *,
    samples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Return case-clustered metric CIs and paired differences.

    A sampled case contributes every generator, variant, and claim it has in the
    canonical tables. Claim failures remain outside claim-rate denominators; N/A
    coverage is separately computed from explanation-level checkpoint rows.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    clusters = build_case_clusters(verified_claims, checkpoint)
    cases, numerators, denominators = _ordered_cluster_arrays(clusters)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cases), size=(samples, len(cases)))
    estimate_rows = []
    bootstrap_rates: dict[str, np.ndarray] = {}
    for metric in [*METRICS, NA_COVERAGE]:
        numerator = numerators[metric]
        denominator = denominators[metric]
        point = numerator.sum(axis=0) / denominator.sum(axis=0)
        sampled_numerator = numerator[draws].sum(axis=1)
        sampled_denominator = denominator[draws].sum(axis=1)
        rates = np.divide(
            sampled_numerator,
            sampled_denominator,
            out=np.full(sampled_numerator.shape, np.nan, dtype=float),
            where=sampled_denominator != 0,
        )
        bootstrap_rates[metric] = rates
        for index, variant in enumerate(VARIANTS):
            lower, upper = _percentile_interval(rates[:, index], confidence_level)
            generation_applicable = (
                metric != "generation_available_evidence_support" or variant != "no_rag"
            )
            estimate_rows.append(
                {
                    "metric": metric,
                    "grounding_variant": variant,
                    "applicable": generation_applicable,
                    "numerator": int(numerator[:, index].sum()),
                    "denominator": int(denominator[:, index].sum()),
                    "estimate": float(point[index]) if generation_applicable else np.nan,
                    "ci_lower": lower if generation_applicable else np.nan,
                    "ci_upper": upper if generation_applicable else np.nan,
                    "cluster_count": len(cases),
                    "bootstrap_samples": samples,
                    "seed": seed,
                }
            )

    difference_rows = []
    for metric in ("any_permitted_evidence_support", "unsupported"):
        point = numerators[metric].sum(axis=0) / denominators[metric].sum(axis=0)
        rates = bootstrap_rates[metric]
        for first, second in PAIRED_COMPARISONS:
            first_index, second_index = VARIANTS.index(first), VARIANTS.index(second)
            differences = rates[:, first_index] - rates[:, second_index]
            lower, upper = _percentile_interval(differences, confidence_level)
            finite_differences = differences[np.isfinite(differences)]
            raw_p_value = 2 * min(
                float((finite_differences <= 0).mean()),
                float((finite_differences >= 0).mean()),
            )
            raw_p_value = min(raw_p_value, 1.0)
            difference_rows.append(
                {
                    "metric": metric,
                    "variant_a": first,
                    "variant_b": second,
                    "difference_percentage_points": float(
                        100 * (point[first_index] - point[second_index])
                    ),
                    "ci_lower_percentage_points": float(100 * lower),
                    "ci_upper_percentage_points": float(100 * upper),
                    "p_value": raw_p_value if raw_p_value else 1.0 / samples,
                    "p_value_display": f"< 1/{samples}"
                    if raw_p_value == 0
                    else f"{raw_p_value:.6g}",
                    "p_value_raw": raw_p_value,
                    "cluster_count": len(cases),
                    "bootstrap_samples": samples,
                    "seed": seed,
                }
            )
    metadata: dict[str, int | float] = {
        "test_cases": len(cases),
        "case_variant_clusters": len(clusters),
        "bootstrap_samples": samples,
        "confidence_level": confidence_level,
        "seed": seed,
    }
    return pd.DataFrame(estimate_rows), pd.DataFrame(difference_rows), metadata
