"""Score a completed blinded audit; this script never calls a model."""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path("outputs/final_eval_v2/manual_audit")
LABELS = ("supported_by_rule_evidence", "supported_by_item_evidence", "supported_by_query_or_locked_item", "unsupported", "contradicted", "not_verifiable")


def load_completed_annotations(path: Path = ROOT / "blinded_360_claims.csv") -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if {"anonymous_audit_id", "human_label", "human_notes"} - set(frame):
        raise ValueError("Annotation file has an invalid schema.")
    if len(frame) != 360 or frame.anonymous_audit_id.duplicated().any():
        raise ValueError("Annotation file must retain 360 unique audit IDs.")
    if (frame.human_label.astype(str).str.strip() == "").any():
        raise ValueError("Refusing to score: complete every human_label first.")
    unknown = set(frame.human_label) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown human labels: {sorted(unknown)}")
    return frame


def _metrics(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    return {
        "overall_agreement": float(np.average(frame.support_label == frame.human_label, weights=weights)),
        "cohen_kappa": float(cohen_kappa_score(frame.support_label, frame.human_label, labels=LABELS, sample_weight=weights)),
        "macro_f1": float(f1_score(frame.support_label, frame.human_label, labels=LABELS, average="macro", zero_division=0, sample_weight=weights)),
        "weighted_f1": float(f1_score(frame.support_label, frame.human_label, labels=LABELS, average="weighted", zero_division=0, sample_weight=weights)),
    }


def score_completed_annotations(annotations: Path = ROOT / "blinded_360_claims.csv", key_path: Path = ROOT / "audit_key_DO_NOT_OPEN_UNTIL_ANNOTATION_COMPLETE.csv", output_dir: Path = ROOT / "scored") -> None:
    """Score only a completed annotation file; produce no model output."""
    frame = load_completed_annotations(annotations).merge(pd.read_csv(key_path), on="anonymous_audit_id", validate="one_to_one")
    if len(frame) != 360:
        raise ValueError("Annotation/key merge is incomplete.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"analysis": "unweighted_audit_sample", **_metrics(frame)}, {"analysis": "sampling_weighted_population", **_metrics(frame, frame.sampling_weight.to_numpy())}]).to_csv(output_dir / "agreement_summary.csv", index=False)
    p, r, f, n = precision_recall_fscore_support(frame.support_label, frame.human_label, labels=LABELS, zero_division=0)
    pd.DataFrame({"label": LABELS, "precision": p, "recall": r, "f1": f, "support": n}).to_csv(output_dir / "per_label_metrics.csv", index=False)
    pd.DataFrame(confusion_matrix(frame.support_label, frame.human_label, labels=LABELS), index=LABELS, columns=LABELS).to_csv(output_dir / "confusion_matrix.csv")
    frame[frame.support_label != frame.human_label].to_csv(output_dir / "disagreement_examples.csv", index=False)
    rng, cases, reps = np.random.default_rng(42), frame.paper_case_id.unique(), []
    for _ in range(5000):
        draw = rng.choice(cases, size=len(cases), replace=True)
        reps.append(_metrics(pd.concat([frame[frame.paper_case_id == case] for case in draw], ignore_index=True)))
    interval = pd.DataFrame(reps).quantile([0.025, 0.975]).T.reset_index(names="metric")
    interval.columns = ["metric", "ci_low", "ci_high"]
    interval.to_csv(output_dir / "case_clustered_95ci.csv", index=False)


if __name__ == "__main__":
    score_completed_annotations()
