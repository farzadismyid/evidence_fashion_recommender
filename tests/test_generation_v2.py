import json
from pathlib import Path

import pandas as pd

from evidence_fashion_recommender.cache import file_fingerprint
from evidence_fashion_recommender.evaluation.generation_v2 import validate_generation_inputs


def test_generation_requires_corrected_frozen_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cases = pd.DataFrame(
        [
            {
                "paper_case_id": "T1",
                "research_split": "test",
                "stage1_packet_protocol": "final_eval_v2_selected",
                "stage1_packet_hash": "packet",
                "query_text": "blazer",
                "user_request": "accessory",
                "recommended_text": "scarf",
                "item_evidence_text": "scarf",
                "rule_evidence_ids": "[]",
                "rule_evidence_text": "rule",
            }
        ]
    )
    packets = Path("packets.csv")
    cases.to_csv(packets, index=False)
    reranking = Path("reranking.json")
    reranking.write_text(
        json.dumps(
            {
                "selection_policy": "evidence_in_loop_pareto_v2",
                "clip_weight": 0.75,
                "evidence_weight": 0.25,
            }
        ),
        encoding="utf-8",
    )
    hybrid = Path("hybrid.json")
    hybrid.write_text(
        json.dumps(
            {
                "selected_on": "validation",
                "candidate_type": "hybrid",
                "item_count": 2,
                "max_words": 35,
                "rule_limit": 5,
                "prompt_order": "item_first",
            }
        ),
        encoding="utf-8",
    )
    decision = Path("decision.json")
    decision.write_text(json.dumps({"decision": "regenerate_all_variants"}), encoding="utf-8")
    freeze = Path("freeze.json")
    freeze.write_text(
        json.dumps(
            {
                "source": {"commit": "abc", "dirty": False},
                "input_hashes": {str(packets): file_fingerprint(packets)},
                "selections": {
                    "reranking": {"sha256": file_fingerprint(reranking)},
                    "hybrid": {"sha256": file_fingerprint(hybrid)},
                },
                "gate_definition": {"decision": "regenerate_all_variants"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evidence_fashion_recommender.evaluation.generation_v2.git_revision",
        lambda: {"commit": "abc", "dirty": False},
    )
    selected, packet_hash = validate_generation_inputs(
        cases,
        input_path=packets,
        reranking_selection_path=reranking,
        hybrid_selection_path=hybrid,
        decision_path=decision,
        freeze_path=freeze,
    )
    assert selected["item_count"] == 2
    assert packet_hash == "packet"
