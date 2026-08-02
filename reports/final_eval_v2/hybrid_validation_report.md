# Final Evaluation v2 Hybrid Validation

Stage 1 packet hash: `c561b2016ba8ad7b8d06aac7cb3dd8a56793e91adfa5f4f70186a448ae3188f9`

Screened 36 configurations on 50 balanced validation cases; evaluated 6 finalists on all 300 validation cases.

Selection used the frozen priority hierarchy with no weighted composite.

## Selected configuration

```json
{
  "name": "hybrid_w35_r5_i2_item_first",
  "max_words": 35,
  "rule_limit": 5,
  "item_count": 2,
  "item_limit": 2,
  "prompt_order": "item_first",
  "candidate_type": "hybrid",
  "selected_on": "validation",
  "selection_protocol": "priority_v2_no_weighted_composite",
  "stage1_packet_hash": "c561b2016ba8ad7b8d06aac7cb3dd8a56793e91adfa5f4f70186a448ae3188f9",
  "packet_source_protocol": "final_eval_v2_selected",
  "generator_model": "llama3.2@a80c4f17acd5:temperature=0.0:max_tokens=240:think=False",
  "judge_model": "qwen3:8b@500a1f067a9f:temperature=0.0:max_tokens=400:think=False",
  "practical_tie": 0.01
}
```
