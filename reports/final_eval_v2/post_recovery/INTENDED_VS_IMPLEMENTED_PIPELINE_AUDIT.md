# Intended versus implemented pipeline audit

This is a code-and-frozen-packet audit. No model was run and no frozen artifact was changed.

## Observed end-to-end implementation

1. The query image is the Hugging Face/Polyvore `image` for `query_item_id`, located through `original_dataset_index`, converted to RGB and encoded by CLIP. Query CLIP text is `Fashion item: {category} | Description: {text} | Input group: {query group} | Request: {request} | Recommend: {target category}` (`evaluation/controlled.py:74-105`). Image and text embeddings are normalized and fused as `0.40 image + 0.60 text`, then normalized again for final v2 (`models/multimodal.py:13-29`; frozen fusion selection). Candidates use cached CLIP image/text target embeddings; the controlled pool is same-outfit positives plus up to 99 deterministic other-outfit negatives.
2. Candidate-specific rule retrieval/scoring uses query category, query group, query description, candidate raw category, candidate broad category, candidate raw text, request, and target category (`evaluation/evidence_ranking.py:15-31`). It retrieves top five category-restricted semantic rules, reliability-weights similarities, and reranks pool-local min--max CLIP/evidence scores at 0.75/0.25.
3. The locked recommendation is `ranked.iloc[0]` after that reranking (`v2_sources.py:220-269`). The rules supplied for C are retrieved again for that locked item by `scorer.retrieve`; they use the same candidate-specific retrieval text, category restriction, top-k, and accessory type filter as retrieval. However, `score()` selects top-k before the accessory type filter whereas `retrieve()` may filter accessories, so exact score-contributor IDs and explanation C IDs are not guaranteed identical for accessory cases.
4. Explanation generation receives no image, image caption, image-derived attributes, or structured JSON packet. It receives only `query_text`, request, `recommended_text`, optional `item_evidence_text`, and rule ID/text lines (`generation.py:8-72`).

## A, B, and C as implemented

Common A is not minimal identity: it contains query textual description, request, the full locked `item_text`, and the literal grounding-variant label. B is a `Retrieved catalogue context` block. C is up to five `[rule_id] rule_text` lines plus a citation instruction.

The frozen selected-case constructor sets `recommended_text = item['item_text']` and `item_evidence_text = item['item_text']` (`v2_sources.py:258-269`). The equality audit confirms **3,600/3,600 (100%) exact and normalized equality**, including 900/900 in each variant. Thus B supplies no factual metadata beyond the locked item. Item-RAG and Hybrid-RAG should not be described as receiving rich additional product metadata. `item_evidence_packet` does contain category, text, item ID, outfit ID, position, broad category and original index, but none of that JSON is passed to the generator; the source dataset also has image, category, text and parsed outfit/item position, but no documented structured colour/material/caption attributes. A valid distinct B can be created without invention by passing deterministic, non-overlapping Polyvore fields (e.g., raw category and item position, and/or a documented image caption/preprocessing output) while removing them from A; it cannot be retrospectively claimed for this frozen evaluation.

The verifier packet includes both duplicated item field and locked item field, and its parser validates only schema/labels/claim IDs. It contains no deterministic source-priority rule; an LLM may label the same fact `supported_by_item_evidence` or `supported_by_query_or_locked_item`. This materially weakens interpretation of the Item-RAG ablation.

## Current prompt conditions (representative case)

For `V2_TEST_0000_accessories`, common text is query `open toe floral block heels black multi`, request `recommend accessories that complete this outfit`, and locked item `Clutches | shein sheinside black rivet floral clutch bag`. Current No-RAG is A plus instruction text; Item-RAG adds exactly the duplicated line below; Rule-RAG adds the five frozen `[R050] ...` rule lines; Hybrid adds both. The implementation’s exact differences are:

```text
No-RAG:    no Retrieved catalogue context; no Retrieved expert rules.
Item-RAG:  Retrieved catalogue context:
           Clutches | shein sheinside black rivet floral clutch bag
Rule-RAG:  Retrieved expert rules:
           [R050] ... [R045] ... [R042] ... [R052] ... [R030] ...
           The explanation must cite at least one provided rule ID exactly in square brackets.
Hybrid-RAG: both Item-RAG and Rule-RAG blocks (rules first under the frozen selected order).
```

All four also contain `Grounding variant: {variant}`, a condition not intended in A and an experimental-label leakage risk within the generator prompt (not the blinded audit). The full constructor is the authoritative rendered template in `generation.py:8-72`; frozen exact rule text is in `explanations.csv:rule_evidence_text` for this case.

## Corrected prospective prompt design (do not apply to frozen evaluation)

Define A strictly as request + query textual context + minimal locked identity (`locked category` and item name/ID). Define B as a deterministic metadata block not repeated in A: query/locked raw category, item position, and separately documented caption or attributes only if actually derived and stored. Define C as the exact rule IDs/texts used by the locked candidate’s scoring trace, with that trace stored.

```text
No-RAG: A only. Explain why the locked item fits; do not infer unprovided attributes.
Item-RAG: A + B. Use only the additional deterministic metadata when it explicitly supports a claim.
Rule-RAG: A + C. Cite only an exact supplied rule ID when its text entails the claim.
Hybrid-RAG: A + B + C. Prefer the most direct source; do not repeat A as B.
```

This prospective design is cleanly non-overlapping and makes the item ablation interpretable. It requires a new preregistered evaluation; it must not be retrofitted onto existing results.
