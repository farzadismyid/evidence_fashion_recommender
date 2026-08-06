# Blinded claim-verification annotation guide

Annotate each atomic claim using only the displayed query/request, locked item, retrieved item evidence, and retrieved expert-rule evidence. Do not infer the generation condition or automatic decision.

Use exactly one label: `supported_by_rule_evidence` when rule evidence semantically entails the claim; `supported_by_item_evidence` when retrieved item evidence entails it; `supported_by_query_or_locked_item` when query/request/locked-item information entails it; `unsupported` when available evidence does not support it; `contradicted` when available evidence conflicts with it; `not_verifiable` when evidence is insufficient, ambiguous, or cannot settle it.

Apply semantic support, not word overlap. A related rule is not support unless it entails the specific claim. If more than one source supports a claim, choose the most direct source (rule, item, then query/locked item) and explain the additional support in notes. Use `contradicted` only for affirmative conflict, not absence. Use `not_verifiable` for genuinely indeterminate claims and `unsupported` for claims that are not entailed despite adequate relevant context. Record concise reasoning in `human_notes`.

Source-selection rule: choose the evidence source that most directly and explicitly entails the claim. Use supported_by_query_or_locked_item only when explicit in query/request/locked-item metadata; supported_by_item_evidence only when additional retrieved item evidence supports it beyond the locked item; supported_by_rule_evidence for styling, compatibility, formality, occasion, colour, or relational claims entailed by an expert rule. If multiple sources are equally direct, choose the most specific and record alternatives in human_notes.
