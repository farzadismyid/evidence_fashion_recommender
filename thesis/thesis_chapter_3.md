# Chapter 3

# Methodology

## 3.1 Introduction

This chapter describes the research design, implementation, and evaluation procedure used to investigate evidence-constrained multimodal fashion recommendation. The central methodological problem was not simply to produce a plausible recommendation. It was to separate three questions that are often conflated: whether a multimodal ranker retrieves compatible items; whether expert evidence materially participates in the ranking decision; and whether a natural-language explanation is faithful to the stored decision trace without adding unverified product attributes. The experiment was therefore organised as a staged, frozen pipeline. Earlier stages prepared the data, representations, candidate pools, and rule base; validation-only stages fixed every tunable choice; the confirmatory recommendation experiment then locked one recommendation and its exact five-rule trace for each case; and the explanation experiment compared two texts for the same locked decision.

The design follows a paired-comparison principle. For each explanation case, common context A contained the user request, query-item text and identity, and locked recommended-item text and identity. Exact trace B contained the five rules that contributed to the Stage 6 evidence score. The No-RAG generator received A alone. The Rule-RAG generator received A and B, while recommendation identity was held constant. This intervention isolates access to the decision trace at the explanation stage. It does not isolate every possible effect of prompt wording or output length, and it does not turn the No-RAG condition into a grounded explanation when its text happens to agree with B. Accordingly, B agreement is termed *post-hoc decision-trace alignment* for No-RAG and *decision-trace faithfulness* for Rule-RAG.

The work is an offline systems experiment rather than a user study. Recommendation relevance comes from co-occurrence within held-out Polyvore outfits. Explanation assessment uses a frozen secondary model to extract atomic claims, verify their relationship to A and B, and apply general quality scales. Study-specific deterministic post-processing converts those saved records into Decision-Trace Alignment (DTA), Unsupported Item-Attribute Rate (UIAR), citation precision, citation coverage, and verbosity-normalised attribute density. These are operational measures for this study, not universal benchmarks. No human or independent external audit is included in the final experimental boundary, so the automated evaluation is treated as system-level evidence with explicit limitations.

## 3.2 Research questions and experimental logic

The methodology addresses four linked research questions. RQ1 asks whether making the exact expert-rule trace visible during generation improves explanation alignment with the actual reranking decision. RQ2 asks whether the same intervention reduces concrete item-attribute assertions that are unsupported by the supplied evidence. RQ3 asks whether broad automated quality judgments agree with the more targeted faithfulness measures. RQ4 asks whether observed differences are stable across language generators, fashion categories, aggregation levels, and a predeclared length sensitivity.

The causal contrast is deliberately narrow. Stage 6 produces a locked candidate \(i^*\) for case \(q\) and a trace \(B_q\). Stage 7 then generates

\[
E^{N}_{q,g}=G_g(A_q), \qquad E^{R}_{q,g}=G_g(A_q,B_q),
\]

where \(g\) denotes one of three generators, \(E^N\) is the No-RAG text, and \(E^R\) is the Rule-RAG text. Both texts explain the same \(i^*\). Candidate quality, query content, generator identity, decoding seed, and generation settings are paired. The manipulated factor is whether B is displayed and explicitly cited. A corrective follow-up applied the same numerical instruction—at most 75 words—to both conditions. The prompts still differ necessarily in evidence, citation, and grounding instructions, and actual lengths remain model-controlled outcomes. Headline claim measures are therefore rates, unsupported attributes are additionally expressed per 100 words, and a 30-pair closest-length cohort is retained as a sensitivity analysis.

The study does not claim that B is a complete account of a neural model’s internal computation. B is instead the exact, inspectable symbolic trace used by the evidence component of the deployed reranker. The term “decision trace” is thus architectural and operational: it identifies the five expert rules, their similarities, reliability weights, ordering, and contributions used to compute the evidence score. Faithfulness is measured relative to that trace. This boundary follows the distinction in explainable-AI research between a convincing rationale and an account tied to the mechanism being explained [1,2].

## 3.3 Staged research design and freezing policy

The project was implemented in ten numbered stages, with Stage 9 ultimately removed when independent auditing was excluded from the final scope. Stage 1 created a clean repository and migration inventory. Stage 2 prepared data, split outfits, resolved exact-image leakage, and constructed deterministic evaluation cases. Stage 3 produced pinned MiniLM and CLIP representations. Stage 4 implemented rule retrieval, evidence scoring, reranking, validation searches, and candidate-pool sensitivity. Stage 5 optimised only explanation-side Rule-RAG variables on validation data, ran a frozen pilot, and later added the explicitly labelled length-control follow-up. Stage 6 executed the confirmatory recommendation evaluation and locked cases; additive Stage 6b tested FashionCLIP without changing those locks. Stage 7 generated the complete paired explanation corpus under the corrected shared word budget. Stage 8 extracted with Qwen3, verified with Mistral, judged with Qwen3, and post-processed claims. Stage 10 performed release review, artifact hashing, test execution, and consistency checks.

Freezing served two purposes. First, it prevented test performance from influencing model weights, fusion weights, evidence weights, pool size, rule count, or prompts. Second, it preserved a stable provenance chain. Each major artifact was written once, bound to a SHA-256 digest, and named in a stage manifest. Model identifiers were accompanied by immutable revisions or local model digests. Configuration objects were canonicalised and hashed. Derived publication analyses used the saved Stage 7 and Stage 8 records and made zero new model calls.

The final confirmatory settings were fixed before the Stage 6 test evaluation: image/text CLIP fusion was 0.40/0.60; evidence reranking was CLIP/evidence 0.75/0.25; exactly five expert rules contributed to each trace; and the primary candidate pool contained approximately 100 candidates. Larger pools of approximately 500 and 1,000 were sensitivity settings rather than alternative primary estimates. The explanation generators, seeds, decoding settings, Rule-RAG prompt form, and claim schemas were frozen before full generation. The corrective extension changed only the No-RAG word instruction and verifier role, preserving unchanged Rule-RAG generations through output-hash validation and recording the extension separately from the original frozen run.

## 3.4 Data source, unit of analysis, and preprocessing

### 3.4.1 Dataset and fields

The data source was the `Marqo/polyvore` dataset at immutable revision `8c782ee447faf2d2a0402ac883cf07d3b3f43e1c`, configuration `default`, source split `data`, and fingerprint `9c97dc763773e2a2`. Polyvore-derived outfit data are widely used for fashion compatibility research because an outfit supplies a set of items curated to appear together [3,4]. The present study used only the raw item identifier, category, product text, outfit association, and image. Textual explanation evidence was intentionally limited: images entered the recommendation representation but were never captioned, classified, or converted into attributes for A or B.

The raw pinned source contained 94,096 items and 21,587 outfits. A validated keyword mapping assigned items to five broad target categories: accessories, bottoms, outerwear, shoes, and tops. The mapping reproduced 67,524 target items in the initial audit, whereas an earlier proposal table stated 66,749. Because no documented filtering rule reproduced the smaller number, the implementation retained the directly reproducible count and recorded the discrepancy rather than inventing a post-hoc exclusion. Subsequent leakage resolution and confirmatory eligibility produced 69,725 prepared items and 20,225 prepared outfits in the final Stage 6 table; differences between intermediate counts reflect the stage-specific prepared universe and are preserved in their respective manifests.

### 3.4.2 Outfit-disjoint splitting

The outfit, not the item row, was the primary split unit. Outfit IDs were ordered by SHA-256 over a fixed seed and assigned to exact quotas of 15,267 development outfits, 3,147 validation outfits, and 3,173 test outfits. This procedure is deterministic and prevents items from the same outfit appearing on opposite sides of the research split. Within the test partition, case selection used a separate seeded SHA-256 order and sampled 200 cases for each broad category, producing 1,000 confirmatory recommendation cases.

Exact duplicate images were audited by hashing image bytes. Twenty-one duplicate groups were identified, of which eleven crossed an initial research split. Outfits connected by shared exact-image hashes were treated as connected components. Each cross-split component was moved to the split of its lowest seeded-hash anchor; singleton outfits were then moved in a separate deterministic order to restore the original quotas. Thirteen outfits changed assignment in total: eleven duplicate-linked moves and two rebalancing moves. The final split retained its exact quotas and contained no cross-split outfit or exact-image leakage. This is stricter than relying on unique item IDs, because separately identified catalogue rows can still carry identical visual content.

### 3.4.3 Query construction and candidate pools

Each case selected a query item and a target category representing the missing outfit component. All known same-outfit positives in that target category were retained. Negatives were sampled from items of the same target category belonging to other test outfits. The query item was always excluded. Category restriction makes the task a controlled within-category ranking problem: the model chooses *which* pair of shoes or *which* top, rather than receiving credit merely for predicting the requested item type.

Stage 2 initially materialised pools with up to 999 negatives plus all positives to support later sensitivity analysis. The primary Stage 6 evaluation used up to 99 negatives plus positives, giving 99,238 candidate rows across 1,000 cases, a mean pool size of 99.238, and a range from 60 to 102. Values above 100 occur when more than one known positive must be retained. The design is sampled controlled-pool ranking, not full-catalogue retrieval. Consequently, absolute ranking values must be interpreted at this pool size, while the 500- and 1,000-candidate analyses show how performance changes as the task becomes more difficult.

## 3.5 Representation learning and baseline retrieval

### 3.5.1 Text representation

The text-only baseline used `sentence-transformers/all-MiniLM-L6-v2` at immutable revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Sentence-BERT-style encoders map sentences into a shared dense space suitable for cosine retrieval [5], while MiniLM distils transformer self-attention into a more compact architecture [6]. Product text was encoded into 384-dimensional float32 vectors and L2 normalised. Texts were truncated only according to the pinned encoder limit of 256 tokens. This pathway provided a lightweight semantic baseline distinct from CLIP text encoding.

### 3.5.2 CLIP image and text representation

The multimodal ranker used `openai/clip-vit-base-patch32` at immutable revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. CLIP learns aligned visual and language representations from image–text pairs [7]. Item images and product text were embedded into 512-dimensional float32 vectors. All vectors were L2 normalised, and Stage 3 validation confirmed maximum norm errors below \(1.2\times10^{-7}\). Distinct input items also produced non-zero pairwise distances, ruling out degenerate caches.

For a query with normalised image vector \(v_q\) and text vector \(t_q\), the fused query representation was

\[
f_q=\frac{\alpha v_q+(1-\alpha)t_q}{\lVert \alpha v_q+(1-\alpha)t_q\rVert_2},
\]

with \(\alpha=0.40\) in the frozen confirmatory system. Candidate \(i\) received CLIP compatibility

\[
s_{\mathrm{CLIP}}(q,i)=f_q^{\top}f_i,
\]

which is cosine similarity because the vectors are normalised. Separate CLIP-image and CLIP-text baselines used their respective query and candidate modalities without fusion. Deterministic item-ID ordering resolved exact score ties.

### 3.5.3 Validation of fusion and computation boundary

Fusion weights were examined on validation data only. The formal search and the final confirmatory value belong to different documented stages: the early broad validation search established the behaviour of the fusion surface, while the authoritative study specification fixed 0.40 image/0.60 text for Stage 6. The test set was not used to retune this choice. Representations were cached and hash-bound so that ranking and statistical calculations did not repeatedly invoke the encoders.

The experiment records parameter counts, quantisation, device choices, token counts, latency, and model digests where available. It does not report floating-point operations (FLOPs). Accurate FLOP accounting for cached transformer embeddings and quantised Ollama generation would require kernel-level profiling, batch-shape accounting, and a definition of how integer/quantised operations are converted to FLOPs. Those counters were not captured during the frozen runs. A retrospective theoretical estimate would therefore create spurious precision and is not required for any effectiveness or faithfulness claim. The publication analysis itself made zero model calls.

## 3.6 Expert rule base and exact decision trace

### 3.6.1 Knowledge-base construction

The expert knowledge base contained 126 curated rules divided across the five recommended categories. High- and medium-reliability rules were preserved separately: accessories contained 15 high and 11 medium rules; bottoms 17 and 8; outerwear 15 and 10; shoes 14 and 12; and tops 14 and 10. Every record carried a stable rule identifier, rule text, recommended category, applicable query grouping, and reliability label. The knowledge base was supplied and frozen; the project did not use automatic rule authoring.

Rules express styling relationships rather than verified catalogue facts. Examples include complementarity between apparel types, consistency of formality, seasonal layering, or matching dominant colours. This distinction is essential. A generic instruction to recommend colour-compatible items can support the reason for seeking compatibility, but it cannot prove the actual colour of a particular candidate or that a particular pair definitely matches. UIAR therefore uses strict instance-level entailment and does not convert prescriptive advice into product metadata.

### 3.6.2 Rule retrieval and scoring

For each query–candidate pair, the system created a textual representation from the user request, query group, candidate category, and item text. Candidate rules were first filtered by the requested recommendation category. Semantic similarity between the pair representation and rule text was calculated using normalised MiniLM vectors. A query-group bonus and a reliability weight were then applied. In simplified form, rule \(r\) received

\[
u(q,i,r)=\bigl(\cos(e_{q,i},e_r)+b(q,r)\bigr)w_{\mathrm{rel}}(r),
\]

where \(e_{q,i}\) is the candidate representation, \(e_r\) is the rule embedding, \(b\) is the documented query-group bonus, and \(w_{\mathrm{rel}}\) is the reliability multiplier. Rules were sorted by weighted contribution with stable rule-ID tie-breaking. The top \(k=5\) rules were retained.

The candidate evidence score was the mean contribution of those five rules,

\[
s_E(q,i)=\frac{1}{5}\sum_{r\in R_5(q,i)}u(q,i,r).
\]

The stored trace included every element needed to reproduce this number: the candidate ID; representation hash; rules before and after category filtering; rules excluded and not selected; retrieval rank; similarity; reliability label and weight; query-group bonus; weighted contribution; and final evidence score. B is therefore not a later summary generated for the explanation model. It is the exact trace of the rules that participated in the reranking score.

## 3.7 Evidence-aware reranking and recommendation locking

Raw CLIP and evidence scores have different ranges. Within each candidate pool, min–max normalisation converted each to \([0,1]\):

\[
\widetilde{s}(i)=\frac{s(i)-\min_j s(j)}{\max_j s(j)-\min_j s(j)}.
\]

If all candidates shared the same score, the deterministic implementation handled the zero range without introducing random noise. The final reranking score was

\[
s_R(q,i)=0.75\,\widetilde{s}_{\mathrm{CLIP}}(q,i)+0.25\,\widetilde{s}_{E}(q,i).
\]

The 0.75/0.25 setting was chosen through a validation-only Pareto procedure that considered recommendation effectiveness and evidence participation rather than optimising a single metric. Candidate-pool size, rule count, and weights were frozen before testing. The top-ranked reranked item became the locked recommendation. Stage 7 could explain this item but could not replace it. This lock is the methodological bridge between recommendation and explanation experiments: both conditions refer to the same decision, and the trace used for Rule-RAG is the trace stored for that item at Stage 6.

Evidence participation was evaluated independently of hit-rate effectiveness. Diagnostics included whether reranking changed the top item or ordered top five, set overlap at five and ten, mean absolute rank shift in the top-ten union, gains in evidence score at several cut-offs, rule coverage, and Shannon entropy of rule use. This avoids the erroneous inference that a non-significant accuracy difference means the evidence component did nothing. Conversely, substantial reordering does not establish improved recommendation accuracy.

## 3.8 Recommendation evaluation

### 3.8.1 Relevance and metrics

An item was relevant if it belonged to the query outfit and matched the requested target category. Multiple positives were allowed. For ranked relevance sequence \(rel_k\), hit rate at \(K\) was

\[
HR@K=\mathbb{1}\left(\sum_{k=1}^{K}rel_k>0\right).
\]

Discounted cumulative gain was

\[
DCG@K=\sum_{k=1}^{K}\frac{2^{rel_k}-1}{\log_2(k+1)},
\]

and \(NDCG@K=DCG@K/IDCG@K\), where IDCG is the ideal ranking for that case [8]. Reciprocal rank was \(RR=1/r\) for the rank \(r\) of the first relevant result and zero if no relevant result appeared. Mean reciprocal rank averaged RR across cases. The reported primary cut-off was ten, with ranks one and five retained for diagnostic resolution.

Five frozen confirmatory methods were compared: MiniLM text, CLIP image, CLIP text, fused CLIP, and evidence reranking. Metrics were reported as micro means over cases, by broad category, and as category macro means. The main scientific contrast compared fused CLIP with evidence reranking, because this isolates the addition of the evidence score to the same multimodal base. An additive Stage 6b analysis subsequently evaluated pinned FashionCLIP 2.0 image, text, and 0.40/0.60 fused representations on the identical 1,000 cases and candidate pools. This domain baseline was not used to retune fusion or select the primary system; it tests whether fashion-domain representation learning changes the bounded recommendation result.

### 3.8.2 Statistical inference

Cases were not treated as fully independent when they shared a query outfit. Confidence intervals and paired contrasts therefore used the query outfit as the bootstrap cluster. For each of 5,000 replicates, complete query-outfit clusters were sampled with replacement, all their cases were retained, and the statistic was recomputed. The 2.5th and 97.5th percentiles formed a 95% interval [9]. Pairing occurred at case level because every method ranked the same candidate pool. The family of 28 predeclared primary method–metric contrasts was corrected using Holm’s sequential procedure [10]. The test-set analysis did not choose a weight, prompt, or model.

## 3.9 Explanation-generation experiment

### 3.9.1 Common context A

For every locked case, A contained the complete frozen common information made available to both generators: the user request; query-item ID, category, and product text; and locked recommended-item ID, category, and product text. The displayed prompt used minimal names, while the saved packet retained identity and category provenance. No image caption, visual detector, outside catalogue lookup, brand knowledge, or inferred fashion attribute was added.

### 3.9.2 Exact trace B and conditions

B contained the five-rule trace described in Section 3.6. It was byte-identical for both conceptual comparisons, although hidden from No-RAG generation. The No-RAG prompt asked for an explanation of why the locked item suited the request using A and imposed the same at-most-75-word instruction used by Rule-RAG. The Rule-RAG prompt displayed A and B in labelled blocks, ordered rules by weighted score, included rule IDs, reliability labels, and scores, required citations, and used the same numerical cap. A/B hashes were stored with every generation record. Thus word-budget assignment is controlled, although observed word counts and the additional Rule-RAG instructions are not identical.

The No-RAG condition represents an unconstrained post-hoc rationale, not a condition with no information: it can use explicit product text in A. The Rule-RAG condition represents trace-assisted generation. A claim that restates an explicit material term from a product title may therefore be A-supported in either condition. A claim that follows a styling rule may be B-supported. A claim that invents “waterproof”, “premium leather”, or a definite instance-level colour match without such evidence is unsupported by the supplied evidence, but is not automatically factually false.

### 3.9.3 Generators and decoding

Three local, digest-pinned instruction models generated the explanations: Llama 3.2 with 3.2 billion parameters, Mistral with 7.2 billion parameters, and Gemma 3 with 12.2 billion parameters. Each used Q4_K_M quantisation through Ollama 0.32.7. Decoding was deterministic: temperature 0, top-p 1, top-k 1, and seed 42, with a 512-token safety ceiling. The generator roster was frozen before the full experiment.

Five hundred locked cases were sampled from Stage 6, balanced with 100 cases per broad category. Crossing 500 cases, three generators, and two conditions produced 3,000 explanations and 1,500 paired comparisons. Every prompt, response, latency, evaluation count, word count, model digest, and content hash was preserved. Outputs were immutable after generation. Failed-key recovery was permitted only for technical failure; the completed run had no retries, empty texts, malformed texts, trace mismatches, prompt mismatches, or exhausted keys.

## 3.10 Validation-only explanation optimisation

Before the full run, Stage 5 compared six Rule-RAG prompt configurations on validation data. Variables included one, three, or five rules; concise versus detailed grounding instructions; citation requirements; reliability labels; rule scores; ordering; and the word limit. Selection considered support, unsupported rate, contradiction, citation entailment, general quality, clarity, specificity, word-count proximity, malformed output, and latency. Normalised objectives were combined only after identifying the Pareto frontier, so no single noisy metric dictated the prompt.

Configuration `rag_c3` was frozen. It used five rules, labelled text evidence, weighted-score order, displayed reliability labels and scores, required citations, used the detailed grounding prompt, and requested no more than 75 words. On its primary validation cohort it achieved support 0.9322, unsupported 0.0678, citation entailment 0.9544, general quality 4.50, clarity 4.63, specificity 4.53, and a mean of 57.07 words. The original 50-case, three-generator pilot was retained as historical validation evidence. A corrective additive follow-up regenerated only its No-RAG side under the shared 75-word instruction: No-RAG/Rule-RAG means were 53.00/59.73 words overall, with zero No-RAG cap violations. Pilot results are not combined with the confirmatory Stage 7–8 estimates.

## 3.11 Atomic-claim extraction and verification

### 3.11.1 Extraction

The complete explanation, rather than a sentence sample, was passed to a frozen Qwen3 8B model with Q4_K_M quantisation. The extractor was instructed to enumerate every independent atomic fashion or styling proposition, split conjunctions where propositions could be independently checked, assign sequential claim IDs, and choose among a fixed schema: body fit, colour, comfort, formality, item type, material, occasion, season, styling relation, trend, visual match, or other. Extraction assessed neither truth nor support. The revised corpus yielded 20,838 claims. Two explanations remained extraction N/A after resumable structured retries; no refusal was detected.

Claim role was assigned deterministically after extraction. `item_type` claims were identity/context claims because they commonly state what item is being recommended. The other schema labels were treated as substantive because even simple categories such as colour and material can carry explanatory content. After verification eligibility, the study-specific layer contained 10,703 substantive No-RAG claims and 8,666 substantive Rule-RAG claims. N/A rows were retained in the 3,000-explanation accounting but did not receive invented claim labels.

### 3.11.2 Multi-source verification

The verifier received the extracted claims, complete A, exact B, and observed citations. For each claim it returned support status, support sources, supporting rule IDs, citation entailment, and a brief reason. Support status had four states: supported, unsupported, contradicted, and not verifiable. “Unsupported” means not entailed by the supplied A/B evidence. “Contradicted” was reserved for affirmative conflict. “Not verifiable” covered evaluative or otherwise non-checkable propositions. Support sources were multi-label: A, B, both, or neither.

Role independence was strengthened by assigning verification to digest-pinned Mistral 7.2B while Qwen3 8B remained the extractor and paired judge. Extraction and verification were therefore cross-model, although Mistral was also one of the three generators and no human validation was introduced. Twenty verification rows remained N/A, giving 2,980 actual verifier outputs and 20,618 verified claims. A deterministic normaliser removed duplicate sources, invalid rule identifiers, inappropriate citation values on uncited claims, and inconsistent source/status combinations according to conservative rules. It never generated a new semantic judgment. Exactly 1,280 verifier outputs (42.95%) required at least one structural normalisation; normalised and unnormalised strata were retained for sensitivity description.

## 3.12 Study-specific explanation metrics

### 3.12.1 Decision-Trace Alignment

DTA uses B as a common reference for both conditions and includes only substantive claims:

\[
DTA(E)=\frac{\#\{c\in C_S(E):B\models c\}}{|C_S(E)|},
\]

where \(C_S(E)\) is the set of substantive claims. A claim counts as entailed only when the saved verifier labels it supported by rule evidence and supplies at least one exact supporting rule ID. Micro DTA pools claims before division. Macro DTA computes the rate within each eligible explanation and averages explanation rates. For No-RAG, the result is post-hoc B alignment because B was not visible at generation. For Rule-RAG, it is decision-trace faithfulness because B was supplied and requested as the basis of the explanation.

### 3.12.2 Unsupported Item-Attribute Rate

UIAR targets a conservative subset of concrete claims about the actual query or recommended item: colour, material, pattern or design detail, fit or silhouette, construction, physical characteristics, concrete product or brand properties, and comfort or season when asserted as an item property. Subjective styling judgments—such as “the pairing is sophisticated” or “one style balances another”—remain outside UIAR unless they also assert a concrete property.

A deterministic classifier uses extracted claim type, lexical physical-property markers, predicate form, and whether the claim identifies an actual case item. Uncertain mixed claims enter an `ambiguous` bucket rather than the denominator. A support is evaluated against the complete frozen context, including product text rather than only an item name. Outside brand knowledge and visual inference are prohibited. In Rule-RAG, B may add support only if a rule strictly entails that instance-level property; generic prescriptions cannot establish it. The metric is

\[
UIAR(E)=\frac{\#\{c\in C_A(E): A\not\models c\ \land\ B_{visible}\not\models c\}}{|C_A(E)|},
\]

where \(C_A(E)\) is the eligible item-attribute subset and \(B_{visible}=\varnothing\) for No-RAG. Zero-eligible explanations receive no invented zero and are absent from macro UIAR. The paired contrast uses only matched pairs where both sides contain at least one eligible claim.

Unsupported Attribute Density addresses verbosity:

\[
UAD(E)=100\times\frac{\#\text{ unsupported eligible attribute claims}}{\text{explanation word count}}.
\]

This is a sensitivity metric, not the primary measure, because the opportunity to make claims does not necessarily grow linearly with words.

### 3.12.3 Citation precision and coverage

Citation metrics apply only to Rule-RAG. A cited claim is valid only when an observed rule citation intersects the verifier’s exact supporting rule IDs, the rule supports the claim, and citation entailment is true:

\[
Precision_{cite}=\frac{\#\text{ validly cited claims}}{\#\text{ claims carrying an evaluated citation}},
\]

\[
Coverage_{cite}=\frac{\#\text{ rule-requiring substantive claims with a valid citation}}{\#\text{ substantive claims requiring rule support}}.
\]

No-RAG is reported as N/A, not zero. Precision and coverage are distinct from citation presence. A model can cite frequently yet attach the wrong rule, or cite validly on selected claims while leaving other rule-dependent claims uncited.

### 3.12.4 Secondary grounding views

Visible-evidence grounding preserves the originally defined evidence boundary: No-RAG is checked against A, while Rule-RAG is checked against A+B. It remains secondary because the comparison changes the visible evidence set and therefore does not use a common reference. A shared-reference A+B view is also retained: for No-RAG it is explicitly post-hoc alignment, while for Rule-RAG it coincides with visible support. Neither secondary view replaces DTA or UIAR.

An Evidence Overreach Rate is not reported. Reliable separation of fully supported, partially supported/overstated, and unsupported claims would require independent annotation not present in the final study. Creating an automated value from the existing binary/nominal verifier fields would imply resolution the data do not contain.

## 3.13 General explanation-quality judging

Each pair was presented to the frozen Qwen3 judge with position randomised deterministically. The judge scored each explanation separately from 1 (very poor) to 5 (excellent) on input consistency, general quality, clarity, specificity, hallucination control, and evidence-use correctness. Higher values always indicated better performance. The prompt prohibited naming conditions and excluded word-limit compliance from the scales. Position assignment and the two explanation hashes were saved.

The publication analysis calculated paired Rule-RAG minus No-RAG differences for all 1,500 case–generator pairs. Ninety-five per cent confidence intervals used 2,000 case-cluster bootstrap replicates. Two-sided Wilcoxon signed-rank tests assessed paired score differences, Holm-adjusted across the six dimensions within each reported scope. Paired standardised effect size was

\[
d_z=\frac{\overline{D}}{s_D},
\]

where \(D\) is the within-pair score difference. This scale is descriptive because integer judge scores are bounded and concentrated near five in Rule-RAG.

To address RQ3, Spearman rank correlations related explanation-level DTA, UIAR, unsupported-attribute density, visible support, and word count to each judge dimension within condition [11]. Within-condition estimation avoids producing a large correlation merely because both metrics differ between conditions. A second analysis correlated paired changes in each targeted metric with paired judge-score changes. Correlation indicates concordance, not interchangeable validity.

## 3.14 Heterogeneity, robustness, and qualitative analysis

Primary DTA and UIAR differences were estimated separately for the three generators and five categories using the same paired definitions. A cluster-bootstrap covariance matrix supported a Wald test of equality across subgroup effects. These interaction-style tests answer whether the magnitude, rather than merely the sign, differs by generator or category. UIAR subgroup results are interpreted cautiously because only 362 pairs are jointly eligible.

Leave-one-generator-out sensitivity recomputed the pooled DTA and UIAR contrast three times, omitting Gemma, Llama, or Mistral. A conclusion that reverses after removing one generator would be dependent on that model; a stable sign and interval indicates broader directional robustness within the tested roster. This is still not evidence for all language models.

The predeclared length sensitivity selected the ten smallest absolute word-gap pairs for each generator, giving 30 pairs. It is explicitly sensitivity-only. In the revised corpus these pairs averaged 54.17 No-RAG and 54.37 Rule-RAG words, with a mean absolute paired gap of 0.33 words. Selection on observed length still creates a specialised cohort, and UIAR remains sensitive to how many explanations contain eligible concrete attributes, so this analysis complements rather than replaces the full paired estimates.

Qualitative examples were selected deterministically to avoid manual cherry-picking. Within each category, the algorithm found the pair closest to that category’s median paired DTA change and median unsupported-attribute-density change; SHA-256 ordering broke ties. The resulting five pairs present common A, exact B, No-RAG and Rule-RAG texts, and their automated metrics. Examples illustrate mechanisms and failure modes but do not replace corpus estimates.

## 3.15 Reproducibility, software, and ethical boundaries

The implementation used Python 3.12 on Windows 11. Embedding and LLM stages were configured for CUDA where available; the clean Stage 3 validation also recorded CPU float32 checks. The repository pins dependencies through `uv.lock`, model revisions or immutable local digests, dataset revision, random seeds, and canonical configuration hashes. Major tables, figures, JSONL corpora, and manifests are SHA-256 bound. Ruff linting and a 72-test suite cover data leakage, deterministic ranking, rule retrieval, prompt separation, schemas, refusal handling, statistics, and artifact integrity. Stage 10 validates all principal output hashes and records the release boundary.

The study uses catalogue images and descriptions from a research dataset and makes no inference about protected personal attributes. It is not a safety-critical wardrobe adviser, product-authentication system, or factual catalogue service. Product text may be noisy, and outfit co-occurrence reflects platform curation rather than universal taste. Generated explanations must therefore be read as system outputs about supplied records, not authoritative fashion or product claims.

Automated assessment presents a more important validity limitation. Extraction and judging use Qwen3 8B, while verification uses Mistral 7.2B; this separates claim construction from entailment but does not establish either model's semantic accuracy. The deterministic UIAR classifier improves transparency but cannot resolve every mixed natural-language claim; its ambiguous bucket trades coverage for precision. No human ratings, independent external annotations, or EOR audit are included. The strongest warranted conclusion is comparative: under this frozen system evaluator, access to the exact five-rule trace changes measured visible grounding, unsupported concrete assertions, citations, and quality scores. It does not establish human preference, factual correctness in the world, or causal faithfulness to every internal neural computation.

## 3.16 Chapter summary

The methodology connects recommendation and explanation through a locked decision and an exact symbolic trace. Multimodal CLIP retrieval supplies the base ranking; a category-filtered expert rule base contributes an evidence score; validation-only selection freezes the fusion and reranking design; and the confirmatory evaluation quantifies both recommendation effectiveness and ranking change. The explanation experiment holds recommendation identity constant while varying access to B across three generators and five categories. Atomic claims, strict source attribution, DTA, UIAR, citation integrity, general judging, cluster-aware paired inference, heterogeneity tests, and deterministic examples provide complementary views of the output.

The design’s principal strength is traceability. Every reported explanation is connected to a specific A packet, B trace, generator digest, locked candidate, claim list, verification record, and judge record. Its principal limitation is equally clear: the final thesis experiment stops at automated system evaluation. Chapter 4 therefore reports strong quantitative differences without treating them as independently human-validated truth.

## 3.17 Validity controls and decision rules

Several controls were applied before results were interpreted. Construct validity was protected by refusing to collapse all explanation properties into a single “grounding” score. DTA addresses agreement with the actual rule trace, UIAR addresses a narrow class of concrete product assertions, citation precision addresses whether an attached citation entails its claim, and the six judge dimensions address user-facing form. These measures can move differently. For example, an explanation can accurately paraphrase B while adding an unsupported material claim, or it can be concise and clear while citing only some of its rule-dependent propositions. Reporting each rate separately prevents one favourable dimension from concealing another failure.

Internal validity depended on pairing and freezing. The recommendation, A packet, B packet, generator, and case were identical within each No-RAG/Rule-RAG comparison. Outputs were not edited to equalise length, remove refusals, repair citations, or improve examples. Stage 7 generation records were hashed before assessment, and Stage 8 outputs were preserved before deterministic normalisation and metric derivation. Model selection and hyperparameter search were confined to development or validation data. These controls do not remove the prompt-and-length difference intrinsic to the intervention, but they prevent test-driven tuning and recommendation changes from contaminating the explanation contrast.

Statistical-conclusion validity was addressed through the appropriate dependence unit. Recommendation cases can share an outfit, so recommendation intervals resample query outfits. Explanation texts from three generators can share a case, so publication analyses resample case IDs. Micro and macro estimates answer different questions and are not treated as interchangeable: micro rates weight explanations in proportion to their number of claims, whereas macro rates weight each eligible explanation equally. The UIAR paired result has the additional jointly eligible restriction; it therefore describes texts that actually make concrete attribute claims on both sides and is not extrapolated to all 1,500 pairs.

External validity is bounded by the sampled catalogue, five broad categories, controlled pools, three local quantised generators, and supplied rule base. The system does not model individual user histories, price, availability, body measurements, culture-specific dress norms, or time-varying trends. Outfit co-occurrence is a useful offline relevance signal but does not prove that every held-out item would be preferred by a new user. Larger candidate pools show the expected reduction in absolute ranking performance and caution against treating the approximately 100-candidate estimates as full-catalogue serving results.

Reliability was supported by deterministic execution and independent re-computation from stored artifacts. SHA-256 ordering replaced random iteration in split, case, tie, and example selection. Generation used greedy deterministic settings, while structured assessment calls retained model digests and raw-response hashes. Each derived table can be traced to a manifest listing input and output hashes. Re-running the publication analysis does not invoke a model; it combines immutable records, recomputes rates and cluster resamples, and recreates figures and examples.

Finally, interpretation followed pre-specified language rules. A claim labelled unsupported is described as “unsupported by supplied evidence” or an “unverified item-specific assertion”, never as factually false unless affirmative contradiction exists. No-RAG agreement with hidden B is post-hoc alignment, never grounding. Visible-evidence support is secondary because its evidence boundary differs between conditions. The 30-pair length cohort is sensitivity-only, and UIAR within it is explicitly inconclusive. Evidence reranking is described as materially changing ranks and evidence alignment, not as improving recommendation accuracy, because the fused-CLIP contrasts are non-significant. These linguistic constraints are part of the methodology: they keep the thesis claims proportional to the estimands actually measured.

## References

[1] Jacovi, A. and Goldberg, Y. (2020) ‘Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?’, *Proceedings of ACL 2020*, pp. 4198–4205. https://doi.org/10.18653/v1/2020.acl-main.386.

[2] Wiegreffe, S. and Pinter, Y. (2019) ‘Attention is not not explanation’, *Proceedings of EMNLP-IJCNLP 2019*, pp. 11–20. https://doi.org/10.18653/v1/D19-1002.

[3] McAuley, J., Targett, C., Shi, Q. and van den Hengel, A. (2015) ‘Image-based recommendations on styles and substitutes’, *Proceedings of SIGIR 2015*, pp. 43–52. https://doi.org/10.1145/2766462.2767755.

[4] Han, X., Wu, Z., Jiang, Y.-G. and Davis, L.S. (2017) ‘Learning fashion compatibility with bidirectional LSTMs’, *Proceedings of the 25th ACM International Conference on Multimedia*, pp. 1078–1086. https://doi.org/10.1145/3123266.3123394.

[5] Reimers, N. and Gurevych, I. (2019) ‘Sentence-BERT: Sentence embeddings using Siamese BERT-networks’, *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992. https://doi.org/10.18653/v1/D19-1410.

[6] Wang, W. et al. (2020) ‘MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers’, *Advances in Neural Information Processing Systems*, 33, pp. 5776–5788.

[7] Radford, A. et al. (2021) ‘Learning transferable visual models from natural language supervision’, *Proceedings of the 38th International Conference on Machine Learning*, 139, pp. 8748–8763.

[8] Järvelin, K. and Kekäläinen, J. (2002) ‘Cumulated gain-based evaluation of IR techniques’, *ACM Transactions on Information Systems*, 20(4), pp. 422–446. https://doi.org/10.1145/582415.582418.

[9] Efron, B. and Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*. New York: Chapman & Hall/CRC.

[10] Holm, S. (1979) ‘A simple sequentially rejective multiple test procedure’, *Scandinavian Journal of Statistics*, 6(2), pp. 65–70.

[11] Spearman, C. (1904) ‘The proof and measurement of association between two things’, *The American Journal of Psychology*, 15(1), pp. 72–101. https://doi.org/10.2307/1412159.
