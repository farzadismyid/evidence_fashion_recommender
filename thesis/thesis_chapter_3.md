# Chapter 3

# Methodology

## 3.1 Introduction

This chapter describes the research design, implementation, and evaluation procedure used to investigate evidence-constrained multimodal fashion recommendation. The central methodological problem was not simply to produce a plausible recommendation. It was to separate three questions that are often conflated: whether a multimodal ranker retrieves compatible items; whether expert evidence materially participates in the ranking decision; and whether a natural-language explanation is faithful to the stored decision trace without adding unverified product attributes. The experiment was therefore organised as a staged, frozen pipeline. Earlier stages prepared the data, representations, candidate pools, and rule base; validation-only stages fixed every tunable choice; the confirmatory recommendation experiment then locked one recommendation and its exact five-rule trace for each case; and the explanation experiment compared two texts for the same locked decision.

The design follows a paired-comparison principle. For each explanation case, common context A contained the request, query-item text and identity, and locked recommended-item text and identity. Exact trace B contained the five rules that contributed to the frozen evidence score. The No-RAG generator received A alone. The Rule-RAG generator received A and B, while recommendation identity was held constant. This intervention isolates access to the decision trace at the explanation stage. It does not isolate every possible effect of prompt wording or output length, and it does not turn the No-RAG condition into a grounded explanation when its text happens to agree with B. Accordingly, B agreement in No-RAG is post-hoc alignment, whereas Rule-RAG support is evidence-grounded because B was visible during generation.

The work is an offline systems experiment rather than a user study. Recommendation relevance comes from co-occurrence within held-out Polyvore outfits. Explanation assessment uses Qwen 3.5 to extract atomic claims and Phi-4 to verify their relationship to the exact trace, the full KB packet, common reference evidence, and citations. Deterministic post-processing derives support rates and trace-supported-claim density from saved records. These are operational measures for this study, not universal benchmarks. No human or independent external audit is included in the final experimental boundary, so the automated evaluation is treated as system-level evidence with explicit limitations.

## 3.2 Research questions and experimental logic

The methodology addresses four linked research questions. RQ1 asks whether making the exact expert-rule trace visible during generation improves support by the actual reranking trace and by the final KB packet. RQ2 asks whether the same intervention changes the rate of eligible concrete item-fact claims that are unsupported by a common reference packet. RQ3 asks whether syntactically present Rule-RAG citations are entailed by the cited rules at claim level. RQ4 asks whether the primary support effects remain directionally stable across the three generators and five target categories.

The causal contrast is deliberately narrow. For case \(q\), let \(r_q\) be the recommendation locked in Stage 2, \(A_q\) the common case context, and \(B_q\) the exact five-rule reranking trace. For generator \(g\), the two outputs are

\[
\begin{aligned}
E^{\mathrm{NoRAG}}_{qg} &= G_g(A_q),\\
E^{\mathrm{RuleRAG}}_{qg} &= G_g(A_q,B_q).
\end{aligned}
\tag{3.1}
\]

Both outputs explain the same \(r_q\). The paired elements are the case, locked recommendation, generator, decoding configuration, and common context. The intervention is the availability of \(B_q\): Rule-RAG receives the exact trace and associated grounding instruction, whereas No-RAG does not. Both conditions receive the same at-most-75-word instruction. The contrast therefore identifies the effect of trace-grounded prompting, while recognising that the evidence and citation instructions are part of that intervention. Claim outcomes are reported as rates and, where appropriate, per 100 generated words.

Equation (3.1) is not intended to claim that the generator has access to hidden neural states. It formalises the visible-information boundary of the experiment. The recommendation is already fixed before either call, and \(B_q\) is a recorded symbolic artifact rather than a newly retrieved explanatory document. This ordering is what permits a comparison of explanation grounding without conflating it with a change in the recommendation itself.

The study does not claim that B is a complete account of a neural model’s internal computation. B is instead the exact, inspectable symbolic trace used by the evidence component of the deployed reranker. The term “decision trace” is thus architectural and operational: it identifies the five expert rules, their similarities, reliability weights, ordering, and contributions used to compute the evidence score. Faithfulness is measured relative to that trace. This boundary follows the distinction in explainable-AI research between a convincing rationale and an account tied to the mechanism being explained [1,2].

## 3.3 Staged research design and freezing policy

The final project was implemented in five frozen stages. Stage 1 performed preflight, audited the final 200-rule KB, fixed the dataset/split and validation-only settings, and bound prompts and configurations. Stage 2 executed 1,000 recommendations and 3,000 fresh explanation attempts. Stage 3 extracted atomic claims from accepted explanations. Stage 4 verified claims against the frozen evidence packets and applied the documented deterministic logical-consistency invariant. Stage 5 derived final tables, figures, paired bootstrap inference, release hashes, and quality checks without new model calls.

Freezing served two purposes. First, it prevented test performance from influencing model weights, fusion weights, evidence weights, pool size, rule count, or prompts. Second, it preserved a stable provenance chain. Each major artifact was written once, bound to a SHA-256 digest, and named in a stage manifest. Model identifiers were accompanied by immutable revisions or local model digests. Configuration objects were canonicalised and hashed. Stage 5 derived all final analysis from the saved Stage 1--4 records and made zero new model calls.

The final confirmatory settings were fixed before Stage 2: image/text CLIP fusion was 0.40/0.60; evidence reranking was CLIP/evidence 0.75/0.25; exactly five expert rules contributed to each trace; and the primary candidate pool contained approximately 100 candidates. Larger pools were validation sensitivity settings rather than alternative primary estimates. Generator identities, decoding settings, Rule-RAG prompt form, and claim schemas were frozen before full generation. The authorised verifier-contract correction is separately recorded in final provenance and binds the release to the actual Stage-4 prompt hash.

## 3.4 Data source, unit of analysis, and preprocessing

### 3.4.1 Dataset and fields

The data source was the `Marqo/polyvore` dataset at immutable revision `8c782ee447faf2d2a0402ac883cf07d3b3f43e1c`, configuration `default`, source split `data`, and fingerprint `9c97dc763773e2a2`. Polyvore-derived outfit data are widely used for fashion compatibility research because an outfit supplies a set of items curated to appear together [3,4]. The present study used only the raw item identifier, category, product text, outfit association, and image. Textual explanation evidence was intentionally limited: images entered the recommendation representation but were never captioned, classified, or converted into attributes for A or B.

The raw pinned source contained 94,096 items and 21,587 outfits. A validated mapping assigned items to the five final target categories: bags, bottoms, outerwear, shoes, and tops. Subsequent leakage resolution and confirmatory eligibility produced the prepared universe recorded in the final manifests. Differences between intermediate preparation counts are stage-specific and are not used as confirmatory results.

### 3.4.2 Outfit-disjoint splitting

The outfit, not the item row, was the primary split unit. Outfit IDs were ordered by SHA-256 over a fixed seed and assigned to exact quotas of 15,267 development outfits, 3,147 validation outfits, and 3,173 test outfits. This procedure is deterministic and prevents items from the same outfit appearing on opposite sides of the research split. Within the test partition, case selection used a separate seeded SHA-256 order and sampled 200 cases for each broad category, producing 1,000 confirmatory recommendation cases.

Exact duplicate images were audited by hashing image bytes. Twenty-one duplicate groups were identified, of which eleven crossed an initial research split. Outfits connected by shared exact-image hashes were treated as connected components. Each cross-split component was moved to the split of its lowest seeded-hash anchor; singleton outfits were then moved in a separate deterministic order to restore the original quotas. Thirteen outfits changed assignment in total: eleven duplicate-linked moves and two rebalancing moves. The final split retained its exact quotas and contained no cross-split outfit or exact-image leakage. This is stricter than relying on unique item IDs, because separately identified catalogue rows can still carry identical visual content.

### 3.4.3 Query construction and candidate pools

Each case selected a query item and a target category representing the missing outfit component. All known same-outfit positives in that target category were retained. Negatives were sampled from items of the same target category belonging to other test outfits. The query item was always excluded. Category restriction makes the task a controlled within-category ranking problem: the model chooses *which* pair of shoes or *which* top, rather than receiving credit merely for predicting the requested item type.

Stage 2 used up to 99 same-category negatives together with all known positives for each of the 1,000 confirmatory cases. Pool sizes can exceed 100 when a case has multiple retained positives. The design is therefore sampled controlled-pool ranking rather than full-catalogue retrieval. Absolute ranking values must be interpreted within this fixed test regime; they do not estimate production-scale catalogue retrieval performance.

## 3.5 Representation learning and baseline retrieval

### 3.5.1 Text representation

The text-only baseline used `sentence-transformers/all-MiniLM-L6-v2` at immutable revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Sentence-BERT-style encoders map sentences into a shared dense space suitable for cosine retrieval [5], while MiniLM distils transformer self-attention into a more compact architecture [6]. Product text was encoded into 384-dimensional float32 vectors and L2 normalised. Texts were truncated only according to the pinned encoder limit of 256 tokens. This pathway provided a lightweight semantic baseline distinct from CLIP text encoding.

### 3.5.2 CLIP image and text representation

The multimodal ranker used `openai/clip-vit-base-patch32` at immutable revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. CLIP learns aligned visual and language representations from image–text pairs [7]. Item images and product text were embedded into 512-dimensional float32 vectors. All vectors were L2 normalised, and Stage-1 preflight confirmed maximum norm errors below \(1.2\times10^{-7}\). Distinct input items also produced non-zero pairwise distances, ruling out degenerate caches.

For a query with normalised image vector \(v_q\) and text vector \(t_q\), the fused query representation was

\[
z_q=\alpha v_q+(1-\alpha)t_q,
\qquad
f_q=\frac{z_q}{\lVert z_q\rVert_2},
\qquad \alpha=0.40.
\tag{3.2}
\]

with \(\alpha=0.40\) in the frozen confirmatory system. Candidate \(i\) received CLIP compatibility

\[
s_{\mathrm{CLIP}}(q,i)=f_q^{\top}f_i.
\tag{3.3}
\]

which is cosine similarity because the vectors are normalised. Separate CLIP-image and CLIP-text baselines used their respective query and candidate modalities without fusion. Deterministic item-ID ordering resolved exact score ties.

### 3.5.3 Validation of fusion and computation boundary

Fusion weights were examined on validation data only. The validation search established the behaviour of the fusion surface, after which the authoritative study specification fixed 0.40 image/0.60 text for the Stage-2 confirmatory run. The test set was not used to retune this choice. Representations were cached and hash-bound so that ranking and statistical calculations did not repeatedly invoke the encoders.

The experiment records parameter counts, quantisation, device choices, token counts, latency, and model digests where available. It does not report floating-point operations (FLOPs). Accurate FLOP accounting for cached transformer embeddings and quantised Ollama generation would require kernel-level profiling, batch-shape accounting, and a definition of how integer/quantised operations are converted to FLOPs. Those counters were not captured during the frozen runs. A retrospective theoretical estimate would therefore create spurious precision and is not required for any effectiveness or faithfulness claim. The publication analysis itself made zero model calls.

## 3.6 Expert rule base and exact decision trace

### 3.6.1 Knowledge-base construction

The final knowledge base contained 200 curated rules, exactly 40 for each recommended category: bags, bottoms, outerwear, shoes, and tops. Every record carried a stable rule identifier, rule text, recommended category, applicable query grouping, and provenance. The knowledge base was supplied and frozen; the project did not use automatic rule authoring.

Rules express styling relationships rather than verified catalogue facts. Examples include complementarity between apparel types, consistency of formality, seasonal layering, or matching dominant colours. This distinction is essential. A generic instruction to recommend colour-compatible items can support the reason for seeking compatibility, but it cannot prove the actual colour of a particular candidate or that a particular pair definitely matches. The common-reference item-fact analysis therefore uses strict instance-level entailment and does not convert prescriptive advice into product metadata.

### 3.6.2 Rule retrieval and scoring

For each query–candidate pair, the system created a textual representation from the user request, query group, candidate category, and item text. Candidate rules were first filtered by the requested recommendation category. Semantic similarity between the pair representation and rule text was calculated using normalised MiniLM vectors. A query-group bonus and a reliability weight were then applied. In simplified form, rule \(r\) received

\[
u(q,i,r)=\bigl(\cos(e_{q,i},e_r)+b(q,r)\bigr)w_{\mathrm{rel}}(r).
\tag{3.4}
\]

where \(e_{q,i}\) is the candidate representation, \(e_r\) is the rule embedding, \(b\) is the documented query-group bonus, and \(w_{\mathrm{rel}}\) is the reliability multiplier. Rules were sorted by weighted contribution with stable rule-ID tie-breaking. The top \(k=5\) rules were retained.

The candidate evidence score was the mean contribution of those five rules,

\[
s_E(q,i)=\frac{1}{5}\sum_{r\in R_5(q,i)}u(q,i,r).
\tag{3.5}
\]

The stored trace included every element needed to reproduce this number: the candidate ID; representation hash; rules before and after category filtering; rules excluded and not selected; retrieval rank; similarity; reliability label and weight; query-group bonus; weighted contribution; and final evidence score. B is therefore not a later summary generated for the explanation model. It is the exact trace of the rules that participated in the reranking score.

## 3.7 Evidence-aware reranking and recommendation locking

Raw CLIP and evidence scores have different ranges. Within each candidate pool, min–max normalisation converted each to \([0,1]\):

\[
\widetilde{s}(i)=
\begin{cases}
\dfrac{s(i)-\min_j s(j)}{\max_j s(j)-\min_j s(j)}, & \max_j s(j)>\min_j s(j),\\
0, & \max_j s(j)=\min_j s(j).
\end{cases}
\tag{3.6}
\]

If all candidates shared the same score, the deterministic implementation handled the zero range without introducing random noise. The final reranking score was

\[
s_R(q,i)=0.75\,\widetilde{s}_{\mathrm{CLIP}}(q,i)+0.25\,\widetilde{s}_{E}(q,i).
\tag{3.7}
\]

The 0.75/0.25 setting was chosen through a validation-only Pareto procedure that considered recommendation effectiveness and evidence participation rather than optimising a single metric. Candidate-pool size, rule count, and weights were frozen before testing. The top-ranked reranked item became the locked recommendation. The Stage-2 explanation calls could explain this item but could not replace it. This lock is the methodological bridge between recommendation and explanation experiments: both conditions refer to the same decision, and the trace used for Rule-RAG is the trace stored for that item during reranking.

Evidence participation was evaluated independently of hit-rate effectiveness. Diagnostics included whether reranking changed the top item or ordered top five, set overlap at five and ten, mean absolute rank shift in the top-ten union, gains in evidence score at several cut-offs, rule coverage, and Shannon entropy of rule use. This avoids the erroneous inference that a non-significant accuracy difference means the evidence component did nothing. Conversely, substantial reordering does not establish improved recommendation accuracy.

## 3.8 Recommendation evaluation

### 3.8.1 Relevance and metrics

An item was relevant if it belonged to the query outfit and matched the requested target category. Multiple positives were allowed. For ranked relevance sequence \(rel_k\), hit rate at \(K\) was

\[
\operatorname{HR@}K=\mathbb{1}\!\left(\sum_{k=1}^{K} rel_k>0\right).
\tag{3.8}
\]

Discounted cumulative gain was

\[
\operatorname{DCG@}K=\sum_{k=1}^{K}\frac{2^{rel_k}-1}{\log_2(k+1)},
\qquad
\operatorname{NDCG@}K=\frac{\operatorname{DCG@}K}{\operatorname{IDCG@}K}.
\tag{3.9}
\]

and \(NDCG@K=DCG@K/IDCG@K\), where IDCG is the ideal ranking for that case [8]. Reciprocal rank was \(RR=1/r\) for the rank \(r\) of the first relevant result and zero if no relevant result appeared. Mean reciprocal rank averaged RR across cases. The reported primary cut-off was ten, with ranks one and five retained for diagnostic resolution.

Five frozen confirmatory methods were compared: MiniLM text, CLIP image, CLIP text, fused CLIP, and evidence reranking. Metrics were reported as micro estimates over cases, by target category, and as category macro estimates. The main scientific contrast compared fused CLIP with evidence reranking, because this isolates the addition of the evidence score to the same multimodal base. The frozen run did not add post-confirmatory representation baselines or retune the primary system.

### 3.8.2 Statistical inference

Cases were not treated as fully independent when they shared a query outfit. Confidence intervals and paired contrasts therefore used the query outfit as the bootstrap cluster. For each of 5,000 replicates, complete query-outfit clusters were sampled with replacement, all their cases were retained, and the statistic was recomputed. The 2.5th and 97.5th percentiles formed a 95% interval [9]. Pairing occurred at case level because every method ranked the same candidate pool. The family of 28 predeclared primary method–metric contrasts was corrected using Holm’s sequential procedure [10]. The test-set analysis did not choose a weight, prompt, or model.

## 3.9 Explanation-generation experiment

### 3.9.1 Common context A

For every locked case, A contained the complete frozen common information made available to both generators: the user request; query-item ID, category, and product text; and locked recommended-item ID, category, and product text. The displayed prompt used minimal names, while the saved packet retained identity and category provenance. No image caption, visual detector, outside catalogue lookup, brand knowledge, or inferred fashion attribute was added.

### 3.9.2 Exact trace B and conditions

B contained the five-rule trace described in Section 3.6. It was byte-identical for both conceptual comparisons, although hidden from No-RAG generation. The No-RAG prompt asked for an explanation of why the locked item suited the request using A and imposed the same at-most-75-word instruction used by Rule-RAG. The Rule-RAG prompt displayed A and B in labelled blocks, ordered rules by weighted score, included rule IDs, reliability labels, and scores, required citations, and used the same numerical cap. A/B hashes were stored with every generation record. Thus word-budget assignment is controlled, although observed word counts and the additional Rule-RAG instructions are not identical.

The No-RAG condition represents an unconstrained post-hoc rationale, not a condition with no information: it can use explicit product text in A. The Rule-RAG condition represents trace-assisted generation. A claim that restates an explicit material term from a product title may therefore be A-supported in either condition. A claim that follows a styling rule may be B-supported. A claim that invents “waterproof”, “premium leather”, or a definite instance-level colour match without such evidence is unsupported by the supplied evidence, but is not automatically factually false.

### 3.9.3 Generators and decoding

Three local, digest-pinned instruction models generated the explanations: Gemma 4 12B, Llama 3.1 8B Instruct Q8_0, and Ministral 3 14B Instruct Q4_K_M. The generator roster and decoding configuration were frozen before the full run. Each case--generator combination was attempted in both conditions under the same locked recommendation and common context; only the Rule-RAG prompt received the stored trace.

Five hundred locked, evidence-eligible cases were sampled from the final recommendation run, balanced with 100 cases each for bags, bottoms, outerwear, shoes, and tops. Crossing 500 cases, three generators, and two conditions produced 3,000 attempted explanation cells. Every prompt, response, latency, word count, model digest, and content hash was preserved. Outputs were immutable after generation. Stage 2 accepted 2,969 cells; the 31 terminal failures were all Llama Rule-RAG outputs that exceeded the shared 75-word limit after the permitted retries. Final comparisons therefore use generator-specific complete pairs only.

## 3.10 Validation-only configuration freezing

Before the full run, validation-only sensitivity grids examined fusion, candidate-pool, and evidence-reranking settings. They were used to freeze an image/text mixture of 0.40/0.60, a CLIP/evidence mixture of 0.75/0.25, and five retrieved rules per candidate. The grids are reported as validation evidence rather than pooled with the confirmatory results; no final recommendation, explanation, extraction, or verification outcome was selected after observing the test results.

The final Rule-RAG configuration was also frozen before generation. It displayed five rules in weighted-score order with stable IDs, reliability labels, and contribution information; it asked the generator to cite rules where relevant and imposed the same at-most-75-word cap used in No-RAG. Validation work set this contract and the ranking settings, but no pilot outcome is combined with the final estimates. The confirmatory corpus is solely the 3,000 Stage-2 attempted cells and the frozen Stage-3 and Stage-4 assessments derived from accepted outputs.

## 3.11 Atomic-claim extraction and verification

### 3.11.1 Extraction

The complete explanation, rather than a sentence sample, was passed to a frozen Qwen 3.5 9B extractor. The extractor enumerated independent atomic fashion or styling propositions, split independently checkable conjunctions, assigned sequential claim IDs, and applied the frozen claim schema. Extraction assessed neither truth nor support. Of 2,969 accepted explanations, 2,965 accepted extractions yielded 17,710 atomic claims; four terminal extraction failures were retained as missing records rather than repaired.

Claim role was assigned deterministically after extraction. `item_type` claims were identity/context claims because they commonly state what item is being recommended. The other schema labels were treated as substantive because even simple categories such as colour and material can carry explanatory content. After verification eligibility, the study-specific layer contained 10,703 substantive No-RAG claims and 8,666 substantive Rule-RAG claims. N/A rows were retained in the 3,000-explanation accounting but did not receive invented claim labels.

### 3.11.2 Multi-source verification

Phi-4 14B verified each extracted claim against complete context A, exact trace B, the record-specific full-KB candidate packet, and observed citations. The final schema preserves four distinct outcomes: `trace_support`, `full_kb_support`, `common_reference_support`, and `citation_entailment`. A `not_supported` value means that the specified evidence packet does not entail the claim under the frozen closed-world protocol; it is not a statement that the claim is false.

Using Phi-4 for verification separates the verifier from the Qwen 3.5 extractor and from the three explanation generators. Stage 4 accepted 2,861 verification records covering 16,804 claims; 104 terminal verification records were retained rather than imputed. A deterministic consistency correction then changed only `full_kb_support` from `not_supported` to `supported` for 163 claims that were trace-supported after confirming that the trace rule occurred in the same record's full-KB packet. This enforces the trace-subset invariant without adding a semantic judgement or rerunning a model.

## 3.12 Study-specific explanation metrics

### 3.12.1 Reranking-trace claim support

For an explanation \(E\) with extracted claims \(C(E)\), reranking-trace claim support is the proportion supported by the exact trace \(B_q\):

\[
\operatorname{TraceSupport}(E)=
\frac{\sum_{c\in C(E)}\mathbb{1}\!\left[\operatorname{trace\_support}(c)=\mathrm{supported}\right]}
{\lvert C(E)\rvert}.
\tag{3.10}
\]

The numerator counts only claims with the canonical `trace_support = supported` label. The denominator is the number of verified claims in that explanation. For No-RAG, a positive label is post-hoc agreement with hidden B; for Rule-RAG, it is support from evidence made visible at generation. Aggregate results use generator-specific complete pairs and case-clustered resampling rather than unpaired marginal totals.

### 3.12.2 Full-KB and common-reference support

Full-KB support uses the same verified claim set but asks whether a claim is supported anywhere in that record's final KB candidate packet. It is broader than trace support: it establishes knowledge-grounding, not that the supporting rule numerically contributed to reranking. The final invariant requires every trace-supported claim to be full-KB-supported.

\[
\operatorname{FullKBSupport}(E)=
\frac{\sum_{c\in C(E)}\mathbb{1}\!\left[\operatorname{full\_kb\_support}(c)=\mathrm{supported}\right]}
{\lvert C(E)\rvert}.
\tag{3.11}
\]

For UIFR, \(C_I(E)\) contains only eligible concrete item-fact claims. Explanations with no eligible claim receive no invented zero, and the paired UIFR comparison includes only cases eligible in both conditions. This deliberately restrictive denominator is why UIFR is secondary and much less precise than the support-rate outcomes.

For the restricted eligible item-fact claims \(C_I(E)\), the common-reference Unsupported Item-Fact Rate is

\[
\operatorname{UIFR}(E)=
\frac{\sum_{c\in C_I(E)}\mathbb{1}\!\left[\operatorname{common\_reference\_support}(c)=\mathrm{not\_supported}\right]}
{\lvert C_I(E)\rvert}.
\tag{3.12}
\]

This is a sensitivity metric, not the primary measure, because the opportunity to make claims does not necessarily grow linearly with words.

### 3.12.3 Citation entailment

Citation metrics apply only to Rule-RAG. A cited claim is valid only when an observed rule citation intersects the verifier’s exact supporting rule IDs, the rule supports the claim, and citation entailment is true:

\[
\operatorname{CitationEntailment}(E)=
\frac{\sum_{c\in C_{\mathrm{cited}}(E)}\mathbb{1}\!\left[\operatorname{citation\_entailment}(c)=\mathrm{entails}\right]}
{\lvert C_{\mathrm{cited}}(E)\rvert}.
\tag{3.14}
\]

No-RAG is reported as N/A, not zero. Precision and coverage are distinct from citation presence. A model can cite frequently yet attach the wrong rule, or cite validly on selected claims while leaving other rule-dependent claims uncited.

### 3.12.4 Scope of the metric set

The final metric set deliberately keeps distinct evidence boundaries rather than collapsing them into one broad grounding score. Trace support is the primary decision-faithfulness outcome, full-KB support is the broader rule-grounding outcome, UIFR is the restricted common-reference item-fact sensitivity, and citation entailment evaluates a cited Rule-RAG relation. No separate condition-specific support score or evidence-overreach rate is reported, because the frozen Stage-4 schema does not provide a common denominator or an independently annotated partial-entailment scale for either construct.

## 3.13 Final paired inference

For every final explanation outcome, the estimand was the paired Rule-RAG minus No-RAG difference. Analysis first retained complete pairs within each generator: 474 for Gemma, 438 for Llama, and 456 for Ministral. Generator-specific differences were then averaged within a case whenever more than one generator pair was available, yielding 498 paired cases for the primary analysis. This avoids treating unequal marginal condition totals as if they formed paired observations.

For metric \(m\), the case-level contrast was

\[
\widehat{\Delta}_m=
\frac{1}{|Q|}\sum_{q\in Q}
\left(\frac{1}{|G_q|}\sum_{g\in G_q}d_{qg}^{(m)}\right).
\tag{3.15}
\]

where \(d_{qg}^{(m)}\) is the generator-specific paired difference and \(G_q\) is the set of complete generator pairs for case \(q\). Percentile 95% confidence intervals were obtained from 5,000 bootstrap resamples of case IDs. Holm correction was applied to the predeclared family of primary support, UIFR, and trace-density contrasts. Citation entailment is summarised descriptively for Rule-RAG because it has no No-RAG counterpart.

## 3.14 Heterogeneity, robustness, and qualitative analysis

Primary trace-support, full-KB-support, and trace-density differences were estimated separately for the three generators and five categories using the same complete-pair rule. These subgroup estimates test directional stability rather than generalisation beyond the evaluated roster. UIFR subgroup analyses are especially cautious because the common-reference item-fact criterion yields a small jointly eligible set.

Missingness was retained as an observable property of the pipeline. The 31 terminal explanation failures were confined to Llama Rule-RAG outputs that exceeded the shared word cap after permitted retries; this produces the smaller Llama complete-pair count and rules out a misleading all-generator balanced-total calculation. Four accepted explanations did not yield accepted claim extraction, and 104 verification records remained terminal. Neither source corpus was silently completed, imputed, or regenerated. Stage 5 reports attempted and accepted denominators alongside the complete-pair analysis, and support rates are calculated only from the verified claims available under the frozen schema. This is conservative: it preserves the distinction between an unobserved assessment and a claim that the verifier labelled not supported.

The bootstrap unit also follows the pairing structure rather than the number of generated texts. A case can contribute up to three generator-specific pairs, so resampling every text independently would understate uncertainty by treating correlated outputs as unrelated observations. Resampling case IDs retains all complete generator pairs for each selected case and repeats the within-case averaging rule. Generator-specific estimates retain their own complete pairs, while category summaries are based on the same frozen case and generator identifiers. This makes the overall and subgroup analyses traceable to one coherent unit of inference, even though the available pair counts differ across generators.

The analysis distinguishes support prevalence from claim opportunity. A shorter text can contain fewer substantive propositions, whereas a longer text can provide more opportunities both for supported relational statements and for unsupported item details. Trace-supported-claim density, expressed per 100 generated words, therefore complements the support-rate outcomes. UIFR does not receive a forced denominator adjustment because its eligibility is defined by the presence of concrete item-fact claims in both conditions; its restricted complete-pair estimate is reported separately rather than extrapolated to explanations that made no eligible assertion. These choices make the final estimands narrower, but also prevent a broad claim of factual reliability from being inferred from a source-specific support result.

Qualitative examples were selected from frozen records to illustrate the distinction between relational styling statements and unsupported candidate attributes, along with the use and limits of rule citations. Examples preserve their common context, trace, generated text, and canonical assessment fields. They aid interpretation but do not replace corpus-level estimates.

## 3.15 Reproducibility, software, and ethical boundaries

The implementation used Python 3.12 on Windows 11. Embedding and LLM stages were configured for CUDA where available. The repository pins dependencies through `uv.lock`, model revisions or immutable local digests, dataset revision, random seeds, and canonical configuration hashes. Major tables, figures, JSONL corpora, and manifests are SHA-256 bound. Ruff linting and the project test suite cover data leakage, deterministic ranking, rule retrieval, prompt separation, schemas, refusal handling, statistics, artifact integrity, and the trace-support-to-full-KB-support invariant. Stage 5 validates principal output hashes and records the release boundary.

The study uses catalogue images and descriptions from a research dataset and makes no inference about protected personal attributes. It is not a safety-critical wardrobe adviser, product-authentication system, or factual catalogue service. Product text may be noisy, and outfit co-occurrence reflects platform curation rather than universal taste. Generated explanations must therefore be read as system outputs about supplied records, not authoritative fashion or product claims.

Automated assessment presents a more important validity limitation. Qwen 3.5 9B extracts claims and Phi-4 14B verifies them; this separates claim construction from entailment but does not establish either model's semantic accuracy. The restricted UIFR definition improves source-boundary transparency but cannot resolve every mixed natural-language claim. No human ratings, independent external annotations, or partial-entailment audit are included. The strongest warranted conclusion is comparative: under this frozen system evaluator, access to the exact five-rule trace changes measured trace support, full-KB support, trace-supported-claim density, and citation-related provenance. It does not establish human preference, factual correctness in the world, or causal faithfulness to every internal neural computation.

## 3.16 Chapter summary

The methodology connects recommendation and explanation through a locked decision and an exact symbolic trace. Multimodal CLIP retrieval supplies the base ranking; a category-filtered expert rule base contributes an evidence score; validation-only selection freezes the fusion and reranking design; and the confirmatory evaluation quantifies both recommendation effectiveness and ranking change. The explanation experiment holds recommendation identity constant while varying access to B across three generators and five categories. Atomic claims, trace and full-KB support, common-reference UIFR, citation entailment, complete-pair inference, subgroup estimates, and frozen qualitative examples provide complementary views of the output.

The design’s principal strength is traceability. Every reported explanation is connected to a specific A packet, B trace, generator digest, locked candidate, claim list, and verification record. Its principal limitation is equally clear: the final thesis experiment stops at automated system evaluation. Chapter 4 therefore reports quantitative differences without treating them as independently human-validated truth.

## 3.17 Validity controls and decision rules

Several controls were applied before results were interpreted. Construct validity was protected by refusing to collapse all explanation properties into a single “grounding” score. Trace support addresses agreement with the actual rule trace, full-KB support addresses broader rule grounding, UIFR addresses a narrow class of concrete product assertions, and citation entailment addresses whether an attached rule supports its claim. These measures can move differently. For example, an explanation can accurately paraphrase B while adding an unsupported material claim, or it can use a citation that does not entail the proposition it accompanies. Reporting each rate separately prevents one favourable dimension from concealing another failure.

Internal validity depended on pairing and freezing. The recommendation, A packet, B packet, generator, and case were identical within each No-RAG/Rule-RAG comparison. Outputs were not edited to equalise length, remove refusals, repair citations, or improve examples. Stage-2 generation records were hashed before assessment, and Stage-3 and Stage-4 outputs were preserved before deterministic metric derivation. Model selection and hyperparameter search were confined to development or validation data. These controls do not remove the prompt-and-length difference intrinsic to the intervention, but they prevent test-driven tuning and recommendation changes from contaminating the explanation contrast.

Statistical-conclusion validity was addressed through the appropriate dependence unit. Recommendation cases can share an outfit, so recommendation intervals resample query outfits. Explanation texts from three generators can share a case, so Stage-5 analyses resample case IDs after averaging complete generator-specific pairs within case. Micro and macro estimates answer different questions and are not treated as interchangeable: micro rates weight explanations in proportion to their number of claims, whereas macro rates weight each eligible explanation equally. The UIFR paired result has the additional jointly eligible restriction; it therefore describes texts that actually make concrete item-fact claims on both sides and is not extrapolated to all complete pairs.

External validity is bounded by the sampled catalogue, five broad categories, controlled pools, three local quantised generators, and supplied rule base. The system does not model individual user histories, price, availability, body measurements, culture-specific dress norms, or time-varying trends. Outfit co-occurrence is a useful offline relevance signal but does not prove that every held-out item would be preferred by a new user. Larger candidate pools show the expected reduction in absolute ranking performance and caution against treating the approximately 100-candidate estimates as full-catalogue serving results.

Reliability was supported by deterministic execution and independent re-computation from stored artifacts. SHA-256 ordering replaced random iteration in split, case, tie, and example selection. Generation used greedy deterministic settings, while structured assessment calls retained model digests and raw-response hashes. Each derived table can be traced to a manifest listing input and output hashes. Re-running the publication analysis does not invoke a model; it combines immutable records, recomputes rates and cluster resamples, and recreates figures and examples.

Finally, interpretation followed pre-specified language rules. A claim labelled unsupported is described as “unsupported by supplied evidence” or an “unverified item-specific assertion”, never as factually false unless affirmative contradiction exists. No-RAG agreement with hidden B is post-hoc alignment, never grounding. UIFR is explicitly described as inconclusive because its restricted eligible set is small. Evidence reranking is described as materially changing ranks and evidence alignment, not as improving recommendation accuracy, because the fused-CLIP contrasts do not establish an improvement. These linguistic constraints are part of the methodology: they keep the thesis claims proportional to the estimands actually measured.

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
