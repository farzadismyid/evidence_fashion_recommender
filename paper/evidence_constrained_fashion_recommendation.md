# Evidence-Constrained Multimodal Fashion Recommendation with Trace-Grounded Explanations

**[Author name]**  
**[Affiliation]**  
**Corresponding author:** [email address]

## Abstract

Generative explanations can make a recommendation appear transparent without describing evidence that actually contributed to its selection. This paper evaluates a trace-grounded alternative for multimodal fashion recommendation. A hybrid system first ranks category-constrained candidates with frozen CLIP image--text representations and then reranks them using a curated expert-rule score. Crucially, the exact rules that contribute to the reranking score are stored before explanation generation. For each selected item, we compare paired No-RAG and Rule-RAG explanations: both conditions receive the same request, query-item and recommended-item text, while only Rule-RAG receives the exact stored rule trace. The final experiment contains 1,000 recommendation cases across five fashion categories and 500 evidence-eligible explanation cases. Three local generators produce 3,000 attempted explanations; Qwen 3.5 extracts atomic claims and Phi-4 verifies each claim against the trace, the record-specific full knowledge-base packet, common item-fact evidence, and observed citations. Evidence reranking changed the top recommendation in 26.5% of cases but did not improve conventional relevance over fused CLIP. In generator-specific complete-pair, case-clustered analysis, trace exposure increased trace-supported claim rate by 21.02 percentage points (95% CI 19.72--22.37) and full-KB-supported claim rate by 21.40 points (20.11--22.71). It also increased trace-supported claims per 100 words by 1.60 (1.48--1.72). The common-reference Unsupported Item-Fact Rate was inconclusive because only 53 complete pairs were eligible. The contribution is therefore not a claim that rule reranking improves recommendation accuracy or that generated prose is universally factual. It is an auditable experimental pattern: preserve evidence that participates in a symbolic ranking component, expose that exact artifact during generation, and evaluate generated claims against explicit source boundaries.

**Keywords:** explainable recommendation; multimodal retrieval; retrieval-augmented generation; faithfulness; fashion compatibility; claim verification

## 1 Introduction

Fashion recommendation is a useful setting for studying the gap between a good ranking and a trustworthy explanation. Compatibility is not mere visual similarity: two visually similar garments can be substitutes, whereas dissimilar items can work together because they play complementary roles in an outfit. Research on outfit compatibility has consequently used images, category structure, context, and learned representations to rank candidate items [1--4]. Modern catalogues also provide product text, and contrastive vision--language models offer a practical way to combine textual and visual signals in a shared representation [5].

Ranking, however, is only one side of an intelligent information system. A user who asks why an item was recommended needs an account that is connected to evidence, rather than polished prose that could have been written for any plausible candidate. This distinction is central to explainable recommendation [6--9] and to the wider distinction between plausible explanations and faithful explanations [10--12]. Large language models make the distinction operationally urgent: they can produce appealing fashion advice, but may also infer colours, materials, occasions, comfort, or quality that are not present in the supplied record.

Retrieval-augmented generation (RAG) can ground a response in external material [13], but a post-hoc retrieval set is not necessarily decision evidence. If a recommender retrieves rules only after it has selected an item, those rules can support a persuasive story without demonstrating that they influenced the selection. Citation markers do not solve this problem by themselves: a citation is meaningful only if the cited source entails the claim to which it is attached [14,15].

We address this problem with a deliberately narrow intervention. Our hybrid recommender combines CLIP image and text compatibility with an expert-rule evidence score. For every query--candidate pair, the system records the retrieved rule IDs, similarities, reliability weights, bonuses, ordering, and weighted contributions. The top-ranked reranked candidate is then locked. The same exact rule trace that participated in the symbolic reranking component is supplied to the Rule-RAG explanation condition; it is hidden from the paired No-RAG condition. Both conditions explain the same locked recommendation with the same common context, generator, decoding configuration, and at-most-75-word instruction.

This design makes four contributions:

- An end-to-end evidence-aware recommender in which an inspectable expert-rule trace is produced during reranking rather than reconstructed after selection.
- A paired explanation intervention that holds the recommendation fixed and isolates access to the stored trace.
- A claim-level evaluation protocol separating trace support, full-KB support, common-reference item-fact support, and citation entailment.
- A frozen, reproducible five-stage release with leakage controls, validation-only configuration selection, complete-pair analysis, case-clustered bootstrap inference, and transparent failure accounting.

The results are intentionally mixed. Evidence reranking materially changes rank order, but fused CLIP remains stronger on conventional controlled-pool relevance measures. In contrast, exposing the stored trace substantially improves source-specific explanation support. This separation is a feature of the study: recommendation effectiveness, traceability, citation validity, and factual restraint should not be compressed into a single unqualified claim that an explanation is “better.”

## 2 Related Work

### 2.1 Fashion compatibility and multimodal retrieval

Fashion compatibility modelling has progressed from visual similarity and substitute discovery [1] to outfit-level sequence models [2], type-aware compatibility embeddings [3], and context-aware visual prediction [4]. These approaches show that a suitable item need not resemble the query; it must occupy a compatible role in a partially observed outfit. In catalogue settings, text adds useful category, material, and style terms that images may omit, while images retain appearance information absent from titles. CLIP provides a well-established shared embedding space for such multimodal signals [5].

The present paper does not propose a new foundation representation or claim state-of-the-art outfit completion. Instead, it treats frozen MiniLM and CLIP pathways as reproducible retrieval components and asks a different systems question: can an evidence component that contributes to reranking subsequently support an auditable explanation?

### 2.2 Explainable recommendation and faithfulness

Explainable recommender systems use features, reviews, rules, graphs, paths, and natural-language rationales to communicate recommendations [6--9]. Their user-facing value is clear, but a readable rationale does not necessarily establish a relationship with the computation that selected an item. The faithfulness literature cautions that explanations can appear convincing without being causally or evidentially tied to the model behaviour they describe [10--12].

Our claim is bounded to a hybrid system. The stored trace is not presented as a complete explanation of CLIP's latent computation. It is the exact account of the expert-rule component that numerically participates in reranking. This distinction permits a precise test: whether showing that recorded artifact changes the support of generated claims relative to the same artifact.

### 2.3 RAG, citations, and claim-level assessment

RAG improves access to external information during generation [13], while citation-aware generation aims to make source use inspectable [14,15]. Yet source presence, citation syntax, and claim entailment are different properties. A rule may be displayed but not used; it may be cited but not support the surrounding sentence; and a generic styling rule may support a relational rationale without proving an item-specific attribute.

Accordingly, our evaluation uses four evidence boundaries. `trace_support` asks whether the exact five-rule reranking trace supports a claim. `full_kb_support` asks whether a record-specific packet from the final 200-rule KB supports it. `common_reference_support` is restricted to eligible concrete item-fact claims shared by both conditions. `citation_entailment` evaluates a Rule-RAG citation as a claim--rule relation rather than as a marker. This source-specific design follows the principle that evaluators should measure the particular property an explanation claims to provide, rather than generic fluency alone.

## 3 Method

### 3.1 Design overview

The final project is organised as five frozen stages. Stage 1 audits the data, final 200-rule KB, splits, prompts, model identities, and validation-only settings. Stage 2 executes recommendations and paired explanation generation. Stage 3 extracts atomic claims. Stage 4 verifies claims against frozen evidence packets. Stage 5 performs deterministic analysis from saved outputs only. No model calls are made during final statistical analysis.

The study uses the pinned `Marqo/polyvore` dataset revision `8c782ee447faf2d2a0402ac883cf07d3b3f43e1c`. The split unit is the outfit. Exact-image duplicate groups are resolved before final quota restoration, preventing cross-split outfit and exact-image leakage. The test partition supplies 1,000 recommendation cases, balanced at 200 cases for each target category: bags, bottoms, outerwear, shoes, and tops. Each case ranks same-category candidates from a controlled pool containing up to 99 negatives plus all known positives. The task is therefore a sampled controlled-pool evaluation, not full-catalogue or personalised production recommendation.

### 3.2 Multimodal ranking and evidence-aware reranking

MiniLM encodes product text, while `openai/clip-vit-base-patch32` supplies frozen image and text embeddings. All vectors are L2-normalised. For query (q), the fused CLIP representation is

\[
f_q=\frac{\alpha v_q+(1-\alpha)t_q}{\lVert\alpha v_q+(1-\alpha)t_q\rVert_2}, \qquad \alpha=0.40,
\tag{1}
\]

and candidate compatibility is (s_{\mathrm{CLIP}}(q,i)=f_q^{\top}f_i). The image/text fusion weight was fixed on validation data only.

The final KB, `fashion_rules.csv`, contains 200 curated styling rules: 40 for each target category. Rules express relational styling knowledge--for example, compatibility of formality, colour coordination, or layering--rather than verified attributes of a particular product. For each query--candidate pair, rules are category-filtered and scored using semantic similarity, an applicable query-group bonus, and a reliability weight. The retained top five rules define the evidence score:

\[
s_E(q,i)=\frac{1}{5}\sum_{r\in R_5(q,i)}u(q,i,r).
\tag{2}
\]

Within each candidate pool, CLIP and evidence scores are min--max normalised and combined as

\[
s_R(q,i)=0.75\,\widetilde{s}_{\mathrm{CLIP}}(q,i)+0.25\,\widetilde{s}_{E}(q,i).
\tag{3}
\]

The top-ranked reranked item is locked before any explanation call. Its stored trace includes the rules, rule IDs, contributions, retrieval ranks, weights, and the final evidence score. Thus, the Rule-RAG evidence is not a later retrieval or a manually written explanation.

### 3.3 Paired explanation intervention

Five hundred evidence-eligible locked recommendations, 100 per category, form the explanation set. Let (A_q) denote the common context for case (q): the request plus query and locked-item identities, categories, and product text. Let (B_q) denote the exact stored reranking trace. For generator (g), the two outputs are

\[
E^{\mathrm{NoRAG}}_{qg}=G_g(A_q), \qquad E^{\mathrm{RuleRAG}}_{qg}=G_g(A_q,B_q).
\tag{4}
\]

Both conditions explain the same locked item. The generators are Gemma 4 12B, Llama 3.1 8B Instruct Q8_0, and Ministral 3 14B Instruct Q4_K_M. Each receives the same word cap and common information; Rule-RAG additionally receives B, an explicit grounding instruction, and citation guidance. The intervention therefore estimates the effect of trace-grounded prompting, not the effect of changing the recommendation.

### 3.4 Claim extraction and verification

Qwen 3.5 9B decomposes each accepted explanation into independently checkable atomic claims without judging truth or support. Phi-4 14B then verifies each claim against common context A, exact trace B, the record-specific full-KB candidate packet, and observed rule citations. A `not_supported` label means that the supplied closed-world packet does not entail the claim; it is not an assertion that the claim is false in the world.

For explanation (E) and extracted verified claims (C(E)), trace support is

\[
\operatorname{TraceSupport}(E)=\frac{\sum_{c\in C(E)}\mathbb{1}[\operatorname{trace\_support}(c)=\mathrm{supported}]}{\lvert C(E)\rvert}.
\tag{5}
\]

Full-KB support uses the same form with `full_kb_support`. The common-reference Unsupported Item-Fact Rate (UIFR) is calculated only for eligible concrete item-fact claims, with no invented zero for explanations containing none. Citation entailment is reported only for citations observed in Rule-RAG. A deterministic post-verification correction enforced the logical invariant `trace_support = supported` implies `full_kb_support = supported`: after confirming the trace rule was in the record-specific full-KB packet, 163 logically impossible `not_supported` full-KB labels were changed to `supported`. No model was rerun and no other field changed.

### 3.5 Inference and missingness

Recommendation confidence intervals use 5,000 percentile bootstrap samples clustered on query outfit (734 unique outfits). For explanation outcomes, analysis retains generator-specific complete No-RAG/Rule-RAG pairs: Gemma 474, Llama 438, and Ministral 456. Available generator-specific differences are averaged within case, yielding 498 underlying paired cases. The primary confidence intervals use 5,000 case-cluster bootstrap replicates; Holm adjustment is applied across the four prespecified overall contrasts.

Failures are retained rather than imputed. Of 3,000 attempted cells, 2,969 explanations were accepted. Thirty-one terminal generation failures all occurred in Llama Rule-RAG outputs that exceeded the shared word cap after permitted retries. Of the accepted explanations, 2,965 yielded accepted extractions with 17,710 claims, and 2,861 yielded accepted verifications covering 16,804 claims. Unequal raw condition totals are never used as paired comparisons.

## 4 Experimental Results

### 4.1 Recommendation effectiveness and evidence participation

Table 1 reports controlled-pool recommendation effectiveness. Fused CLIP provides the strongest aggregate ranking outcomes among the evaluated pathways. Evidence reranking does not improve those outcomes; it is retained because it produces the symbolic decision trace evaluated in the explanation experiment.

| Method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM text | 4.8% | 12.0% | 17.7% | 8.4% | 10.3% | 10.2% |
| CLIP image | 3.2% | 12.6% | 22.0% | 7.9% | 10.9% | 10.0% |
| CLIP text | 3.5% | 13.8% | 21.2% | 8.6% | 11.0% | 10.1% |
| Fused CLIP | 4.5% | 14.3% | 23.1% | 9.4% | 12.2% | 11.4% |
| Evidence rerank | 3.8% | 13.2% | 22.5% | 8.5% | 11.4% | 10.6% |

**Table 1.** Micro recommendation metrics over 1,000 cases. All results use the frozen approximately 100-candidate controlled-pool regime.

Despite the absence of an accuracy gain, evidence reranking changes the top item in 26.5% of cases. Mean top-five overlap between fused CLIP and reranking is 3.859 items; the selected item's mean evidence-score gain is 0.1473; and its mean pre-to-post rank shift is 0.566. At least one locked trace contains 148 of the 200 final rules. The appropriate interpretation is therefore evidence participation and rank reordering, not improved relevance.

### 4.2 Explanation grounding

Table 2 contains the primary paired contrasts. Rule-RAG improves support from both the exact evidence that participated in reranking and the larger record-specific KB packet. It also increases the amount of trace-supported content relative to output length. UIFR is inconclusive: its strict common-reference definition provides only 53 eligible complete pairs.

| Outcome, Rule-RAG minus No-RAG | Paired cases | Estimate | 95% CI | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| Trace claim support rate | 498 | +21.02 pp | +19.72 to +22.37 pp | 0.0016 |
| Full-KB claim support rate | 498 | +21.40 pp | +20.11 to +22.71 pp | 0.0016 |
| Unsupported Item-Fact Rate | 53 | +0.63 pp | -5.03 to +5.66 pp | 0.9042 |
| Trace-supported claims per 100 words | 498 | +1.60 | +1.48 to +1.72 | 0.0016 |

**Table 2.** Generator-specific complete-pair, case-clustered bootstrap contrasts. Positive UIFR values are undesirable; the interval spans zero.

The central result is source-specific. No-RAG sometimes agrees with hidden B, but such agreement is post-hoc because B was unavailable during generation. Rule-RAG receives B and therefore permits an auditable link between the text and evidence used by the symbolic reranker. The full-KB result acts as a broader cross-check: it is not merely a narrow trace-label effect.

### 4.3 Robustness across generators and categories

The primary effects are positive for all tested generators (Table 3). Their confidence intervals exclude zero for trace support, full-KB support, and trace-supported-claim density. All five categories also have positive support differences with confidence intervals excluding zero. This is directional robustness within the evaluated roster; it is not a claim about all language models or all fashion domains.

| Generator | Complete pairs | Trace support difference | Full-KB support difference | Trace claims / 100 words difference |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 12B | 474 | +25.09 pp | +25.22 pp | +1.79 |
| Llama 3.1 8B | 438 | +26.74 pp | +27.26 pp | +1.91 |
| Ministral 3 14B | 456 | +11.35 pp | +11.77 pp | +1.13 |

**Table 3.** Generator-specific complete-pair estimates. The smaller Llama denominator reflects 31 Rule-RAG word-cap failures, not a zero-valued performance outcome.

### 4.4 Citations and qualitative audit

Citation syntax is not counted as proof of support. Among the final verified claims, 1,820 citation relations entail their claim, 5,502 do not entail it, and 9,482 are not applicable. Strict citation precision is 87.5% for the small evaluated citation-relation subset (21 valid of 24), while verified citation coverage is 0.23% (19 validly verified citations among 8,275 claims classified as requiring rule support). These sparse figures do not support a sweeping citation-quality claim; they show why citations must be assessed as relations.

One frozen paired record illustrates the mechanism and its limits. The query is a “marc jacobs st. marc small leather shoulder bag,” and the locked recommendation is “marni embellished draped satin crop top.” No-RAG describes “a sophisticated contrast,” “high-fashion aesthetic,” and an “elevated evening look.” Rule-RAG receives rule K185, which states that structured shoulder bags pair well with several neat or fitted top types and that prominent hardware should remain a focal point. It explains that the structured leather bag coordinates with a draped top for a polished city look and cites K185. The example is useful because it preserves the original trace and the generated text; it is not used as substitute evidence for the corpus estimates. It also shows why the output must not be treated as a verified product description: “premium craftsmanship” is not established by the rule.

## 5 Discussion

The experiment separates two objectives that are often merged in descriptions of explainable recommender systems. The first is retrieval effectiveness. Under the fixed controlled-pool protocol, fused CLIP is strongest. The second is whether explanation claims can be connected to stored decision evidence. Under the paired intervention, Rule-RAG is substantially stronger on trace support, full-KB support, and trace-supported claim density. The results do not establish that an evidence-aware system is more accurate simply because it is more inspectable.

The stored-trace design is important. A generic RAG system can attach relevant rules after selection, but cannot demonstrate that those rules participated in selection. Here the exact contributing rules are recorded before generation and supplied unchanged to Rule-RAG. This gives the resulting language a clear provenance target while avoiding a claim to expose hidden chain-of-thought or every neural computation in CLIP.

The negative and inconclusive findings sharpen rather than weaken the paper. UIFR is not shown to improve because the common-reference item-fact criterion is deliberately strict and sparsely eligible. Citation markers are not accepted as citation entailment. Llama's Rule-RAG word-cap failures introduce a condition-specific missingness pattern, addressed through generator-specific complete pairs rather than raw unequal totals. These facts limit the conclusion, but also make the remaining positive result more credible.

For intelligent information systems, the practical implication is architectural: evidence should be created during decision making and retained as typed, inspectable data. A system can then expose the same artifact to a generator, an auditor, or a user interface. Rules should be used to justify relational styling guidance, not to assert unsupported item facts. Citation interfaces should preserve rule IDs and claim-level support information rather than rewarding decorative references.

## 6 Limitations

Several limitations bound the result. First, relevance derives from held-out outfit co-occurrence in sampled same-category pools. It does not represent individual taste, pricing, availability, temporal trends, or full-catalogue serving. Second, the 200-rule KB is finite and curated. Trace support means that a claim aligns with evidence used by the symbolic component; it does not prove that the rule itself is universally correct or that it captures every relevant fashion relationship.

Third, images influence ranking but are not converted into textual evidence. This protects the closed-world explanation boundary, but means a visually true attribute can remain unsupported if it is not explicit in product text or a rule. Fourth, Qwen 3.5 extraction and Phi-4 verification are automated. Their role separation reduces direct same-model confirmation, but neither provides human ground truth. Confidence intervals quantify case-sampling variation, not evaluator uncertainty.

Finally, the intervention includes both trace availability and the associated grounding/citation instruction. It measures the practical Rule-RAG contract, not a pure isolation of every prompt component. A future factorial study could separately manipulate trace visibility, citation instruction, and length regime. Human judgments, independently calibrated verifiers, validated visual attributes, counterfactual rule removal, and larger catalogue pools are important next steps.

## 7 Conclusion

This paper presents a reproducible approach to evidence-constrained multimodal fashion recommendation in which a symbolic rule trace participates in reranking and is then reused for explanation. The final experiment demonstrates a meaningful trade-off: evidence reranking changes decisions but does not improve conventional controlled-pool relevance over fused CLIP; trace exposure substantially improves trace and full-KB claim support in paired explanations.

The appropriate conclusion is specific. Within the frozen dataset, models, KB, and automated evaluator, showing the exact five-rule reranking trace produces explanations with more claims supported by that trace and by the associated full-KB packet. It does not prove universal factual correctness, human preference superiority, complete neural faithfulness, or production-scale recommendation performance. By retaining the trace, separating evidence boundaries, preserving failures, and analysing complete pairs at the case level, the study offers a reviewer-auditable pattern for evaluating evidence-grounded explanations in recommender systems.

## Declarations

**Funding.** No external funding was received for this study.

**Conflict of interest.** The author declares no conflict of interest.

**Data availability.** The frozen release package, manifests, canonical result tables, and code required to reproduce the reported analyses without additional LLM calls are available in the accompanying project repository. Dataset access remains subject to the source dataset's terms.

**Code availability.** The project repository contains the frozen configurations, manifests, tests, and deterministic analysis scripts.

**Ethics approval.** Not applicable. The study uses public catalogue-style research data and contains no human-participant experiment.

## References

1. McAuley, J., Targett, C., Shi, Q., van den Hengel, A.: Image-based recommendations on styles and substitutes. In: *Proceedings of SIGIR 2015*, pp. 43--52 (2015). https://doi.org/10.1145/2766462.2767755
2. Han, X., Wu, Z., Jiang, Y.-G., Davis, L.S.: Learning fashion compatibility with bidirectional LSTMs. In: *Proceedings of ACM Multimedia 2017*, pp. 1078--1086 (2017). https://doi.org/10.1145/3123266.3123394
3. Vasileva, M.I., et al.: Learning type-aware embeddings for fashion compatibility. In: *ECCV 2018*, pp. 390--405 (2018).
4. Cucurull, G., Taslakian, P., Vazquez, D.: Context-aware visual compatibility prediction. In: *CVPR 2019*, pp. 12617--12626 (2019).
5. Radford, A., et al.: Learning transferable visual models from natural language supervision. In: *ICML 2021*, vol. 139, pp. 8748--8763 (2021).
6. Zhang, Y., Chen, X.: Explainable recommendation: A survey and new perspectives. *Foundations and Trends in Information Retrieval* 14(1), 1--101 (2020). https://doi.org/10.1561/1500000066
7. Xian, Y., et al.: Reinforcement knowledge graph reasoning for explainable recommendation. In: *SIGIR 2019*, pp. 285--294 (2019). https://doi.org/10.1145/3331184.3331203
8. Zhu, Y., et al.: Faithfully explainable recommendation via neural logic reasoning. In: *NAACL 2021*, pp. 3083--3090 (2021). https://doi.org/10.18653/v1/2021.naacl-main.245
9. Knijnenburg, B.P., et al.: Explaining the user experience of recommender systems. *User Modeling and User-Adapted Interaction* 22, 441--504 (2012). https://doi.org/10.1007/s11257-011-9118-4
10. Jacovi, A., Goldberg, Y.: Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? In: *ACL 2020*, pp. 4198--4205 (2020). https://doi.org/10.18653/v1/2020.acl-main.386
11. Wiegreffe, S., Pinter, Y.: Attention is not not explanation. In: *EMNLP-IJCNLP 2019*, pp. 11--20 (2019). https://doi.org/10.18653/v1/D19-1002
12. Lyu, Q., Apidianaki, M., Callison-Burch, C.: Towards faithful model explanation in NLP: A survey. *Computational Linguistics* 50(2), 657--723 (2024). https://doi.org/10.1162/coli_a_00511
13. Lewis, P., et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. In: *NeurIPS 2020*, vol. 33, pp. 9459--9474 (2020).
14. Gao, T., et al.: Enabling large language models to generate text with citations. In: *EMNLP 2023*, pp. 6465--6488 (2023). https://doi.org/10.18653/v1/2023.emnlp-main.398
15. Zhang, W., et al.: Towards fine-grained citation evaluation in generated text: A comparative analysis of faithfulness metrics. In: *INLG 2024*, pp. 427--439 (2024). https://doi.org/10.18653/v1/2024.inlg-main.35
16. Järvelin, K., Kekäläinen, J.: Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems* 20(4), 422--446 (2002). https://doi.org/10.1145/582415.582418
17. Efron, B., Tibshirani, R.J.: *An Introduction to the Bootstrap*. Chapman & Hall/CRC, New York (1993).
18. Holm, S.: A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65--70 (1979).
