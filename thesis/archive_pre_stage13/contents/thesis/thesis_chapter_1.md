# Chapter 1

# Introduction

## 1.1 Background and motivation

Fashion recommendation is a demanding application of information retrieval because compatibility is not reducible to simple visual similarity. Two garments may look alike yet serve the same role and therefore be substitutes, while visually dissimilar items may complement one another within an outfit. Early work demonstrated that product images can support large-scale modelling of compatibility and substitution [1]. Later sequence, type-aware, and context-aware approaches represented outfits as structured combinations rather than isolated item pairs [2–4]. These developments established that visual and categorical information can improve the ranking of candidate fashion items.

Modern fashion catalogues also contain product text. Titles and descriptions may identify category, brand, material, or style terms that are difficult to infer reliably from pixels, while images contain appearance information that catalogue text may omit. Contrastive vision-language models provide a practical way to combine these signals within a shared representation [5]. A multimodal ranker can therefore exploit complementary image and text evidence without requiring every visual property to be translated into language.

Improved ranking does not, however, make a recommendation self-explanatory. A user may reasonably ask why a particular item was selected, and a system can produce a fluent answer without revealing which information actually influenced the ranking. Explainable-recommendation research has proposed feature-based, review-based, knowledge-graph, path-based, and natural-language rationales [6–9]. These approaches can improve transparency or persuasiveness, but the existence of a readable explanation does not by itself establish faithfulness to the underlying decision process.

This distinction has become more important with generative language models. Such models can turn sparse catalogue information into polished style advice, but they can also add likely-sounding colours, materials, comfort claims, occasions, or visual relationships that were not supplied to them. In explainable artificial intelligence, plausibility concerns whether an explanation appears reasonable to a reader, whereas faithfulness concerns whether it accurately reflects the mechanism or evidence being explained [10–12]. A plausible fashion rationale may therefore be useful as advice while remaining a post-hoc interpretation of the selected product.

Retrieval-augmented generation offers a partial response by displaying external context during generation [13]. Yet retrieved context and decision evidence are not necessarily the same thing. Documents retrieved after an item has been selected may support a persuasive account without showing that those documents participated in selection. Furthermore, access to relevant evidence does not ensure that every generated claim is entailed by it. Citation identifiers can improve auditability, but a citation is only informative when its source actually supports the associated claim [14,15].

The central motivation of this thesis is therefore to connect recommendation and explanation through an inspectable artifact produced during ranking. The implemented system combines CLIP image and text representations with an expert-rule score. For every candidate, the rule component retains the five retrieved rules, their similarities, reliability weights, bonuses, and weighted contributions. After reranking, the selected recommendation and its exact symbolic trace are frozen. This trace is not a reconstruction created by the explanation model; it records evidence that numerically participated in the ranking decision.

The explanation experiment then holds the selected item constant. For each case and language generator, a No-RAG explanation receives common context A: the request and frozen query and recommended-item identities, categories, and catalogue text. Its paired Rule-RAG explanation receives the same A plus exact trace B. This design isolates access to the symbolic decision trace more closely than comparing explanations of different recommendations. It also permits a necessary terminological distinction: agreement with hidden B is post-hoc decision-trace alignment for No-RAG, while agreement with visible B is decision-trace faithfulness for Rule-RAG.

The study remains deliberately bounded. Images participate in representation and ranking, but they are not captioned and do not become textual explanation evidence. The task is sampled, same-category, controlled-pool ranking over held-out Polyvore outfits rather than personalised, full-catalogue serving. The symbolic trace explains the expert-rule component of a hybrid reranker, not every internal computation performed by CLIP. Explanation outcomes are measured through a frozen automated pipeline in which Qwen3 extracts claims, Mistral verifies them, and Qwen3 judges paired quality. Cross-model verification strengthens role separation, but the results remain operational system evaluation rather than human-validated ground truth.

## 1.2 Problem statement

Existing fashion recommendation research provides increasingly capable representations of item compatibility, but conventional ranking metrics answer only whether relevant items are placed near the top of a candidate list. They do not show whether an explanation accurately describes information used by the system. This creates a gap between recommendation effectiveness and explanation faithfulness.

The gap has three parts. First, high-performing multimodal representations are generally latent. Although an image and its catalogue text can jointly influence a similarity score, the score does not naturally provide a concise verbal account of the decision. Translating latent similarity into unrestricted prose risks presenting an interpretation as if it were an observed causal trace.

Second, post-hoc natural-language explanations can exploit information that did not participate in the recommendation or introduce details absent from the supplied context. In fashion, this problem is particularly visible because descriptions of colour, material, silhouette, formality, season, comfort, and occasion are easy to generate plausibly. An unsupported assertion is not necessarily false in the world, but it is not justified by the evidence available to the generator. Treating unsupported and contradicted claims as equivalent would overstate what the experiment can establish.

Third, standard explanation-quality scores can conceal source-specific failures. Fluency, clarity, specificity, and general usefulness do not directly measure whether claims match the actual decision trace or whether cited rules entail them. A system may receive a high holistic score while misapplying a conditional rule or attaching a valid-looking identifier to an unsupported claim. Explanation evaluation therefore requires measures aligned with the precise evidence boundary.

This thesis addresses these problems through an evidence-aware hybrid architecture and a paired explanation experiment. Expert rules contribute numerically to reranking and are stored as exact trace B. Recommendation effectiveness is evaluated separately from evidence participation. For the same locked recommendation, trace-visible and trace-hidden explanations are compared using Decision-Trace Alignment (DTA), Unsupported Item-Attribute Rate (UIAR), citation precision and coverage, unsupported-attribute density, visible-evidence support, and general automated quality judgments.

The investigation does not assume that rule reranking improves accuracy. Indeed, the research design permits a negative recommendation result alongside a positive explanation result. This separation is important: a decision may become easier to audit without becoming more relevant, and an explanation may become more faithful to an evidence component whose domain coverage remains imperfect.

## 1.3 Aim, objectives, and research questions

The aim of this thesis is to develop and evaluate an evidence-constrained multimodal fashion recommendation framework in which generated explanations can be assessed against the exact expert-rule trace used during recommendation.

The first objective is to construct a reproducible controlled-pool fashion-ranking pipeline that combines image and text representations while preventing outfit and exact-image leakage across research splits. This includes deterministic case construction, category-controlled candidate pools, pinned model and dataset revisions, and conventional ranking evaluation.

The second objective is to make expert evidence participate directly in ranking. A curated rule base is retrieved and scored for each query-candidate pair, the top five contributions form an evidence score, and a frozen mixture of multimodal compatibility and evidence scores reranks the pool. The complete contributing trace is stored before explanation generation.

The third objective is to compare post-hoc and trace-visible natural-language explanations under a paired design. Both conditions explain the same locked recommendation, use the same generator, decoding settings, and at-most-75-word instruction, and differ in access to exact trace B and the associated evidence/citation instructions.

The fourth objective is to evaluate explanation behaviour at claim level. The study distinguishes alignment with the recommendation trace, unsupported concrete item attributes, citation correctness and completeness, broader visible-evidence grounding, and general explanation quality. It also examines generator and category heterogeneity, aggregation choices, length sensitivity, and leave-one-generator-out stability.

These objectives lead to four research questions:

- **RQ1:** Does displaying the exact five-rule decision trace during generation increase explanation alignment with that trace relative to a No-RAG post-hoc baseline?
- **RQ2:** Does trace access reduce concrete item-attribute assertions that are unsupported by the supplied evidence?
- **RQ3:** Do broad automated explanation-quality judgments provide the same information as targeted trace-alignment, unsupported-attribute, and citation measures?
- **RQ4:** Are the observed explanation differences directionally stable across generators, fashion categories, aggregation levels, and the predeclared length sensitivity?

Recommendation effectiveness and evidence participation are supporting questions needed to interpret these RQs. The thesis tests whether multimodal fusion improves ranking relative to text pathways, whether expert evidence materially changes selection, and whether any change in conventional accuracy is statistically supported. These tests establish what kind of decision the explanations are describing; they are not used to redefine explanation faithfulness as recommendation accuracy.

## 1.4 Scope and boundaries

The empirical dataset is the pinned `Marqo/polyvore` release described in Chapter 3. Held-out outfit co-occurrence supplies an offline relevance proxy. The primary evaluation contains approximately 100 same-category candidates per case and therefore represents sampled controlled-pool ranking. It does not model inventory, price, body fit, temporal preference, individual purchase history, or commercial serving constraints.

The multimodal pathway uses catalogue images and product text. It does not accept unconstrained real-world scene photographs, infer body characteristics, or perform image captioning. Keeping image pixels outside explanation evidence prevents unverified visual descriptions from entering A or B, but it also restricts what a legitimate explanation can say about visible properties.

The expert knowledge base contains 126 curated, generic styling rules. These rules provide an inspectable vocabulary for complementarity, formality, season, colour, and related styling considerations. They are not complete fashion knowledge, verified product metadata, or universal prescriptions. Faithfulness to a retrieved rule does not establish that the rule is correct for every person or context.

The final experiment is an automated offline systems study. Three quantised local generators produce the explanations. Qwen3 8B extracts claims and supplies broad paired judgments, while digest-pinned Mistral 7.2B performs claim verification. Confidence intervals quantify variation across sampled cases, not evaluator error. Human preference, trust, factual catalogue verification, and independent semantic annotation remain outside the completed experimental boundary.

## 1.5 Contributions

This thesis makes four principal contributions. First, it implements an end-to-end architecture in which an expert-rule trace is produced during multimodal reranking and retained for later explanation. Second, it introduces a paired evaluation design that locks recommendation identity, preventing recommendation differences from contaminating the explanation contrast. Third, it operationalises complementary measures—DTA, UIAR, citation precision and coverage, and unsupported-attribute density—that expose failures hidden by holistic quality ratings. Fourth, it provides a reproducible staged workflow with leakage control, validation-only tuning, frozen settings, immutable outputs, configuration hashes, and case-clustered statistical analysis.

The contribution is intentionally not framed as a new state-of-the-art fashion ranker or proof that expert evidence increases relevance. The completed results show that evidence reranking changes ranking substantially without a statistically significant improvement over fused CLIP. FashionCLIP image strengthens the representation baseline, while fixed-weight FashionCLIP fusion does not significantly exceed fused general CLIP. For explanations, trace access produces much stronger visible-evidence grounding, lower unsupported item-attribute rates, and higher holistic scores, but not higher common-reference DTA under the cross-model verifier. Preserving that mixed result is part of the empirical contribution.

## 1.6 Thesis structure

Chapter 2 reviews fashion compatibility modelling, multimodal ranking, explainable recommendation, retrieval-augmented generation, faithfulness, citation evaluation, and automated assessment. It derives the gap addressed by the implemented study without claiming that evidence can guarantee perfect generation.

Chapter 3 specifies the staged methodology, data controls, multimodal representations, expert-rule retrieval, reranking, locked explanation intervention, operational metrics, and statistical procedures. Chapter 4 reports recommendation, explanation, heterogeneity, sensitivity, and qualitative results. Chapter 5 interprets those findings, separates accuracy from evidence alignment, states the limitations, and identifies controlled length, independent assessment, verified visual evidence, and user-centred evaluation as future work.

## 1.7 Chapter summary

The research problem is not merely how to generate a compatible fashion recommendation or a fluent explanation. It is how to preserve an inspectable link between part of the ranking process and the language used to justify the selected item, while measuring residual unsupported claims honestly. The thesis addresses this problem through an evidence-participating hybrid reranker, an exact symbolic trace, paired generation for a locked recommendation, and claim-level automated evaluation. The following chapter positions this design against the relevant literature.

## References

[1] McAuley, J., Targett, C., Shi, Q. and van den Hengel, A. (2015) ‘Image-based recommendations on styles and substitutes’, *Proceedings of SIGIR 2015*, pp. 43–52. https://doi.org/10.1145/2766462.2767755.

[2] Han, X., Wu, Z., Jiang, Y.-G. and Davis, L.S. (2017) ‘Learning fashion compatibility with bidirectional LSTMs’, *Proceedings of ACM Multimedia 2017*, pp. 1078–1086. https://doi.org/10.1145/3123266.3123394.

[3] Vasileva, M.I. et al. (2018) ‘Learning type-aware embeddings for fashion compatibility’, *Proceedings of ECCV 2018*, pp. 390–405.

[4] Cucurull, G., Taslakian, P. and Vazquez, D. (2019) ‘Context-aware visual compatibility prediction’, *Proceedings of CVPR 2019*, pp. 12617–12626.

[5] Radford, A. et al. (2021) ‘Learning transferable visual models from natural language supervision’, *Proceedings of ICML 2021*, 139, pp. 8748–8763.

[6] Zhang, Y. and Chen, X. (2020) ‘Explainable recommendation: A survey and new perspectives’, *Foundations and Trends in Information Retrieval*, 14(1), pp. 1–101. https://doi.org/10.1561/1500000066.

[7] Xian, Y. et al. (2019) ‘Reinforcement knowledge graph reasoning for explainable recommendation’, *Proceedings of SIGIR 2019*, pp. 285–294. https://doi.org/10.1145/3331184.3331203.

[8] Zhu, Y. et al. (2021) ‘Faithfully explainable recommendation via neural logic reasoning’, *Proceedings of NAACL 2021*, pp. 3083–3090. https://doi.org/10.18653/v1/2021.naacl-main.245.

[9] Knijnenburg, B.P. et al. (2012) ‘Explaining the user experience of recommender systems’, *User Modeling and User-Adapted Interaction*, 22, pp. 441–504. https://doi.org/10.1007/s11257-011-9118-4.

[10] Jacovi, A. and Goldberg, Y. (2020) ‘Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?’, *Proceedings of ACL 2020*, pp. 4198–4205. https://doi.org/10.18653/v1/2020.acl-main.386.

[11] Wiegreffe, S. and Pinter, Y. (2019) ‘Attention is not not explanation’, *Proceedings of EMNLP-IJCNLP 2019*, pp. 11–20. https://doi.org/10.18653/v1/D19-1002.

[12] Lyu, Q., Apidianaki, M. and Callison-Burch, C. (2024) ‘Towards faithful model explanation in NLP: A survey’, *Computational Linguistics*, 50(2), pp. 657–723. https://doi.org/10.1162/coli_a_00511.

[13] Lewis, P. et al. (2020) ‘Retrieval-augmented generation for knowledge-intensive NLP tasks’, *Advances in Neural Information Processing Systems*, 33, pp. 9459–9474.

[14] Gao, T. et al. (2023) ‘Enabling large language models to generate text with citations’, *Proceedings of EMNLP 2023*, pp. 6465–6488. https://doi.org/10.18653/v1/2023.emnlp-main.398.

[15] Zhang, W. et al. (2024) ‘Towards fine-grained citation evaluation in generated text: A comparative analysis of faithfulness metrics’, *Proceedings of INLG 2024*, pp. 427–439. https://doi.org/10.18653/v1/2024.inlg-main.35.
