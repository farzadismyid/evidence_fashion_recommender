# Chapter 5

# Discussion and Conclusion

## 5.1 Introduction

This chapter interprets the findings in relation to the research problem and literature reviewed in Chapter 2. The study began from a practical tension. Multimodal fashion recommenders can rank visually and semantically compatible products, while language models can explain almost any selected item fluently. Neither capability ensures that the explanation describes the evidence that actually influenced the recommendation. The implemented system addresses this gap by storing the expert-rule component of the reranking decision and exposing that same trace to an explanation generator.

The completed experiment supports a bounded and deliberately mixed conclusion. Expert-rule reranking materially changed rank order but did not significantly improve or degrade recommendation effectiveness relative to fused general-purpose CLIP. FashionCLIP strengthened image-only retrieval, although its fixed fused version did not significantly beat the original fusion. When the exact five-rule trace was displayed during explanation generation under the same 75-word instruction as No-RAG, visible grounding, UIAR, unsupported-attribute density, and all six automated quality scores improved. Common-reference Decision-Trace Alignment (DTA) did not improve: post-hoc No-RAG alignment was higher under the independent Mistral verifier. These results do not establish human preference, world-factual correctness, or faithful access to every internal computation of a neural ranker.

## 5.2 Interpretation of recommendation findings

### 5.2.1 Value of multimodal fusion

Fused CLIP achieved HR@10 of 0.219 and NDCG@10 of 0.110, significantly exceeding MiniLM text and CLIP text on both top-ten measures after Holm correction. This supports the foundational premise that visual and textual catalogue signals are complementary. Product text identifies category, brand, material, and design terms, but images encode shape, colour distribution, and appearance that may be absent or noisy in titles. Conversely, text can distinguish semantically important properties that a generic visual encoder does not represent reliably. The fixed 0.40 image/0.60 text fusion benefited from both.

The result aligns with multimodal fashion research in which learned visual representations improve style compatibility and category-aware outfit modelling [1–3]. It also qualifies claims about multimodality. Fused CLIP was not significantly better than CLIP image after correction, and absolute metrics declined at larger pools. The experiment demonstrates benefit over specific text baselines in a controlled held-out task, not universal superiority over specialised fashion encoders or catalogue-scale retrieval systems.

The added FashionCLIP 2.0 baseline tests that final qualification directly [9]. FashionCLIP image achieved HR@10 0.266 and NDCG@10 0.1387, compared with 0.204 and 0.1022 for general CLIP image. The paired gains of 0.062 and 0.0365 remained significant after Holm correction (both adjusted \(p=0.0084\)); HR@5, NDCG@5, and MRR also improved significantly. This is evidence that fashion-domain representation learning matters for image retrieval in the present task.

The advantage did not transfer automatically through the locked 0.40 image/0.60 text fusion. FashionCLIP fused HR@10 was 0.225 and NDCG@10 0.1140, only 0.006 and 0.0040 above general fused CLIP, and none of the seven fused contrasts was significant after correction. FashionCLIP text was also statistically indistinguishable from general CLIP text. The most plausible interpretation is that the fixed weighting gives the text tower enough influence to dilute the domain-adapted image gain. The baseline therefore adds weight to the thesis while also identifying fusion calibration as an unresolved optimisation problem.

### 5.2.2 Accuracy and evidence alignment are different objectives

Evidence reranking produced HR@10 0.220, NDCG@10 0.1078, and MRR 0.0995. None differed significantly from fused CLIP. It would therefore be incorrect to present the rule component as an accuracy improvement. Yet the component changed 50.5% of top recommendations, 99.1% of ordered top-five rankings, and increased the evidence score of selected items. Its effect was both material and orthogonal to aggregate relevance.

This finding exposes a limitation of evaluating an evidence-aware recommender only with hit rate or NDCG. If two methods have similar mean relevance but select different items, one may produce decisions that are easier to justify through an explicit knowledge source. Conversely, optimising evidence alignment can move a model away from the latent taste signal encoded in outfit co-occurrence. The present 0.75/0.25 reranker found a region where substantial reordering occurred without a statistically detectable effectiveness penalty, but the confidence intervals do not prove equivalence. A future non-inferiority design would require a predeclared margin and adequate power.

The 126-rule base also gives the symbolic component a clear boundary. It cannot represent every fashion relationship, and rules are generic prescriptions rather than candidate facts. Its contribution is not to replace CLIP with a full expert system. It provides a compact decision vocabulary that can participate in ranking and later be inspected. The use of 112 rules and high entropy suggests that this vocabulary was not reduced to a few boilerplate explanations.

## 5.3 Interpretation of explanation faithfulness

### 5.3.1 From plausible rationale to trace-linked account

No-RAG post-hoc DTA was 90.08% micro and 91.45% macro. This unexpectedly high baseline is itself an important result: general language models can generate fashion reasoning that overlaps heavily with expert rules they never saw. Complementary categories, colour harmony, formality, and seasonal appropriateness are common concepts in pretraining and in the rule base. Agreement alone therefore cannot demonstrate use of the recommendation trace.

Rule-RAG achieved 84.95% micro and 86.36% macro DTA. Across 1,480 eligible pairs, Rule-RAG minus No-RAG was −5.13 percentage points (95% CI −6.51 to −3.77; \(p=0.0004\)). The directional hypothesis was therefore not supported by the revised cross-model assessment. This negative finding replaces the earlier same-model result rather than being hidden. It suggests that DTA, when defined only as entailment by B, measures semantic compatibility with the rule bank but cannot on its own distinguish causal evidence use from a plausible post-hoc reconstruction.

That limitation does not make the stored trace unimportant. Against evidence visible to the generator, support was only 2.31% micro for No-RAG and 89.12% for Rule-RAG. No-RAG’s 9,632 B-only supported claims are post-hoc matches because B was unavailable; Rule-RAG’s B-supported claims can be provenance-linked because B was in its prompt. Faithfulness is therefore more credible as a conjunction of evidence availability, support, and provenance than as DTA alone. This conclusion is consistent with literature distinguishing plausible rationales from faithful explanations [4,5], but it also exposes a weakness in the thesis’s initial operational hypothesis.

### 5.3.2 Unsupported item-specific assertions

UIAR addresses a different failure: concrete properties of actual products. No-RAG micro UIAR was 62.38%, compared with 37.11% for Rule-RAG. In the 192 jointly eligible pairs, the average reduction was 17.45 percentage points (95% CI −24.61 to −10.56; \(p=0.0004\)). Unsupported Attribute Density also fell from 0.728 to 0.317 per 100 words. Because both conditions received the same 75-word instruction, the result is no longer confounded by an explicitly free-form No-RAG prompt.

The qualitative examples show the form of these details. No-RAG texts described a clutch as leather or suede, inferred beads and feathers from a vague “hippie” title, claimed trousers were burgundy, assigned luxury and quality to a brand, and inferred comfort or warm-weather suitability. These claims may sometimes be true in the world. The problem is that the generator was not supplied with evidence for them. In a user-facing recommender, such specificity creates an impression of catalogue knowledge that the system does not possess.

Rule-RAG did not eliminate the issue. More than one third of its eligible attribute claims remained unsupported under strict A+B entailment. Some rules encouraged discussion of colour or formality, and the generator converted this into a definite candidate-level match. This is a known risk in retrieval-augmented generation: retrieval supplies relevant context, but generation can overgeneralise or blend that context with parametric knowledge [6,7]. The system’s generic rules are particularly vulnerable because they express what a recommender should seek, not what a product definitively is.

### 5.3.3 Citation integrity

Strict citation precision was 87.5%, but this was based on only 24 evaluated citation relations, of which 21 were valid. Strict coverage was 0.23%: only 19 of 8,275 claims classified as requiring rule support received a validly verified citation. These figures cannot sustain the earlier conclusion of substantial citation integrity. They instead show a mismatch between visibly frequent rule identifiers and the cross-model verifier’s ability to attach and validate those identifiers at atomic-claim level.

Citations still improve inspectability because a developer can follow a displayed rule ID to its source. They are not self-validating, however, and the automated estimate is evaluator-dependent. A system that displays references without reliably checking the claim–source relation can create “evidence theatre.” The defensible conclusion is therefore limited: the architecture preserves citation provenance, but this experiment does not demonstrate comprehensive valid citation coverage.

## 5.4 Relationship between faithfulness and user-facing quality

All six automated quality dimensions favoured Rule-RAG. The largest difference was evidence-use correctness (+3.335 on a five-point scale), followed by specificity (+1.506), hallucination control (+1.155), and input consistency (+1.040). General quality improved by 0.663 and clarity by 0.427. The result counters the assumption that evidence constraints necessarily make explanations mechanical or unreadable. Under the common cap, Rule-RAG was slightly longer overall but judged clearer and better.

However, DTA moved in the opposite direction. The holistic judge recognised the overall Rule-RAG regime, visible provenance, and cautious style, while common-reference entailment rewarded plausible No-RAG overlap with B. The disagreement reinforces the multidimensional account of explanation quality: fluency, usefulness, visible grounding, unsupported specificity, and trace compatibility are related but not interchangeable [4,8]. A single preference score cannot reveal which failure occurred.

This separation reinforces the multidimensional account of explanation quality described in the literature. Fluency, plausibility, usefulness, and faithfulness are related but not interchangeable [4,8]. A concise explanation may be clear and useful while citing the wrong rule. A long explanation may contain several correct reasons and one unsupported material assertion. A single preference score cannot reveal which failure occurred.

The current results should also be interpreted conservatively because evaluation is automated. Qwen3 served as extractor and holistic judge, while Mistral independently verified extracted claims. This separation reduces direct extraction–verification self-confirmation, but it does not provide human ground truth. Structural normalisation was needed for 42.95% of actual verifier outputs, and 20 verifications plus two extractions remained missing after recovery. The saved raw responses and explicit denominators make this uncertainty inspectable rather than eliminating it.

## 5.5 Generator and category heterogeneity

The revised pipeline retained balanced condition coverage for all three generators and five categories. DTA paired eligibility was 498 for Gemma and 491 each for Llama and Mistral; category counts ranged from 292 to 299. UIAR intersections were smaller because many concise explanations made no concrete item-attribute claim. Subgroup tables therefore help diagnose where models and categories differ, but they are not used to override the primary overall contrast.

Three quantised open-weight generators remain a limited roster. Their compliance with the common cap differed: Gemma averaged 46.86/43.14 words for No-RAG/Rule-RAG, Llama 63.83/59.73, and Mistral 47.83/78.77. This variation is a behavioural result and a reminder that a shared instruction does not guarantee identical realised length. Proprietary models, domain-fine-tuned generators, or citation-trained models could behave differently. Category variation may likewise reflect rule conditionality and product-title quality rather than an inherent benefit for a fashion category.

## 5.6 The role of explanation length

Both conditions were instructed to use at most 75 words. No-RAG averaged 52.84 words and Rule-RAG 60.55, reducing the earlier design gap to 7.71 words. The residual direction varied by generator, and the Stage 5 follow-up reproduced a small aggregate difference under the same instruction. Length is therefore still a behavioural covariate, but it is no longer an imposed free-form-versus-capped contrast.

All headline measures are rates, and unsupported-attribute density directly divides by word count while retaining a substantial Rule-RAG advantage. The 30 closest-length pairs averaged 54.17 and 54.37 words, with a mean absolute gap of only 0.33. Their DTA difference was −5.51 points with an interval crossing zero; UIAR was descriptively lower for Rule-RAG but only eight pairs were jointly eligible. A full factorial short/free experiment would estimate evidence, length regime, and their interaction more cleanly, but it would require regenerating and re-verifying four cells. It is appropriately left outside the completed thesis rather than added as an underpowered late rerun.

## 5.7 Contributions of the research

The first contribution is an end-to-end architecture that connects multimodal ranking to a reusable explanation trace. CLIP image and text representations support candidate retrieval; expert rules participate numerically in reranking; and the exact top-five contributions are stored before explanation generation. This avoids constructing “evidence” after the recommendation has already been chosen.

The second contribution is a paired experimental design that locks recommendation identity. No-RAG and Rule-RAG explain the same item for the same user request and generator. Differences in explanation cannot be attributed to one condition receiving an easier or more compatible recommendation. This is a practical design pattern for evaluating explanation interventions in recommender systems.

The third contribution is a study-specific measurement framework whose disagreement is itself informative. DTA measures compatibility with B and exposes the high post-hoc No-RAG baseline; visible support incorporates evidence availability; UIAR isolates concrete item assertions; and unsupported-attribute density controls exposure by length. Citation precision and coverage reveal the fragility of automated claim–source relation checking rather than rewarding marker presence. Together, these measures make the evidence boundary explicit and prevent one favourable score from standing in for faithfulness.

The fourth contribution is reproducibility. Data revisions, model revisions and digests, configuration hashes, prompt hashes, rule traces, response hashes, cluster units, and derived artifacts are preserved. The publication analyses—paired judge statistics, metric associations, formal subgroup heterogeneity, leave-one-generator-out sensitivity, figures, and qualitative selection—operate entirely on saved records with zero new inference. This is particularly valuable for local generative experiments whose outputs can change with server or model updates.

Finally, the study offers two useful negative contributions. Evidence-aware reranking should not be sold as more accurate merely because it creates inspectable decisions: its effectiveness was statistically indistinguishable from fused CLIP. Likewise, displaying the trace should not be said to increase DTA when a cross-model verifier finds the opposite. Separating those results from the strong visible-grounding, UIAR, quality, and FashionCLIP-image findings makes the contribution more credible.

## 5.8 Limitations

### 5.8.1 Offline relevance and controlled pools

Same-outfit co-occurrence is an imperfect relevance proxy. It reflects one curated outfit rather than all acceptable combinations and labels plausible alternatives as negatives. Candidate pools contain sampled same-category items, not a live catalogue with inventory, price, user history, or business constraints. Metrics at approximately 100 candidates cannot be transferred directly to full-catalogue serving. The larger-pool sensitivity confirms that absolute performance falls as competition increases.

### 5.8.2 Visual representation and fusion

General CLIP ViT-B/32 is not fine-tuned on outfit compatibility, body fit, or subtle fashion attributes. FashionCLIP improves the image-only baseline but is still trained for fashion product representation rather than the exact outfit-completion objective, and its improvement was diluted by the frozen fusion weight. The study deliberately avoided visual captioning so that images could not leak unverified text into explanations. This protects the evidence boundary but means the explanation cannot mention genuine visual properties unless product text or a rule supports them.

### 5.8.3 Rule-base coverage and semantics

The 126 rules are curated, finite, and generic. Reliability labels do not constitute empirical truth probabilities, and rule retrieval by semantic similarity can select a rule whose antecedent only partly matches the case. The trace faithfully records what the algorithm used, but the algorithm’s rule choice can itself be questionable. DTA rewards alignment with B, not correctness of B. Recommendation metrics partly evaluate the consequence, but do not validate individual rules.

### 5.8.4 Automated evaluation

No human audit is part of the final study. Qwen3-generated claim boundaries and Mistral support labels can be wrong. Cross-model separation is stronger than using one model for both tasks, but shared training data and general LLM biases remain possible. Structural normalisation, while deterministic and conservative, cannot correct semantic mistakes. The UIAR classifier is transparent but schema-dependent; ambiguous claims are excluded, which improves conservatism at the cost of coverage. Reported confidence intervals quantify sampling variation over cases, not evaluator uncertainty.

Evidence Overreach Rate was removed because the saved labels do not reliably distinguish partial support from full entailment. Some qualitative Rule-RAG outputs appear to overstate conditional rules, but no corpus rate is asserted. This omission is preferable to manufacturing a precise value from inadequate labels.

### 5.8.5 Generalisability and model dependence

The three generators are quantised local models between 3.2B and 12.2B parameters. Greedy decoding improves reproducibility but does not represent common stochastic chat settings. The assessment model is also local and quantised. Results may differ for larger proprietary models, domain-fine-tuned generators, alternative RAG templates, or repeated stochastic samples. The tested categories cover common apparel groups but not dresses as an independent target, cosmetics, sizing, or personalised styling.

### 5.8.6 Computational reporting

The study reports model sizes, quantisation, digests, latency, and output counts but not FLOPs. Retrospective FLOP calculation for cached encoders and quantised generation would require operation-level instrumentation that was not captured. Any theoretical number would depend on sequence length, attention implementation, batching, cache reuse, quantisation kernels, and whether integer operations are counted as floating-point operations. The absence of FLOPs limits hardware-normalised efficiency comparison but does not affect the statistical conclusions. Future work should record energy, peak memory, token throughput, and kernel-level operation counters prospectively.

## 5.9 Implications for system design

For developers, the main implication is that evidence should be produced during the decision and carried forward as a typed artifact. A list of documents retrieved only after selection can explain a candidate without showing that those documents influenced it. The stored B trace instead binds rule identifiers and contributions to the chosen item. This makes later generation and auditing possible.

The second implication is to constrain instance-level language. A system can safely say that a rule recommends considering colour compatibility without asserting that two unknown colours match. Product facts should come from explicit catalogue metadata or verified visual attributes, and rule advice should remain relational. Prompt instructions can encourage this distinction, but deterministic post-generation checks or constrained templates may be needed for high-stakes deployment.

The third implication is to evaluate citations as relations. A valid citation is not a bracketed token; it is a claim–source entailment. Systems should store cited spans, source IDs, and support status at claim level. Precision and coverage can then be monitored separately. Invalid or missing citations should remain inspectable rather than silently repaired in user-visible text.

The fourth implication is to preserve multi-objective evaluation. Recommendation relevance, evidence participation, DTA, UIAR, citation integrity, clarity, and length can conflict. A deployment decision should specify acceptable thresholds rather than collapse these properties into a single utility after seeing test results. The validation Pareto approach used here is one practical starting point.

## 5.10 Future research

The most valuable next experiment is a pre-planned factorial length study. The same locked cases should be crossed with evidence visible/hidden and short/free length instructions. This would determine whether B reduces unsupported attributes beyond the effect of rhetorical concision. Multiple deterministic or stochastic samples per cell would permit variance decomposition across case, generator, prompt, and decoding. It was not added late to the present thesis because it would require a full four-cell generation and assessment rerun.

A second direction is improved machine evaluation. Additional independently trained verifier models, calibrated entailment benchmarks, and deterministic citation-span parsing could test whether the DTA reversal and near-zero strict citation coverage are model-specific. Evaluator agreement should be reported before any consensus or adjudication rule. This stays within an automated research programme while directly addressing the largest remaining measurement uncertainty.

Third, visual evidence could be added under a separate, explicit boundary. A verified attribute extractor could produce an image-evidence block C containing confidence-scored colour, pattern, and silhouette observations. Explanations could then cite A, B, or C. This would allow legitimate visual specificity while maintaining source provenance. FashionCLIP provides a stronger representation starting point, but an attribute benchmark and calibrated thresholds would still be needed.

Fourth, the rule base could be evaluated and expanded. Experts could assess rule correctness, antecedent applicability, redundancy, and category gaps. Learned retrieval could be calibrated against labelled rule relevance, while counterfactual tests could remove or swap rules and observe ranking and explanation changes. Such interventions would more directly test dependence on specific evidence than observational DTA alone.

Fifth, recommendation evaluation could move beyond sampled co-occurrence through temporal hold-outs, larger catalogue pools, stronger compatibility objectives, and calibrated fusion. In particular, the FashionCLIP image gain motivates validation-only tuning of image/text fusion and a fair comparison between fashion-domain and general encoders at their own selected weights. User-centred evaluation remains outside the scope of the present automated thesis.

Finally, efficiency should become a first-class outcome. Prospective profiling could record tokens, joules, latency percentiles, GPU memory, and operation counts for each stage. Smaller generators or extractive templates may recover most faithfulness benefits at lower cost. Because Rule-RAG outputs were shorter and faster in this experiment, evidence constraints may improve both reliability and efficiency, but a controlled throughput study is needed.

## 5.11 Final conclusion

This thesis developed and evaluated an evidence-constrained multimodal fashion recommendation framework in which the explanation can be linked to the exact expert rules used by the reranker. The system combined CLIP image and text representations, a curated 126-rule knowledge base, a frozen five-rule evidence trace, deterministic local language generation, and claim-level assessment.

The recommendation experiment showed that multimodal fusion improved key top-ten measures over text baselines. Adding expert evidence changed ranking substantially but did not significantly improve or degrade effectiveness relative to fused CLIP. FashionCLIP then improved image-only HR@5, HR@10, NDCG@5, NDCG@10, and MRR, while its fixed fused version produced no significant gain. The explanation experiment showed that Rule-RAG greatly increased visible support, reduced unsupported item assertions and their word-normalised density, and improved all six automated quality scores under a shared 75-word instruction.

It did not show increased DTA. Cross-model verification found 90.08% post-hoc No-RAG B alignment against 84.95% Rule-RAG alignment, with a paired macro difference of −5.13 points. Strict citation coverage was also too sparse for a positive integrity claim. These findings narrow, rather than erase, the contribution: trace exposure improves operational grounding and caution, but semantic agreement with a rule reference is neither unique to trace access nor sufficient evidence of faithful use.

The study also shows why careful terminology matters. No-RAG agreement with hidden rules is not grounding. Unsupported by supplied evidence is not factually false. A citation is not valid merely because it is present. A faithful explanation of an evidence-aware decision does not prove that the decision is more accurate. Preserving these distinctions turns an otherwise persuasive demonstration into a defensible experiment.

The final evidence therefore supports a specific claim: within the frozen tested system, exposing the exact five-rule recommendation trace produced explanations with far stronger visible grounding, fewer unverified item-specific assertions, lower unsupported-attribute density, and higher automated quality, without improving common-reference DTA or recommendation accuracy. The FashionCLIP extension additionally shows that domain adaptation helps image retrieval but requires fusion recalibration. The architecture, corrected controls, cross-model verification, negative findings, and reproducible artifacts provide a credible basis for a paper focused on evaluating evidence-grounded recommendation explanations without conflating plausibility, trace compatibility, and provenance.

## References

[1] McAuley, J., Targett, C., Shi, Q. and van den Hengel, A. (2015) ‘Image-based recommendations on styles and substitutes’, *Proceedings of SIGIR 2015*, pp. 43–52. https://doi.org/10.1145/2766462.2767755.

[2] Han, X., Wu, Z., Jiang, Y.-G. and Davis, L.S. (2017) ‘Learning fashion compatibility with bidirectional LSTMs’, *Proceedings of the 25th ACM International Conference on Multimedia*, pp. 1078–1086. https://doi.org/10.1145/3123266.3123394.

[3] Radford, A. et al. (2021) ‘Learning transferable visual models from natural language supervision’, *Proceedings of the 38th International Conference on Machine Learning*, 139, pp. 8748–8763.

[4] Jacovi, A. and Goldberg, Y. (2020) ‘Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?’, *Proceedings of ACL 2020*, pp. 4198–4205. https://doi.org/10.18653/v1/2020.acl-main.386.

[5] Wiegreffe, S. and Pinter, Y. (2019) ‘Attention is not not explanation’, *Proceedings of EMNLP-IJCNLP 2019*, pp. 11–20. https://doi.org/10.18653/v1/D19-1002.

[6] Lewis, P. et al. (2020) ‘Retrieval-augmented generation for knowledge-intensive NLP tasks’, *Advances in Neural Information Processing Systems*, 33, pp. 9459–9474.

[7] Ji, Z. et al. (2023) ‘Survey of hallucination in natural language generation’, *ACM Computing Surveys*, 55(12), Article 248. https://doi.org/10.1145/3571730.

[8] Doshi-Velez, F. and Kim, B. (2017) ‘Towards a rigorous science of interpretable machine learning’, arXiv:1702.08608.

[9] Chia, P.J. et al. (2022) ‘Contrastive language and vision learning of general fashion concepts’, *Scientific Reports*, 12, 18958. https://doi.org/10.1038/s41598-022-23052-9.
