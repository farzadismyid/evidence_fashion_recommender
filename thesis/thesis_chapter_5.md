# Chapter 5

# Discussion and Conclusion

## 5.1 Introduction

This chapter interprets the findings in relation to the research problem and literature reviewed in Chapter 2. The study began from a practical tension. Multimodal fashion recommenders can rank visually and semantically compatible products, while language models can explain almost any selected item fluently. Neither capability ensures that the explanation describes the evidence that actually influenced the recommendation. The implemented system addresses this gap by storing the expert-rule component of the reranking decision and exposing that same trace to an explanation generator.

The completed experiment supports a bounded and deliberately mixed conclusion. Expert-rule reranking materially changed rank order but did not improve conventional recommendation effectiveness relative to fused CLIP. When the exact five-rule trace was supplied during explanation generation, reranking-trace claim support increased by 21.02 percentage points and full-KB claim support increased by 21.40 points under complete-pair, case-clustered inference. Trace-supported claims per 100 words also increased by 1.60. The common-reference UIFR outcome was inconclusive because only 53 complete pairs were eligible. These results do not establish human preference, world-factual correctness, or faithful access to every internal computation of a neural ranker.

## 5.2 Interpretation of recommendation findings

### 5.2.1 Value of multimodal fusion

Fused CLIP achieved HR@1 of 0.045, HR@5 of 0.143, HR@10 of 0.231, NDCG@5 of 0.0943, NDCG@10 of 0.1223, and MRR of 0.1144 in the frozen controlled-pool evaluation. These values establish the multimodal reference against which the evidence reranker was compared. Product text identifies category, brand, material, and design terms, but images encode shape, colour distribution, and appearance that may be absent or noisy in titles. Conversely, text can distinguish semantically important properties that a generic visual encoder does not represent reliably. The fixed 0.40 image/0.60 text fusion therefore combines complementary catalogue signals.

The result aligns with multimodal fashion research in which learned visual representations improve style compatibility and category-aware outfit modelling [1–3]. It also qualifies claims about multimodality. The experiment evaluates a fixed general-purpose multimodal representation in a controlled held-out task; it does not establish universal superiority over specialised fashion encoders or catalogue-scale retrieval systems.

The final comparison remains deliberately modest. It evaluates frozen MiniLM text, CLIP image, CLIP text, fused CLIP, and the evidence reranker on the same controlled pools; it does not claim to exhaust modern fashion encoders. Its contribution is the reproducible relationship between a multimodal retrieval baseline, a symbolic reranking component, and the exact trace later used for explanation under a documented, fixed protocol.

### 5.2.2 Accuracy and evidence alignment are different objectives

Evidence reranking produced HR@10 of 0.225, NDCG@10 of 0.1145, and MRR of 0.1056, compared with 0.231, 0.1223, and 0.1144 for fused CLIP. It would therefore be incorrect to present the rule component as an accuracy improvement. Yet the component changed 26.5% of top recommendations, increased the selected item's evidence score by 0.1473 on average, and used 148 of the 200 rules at least once. Its effect was material and orthogonal to aggregate relevance.

This finding exposes a limitation of evaluating an evidence-aware recommender only with hit rate or NDCG. If two methods have similar mean relevance but select different items, one may produce decisions that are easier to justify through an explicit knowledge source. Conversely, optimising evidence alignment can move a model away from the latent taste signal encoded in outfit co-occurrence. The present 0.75/0.25 reranker found a region where substantial reordering occurred without a statistically detectable effectiveness penalty, but the confidence intervals do not prove equivalence. A future non-inferiority design would require a predeclared margin and adequate power.

The 200-rule base also gives the symbolic component a clear boundary. It cannot represent every fashion relationship, and rules are generic prescriptions rather than candidate facts. Its contribution is not to replace CLIP with a full expert system. It provides a compact decision vocabulary that can participate in ranking and later be inspected.

## 5.3 Interpretation of explanation faithfulness

### 5.3.1 From plausible rationale to trace-linked account

The final primary outcome is not a generic semantic-overlap score. It is the proportion of extracted claims supported by the exact reranking trace. Across 498 underlying complete-pair cases, Rule-RAG increased this rate by 21.02 percentage points (95% CI +19.72 to +22.37; Holm-adjusted \(p=0.0016\)). It increased full-KB support by 21.40 points (95% CI +20.11 to +22.71; Holm-adjusted \(p=0.0016\)). These effects are large enough to matter substantively as well as statistically.

The interpretation depends on evidence availability. No-RAG claims that happen to match the hidden trace demonstrate post-hoc agreement, not trace use. Rule-RAG claims supported by B can be connected to source material displayed in the prompt and to a symbolic component that genuinely participated in reranking. This does not prove that the generator followed every rule or that the trace is a complete account of neural computation. It establishes a much tighter provenance relation than a post-hoc rule retrieval performed only after the recommendation is selected.

The effect is stable over the tested generators. Trace-support differences were +25.09 points for Gemma, +26.74 for Llama, and +11.35 for Ministral; all confidence intervals excluded zero. The same positive direction appeared for every category. This consistency strengthens the comparative claim while remaining bounded to the three local generators, five categories, and frozen prompt contract.

### 5.3.2 Unsupported item-specific assertions

The final UIFR outcome addresses a different failure: concrete item facts unsupported by the common reference packet. Its estimate was +0.63 percentage points for Rule-RAG minus No-RAG, with a 95% interval from -5.03 to +5.66 and Holm-adjusted \(p=0.9042\). Only 53 complete pairs were eligible. The result is therefore inconclusive, not evidence that Rule-RAG either reduces or increases unsupported item facts.

The qualitative examples show the form of these details. No-RAG texts described a clutch as leather or suede, inferred beads and feathers from a vague “hippie” title, claimed trousers were burgundy, assigned luxury and quality to a brand, and inferred comfort or warm-weather suitability. These claims may sometimes be true in the world. The problem is that the generator was not supplied with evidence for them. In a user-facing recommender, such specificity creates an impression of catalogue knowledge that the system does not possess.

The restricted denominator is itself instructive. Many explanations make no claim that falls within the item-fact schema, and treating those texts as zero-error would be misleading. Generic rules also remain vulnerable to overgeneralisation: a rule can recommend considering colour or formality without proving a particular catalogue item's colour, material, or occasion. The source-boundary design makes this limitation visible but does not eliminate it.

### 5.3.3 Citation integrity

Strict citation precision was 87.5%, but this was based on only 24 evaluated citation relations, of which 21 were valid. Strict coverage was 0.23%: only 19 of 8,275 claims classified as requiring rule support received a validly verified citation. These figures cannot sustain the earlier conclusion of substantial citation integrity. They instead show a mismatch between visibly frequent rule identifiers and the cross-model verifier’s ability to attach and validate those identifiers at atomic-claim level.

Citations still improve inspectability because a developer can follow a displayed rule ID to its source. They are not self-validating, however, and the automated estimate is evaluator-dependent. A system that displays references without reliably checking the claim–source relation can create “evidence theatre.” The defensible conclusion is therefore limited: the architecture preserves citation provenance, but this experiment does not demonstrate comprehensive valid citation coverage.

## 5.4 Relationship between grounding and user-facing quality

The final experiment does not include a holistic preference judge or a human-quality study. It cannot therefore establish that Rule-RAG explanations are clearer, more persuasive, or more useful to users. This absence is not a minor reporting omission: a fluent explanation may be attractive while being unsupported, and a trace-grounded explanation may be cautious or less rhetorically engaging. The study deliberately avoids converting claim-level support into a universal claim of explanation quality.

What the final data do show is a source-specific grounding advantage. Trace support, full-KB support, and trace-supported-claim density all favour Rule-RAG, whereas the common-reference item-fact outcome is inconclusive. These measures answer different questions. A trace-supported claim links the language to evidence used in the symbolic reranker; a full-KB-supported claim links it to a broader available packet; a common-reference item-fact claim tests a much narrower form of catalogue-specific support; and citation entailment asks whether an attached rule actually supports the statement.

This separation reinforces the multidimensional account of explanation quality described in the literature [4,8]. Fluency, usefulness, plausibility, factual restraint, citation validity, and decision-trace grounding are related but not interchangeable. A concise explanation may be useful while citing the wrong rule. A detailed explanation may contain several trace-supported relations and one unsupported material assertion. A single preference score cannot reveal which source-boundary failure occurred.

The results should also be interpreted conservatively because assessment is automated. Qwen 3.5 extracts claims and Phi-4 verifies them; this role separation reduces direct self-confirmation but does not provide human ground truth. Four extractions and 104 verifications remained terminal failures, and the saved raw records and denominators make this uncertainty inspectable rather than eliminating it.

## 5.5 Generator and category heterogeneity

The final pipeline retained three generators and five categories, but accepted coverage was not perfectly balanced. Complete pairs were 474 for Gemma, 438 for Llama, and 456 for Ministral; overall analysis used 498 cases with at least one available generator pair. UIFR intersections were much smaller because only some paired outputs made an eligible common-reference item-fact claim. Subgroup results therefore diagnose stability and do not override the primary overall contrast.

The three local generators remain a limited roster. Llama's 31 Rule-RAG terminal word-count failures also show that a shared instruction does not guarantee identical contract compliance. Proprietary, domain-fine-tuned, citation-trained, or stochastic generators could behave differently. Category variation may likewise reflect rule conditionality and product-title quality rather than an inherent benefit for a fashion category.

## 5.6 The role of explanation length

Both conditions were instructed to use at most 75 words. No-RAG averaged 52.84 words and Rule-RAG 60.55, a residual difference of 7.71 words under the shared instruction. The direction and size of the difference varied by generator. Length is therefore still a behavioural covariate, but it is no longer an imposed free-form-versus-capped contrast.

The primary support outcomes are claim-level rates, and trace-supported-claim density is reported per 100 generated words. Rule-RAG's trace-density advantage therefore cannot be attributed solely to producing more text. Nevertheless, length remains a behavioural covariate: the trace and citation instructions are part of the intervention, and the completed study does not isolate their separate effects. A factorial design crossing evidence visibility with length instructions would estimate those effects more cleanly, but it would require a new generation and verification experiment and is outside the frozen thesis boundary.

## 5.7 Contributions of the research

The first contribution is an end-to-end architecture that connects multimodal ranking to a reusable explanation trace. CLIP image and text representations support candidate retrieval; expert rules participate numerically in reranking; and the exact top-five contributions are stored before explanation generation. This avoids constructing “evidence” after the recommendation has already been chosen.

The second contribution is a paired experimental design that locks recommendation identity. No-RAG and Rule-RAG explain the same item for the same user request and generator. Differences in explanation cannot be attributed to one condition receiving an easier or more compatible recommendation. This is a practical design pattern for evaluating explanation interventions in recommender systems.

The third contribution is a study-specific measurement framework whose distinctions are themselves informative. Trace support measures alignment with B and distinguishes post-hoc No-RAG agreement from Rule-RAG evidence use; full-KB support evaluates the broader final rule packet; UIFR isolates eligible concrete item-fact assertions; and trace-supported-claim density accounts for generated length. Citation entailment reveals the fragility of automated claim–source relation checking rather than rewarding marker presence. Together, these measures make the evidence boundary explicit and prevent one favourable score from standing in for faithfulness.

The fourth contribution is reproducibility. Data revisions, model revisions and digests, configuration hashes, prompt hashes, rule traces, response hashes, cluster units, and derived artifacts are preserved. Stage-5 paired support analysis, subgroup summaries, figures, and qualitative selection operate entirely on saved records with zero new inference. This is particularly valuable for local generative experiments whose outputs can change with server or model updates.

Finally, the study offers two useful negative contributions. Evidence-aware reranking should not be sold as more accurate merely because it creates inspectable decisions: its effectiveness did not improve over fused CLIP. Likewise, displaying the trace should not be treated as proof of comprehensive citation validity or human-preferred explanation quality. Separating those limits from the strong trace-support, full-KB-support, and trace-density findings makes the contribution more credible.

## 5.8 Limitations

### 5.8.1 Offline relevance and controlled pools

Same-outfit co-occurrence is an imperfect relevance proxy. It reflects one curated outfit rather than all acceptable combinations and labels plausible alternatives as negatives. Candidate pools contain sampled same-category items, not a live catalogue with inventory, price, user history, or business constraints. Metrics at approximately 100 candidates cannot be transferred directly to full-catalogue serving. The larger-pool sensitivity confirms that absolute performance falls as competition increases.

### 5.8.2 Visual representation and fusion

The frozen CLIP pathway is not fine-tuned on the exact outfit-completion objective, body fit, or subtle fashion attributes. The study deliberately avoided visual captioning so that images could not leak unverified text into explanations. This protects the evidence boundary but means an explanation cannot legitimately mention genuine visual properties unless product text or a rule supports them.

### 5.8.3 Rule-base coverage and semantics

The 200 rules are curated, finite, and generic. Their provenance does not constitute empirical truth probabilities, and semantic retrieval can select a rule whose antecedent only partly matches the case. The trace faithfully records what the algorithm used, but the algorithm's rule choice can itself be questionable. Trace support rewards alignment with B, not correctness of B. Recommendation metrics partly evaluate the consequence, but do not validate individual rules.

### 5.8.4 Automated evaluation

No human audit is part of the final study. Qwen-generated claim boundaries and Phi support labels can be wrong. Cross-model separation is stronger than using one model for both tasks, but shared training data and general LLM biases remain possible. The deterministic logical-consistency correction cannot correct semantic mistakes; it only enforces the trace-subset invariant. The restricted UIFR schema improves conservatism at the cost of coverage. Reported confidence intervals quantify sampling variation over cases, not evaluator uncertainty.

Evidence Overreach Rate was removed because the saved labels do not reliably distinguish partial support from full entailment. Some qualitative Rule-RAG outputs appear to overstate conditional rules, but no corpus rate is asserted. This omission is preferable to manufacturing a precise value from inadequate labels.

### 5.8.5 Generalisability and model dependence

The three generators are quantised local models between 3.2B and 12.2B parameters. Greedy decoding improves reproducibility but does not represent common stochastic chat settings. The assessment model is also local and quantised. Results may differ for larger proprietary models, domain-fine-tuned generators, alternative RAG templates, or repeated stochastic samples. The tested categories cover common apparel groups but not dresses as an independent target, cosmetics, sizing, or personalised styling.

### 5.8.6 Computational reporting

The study reports model sizes, quantisation, digests, latency, and output counts but not FLOPs. Retrospective FLOP calculation for cached encoders and quantised generation would require operation-level instrumentation that was not captured. Any theoretical number would depend on sequence length, attention implementation, batching, cache reuse, quantisation kernels, and whether integer operations are counted as floating-point operations. The absence of FLOPs limits hardware-normalised efficiency comparison but does not affect the statistical conclusions. Future work should record energy, peak memory, token throughput, and kernel-level operation counters prospectively.

## 5.9 Implications for system design

For developers, the main implication is that evidence should be produced during the decision and carried forward as a typed artifact. A list of documents retrieved only after selection can explain a candidate without showing that those documents influenced it. The stored B trace instead binds rule identifiers and contributions to the chosen item. This makes later generation and auditing possible.

The second implication is to constrain instance-level language. A system can safely say that a rule recommends considering colour compatibility without asserting that two unknown colours match. Product facts should come from explicit catalogue metadata or verified visual attributes, and rule advice should remain relational. Prompt instructions can encourage this distinction, but deterministic post-generation checks or constrained templates may be needed for high-stakes deployment.

The third implication is to evaluate citations as relations. A valid citation is not a bracketed token; it is a claim–source entailment. Systems should store cited spans, source IDs, and support status at claim level. Precision and coverage can then be monitored separately. Invalid or missing citations should remain inspectable rather than silently repaired in user-visible text.

The fourth implication is to preserve multi-objective evaluation. Recommendation relevance, evidence participation, trace support, full-KB support, UIFR, citation entailment, and length can conflict. A deployment decision should specify acceptable thresholds rather than collapse these properties into a single utility after seeing test results. The validation Pareto approach used here is one practical starting point.

## 5.10 Future research

The most valuable next experiment is a pre-planned factorial length study. The same locked cases should be crossed with evidence visible/hidden and short/free length instructions. This would determine whether B reduces unsupported attributes beyond the effect of rhetorical concision. Multiple deterministic or stochastic samples per cell would permit variance decomposition across case, generator, prompt, and decoding. It was not added late to the present thesis because it would require a full four-cell generation and assessment rerun.

A second direction is improved machine evaluation. Additional independently trained verifier models, calibrated entailment benchmarks, and deterministic citation-span parsing could test whether the observed support and citation-entailment estimates are model-specific. Evaluator agreement should be reported before any consensus or adjudication rule. This stays within an automated research programme while directly addressing the largest remaining measurement uncertainty.

Third, visual evidence could be added under a separate, explicit boundary. A verified attribute extractor could produce an image-evidence block C containing confidence-scored colour, pattern, and silhouette observations. Explanations could then cite A, B, or C. This would allow legitimate visual specificity while maintaining source provenance, provided that the extractor is evaluated against an attribute benchmark with calibrated thresholds.

Fourth, the rule base could be evaluated and expanded. Experts could assess rule correctness, antecedent applicability, redundancy, and category gaps. Learned retrieval could be calibrated against labelled rule relevance, while counterfactual tests could remove or swap rules and observe ranking and explanation changes. Such interventions would more directly test dependence on specific evidence than the present trace-support measurements alone.

Fifth, recommendation evaluation could move beyond sampled co-occurrence through temporal hold-outs, larger catalogue pools, stronger compatibility objectives, and calibrated fusion. A future comparison should evaluate general and fashion-domain encoders at their own validation-selected weights, with the same split and candidate protocol. User-centred evaluation remains outside the scope of the present automated thesis.

Finally, efficiency should become a first-class outcome. Prospective profiling could record tokens, joules, latency percentiles, GPU memory, and operation counts for each stage. Smaller generators or extractive templates may recover most faithfulness benefits at lower cost. Because Rule-RAG outputs were shorter and faster in this experiment, evidence constraints may improve both reliability and efficiency, but a controlled throughput study is needed.

## 5.11 Final conclusion

This thesis developed and evaluated an evidence-constrained multimodal fashion recommendation framework in which the explanation can be linked to the exact expert rules used by the reranker. The system combined CLIP image and text representations, a curated 200-rule knowledge base, a frozen five-rule evidence trace, deterministic local language generation, and claim-level assessment.

The recommendation experiment showed that fused CLIP produced the strongest conventional top-five and top-ten effectiveness among the tested methods. Adding expert evidence changed ranking substantially but did not improve effectiveness relative to fused CLIP. The explanation experiment showed that Rule-RAG increased exact-trace support, full-KB support, and trace-supported claim density under the same locked recommendation and common context.

The study did not show a reliable UIFR advantage, because only 53 complete pairs were eligible and the interval crossed zero. Citation syntax was also not treated as a positive integrity result: a citation requires claim--rule entailment. These findings narrow, rather than erase, the contribution. Trace exposure improves auditable evidence grounding under the frozen evaluator; it is neither a universal claim of factual correctness nor sufficient evidence of complete faithfulness to a neural model.

The study also shows why careful terminology matters. No-RAG agreement with hidden rules is not grounding. Unsupported by supplied evidence is not factually false. A citation is not valid merely because it is present. A faithful explanation of an evidence-aware decision does not prove that the decision is more accurate. Preserving these distinctions turns an otherwise persuasive demonstration into a defensible experiment.

The final evidence therefore supports a specific claim: within the frozen tested system, exposing the exact five-rule recommendation trace produced explanations with substantially stronger trace and full-KB claim support, and greater trace-supported-claim density, without improving recommendation accuracy. UIFR was inconclusive, and citation syntax was not accepted as citation entailment. The architecture, controls, cross-model verification, negative findings, and reproducible artifacts provide a credible basis for evaluating evidence-grounded recommendation explanations without conflating plausibility, trace compatibility, and provenance.

This conclusion is deliberately useful rather than maximal. It gives a recommender-system designer a concrete, testable pattern: retain the evidence that participates in a symbolic decision component, pass that exact artifact to the explanatory interface, and assess generated claims against it with explicit source boundaries. It also gives a researcher clear conditions for challenging the result: change the rule base, the candidate-pool regime, the generator set, the verifier, or the evidence packet, and report the corresponding effects rather than assuming that a citation or fluent rationale is faithful by default.

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
