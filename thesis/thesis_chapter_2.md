# Chapter 2

# Literature Review

## 2.1 Introduction

This chapter positions the thesis at the intersection of fashion compatibility modelling, multimodal information retrieval, explainable recommendation, retrieval-augmented generation (RAG), and explanation-faithfulness evaluation. These areas address related but distinct questions. Fashion recommendation asks which item is compatible with a query or partial outfit. Multimodal retrieval asks how image and text signals should be represented and combined. Explainable recommendation asks how the basis of a recommendation can be communicated. RAG supplies external context to a generator. Faithfulness research asks whether an explanation accurately reflects the evidence or process it claims to explain.

The completed study does not attempt to solve every problem in these areas. It does not learn a new foundation model, provide personalised full-catalogue serving, or recover the complete internal reasoning of a neural encoder. Its narrower concern is whether a symbolic evidence component can participate in multimodal reranking, be retained as an exact decision trace, and subsequently improve the evidential behaviour of natural-language explanations for the same locked recommendation.

The review proceeds in six parts. Section 2.2 examines the development of fashion compatibility and outfit recommendation. Section 2.3 discusses multimodal representations and the distinction between latent compatibility and verbal evidence. Section 2.4 reviews explainable and knowledge-aware recommendation. Section 2.5 considers generative explanation, RAG, citations, and unsupported claims. Section 2.6 examines faithfulness definitions and evaluation methods. Section 2.7 compares representative studies and derives the research gap that leads directly to the methodology in Chapter 3.

## 2.2 Fashion compatibility and outfit recommendation

### 2.2.1 From visual similarity to compatibility

Fashion recommendation differs from conventional nearest-neighbour retrieval because similarity and compatibility are not identical. A black shoe and a nearly identical black shoe may be substitutes, while a shoe and a pair of trousers can be complementary despite belonging to different visual and semantic categories. A useful representation must therefore model relations between item types rather than assume that the closest item in a generic feature space is the best addition to an outfit.

McAuley et al. [1] provided an influential early formulation by learning visual relationships for styles and substitutes from large-scale product data. Their work showed that convolutional image features can encode notions of compatibility derived from observed relationships. The contribution established visual appearance as a valuable recommendation signal, but the learned distances remain latent and do not directly express why a particular candidate complements a particular query.

Han et al. [2] moved beyond isolated pairs by treating an outfit as an ordered sequence and learning compatibility with bidirectional LSTMs. The forward and backward structure captures dependencies across multiple outfit components and supports fill-in-the-blank evaluation. This work was important because compatibility depends on the surrounding outfit, not only a single pair. Nevertheless, sequence order is an imposed representation of an outfit, and the model’s learned state is not a readily inspectable natural-language decision record.

Vasileva et al. [3] explicitly distinguished similarity from compatibility through type-aware embeddings. Their model learns item-type-specific relations so that comparisons can be conditioned on categories such as tops, bottoms, or shoes. This matches the intuition that different features matter for different cross-category relationships. The associated Polyvore-Outfits dataset and fill-in-the-blank tasks became important resources for compatibility research. The thesis adopts the category-aware principle operationally by constructing same-target-category candidate pools, although it uses frozen pretrained encoders rather than training a type-aware fashion model.

### 2.2.2 Context, graphs, and outfit structure

Cucurull et al. [4] argued that compatibility should depend on item context and proposed graph neural representations conditioned on products known to be compatible. Their results on Polyvore, Fashion-Gen, and Amazon data demonstrated the benefit of relational context over isolated pairwise comparison. Graph-based methods can model higher-order outfit structure, but their explanatory value depends on whether graph paths or learned relationships are exposed and demonstrably used in prediction.

Tan et al. [5] learned similarity conditions without requiring explicit condition labels at test time. By representing distinct semantic subspaces as latent variables, the model improved generalisation across fashion datasets. This line of work illustrates a recurring trade-off: more flexible latent representations can improve predictive capability while making a human-readable account of an individual decision less direct.

Scene-based work further broadened the input boundary. Kang et al. [6] introduced “Complete the Look,” recommending complementary products from real-world scene images rather than isolated catalogue product shots. Scene images may convey pose, environment, and outfit context, but they also introduce clutter and properties that require additional visual inference. The present thesis does not claim this scene-input capability. It operates on catalogue item images within a controlled held-out outfit task and intentionally prevents image-derived attributes from entering the textual explanation evidence.

These studies demonstrate that fashion compatibility benefits from type, context, and multimodal structure. They also show why ranking success and explainability should be evaluated separately. A latent representation can retrieve relevant items effectively while offering no direct proposition that a language generator can cite. Conversely, an interpretable rule may provide a readable reason while failing to improve ranking. The system developed in this thesis therefore retains conventional ranking evaluation and adds a separate, inspectable symbolic evidence pathway.

### 2.2.3 Evaluation conventions and their limits

Fashion compatibility studies commonly use outfit-compatibility classification, fill-in-the-blank tasks, hit rate, mean reciprocal rank, or normalised discounted cumulative gain. Such metrics measure whether held-out positives appear above sampled negatives and whether relevant items occupy favourable ranks [7]. They are appropriate for comparing rankers under a fixed candidate protocol, but their values depend strongly on candidate-pool construction, negative sampling, category restrictions, and the definition of relevance.

Polyvore co-occurrence is a practical but incomplete relevance proxy. An observed outfit provides positive evidence that items were curated together, yet an unobserved candidate is not necessarily incompatible. Fashion permits many plausible alternatives, and user-created outfits encode time, culture, availability, and individual taste. Consequently, the thesis describes its experiment as sampled controlled-pool ranking and does not equate unobserved items with universally poor fashion choices.

Data leakage is another concern. Item identifiers alone may not detect separately listed products containing identical image bytes. If exact images cross development, validation, and test partitions, retrieval results can exaggerate generalisation. The completed implementation therefore groups exact-image hashes at outfit-component level before freezing split assignments. This control extends the outfit-disjoint logic used in prior work and supports a more defensible evaluation, but it does not transform an offline benchmark into an online user study.

## 2.3 Multimodal representation and evidence boundaries

### 2.3.1 Image and text as complementary signals

Catalogue text can expose explicit product terms, while images capture appearance information missing from descriptions. CLIP learns aligned image and text representations from large-scale image-text training and supports zero-shot cross-modal transfer [8]. Its shared embedding space makes it a practical frozen baseline for multimodal catalogue retrieval. In the present system, normalised CLIP image and text vectors are fused before cosine ranking, while MiniLM provides a separate lightweight sentence-embedding pathway [9,10].

The use of frozen general-purpose encoders has advantages and limitations. It permits a reproducible study without end-to-end fashion-model training and allows the research contribution to focus on evidence traces and explanation. However, CLIP is not specialised for outfit compatibility, fit, or subtle garment attributes. The thesis therefore treats its multimodal retrieval pathway as a reproducible baseline rather than assuming that it is the strongest possible fashion representation.

The completed results support a bounded multimodal claim. Fused CLIP is the strongest tested pathway on the principal top-five and top-ten retrieval measures, while the evidence reranker produces a modest relevance trade-off in exchange for an inspectable symbolic trace. The literature therefore motivates multimodal representation, whereas the experiment reserves its main causal claim for the subsequent evidence-grounded explanation contrast.

### 2.3.2 Representation is not textual evidence

A critical distinction for this thesis is that an image embedding is a ranking signal, not a set of verified textual attributes. A similarity score may be affected by colour, silhouette, pattern, composition, or correlations that are not individually recoverable. Generating an explanation that states “the burgundy trousers match the leather shoes” would require evidence that the items are burgundy and leather. The fact that CLIP processed their images does not establish that these particular propositions were explicitly represented or causally decisive.

The system therefore draws a strict evidence boundary. Images enter the recommendation pathway but are never captioned or classified into explanation facts. Common context A contains the user request and the frozen item identities, categories, and catalogue text. Exact trace B contains the five expert rules and scoring information used by the evidence component. A future image-evidence block could contain separately validated visual attributes, but such a component would require its own benchmark and uncertainty controls.

This boundary avoids an important category error in multimodal explanation. Visual grounding normally asks whether language corresponds to observable image content. Decision faithfulness asks whether language reflects information used in a prediction. These properties can overlap but are not interchangeable. The thesis evaluates faithfulness to B and support from A+B while acknowledging that B is only the symbolic portion of a hybrid score.

## 2.4 Explainable recommender systems

### 2.4.1 Purposes and forms of recommendation explanation

Explanations in recommender systems serve several possible goals: transparency, scrutability, trust calibration, persuasion, satisfaction, and support for better decisions [11,12]. These objectives can conflict. A persuasive explanation may increase acceptance without accurately exposing the recommendation process; a technically faithful trace may be difficult for a user to understand. Evaluation must therefore match the stated purpose.

Zhang and Chen [11] survey explainable recommendation approaches including neighbourhood, matrix-factorisation, topic, graph, deep-learning, and natural-language methods. Feature-based explanations can identify influential attributes; review-based methods can reuse user-authored evidence; path-based systems can expose relations between users and products. Natural-language generation can improve accessibility, but it introduces another model whose output may diverge from the recommendation evidence.

Knijnenburg et al. [12] demonstrate that system effectiveness, explanation properties, user perceptions, and experience are distinct levels of evaluation. Their framework cautions against treating accuracy as a proxy for explanation usefulness or treating subjective appeal as proof of system fidelity. The completed thesis does not include a user study, so it confines its claims to system-level operational measures and automated judgments.

### 2.4.2 Post-hoc and mechanism-linked explanations

Post-hoc explanations are produced after a prediction and may approximate its basis through surrogate features, attention, retrieved examples, or generated rationales. They can be valuable, but their fidelity must be tested rather than assumed. Attention weights, for example, have generated substantial debate over when they constitute explanations and what kind of causal claim they support [13].

Mechanism-linked approaches instead incorporate interpretable components into prediction. Policy-Guided Path Reasoning uses knowledge-graph paths to connect users and recommended items [14]. LOGER combines knowledge-graph embeddings with neural logic rules and uses learned rule importance to guide path reasoning for explainable recommendation [15]. These systems demonstrate that paths or rules can be part of recommendation rather than decoration added afterwards.

The present work shares the goal of mechanism linkage but differs in architecture and question. It does not learn personalised logical rules over a user-item knowledge graph. It retrieves from a curated fashion-rule base, calculates a candidate evidence score from five weighted rule contributions, mixes that score with fused CLIP, and stores the exact contributing trace. It then experimentally tests what happens when a separate language generator can or cannot see that trace while recommendation identity remains locked.

### 2.4.3 Limits of symbolic evidence

An interpretable component is not automatically correct. A generic rule may be sensible in the abstract but only partially applicable to a specific query-candidate pair. Semantic retrieval may select a rule whose antecedent is not established. Reliability categories supplied with a rule base are not empirical probabilities. A faithful explanation can therefore accurately report questionable evidence.

This distinction separates three evaluation targets. Recommendation metrics examine whether the hybrid system ranks held-out outfit items. Evidence-participation diagnostics examine whether the rule component materially changes selection and whether diverse rules are used. Explanation metrics examine whether generated language matches the stored trace and avoids unsupported instance-level claims. No single target validates the other two.

The thesis consequently avoids describing its architecture as perfectly faithful “by construction.” B is exact as a data record of the symbolic scoring calculation, but generated prose can omit, distort, or overgeneralise it. Trace support, full-KB support, and citation-entailment evaluation exist precisely because exposure to the trace does not guarantee correct use.

## 2.5 Retrieval-augmented generation and cited explanations

### 2.5.1 Retrieval as external context

RAG combines parametric generation with retrieved external information [16]. It has become a common strategy for improving factuality and updating knowledge without retraining the generator. In recommendation, retrieved reviews, catalogue fields, knowledge-graph facts, or domain guidance can support conversational answers and item justifications.

However, “retrieved,” “visible,” and “used” describe different relationships. A document can be retrieved but excluded from a prompt; it can be displayed but ignored by the generator; it can influence wording without having influenced item selection. Standard RAG commonly grounds the answer in visible context, but it does not necessarily connect that context to the upstream recommendation decision.

This thesis uses the term Rule-RAG for the explanation condition because rules are retrieved and displayed to the generator. Its stronger architectural property is that the displayed five-rule trace is identical to the trace used in the evidence score for the locked item. Even so, the language generator is prompted rather than token-constrained. It remains capable of adding unsupported content, and the empirical citation results confirm that rule identifiers are not always attached correctly.

### 2.5.2 Hallucination, unsupportedness, and contradiction

Hallucination terminology varies across language-generation research. Ji et al. [17] review factual inconsistency and distinguish errors relative to sources from errors relative to world knowledge. For this study, the supplied evidence boundary is more observable than world truth. A claim is unsupported when A and B do not entail it, contradicted when the supplied evidence entails its opposite, and not verifiable when the evidence is insufficient for the applicable schema.

This vocabulary matters in fashion. If a model calls a bag leather when neither item text nor trace establishes material, the claim may happen to be true. The experiment can show that it is unverified by supplied evidence, not that it is factually false. Conversely, contradiction is rare because sparse catalogue context seldom states the opposite of a generated style assertion. Collapsing unsupported and contradicted categories would inflate the apparent detection of falsehood.

The final study's common-reference item-fact metric focuses conservatively on concrete claims about actual query or recommended items. Subjective relational claims and ambiguous statements are excluded from its denominator. This sacrifices coverage to make the estimand clearer and explains why the final complete-pair UIFR analysis has a much smaller eligible sample than the primary support outcomes.

### 2.5.3 Citations as claim-source relations

Cited generation makes source use inspectable, but citation presence alone is inadequate. Gao et al. [18] evaluate systems that generate text with citations and distinguish correctness from completeness. Later fine-grained work shows that automatic citation-faithfulness metrics vary in their ability to distinguish full, partial, and absent support [19]. These findings motivate claim-level treatment rather than counting bracketed identifiers.

The thesis adopts citation precision and coverage. Precision asks whether a cited rule entails the associated claim. Coverage asks whether claims requiring rule support receive at least one valid citation. A generator can have high citation frequency but low precision if identifiers are decorative or attached to over-broad claims. It can have high precision but low coverage if it cites a small supported subset while leaving many rule-dependent claims uncited.

The exact rule trace enables deterministic identifier validation and source-aware verification. Yet citation entailment is still assessed automatically, and the cross-model verifier frequently returned null or structurally inconsistent citation relations even when identifiers were present. The final results therefore treat strict citation coverage as weak and evaluator-dependent, illustrating that marker presence is not a formal guarantee of claim–source support.

## 2.6 Faithfulness and evaluation

### 2.6.1 Plausibility, faithfulness, and the object of explanation

Jacovi and Goldberg [20] argue that faithfulness must be defined relative to a model and an explanation target. An explanation should correspond to the process it purports to describe rather than merely satisfy human expectations. Wiegreffe and Pinter [13] likewise show that debates about explanation require explicit claims and tests. Lyu et al. [21] organise faithful-explanation research into similarity-based, model-internal, gradient, counterfactual, and self-explaining approaches, illustrating that no single measure applies universally.

For a hybrid recommender, the object of explanation must be stated precisely. The CLIP component produces a latent compatibility score; the rule component produces an auditable score from five contributions. B is a complete record of the latter but not the former. “Decision-trace faithfulness” in this thesis therefore means faithfulness to the symbolic evidence trace used within the decision, not full causal explanation of the hybrid model.

The No-RAG condition creates a useful comparison. A pretrained generator may mention concepts similar to hidden B because fashion rules are common cultural knowledge. Such agreement is evidence of post-hoc alignment, not grounding, because the generator did not receive B. This prevents a high baseline match rate from being misrepresented as evidence use.

### 2.6.2 Functional and human-grounded evaluation

Explanation evaluation can be functionally grounded, application grounded, or human grounded. Functional measures test formal or computational properties without users. Human-grounded experiments use simplified tasks with people, while application-grounded work evaluates explanations with intended users or domain experts. Each supports different claims.

The completed experiment is functionally grounded and automated. Qwen 3.5 extracts atomic claims and Phi-4 verifies each claim against the trace, full-KB packet, common reference, and observed citations. Deterministic post-processing derives the final support rates and density from saved records. This permits evaluation of thousands of claims under reproducible schemas but introduces evaluator dependence.

Automated RAG-evaluation frameworks such as ARES show how model judges can assess context relevance, answer faithfulness, and answer relevance at scale [22]. Such frameworks often use human-labelled examples for calibration or prediction-powered inference. The present thesis does not include independent human calibration, so its confidence intervals capture case-sampling variability rather than semantic evaluator uncertainty. The limitation constrains interpretation but does not erase the value of a controlled system comparison.

### 2.6.3 Atomic claims and aggregation

Whole-response labels can hide mixed support. An explanation may contain one trace-supported styling relation, two unsupported product attributes, and a correct item identity. Atomic claim extraction enables these components to be assessed separately. It also introduces boundary decisions: conjunctions may be split inconsistently, pronouns may obscure the target item, and conditional language may be converted into overly definite propositions.

Both micro and macro aggregation are therefore useful. Claim-micro rates weight explanations with more extracted claims more heavily. Explanation-macro rates first calculate within-explanation proportions and then average units. Paired macro differences compare the same case-generator combination across conditions, while clustered bootstrap intervals recognise that several generator outputs derive from the same recommendation case.

Eligibility creates additional denominator differences. UIFR cannot be calculated for an explanation with no eligible concrete item-fact claims. Marginal condition rates therefore use different eligible explanation populations, while paired estimates use only the intersection in which both explanations contain an eligible claim. Reporting these denominators prevents a visually simple percentage from concealing a change in what the generator chose to claim.

### 2.6.4 Length and evaluator dependence

Explanation length is a major design issue. Longer outputs create more opportunities to state unsupported attributes, while concise evidence-focused prompts may improve both density and readability. Normalising unsupported claims per 100 words reduces but does not remove this difference because evidence access and brevity can affect which claims are selected, not only how many words are produced.

In the corrective experiment, both conditions received the same at-most-75-word instruction. No-RAG averaged 52.84 words and Rule-RAG 60.55; the remaining gap arose mainly from generator compliance, including 243 legacy Rule-RAG cap violations and two No-RAG violations. The predeclared 30-pair sensitivity averaged 54.17 versus 54.37 words with a 0.33-word mean absolute paired gap. Length is therefore controlled more directly than in the first draft, although evidence/citation instructions still change rhetorical content and actual length.

Evaluator independence is related but distinct. Using the same model family for extraction and verification can create correlated errors. The final run therefore uses Qwen 3.5 for atomic-claim extraction and Phi-4 for verification. This is stronger role separation, though neither model is human-calibrated. Cross-model assessment reduces one source of correlated error but does not substitute for annotation or establish semantic correctness.

## 2.7 Comparative synthesis and research gap

### 2.7.1 Comparison of representative research

Table 2.1 compares the main strands that lead to the thesis. The “remaining gap” column identifies a limitation relative to this study’s question, not a defect in the cited work. For example, a fashion-compatibility paper need not generate explanations to make a valid contribution; it simply leaves explanation faithfulness unresolved.

| Study | Research focus | Method and evidence | Main contribution | Remaining gap leading to this thesis |
|---|---|---|---|---|
| McAuley et al. [1] | Visual styles, substitutes, and compatibility | Visual features and learned product relations | Established large-scale visual compatibility modelling | Learned distance does not provide an exact natural-language decision trace |
| Han et al. [2] | Whole-outfit compatibility | Bidirectional sequence model on Polyvore outfits | Modelled dependencies across multiple garments | Sequential hidden state is not directly auditable as explanation evidence |
| Vasileva et al. [3] | Type-aware similarity and compatibility | Category-conditioned visual embeddings and Polyvore-Outfits | Distinguished within-type similarity from cross-type compatibility | Strong representation learning remains separate from claim-level explanation support |
| Chia et al. [23] | Transferable fashion product representation | Fashion-domain contrastive image–text adaptation of CLIP | Demonstrated gains from domain-specific fashion representation learning | Does not connect representation scores to an exact rule trace or assess generated decision explanations |
| Cucurull et al. [4] | Context-aware compatibility | Graph neural network over item context | Demonstrated value of higher-order relational context | Graph representation alone does not establish that generated prose follows the decision mechanism |
| Kang et al. [6] | Scene-based complementary recommendation | Real-world scene images, CNN compatibility, and attention | Extended recommendation beyond isolated catalogue queries | Visual context requires separate verified textualisation before it can support factual explanations |
| Zhang and Chen [11] | Explainable recommendation | Survey of feature-, review-, graph-, and generation-based methods | Organised goals, methods, and evaluation of explainable recommenders | Highlights heterogeneous explanation objectives; no single operational trace test follows automatically |
| Xian et al. [14] | Explainable knowledge-graph recommendation | Policy-guided multi-hop path reasoning | Connected recommendation to interpretable user-item paths | Focuses on personalised KG paths rather than paired LLM explanations of a frozen hybrid decision |
| Zhu et al. [15] | Faithful logic-based recommendation | Neural logic rules, KG embeddings, and path reasoning | Made rule importance part of recommendation and explanation | Does not test unsupported product claims or citation integrity in free natural-language generation |
| Lewis et al. [16] | Retrieval-augmented generation | Retrieved documents combined with parametric generation | Established a general architecture for retrieved-context generation | Retrieved context may be visible without being part of the upstream recommendation decision |
| Gao et al. [18] | Citation-supported generation | Retrieval, cited long-form generation, and citation metrics | Separated citation correctness and completeness | Addresses document-supported generation rather than recommendation-specific decision traces |
| Jacovi and Goldberg [20] | Definition of explanation faithfulness | Conceptual framework for faithfulness and plausibility | Requires explicit model, explanation target, and criteria | Needs a domain-specific operationalisation for hybrid recommenders |
| Lyu et al. [21] | Faithful NLP explanation | Survey of more than 100 explanation approaches | Shows diversity of faithfulness definitions and tests | Does not itself provide metrics for unsupported fashion-item attributes or rule citations |
| Saad-Falcon et al. [22] | Automated RAG evaluation | Synthetic training, lightweight judges, and calibrated estimation | Demonstrates scalable component-level RAG assessment | Automated judges remain estimator-dependent and do not replace explicit decision provenance |

### 2.7.2 Research gaps

The comparison reveals four connected gaps. The first is a **decision-provenance gap**. Fashion rankers increasingly exploit visual, textual, categorical, and relational representations, but their natural-language explanations are often not connected to an exact artifact that participated in selecting the item. A readable account can therefore become a plausible reconstruction of latent compatibility.

The second is an **evidence-use gap**. RAG supplies external information to a generator, but information retrieved for explanation may differ from information used during ranking. Even when the same source is visible, generation remains capable of ignoring, blending, or overgeneralising it. The relevant architectural question is not merely whether rules can be retrieved, but whether the displayed rules are demonstrably the rules used by an evidence component before explanation.

The third is a **claim-specific evaluation gap**. General quality, fluency, and hallucination scores do not isolate agreement with a stored decision trace, unsupported concrete item attributes, or claim-citation entailment. Recommendation explanations require measures that respect the difference between generic styling advice and assertions about actual catalogue items.

The fourth is an **objective-separation gap**. Explainability mechanisms are sometimes justified through accuracy gains, while faithful explanations are sometimes treated as evidence that a recommendation is better. These are different propositions. A rule component can change ranking without improving relevance, and a generator can explain that component faithfully even when the component is incomplete. A defensible evaluation should report recommendation effectiveness, evidence participation, and explanation behaviour separately.

### 2.7.3 How the thesis addresses the gap

Chapter 3 responds with a staged architecture and evaluation design. First, it constructs deterministic outfit-disjoint and exact-image-leakage-resolved splits from a pinned Polyvore release. Second, it establishes MiniLM, CLIP image, CLIP text, and fused CLIP ranking pathways under controlled same-category candidate pools. Third, it retrieves five rules for every query-candidate pair, calculates an evidence score, combines that score with fused CLIP, and preserves the exact scoring trace.

Fourth, the selected recommendation is locked before language generation. The No-RAG and Rule-RAG conditions explain the same item for the same request and generator. Common context A is identical; Rule-RAG additionally receives trace B. Fifth, saved explanations are decomposed into claims and evaluated with source-aware automated schemas. Trace support distinguishes post-hoc agreement from grounding in visible B; full-KB support captures the broader rule packet; common-reference item-fact support remains a restricted secondary outcome; and citation entailment measures a claim--source relation rather than identifier presence. Generator and category analyses provide complementary robustness views.

This design does not make unsupported generation impossible. Instead, it makes the relationship between a symbolic decision component and generated language observable and measurable. It also retains negative evidence: if reranking does not improve recommendation accuracy, or if citations remain invalid, those outcomes constrain rather than invalidate the contribution.

## 2.8 Chapter summary

Fashion recommendation research has progressed from pairwise visual relationships to sequence, type-aware, graph, scene, and multimodal representations. Explainable recommendation has progressed from readable features and reviews to knowledge-graph paths and logic-guided reasoning. RAG and citation research provide mechanisms for supplying and auditing external context, while faithfulness research warns that plausible language and visible evidence do not prove dependence on a decision process.

The unresolved intersection concerns provenance: whether an inspectable evidence trace can participate in recommendation, be preserved before generation, and support a controlled comparison of explanations for the same selected item. The thesis addresses that intersection through a hybrid multimodal and rule-based reranker, paired trace visibility, and claim-level automated evaluation. The next chapter formalises the data, scoring trace, experimental freezing, metrics, and statistical procedures used to test the resulting research questions.

## References

[1] McAuley, J., Targett, C., Shi, Q. and van den Hengel, A. (2015) ‘Image-based recommendations on styles and substitutes’, *Proceedings of SIGIR 2015*, pp. 43–52. https://doi.org/10.1145/2766462.2767755.

[2] Han, X., Wu, Z., Jiang, Y.-G. and Davis, L.S. (2017) ‘Learning fashion compatibility with bidirectional LSTMs’, *Proceedings of ACM Multimedia 2017*, pp. 1078–1086. https://doi.org/10.1145/3123266.3123394.

[3] Vasileva, M.I., Plummer, B.A., Dusad, K., Rajpal, S., Kumar, R. and Forsyth, D. (2018) ‘Learning type-aware embeddings for fashion compatibility’, *Proceedings of ECCV 2018*, pp. 390–405.

[4] Cucurull, G., Taslakian, P. and Vazquez, D. (2019) ‘Context-aware visual compatibility prediction’, *Proceedings of CVPR 2019*, pp. 12617–12626.

[5] Tan, R., Vasileva, M.I., Saenko, K. and Plummer, B.A. (2019) ‘Learning similarity conditions without explicit supervision’, *Proceedings of ICCV 2019*, pp. 10373–10382.

[6] Kang, W.-C., Kim, E., Leskovec, J., Rosenberg, C. and McAuley, J. (2019) ‘Complete the look: Scene-based complementary product recommendation’, *Proceedings of CVPR 2019*, pp. 10532–10541.

[7] Järvelin, K. and Kekäläinen, J. (2002) ‘Cumulated gain-based evaluation of IR techniques’, *ACM Transactions on Information Systems*, 20(4), pp. 422–446. https://doi.org/10.1145/582415.582418.

[8] Radford, A. et al. (2021) ‘Learning transferable visual models from natural language supervision’, *Proceedings of ICML 2021*, 139, pp. 8748–8763.

[9] Reimers, N. and Gurevych, I. (2019) ‘Sentence-BERT: Sentence embeddings using Siamese BERT-networks’, *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992. https://doi.org/10.18653/v1/D19-1410.

[10] Wang, W. et al. (2020) ‘MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers’, *Advances in Neural Information Processing Systems*, 33, pp. 5776–5788.

[11] Zhang, Y. and Chen, X. (2020) ‘Explainable recommendation: A survey and new perspectives’, *Foundations and Trends in Information Retrieval*, 14(1), pp. 1–101. https://doi.org/10.1561/1500000066.

[12] Knijnenburg, B.P., Willemsen, M.C., Gantner, Z., Soncu, H. and Newell, C. (2012) ‘Explaining the user experience of recommender systems’, *User Modeling and User-Adapted Interaction*, 22, pp. 441–504. https://doi.org/10.1007/s11257-011-9118-4.

[13] Wiegreffe, S. and Pinter, Y. (2019) ‘Attention is not not explanation’, *Proceedings of EMNLP-IJCNLP 2019*, pp. 11–20. https://doi.org/10.18653/v1/D19-1002.

[14] Xian, Y., Fu, Z., Muthukrishnan, S., de Melo, G. and Zhang, Y. (2019) ‘Reinforcement knowledge graph reasoning for explainable recommendation’, *Proceedings of SIGIR 2019*, pp. 285–294. https://doi.org/10.1145/3331184.3331203.

[15] Zhu, Y., Xian, Y., Fu, Z., de Melo, G. and Zhang, Y. (2021) ‘Faithfully explainable recommendation via neural logic reasoning’, *Proceedings of NAACL 2021*, pp. 3083–3090. https://doi.org/10.18653/v1/2021.naacl-main.245.

[16] Lewis, P. et al. (2020) ‘Retrieval-augmented generation for knowledge-intensive NLP tasks’, *Advances in Neural Information Processing Systems*, 33, pp. 9459–9474.

[17] Ji, Z. et al. (2023) ‘Survey of hallucination in natural language generation’, *ACM Computing Surveys*, 55(12), Article 248. https://doi.org/10.1145/3571730.

[18] Gao, T. et al. (2023) ‘Enabling large language models to generate text with citations’, *Proceedings of EMNLP 2023*, pp. 6465–6488. https://doi.org/10.18653/v1/2023.emnlp-main.398.

[19] Zhang, W. et al. (2024) ‘Towards fine-grained citation evaluation in generated text: A comparative analysis of faithfulness metrics’, *Proceedings of INLG 2024*, pp. 427–439. https://doi.org/10.18653/v1/2024.inlg-main.35.

[20] Jacovi, A. and Goldberg, Y. (2020) ‘Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?’, *Proceedings of ACL 2020*, pp. 4198–4205. https://doi.org/10.18653/v1/2020.acl-main.386.

[21] Lyu, Q., Apidianaki, M. and Callison-Burch, C. (2024) ‘Towards faithful model explanation in NLP: A survey’, *Computational Linguistics*, 50(2), pp. 657–723. https://doi.org/10.1162/coli_a_00511.

[22] Saad-Falcon, J., Khattab, O., Potts, C. and Zaharia, M. (2024) ‘ARES: An automated evaluation framework for retrieval-augmented generation systems’, *Proceedings of NAACL 2024*, pp. 338–354. https://doi.org/10.18653/v1/2024.naacl-long.20.

[23] Chia, P.J. et al. (2022) ‘Contrastive language and vision learning of general fashion concepts’, *Scientific Reports*, 12, Article 18958. https://doi.org/10.1038/s41598-022-23052-9.
