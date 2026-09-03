# Temporary plan: human validation and a stronger paper

## Bottom line

The current result is paper-shaped: providing the exact reranking trace increased measured
trace-supported and full-KB-supported claims by about 21 percentage points in paired comparisons.
The important limitation is that evidence reranking did not improve conventional relevance over
fused CLIP. The defensible paper claim is therefore narrow:

> Showing the exact stored reranking trace improves evidence alignment of explanations under this
> locked recommendation and explanation protocol.

It should not claim improved recommendation accuracy, universal factual correctness, or user
preference unless those are measured separately.

## Human validation: recommended design

### What humans should validate

Humans should independently judge:

- whether the automated extractor found the important checkable claims;
- whether the exact stored trace supports each claim;
- whether the record-specific full-KB packet supports each claim;
- whether a cited rule actually entails the claim; and
- whether a concrete item-property claim is unsupported by the allowed common item context.

Do not show annotators the Qwen/Phi verdict before they decide. The human labels must be an
independent reference standard, rather than a confirmation of the LLM output.

### A practical protocol

1. Write and freeze the annotation rules before viewing final human results. Define labels,
   handling of mixed claims, sample selection, adjudication, and analysis.
2. Draw a stratified random sample. A realistic starting final sample is 150 paired cases: 10
   cases for each of 5 categories and 3 generators. This produces 300 explanations. If the work
   is too large, first conduct a smaller pilot for timing and instruction refinement, then set the
   final sample before the confirmatory annotation begins.
3. Have two people label every explanation independently. A third person adjudicates
   disagreements using a written rule. Retain the original independent labels as well as the
   adjudicated result.
4. Blind the condition and randomise presentation order. Annotators may see the evidence packet
   where needed for an entailment task, but should not see labels such as “No-RAG”, “Rule-RAG”, or
   the generator name.
5. Report pre-adjudication agreement, then compare human and automated results with precision,
   recall, F1, and confusion matrices for extraction and each support label.
6. Repeat the No-RAG versus Rule-RAG paired comparison using the human labels. Use case-level
   paired confidence intervals and a valid paired null test. Report this human subset separately
   from the full automated corpus.
7. Preserve the protocol, sampling seed, de-identified labels, and adjudication rules where
   ethics approval permits.

This answers the key question: does the automated evaluator track independent human judgement?
If the human-labelled effect has the same direction and useful precision, the paper is much more
convincing.

## An extra general LLM judge

Yes, add one only as a **secondary triangulation measure**. It can assess things that the claim
verifier does not target directly: clarity, explanatory usefulness, specificity, whether the text
communicates a reason, and citation readability.

Use a model distinct from the generator/extractor/verifier where possible. Randomise first versus
second explanation, hide conditions and model names, freeze the prompt/model/version/decoding
settings, and require structured scores plus brief reasons.

Humans should validate the judge by making the same blinded ratings independently first. Compare
the judge to those human ratings using agreement, ordinal-score correlation, and confusion
matrices. Do not show a human the LLM’s rating and ask whether it is correct: that creates
anchoring. The human label is the reference; the LLM judge is scalable secondary evidence.

Do not use a second LLM simply to add favourable metrics. If it knows the condition, uses the same
framing as the verifier, or is never checked against humans, it adds little credibility.

## Are three University of Salford staff enough?

Three people can be enough for a careful **pilot or expert annotation study**. They do not have to
be lecturers. HPA staff, project staff, researchers, or postgraduate staff can participate if they
have relevant competence and their role is transparently described.

What matters is more important than title:

- relevant knowledge, such as LLM evaluation, recommender systems, annotation, fashion/design,
  or evidence/entailment assessment;
- independence from the desired result and, where possible, no direct supervisory or management
  relationship with the researcher;
- a common protocol and brief calibration/training round before final annotation;
- two independent annotations per record and a pre-specified third-person adjudication process;
- reporting the broad expertise, conflicts of interest, time/compensation, and agreement.

Three staff are reasonable for the expert checking proposed here. A sensible allocation is two
primary annotators per record and the third as adjudicator; rotate that role or have the third
person independently label a random subset as well. Three staff alone are not enough for a broad
claim about typical user preference or trust in fashion recommendations. That requires a larger,
more representative participant sample.

## Salford ethics and governance

Confirm the ethics route with the supervisor and relevant School ethics contact before recruiting
or collecting annotations. The University of Salford states that research involving human
participants conducted by staff and students is subject to ethics-panel scrutiny and that approval
should be obtained before data collection. Do not assume colleagues are exempt participants.

Official starting points:

- [Academic Ethics and Ethics Approval](https://www.salford.ac.uk/research/research-culture/research-integrity/academic-ethics-and-ethics-approval)
- [Research Integrity](https://www.salford.ac.uk/research/research-culture/research-integrity)
- [Ethics contacts and application routes](https://apply-ethics.salford.ac.uk/Personalisation/DisplayPage/8)

The application should cover voluntary informed consent, withdrawal, the participant-information
sheet, data minimisation, secure storage, retention/deletion, whether identifiers are collected,
and unequal power relationships. Avoid recruiting direct managers, supervisors, or supervisees
where possible; keep ratings confidential from colleagues and managers.

## Three ways to make the paper more competitive

These are alternatives, not all mandatory. Human validation plus one well-executed extension
would be a substantial improvement.

### 1. A second dataset or a second domain

**Simple meaning:** demonstrate that the result is not only a peculiarity of Polyvore and one
fashion rule base.

**Method:** choose a genuinely separate catalogue/dataset; define its categories, split,
candidate-pool procedure, and rule provenance before using its test set; build a separate scoped
KB; save the exact reranking trace; then repeat the same paired No-RAG versus trace-RAG study.
Report each dataset/domain separately. A second fashion dataset is the lowest-risk extension; a
different compatibility domain such as home decor or recipes is stronger generalisation but needs
an equally careful new rule base.

**Why it helps:** it changes the result from a single-dataset case study to a repeated pattern.

### 2. A relevance-preserving reranker

**Simple meaning:** evidence currently makes explanations more grounded but slightly reduces
retrieval relevance. Improve the ranking policy so evidence only changes a recommendation when it
does not cost too much baseline relevance.

**Methods to choose on validation data only:**

- a **margin gate**: preserve the fused-CLIP top choice unless an alternative is within a small
  CLIP-score margin and has clearly better evidence;
- **constrained reranking**: maximise evidence only within a fused-CLIP top-k or below a fixed
  relevance-loss threshold;
- **lexicographic ranking**: use evidence only as a deterministic tie or near-tie breaker; or
- **Pareto selection**: choose among candidates not dominated on both relevance and evidence.

Freeze the selected policy after validation, run the test set once, and report the trade-off curve:
relevance, top-1-change rate, trace coverage, and grounding. This makes the paper answer a useful
systems question: how much grounding benefit is possible for what relevance cost?

### 3. A human usefulness and trust study

**Simple meaning:** evidence support is technical; test whether explanations actually help people
understand or assess a recommendation.

**Method:** show paired explanations for the same locked recommendation with random order and
hidden conditions. Use short pre-registered questions such as which explanation better explains
the recommendation, which is clearer, and how confident the participant is that the explanation
is supported by displayed evidence. Include a small comprehension question about which stated
reason is supported by the rule, not only preference or trust. Analyse paired choices/ratings at
participant and case levels.

**Why it helps:** it connects technical grounding to an actual human outcome. It is a different
claim from expert annotation, so it needs its own recruitment and ethics plan.

## Suggested order of work

1. Agree the narrow human-validation question and ethics route with the supervisor.
2. Run a small non-confirmatory annotation pilot; revise instructions only before freezing the
   final protocol.
3. Run the blinded expert annotation study and compare humans with automated labels.
4. If supported, add one extension: relevance-preserving reranking is likely the most efficient;
   a second dataset is the strongest robustness test; a user study is the strongest practical-
   impact test.
5. Report any extra general LLM judge as secondary robustness analysis calibrated against the
   independent human labels.
