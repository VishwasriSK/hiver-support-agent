# Architecture & Engineering Decision Log — Hiver Support Agent

This document records the key architectural decisions, empirical trade-offs, and design choices made during the development of the Hiver Support Agent.

---

### Decision 1: Customer Support Intent Taxonomy (8 Categories)
- **Context:** Raw customer interactions span a wide range of inquiries, complaints, and requests. A coarse taxonomy loses utility, whereas a fine-grained taxonomy causes severe class overlap on short tweets.
- **Decision:** Selected 8 distinct intent categories based on exploratory word frequency and bi-gram analysis:
  1. `Delivery & Tracking`
  2. `Order Management`
  3. `Payment & Charges`
  4. `Returns & Refunds`
  5. `Account & Login`
  6. `Prime Membership & Benefits`
  7. `Digital Services & Devices`
  8. `Product Problems`
- **Rationale:** Aligns with standard customer support routing in e-commerce while providing clear escalation targets.

---

### Decision 2: Golden Dataset Creation & Evaluation Leakage Prevention
- **Context:** To ensure uncompromised evaluation, ground truth evaluation labels must be manually verified and strictly isolated from retrieval corpora.
- **Decision:** 
  1. Built a 200-example golden set (`golden_set/golden_200_final.csv`) with full manual inspection.
  2. Maintained a fixed 80/20 stratified split (160 reference/train, 40 test) with `random_state=42`.
  3. **Leakage Prevention:** Removed all 186 golden candidate conversations from the 150k historical retrieval matrix (`evaluation/retrieval_data.csv`). The retrieval index searches only among genuinely historical, unseen support pairs.

---

### Decision 3: Hybrid Grounded Agent (Intent Classification + Historical Retrieval)
- **Context:** Pure generative LLMs without grounding risk hallucinating fake policies, order tracking links, or delivery commitments. Pure classifiers can identify intent but cannot suggest historically validated responses.
- **Decision:** Implemented a two-stage hybrid pipeline:
  1. **Intent Stage:** Predicts customer support intent category and probability via TF-IDF + Logistic Regression.
  2. **Retrieval Stage:** Computes cosine similarity against 150,240 historical customer-support pairs to retrieve the top similar resolved interaction and its official brand reply.
  3. **Composite Confidence:** Overall Confidence = $0.4 \times \text{Intent Confidence} + 0.6 \times \text{Retrieval Similarity}$.

---

### Decision 4: Confidence-Based Escalation Policy (0.30 Threshold)
- **Context:** An arbitrary escalation threshold (e.g. 0.45) escalated nearly 100% of cases due to short query lengths and sparse TF-IDF representations.
- **Decision:** Empirically tuned the operational threshold on 5-fold cross-validation of the training set.
  - Threshold set to `0.30`.
  - At $\ge 0.30$, intent accuracy on suggested responses rises from 50.0% to 55.0% while achieving a 50.0% automated suggestion coverage.
  - For cases $< 0.30$, the system escalates with an explicit diagnosed root cause:
    - *Low intent confidence & low retrieval similarity*
    - *Low intent confidence (ambiguous customer query)*
    - *Low retrieval similarity (insufficient historical precedent)*

---

### Decision 5: Multi-Dimensional Evaluation (Intent, Retrieval, Response Quality, Human Agreement)
- **Context:** Intent classification accuracy alone does not reflect whether the agent's retrieved response is helpful or grounded.
- **Decision:** Separated evaluation into four distinct orthogonal tracks:
  1. **Intent Classification:** Compared against Trivial (25%) and Retrieval Baseline (55%).
  2. **Retrieval Relevance:** Measured Top-1 retrieval success (52.5%).
  3. **Response Quality (LLM-as-a-Judge):** Graded on 6 dimensions (Relevance: 2.50/5, Helpfulness: 2.75/5, Groundedness: 2.60/5, No Unsupported Claims: 4.00/5, Professional Tone: 4.00/5, Overall: 2.85/5).
  4. **Human Agreement Validation:** Validated LLM judge against human evaluation on 20 test responses, demonstrating 95.0% exact agreement, 100% within-1 point agreement, and a 0.8245 Pearson correlation.

---

### Decision 6: Intent-Conditioned Retrieval Masking vs. Global Search
- **Context:** Global TF-IDF retrieval across the entire 150k historical corpus allowed generic phrases (e.g., *"it's not letting me"*, *"what should I do"*) to dominate similarity rankings, causing unrelated cancellation and damaged-item inquiries to retrieve delivery tracking links.
- **Decision:** Restricted cosine similarity search dynamically to the candidate subset belonging to the predicted intent (`data[intent == predicted_intent]`). If the subset contains zero historical candidates, the agent handles the edge case gracefully by triggering safe human escalation.

---

### Decision 7: Balanced Class Weighting (`class_weight='balanced'`)
- **Context:** In the 160-example training set, `Delivery & Tracking` had 39 examples (24.4%) while `Order Management` had only 10 examples (6.25%). An unweighted Logistic Regression model learned a +1.30 log-odds intercept gap favoring delivery, causing 0% recall on three minority classes (`Order Management`, `Product Problems`, `Digital Services & Devices`).
- **Decision:** Added `class_weight='balanced'` to Logistic Regression. This penalizes misclassifications inversely proportional to class frequencies, eliminating the artificial prior bias and restoring non-zero recall across all 8 intent classes.

---

### Decision 8: FeatureUnion of Word-Level and Subword Character N-Grams (`char_wb`, 3–5)
- **Context:** Word-level tokenization with `min_df=2` yielded a vocabulary of only 950 tokens. Misspellings like `"sampoo"` and morphological variations like `"spilt"` were entirely Out-Of-Vocabulary (OOV), stripping essential signal from product complaints.
- **Decision:** Combined word unigrams/bigrams with word-boundary character n-grams (`analyzer='char_wb'`, range 3–5) via `sklearn.pipeline.FeatureUnion`. This provides partial robustness to spelling anomalies and morphological forms without introducing brittle external spell-check dictionaries.

---

### Decision 9: Local Ollama + Qwen 2.5 (3B) as an Autonomous, Triage-Aware Auditor
- **Context:** Commercial API evaluations introduce external dependencies and privacy risks, while naive LLM judge prompts unfairly penalize triage systems for escalating ambiguous or sensitive cases.
- **Decision:** Deployed an automated local LLM judge using Ollama (`qwen2.5:3b`). Designed a triage-aware operational prompt that evaluates `ESCALATE` as a valid, high-value routing action for low-confidence or sensitive queries, and scores professional tone independently from resolution completeness to prevent halo effects.

---

### Decision 10: Strict Refusal to Hardcode Samples, Overrides, or Scores
- **Context:** During error analysis on target failures (Samples 7, 8, 9, 13), it was tempting to add special-case keyword overrides, intent patches, or score defaults.
- **Decision:** Enforced a strict architectural constraint against hardcoding any sample IDs, customer names, intent overrides, or evaluation fallback scores. All improvements were required to stem strictly from generalized machine learning and retrieval refinements.

---

### Decision 11: Documenting Multi-Turn Context Fragmentation as an Intrinsic Constraint
- **Context:** Sample 13 revealed that single-turn retrieval on a top-level tweet ("Not sure I'll be renewing my Prime membership") retrieved a brand response addressing delivery delays ("Has your order still not arrived? Deliveries can be made until 21:00"). This occurred because the historical tweet pair belonged to a multi-turn thread with prior unobserved turns.
- **Decision:** Rather than artificially masking this failure or tampering with the dataset, documented multi-turn context fragmentation as an open architectural limitation in Twitter customer-support data, designating thread disentanglement and dialogue-act modeling as future work.

---

### Decision 12: Dual-Confidence Weighting Favoring Retrieval ($0.4 \times \text{Intent} + 0.6 \times \text{Retrieval}$)
- **Context:** Intent classification identifies the broad domain, but real-world customer satisfaction depends on whether the retrieved brand reply specifically addresses the query text.
- **Decision:** Weighted retrieval similarity at 0.6 and intent confidence at 0.4 in the composite scoring function. Even when the classifier is confident in the intent, if no historical support precedent has high similarity, the composite confidence drops below 0.30 and safely escalates to human agents.
