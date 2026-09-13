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

### Decision 4: Calibrated Escalation Policy
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
  3. **Response Quality (LLM-as-a-Judge):** Graded on 5 dimensions (Relevance: 4.05/5, Groundedness: 4.65/5, Professionalism: 4.70/5, Overall: 4.05/5).
  4. **Human Agreement Validation:** Validated LLM judge against human evaluation on 20 test responses, demonstrating 95.0% exact agreement, 100% within-1 point agreement, and a 0.8245 Pearson correlation.
