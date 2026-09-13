# Hiver Support Agent

An intelligent, retrieval-grounded customer support agent designed to assist support teams by accurately classifying customer inquiries, retrieving historically verified resolutions from past brand interactions, scoring confidence, and safely routing uncertain cases through calibrated escalation.

Built using the Amazon Customer Support Twitter/X dataset (~2.8M interactions) with a clean 8-intent support taxonomy, strict evaluation leakage prevention, reproducible baselines, LLM-as-a-judge response evaluation, and empirical human agreement validation.

---

## 1. Problem Statement

Customer support teams at high-growth organizations receive massive volumes of repetitive incoming customer messages across channels (email, social media, ticketing platforms). Support agents face several operational challenges:
- High latency identifying customer intent and routing tickets.
- Repetitive drafting of routine replies for common issues (shipping delays, returns, credentials).
- Risk of hallucinations or inaccurate commitments when relying purely on ungrounded generative AI.

The **Hiver Support Agent** addresses this by providing a reliable assistant that:
1. Classifies incoming customer text into standardized support intents.
2. Retrieves historically validated, real brand responses from past resolved interactions.
3. Computes a grounded confidence score combining intent and retrieval similarity.
4. Determines whether to safely **suggest a response** or **escalate** to human agents with a diagnosed reason.

---

## 2. End-to-End Pipeline & Architecture

```
                       Customer Message
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Intent Classifier            TF-IDF Retrieval
    (TF-IDF + Logistic Reg)       (150k Historical Index)
               │                             │
       Intent & Confidence         Top-1 Match & Similarity
               │                             │
               └──────────────┬──────────────┘
                              ▼
                     Composite Confidence
            (0.4 × Intent Conf + 0.6 × Retrieval Sim)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Overall Conf ≥ 0.30           Overall Conf < 0.30
               │                             │
      SUGGESTED RESPONSE                  ESCALATE
    (Brand reply grounded)         (Diagnosed root cause +
                                    Safe human routing)
```

### Key Workflow Stages:
1. **Dataset Ingestion & Filtering:** Extract customer-to-brand conversation pairs from AmazonHelp.
2. **Data Cleaning:** Missing value removal, non-English filtering, duplicate elimination, short-message pruning.
3. **Intent Taxonomy:** Definition of 8 core customer support categories.
4. **Golden Set Construction:** 200 high-quality, manually verified evaluation examples with strict leakage prevention.
5. **Baselines:** Trivial majority baseline, TF-IDF + Logistic Regression, and kNN Retrieval baseline.
6. **Retrieval Index:** TF-IDF sparse vector index over 150,240 historical conversation pairs.
7. **Support Agent:** Multi-component decision engine with confidence calibration and policy escalation.
8. **Multi-Track Evaluation:** Intent classification metrics, Top-1 retrieval success, LLM-as-a-judge rubric, human agreement, and failure analysis.

---

## 3. Support Intent Taxonomy

The system classifies inquiries into 8 operational support intents:

| # | Intent Category | Typical Topics Covered |
|---|---|---|
| 1 | **Delivery & Tracking** | Package tracking, courier delays, missing items, delivery estimates |
| 2 | **Order Management** | Order cancellation, modifications, invoice requests, order details |
| 3 | **Payment & Charges** | Unexpected deductions, payment gateway errors, card/EMI inquiries, account credit |
| 4 | **Returns & Refunds** | Refund status, return pickups, exchange requests, drop-off questions |
| 5 | **Account & Login** | Password reset, OTP verification, account suspension/holds, unauthorized access |
| 6 | **Prime Membership & Benefits** | Prime subscription billing, benefits, Prime Video/Music access, renewal cancellation |
| 7 | **Digital Services & Devices** | Amazon Echo, Alexa, Kindle, Fire TV app issues, firmware/streaming glitches |
| 8 | **Product Problems** | Defective items, broken/damaged packaging, incorrect product received |

---

## 4. Dataset Statistics

| Pipeline Stage | Record Count | Description |
|---|---|---|
| **Raw Dataset** | 2,811,774 | Complete customer support Twitter/X conversations |
| **AmazonHelp Filtered** | 169,840 | Tweets authored by or replying to `@AmazonHelp` |
| **Conversation Pairs** | 168,823 | Customer query $\rightarrow$ official brand response pairs |
| **Cleaned Pairs** | 150,426 | Deduplicated, English-filtered, non-empty pairs |
| **Leakage Exclusions** | 186 | Golden candidates removed from retrieval corpus to ensure zero leakage |
| **Final Retrieval Corpus** | **150,240** | Active historical conversation index (`evaluation/retrieval_data.csv`) |
| **Golden Ground Truth Set** | **200** | Manually validated evaluation dataset (`golden_set/golden_200_final.csv`) |
| **Evaluation Split** | 160 / 40 | 80% reference/train, 20% held-out test split (stratified by intent) |

---

## 5. Evaluation Methodology & Leakage Prevention

To ensure rigorous and uncompromised evaluation:
- **Strict Leakage Prevention:** All 186 golden candidate examples were permanently filtered out of the 150,240-row retrieval corpus (`evaluation/retrieval_data.csv`). The retrieval system never searches across test examples.
- **Stratified Split:** Golden set evaluation uses an 80/20 train/test split (160 train, 40 test) stratified across all 8 classes with a fixed random seed (`random_state=42`).
- **No Artificially Inflated Metrics:** Classification accuracy and macro-averaged metrics are reported strictly as measured without arbitrary inflation.
- **Empirical Escalation Threshold:** The escalation threshold was calibrated via 5-fold cross-validation on the 160 training examples rather than picked arbitrarily.

---

## 6. Experimental Results & Baselines

### 6.1 System Comparison (Golden Test Split, N=40)

All metrics are measured on the unseen 40-example test split:

| System / Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Top-1 Retrieval Success | Escalation Rate | Key Characteristics |
|---|---|---|---|---|---|---|---|
| **Trivial Baseline** | 25.0% | 0.0312 | 0.1250 | 0.0500 | N/A | N/A | Always predicts majority class (`Delivery & Tracking`) |
| **TF-IDF + Logistic Regression** | 50.0% | 0.4293 | 0.3813 | 0.3583 | N/A | N/A | Simple ML classifier trained on 160 golden examples |
| **Retrieval Baseline (kNN)** | 55.0% | 0.6078 | 0.5292 | 0.5439 | N/A | N/A | Cosine similarity against 160 golden reference examples |
| **Final Support Agent** | **50.0%** | **0.4293** | **0.3813** | **0.3583** | **52.5%** | **50.0%** | **End-to-End Hybrid:** Intent + 150k retrieval grounding + calibrated escalation |

> **Key Takeaway:** While raw intent classification accuracy is 50.0%, evaluating the agent end-to-end reveals that when confidence exceeds the operational threshold ($\ge 0.30$), **intent accuracy on suggested responses increases to 55.0%**, while unsafe, low-confidence queries (50.0%) are appropriately escalated.

### 6.2 Escalation Policy Breakdown

Using the calibrated threshold of $0.30$:
- **Suggested Responses:** 20 / 40 (50.0%)
- **Escalated to Human:** 20 / 40 (50.0%)

**Escalation Reasons Diagnosed:**
- Low retrieval similarity (insufficient historical precedent): **8 cases**
- Low intent confidence & low retrieval similarity: **6 cases**
- Low intent confidence (ambiguous customer query): **6 cases**

---

## 7. Response Quality Evaluation (LLM-as-a-Judge)

Response quality was evaluated across 20 representative test interactions using an explicit 5-point customer support rubric:

| Evaluation Dimension | Description | Mean Score (1–5 Scale) |
|---|---|---|
| **Relevance** | Addresses the customer's specific inquiry and context | **4.05 / 5.0** |
| **Helpfulness** | Provides clear next steps, resolution links, or escalation | **3.95 / 5.0** |
| **Groundedness** | Faithfully grounded in verified brand replies / safe policy | **4.65 / 5.0** |
| **No Unsupported Claims** | Zero hallucinations of fake order numbers, dates, or promises | **4.95 / 5.0** |
| **Professional Tone** | Empathetic, polite, and aligned with customer support standards | **4.70 / 5.0** |
| **Overall Quality** | Holistic acceptability for customer-facing deployment | **4.05 / 5.0** |

*Results saved to: `evaluation/llm_judge_results.csv`*

---

## 8. Human vs. LLM Judge Agreement

To validate the reliability of the LLM-as-a-Judge evaluation, all 20 responses were independently evaluated under the identical rubric by a human reviewer:

| Agreement Metric | Measured Value | Interpretation |
|---|---|---|
| **Exact Score Agreement** | **95.0%** (19 / 20) | High consistency between human and automated rubric scoring |
| **Within-1 Point Agreement** | **100.0%** (20 / 20) | Zero extreme disagreements or rating discrepancies |
| **Mean Absolute Difference (MAD)** | **0.05 points** | Minor score variance across nuanced boundary cases |
| **Pearson Correlation ($r$)** | **0.8245** | Strong positive linear correlation between human and LLM ratings |

*Results saved to: `evaluation/human_vs_llm.csv`*

---

## 9. Error Analysis & Failure Patterns

Detailed analysis of misclassified queries (`evaluation/error_analysis.csv`) revealed 5 key failure patterns:

1. **Delivery Keyword Bias (40% of errors):**
   - *Pattern:* Customers mentioning words like *"delivery"*, *"shipping"*, or *"package"* while complaining about defective items or subscription delays.
   - *Example:* A user complaining about receiving a defective laptop on 1-day delivery is misclassified as `Delivery & Tracking` instead of `Product Problems`.
2. **Dominant Class Prior (40% of errors):**
   - *Pattern:* Extremely brief tweets (under 10 words) lacking explicit domain tokens default to the high-prior `Delivery & Tracking` category.
3. **Payment & Billing Nuance (10% of errors):**
   - *Pattern:* Inquiries regarding Amazon balance or gift card credits misclassified due to absence of typical banking terms like *"charge"* or *"debit"*.
4. **Digital vs. Prime Overlap (5% of errors):**
   - *Pattern:* App navigation issues on Prime Video or Prime Music confound the boundary between `Prime Membership & Benefits` and `Digital Services & Devices`.
5. **Order Lifecycle Ambiguity (5% of errors):**
   - *Pattern:* Inquiries regarding order cancellations overlap lexically with delivery status tracking.

---

## 10. Example Demonstration

### Query:
```
"My package says it was delivered but I haven't received it."
```

### Agent Output:
```
--------------------------------
HIVER SUPPORT AGENT RESULT
--------------------------------
Predicted Intent:      Delivery & Tracking
Intent Confidence:     0.519
Retrieval Confidence:  0.768
Overall Confidence:    0.669
Status:                SUGGESTED_RESPONSE

Suggested Response:
"@350170 Please don't provide your details here as we consider them to be personal information. 
Our Twitter page is public. Please connect with us securely via direct message so we can trace your parcel."

Top Retrieved Historical Case (Similarity: 0.768):
Customer:  "Said delivered but not here"
Brand:     "@350170 Please don't provide your details here as we consider them to be personal info..."
```

---

## 11. Project Structure

```
hiver-support-agent/
│
├── data/
│   └── processed/
│       ├── clean_pairs.csv             # 150,426 cleaned conversation pairs
│       └── sample_1000.csv             # Exploratory analysis sample
│
├── golden_set/
│   ├── golden_candidates.csv           # Initial 480 candidates (60 per intent)
│   ├── golden_200_to_label.csv         # Candidate subset for annotation
│   └── golden_200_final.csv            # 200 manually verified ground-truth pairs
│
├── evaluation/
│   ├── baseline_classifier.joblib      # Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.joblib         # 50k feature TF-IDF vectorizer
│   ├── tfidf_matrix.joblib             # 150,240 x 50,000 sparse matrix
│   ├── retrieval_data.csv              # 150,240 historical pairs (leakage-free)
│   ├── final_results.csv               # Comparative evaluation summary table
│   ├── error_analysis.csv              # Detailed failure patterns & explanations
│   ├── llm_judge_results.csv           # 5-dimension rubric response evaluations
│   ├── human_vs_llm.csv                # Human agreement comparison (20 samples)
│   └── test_retrieval_cases.csv        # Detailed test retrieval diagnostics
│
├── src/
│   ├── clean_data.py                   # Data cleaning pipeline
│   ├── explore_data.py                 # Word & bi-gram exploration
│   ├── analyze_intents.py              # Intent distribution analysis
│   ├── create_golden_candidate.py      # Golden candidate generation
│   ├── validate_golden.py              # Golden set integrity validation
│   ├── trivial_baseline.py             # Majority-class baseline (25.0%)
│   ├── baseline_classifier.py          # TF-IDF + Logistic Regression (50.0%)
│   ├── retrieval_baseline.py           # kNN retrieval baseline (55.0%)
│   ├── build_retrieval_index.py        # Leakage-free TF-IDF index builder
│   ├── support_agent.py                # Core Hiver Support Agent engine
│   ├── evaluate_agent.py               # Comprehensive evaluation & reports
│   └── llm_judge.py                    # Response quality & human agreement judge
│
├── decision_log.md                     # Architectural & engineering decisions
├── requirements.txt                    # Minimal reproducible dependencies
├── .gitignore                          # Clean repository ignore configuration
└── README.md                           # Complete documentation & reproduction guide
```

---

## 12. Reproduction Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <REPO_URL>
cd hiver-support-agent

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Baselines
```bash
# Trivial Majority Baseline (25% Accuracy)
python src/trivial_baseline.py

# Simple ML Classifier Baseline (50% Accuracy)
python src/baseline_classifier.py

# Retrieval Baseline (55% Accuracy)
python src/retrieval_baseline.py
```

### 3. Run the Support Agent (Interactive CLI)
```bash
python src/support_agent.py
```

### 4. Run the Full Evaluation Suite
```bash
# Runs intent, retrieval, escalation, and error analysis:
# Generates evaluation/final_results.csv & evaluation/error_analysis.csv
python src/evaluate_agent.py

# Runs response-quality evaluation (rubric) & human agreement analysis:
# Generates evaluation/llm_judge_results.csv & evaluation/human_vs_llm.csv
python src/llm_judge.py
```
