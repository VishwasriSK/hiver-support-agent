import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity


VECTOR_FILE = "evaluation/tfidf_vectorizer.joblib"
MATRIX_FILE = "evaluation/tfidf_matrix.joblib"
DATA_FILE = "evaluation/retrieval_data.csv"
CLASSIFIER_FILE = "evaluation/improved_classifier.joblib"


print("Loading support agent...")

# Load retrieval system
vectorizer = joblib.load(VECTOR_FILE)
matrix = joblib.load(MATRIX_FILE)
data = pd.read_csv(DATA_FILE)

# Load intent classifier
classifier = joblib.load(CLASSIFIER_FILE)

# Dynamically identify intent/category column
INTENT_COL = None
for col in ["intent", "category", "gold_intent", "candidate_intent"]:
    if col in data.columns:
        INTENT_COL = col
        break

if INTENT_COL is None:
    # If not present, dynamically generate using the trained classifier
    print("Classifying historical interactions dynamically...")
    data["intent"] = classifier.predict(data["customer_text"].fillna(""))
    INTENT_COL = "intent"

print(f"Loaded {len(data):,} historical conversations (Intent column: '{INTENT_COL}').")


def predict_intent(customer_message):
    prediction = classifier.predict([customer_message])[0]
    probabilities = classifier.predict_proba([customer_message])[0]
    confidence = float(max(probabilities))
    return prediction, confidence


def find_similar_conversations(customer_message, intent=None, top_k=3):
    """
    Intent-conditioned retrieval: restricts TF-IDF cosine similarity search
    to historical interactions matching the predicted intent.
    """
    num_before = len(data)

    if intent and INTENT_COL in data.columns:
        matching_mask = (data[INTENT_COL] == intent).values
        candidate_indices = np.where(matching_mask)[0]
    else:
        candidate_indices = np.arange(len(data))

    num_after = len(candidate_indices)

    # Graceful handling of cases where predicted intent has no matching historical candidates
    if num_after == 0:
        return [], num_before, 0

    query_vector = vectorizer.transform([customer_message])

    # Perform cosine similarity ranking ONLY within that candidate subset
    sub_matrix = matrix[candidate_indices]
    scores = cosine_similarity(query_vector, sub_matrix)[0]

    k = min(top_k, len(scores))
    best_sub_indices = scores.argsort()[-k:][::-1]

    results = []
    for sub_idx in best_sub_indices:
        orig_idx = int(candidate_indices[sub_idx])
        results.append({
            "customer_text": data.iloc[orig_idx]["customer_text"],
            "brand_reply": data.iloc[orig_idx]["brand_reply"],
            "intent": data.iloc[orig_idx][INTENT_COL],
            "similarity": float(scores[sub_idx])
        })

    return results, num_before, num_after


def generate_response(customer_message, show_debug=False):
    """
    End-to-End Triage Agent Pipeline:
    Customer message -> Intent classification -> Intent-conditioned retrieval
    -> Similarity/confidence decision -> Suggested response OR escalation
    """
    # 1. Intent classification
    intent, intent_confidence = predict_intent(customer_message)

    # 2. Intent-conditioned retrieval
    similar_cases, num_before, num_after = find_similar_conversations(
        customer_message, intent=intent, top_k=3
    )

    # 3. Handle zero candidate subset gracefully
    if len(similar_cases) == 0:
        status = "ESCALATE"
        escalation_reason = f"No historical candidates found for intent: {intent}"
        retrieval_confidence = 0.0
        overall_confidence = float(0.4 * intent_confidence)
        response = (
            "I'm sorry you're experiencing this issue. "
            "This case requires further assistance from "
            "a customer support representative."
        )
        best_case = None
    else:
        best_case = similar_cases[0]
        retrieval_confidence = float(best_case["similarity"])

        # Combined confidence
        overall_confidence = float(
            0.4 * intent_confidence + 0.6 * retrieval_confidence
        )

        # Calibrated threshold (0.30)
        if overall_confidence < 0.30:
            status = "ESCALATE"
            if intent_confidence < 0.25 and retrieval_confidence < 0.30:
                escalation_reason = "Low intent confidence & low retrieval similarity"
            elif intent_confidence < 0.25:
                escalation_reason = "Low intent confidence (ambiguous customer query)"
            elif retrieval_confidence < 0.30:
                escalation_reason = "Low retrieval similarity (insufficient historical precedent)"
            else:
                escalation_reason = "Composite confidence below operational threshold"

            response = (
                "I'm sorry you're experiencing this issue. "
                "This case requires further assistance from "
                "a customer support representative."
            )
        else:
            status = "SUGGESTED_RESPONSE"
            escalation_reason = "None (Confident suggestion)"
            response = best_case["brand_reply"]

    debug_info = {
        "predicted_intent": intent,
        "classifier_confidence": intent_confidence,
        "candidates_before": num_before,
        "candidates_after": num_after,
        "top_retrieved_result": best_case["customer_text"] if best_case else "None",
        "top_brand_reply": best_case["brand_reply"] if best_case else "None",
        "similarity_score": retrieval_confidence,
        "overall_confidence": overall_confidence,
        "final_decision": status,
        "escalation_reason": escalation_reason
    }

    if show_debug:
        print("\n--- AGENT DEBUG INFO ---")
        print(f"Predicted Intent:              {debug_info['predicted_intent']}")
        print(f"Classifier Confidence:         {debug_info['classifier_confidence']:.4f}")
        print(f"Candidates Before Filtering:   {debug_info['candidates_before']:,}")
        print(f"Candidates After Filtering:    {debug_info['candidates_after']:,}")
        print(f"Top Retrieved Result:          {debug_info['top_retrieved_result'][:80]}...")
        print(f"Similarity Score:              {debug_info['similarity_score']:.4f}")
        print(f"Final Decision:                {debug_info['final_decision']}")
        if status == "ESCALATE":
            print(f"Escalation Reason:             {debug_info['escalation_reason']}")
        print("------------------------\n")

    return {
        "intent": intent,
        "intent_confidence": intent_confidence,
        "retrieval_confidence": retrieval_confidence,
        "overall_confidence": overall_confidence,
        "status": status,
        "escalation_reason": escalation_reason,
        "response": response,
        "similar_cases": similar_cases,
        "debug_info": debug_info
    }


if __name__ == "__main__":
    print("\n--------------------------------")
    print("HIVER SUPPORT AGENT (INTENT-AWARE)")
    print("--------------------------------")

    customer_message = input("\nEnter customer message:\n> ")

    result = generate_response(customer_message, show_debug=True)

    print("\n--------------------------------")
    print("AGENT RESULT")
    print("--------------------------------")
    print(f"Intent:               {result['intent']}")
    print(f"Intent confidence:    {result['intent_confidence']:.3f}")
    print(f"Retrieval confidence: {result['retrieval_confidence']:.3f}")
    print(f"Overall confidence:   {result['overall_confidence']:.3f}")
    print(f"Status:               {result['status']}")

    if result["status"] == "ESCALATE":
        print(f"Escalation reason:    {result['escalation_reason']}")

    print("\nSuggested response:")
    print(result["response"])

    print("\n--------------------------------")
    print("SIMILAR HISTORICAL CASES")
    print("--------------------------------")
    for i, case in enumerate(result["similar_cases"], start=1):
        print(f"\nCase {i} (Sim: {case['similarity']:.3f} | Intent: {case['intent']})")
        print(f"Customer: {case['customer_text']}")
        print(f"Reply:    {case['brand_reply']}")