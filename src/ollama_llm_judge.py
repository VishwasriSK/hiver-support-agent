"""
ollama_llm_judge.py - Genuine LLM-as-a-Judge using Ollama and local Qwen 2.5 (3B)

Triage-Aware Customer Support Quality Auditor:
Evaluates customer support triage decisions across 6 independent dimensions:
1. Relevance (1-5): Addresses intent OR appropriately routes issue to human support
2. Helpfulness (1-5): Provides appropriate resolution or appropriate escalation next step
3. Groundedness (1-5): Supported by customer message, intent, and operational context
4. No Unsupported Claims (1-5): Freedom from fabricated commitments or fake claims
5. Professional Tone (1-5): Evaluated independently from helpfulness (polite & empathetic)
6. Overall Quality (1-5): Holistic assessment of decision and response quality

Features:
- Dynamic ingestion of all 20 evaluation cases from evaluation/llm_judge_results.csv
- Clear triage operational context (ESCALATE is recognized as a valid, high-value action)
- Independent scoring across dimensions (no halo effect / tone penalty for unhelpful text)
- Strict validation (integers 1-5, no hardcoded scores, no fallback values of 1)
- Automatic retry on JSON formatting errors
- Explicit evaluation status tracking (SUCCESS / FAILED)
- Output saved to evaluation/ollama_llm_judge_results.csv
"""

import sys
import os
import json
import time
import requests
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

INPUT_FILE = "evaluation/llm_judge_results.csv"
OUTPUT_FILE = "evaluation/ollama_llm_judge_results.csv"

MAX_RETRIES = 2
TIMEOUT_SECONDS = 90

SCORE_FIELDS = [
    "relevance",
    "helpfulness",
    "groundedness",
    "no_unsupported_claims",
    "professional_tone",
    "overall"
]


def build_evaluation_prompt(customer_message, gold_intent, agent_status, agent_response):
    """
    Constructs a triage-aware evaluation prompt with explicit rubric definitions
    and dimension independence.
    """
    return f"""You are an expert customer-support quality auditor evaluating an enterprise customer-support triage agent.

[OPERATIONAL ROLE OF THE AGENT]
The agent functions as a frontline triage assistant. Its operational responsibilities:
1. Provide an automated response (SUGGESTED_RESPONSE) ONLY when there is clear historical precedent directly addressing the customer's specific inquiry.
2. Safely route to a human support specialist (ESCALATE) when confidence is low, the request is ambiguous, involves account-level security, billing disputes, or requires manual human intervention.

CRITICAL EVALUATION RULE FOR ESCALATIONS:
An ESCALATE decision is NOT a bad response. In an enterprise support workflow, escalating complex, sensitive, or low-confidence issues to human agents with a polite handover message is the correct, safe, and desirable behavior. Do NOT penalize an escalation simply because it does not resolve the customer's problem on the spot.

[CUSTOMER INTERACTION DATA]
- Customer Message: {customer_message}
- Expected Customer Intent: {gold_intent}
- Agent Decision Status: {agent_status}
- Agent Response: {agent_response}

[EVALUATION GUIDELINES]
1. Assess the customer's intent and whether human escalation is reasonable.
2. If status is ESCALATE: Check whether escalating is appropriate and whether the handover message is polite and professional. If escalation is warranted, it deserves high relevance, high professional tone, high groundedness, and good helpfulness (as an appropriate next step).
3. If status is SUGGESTED_RESPONSE: Check whether the response genuinely answers the customer's intent, or if it retrieved an off-topic/unrelated historical answer.
4. Independent Dimensions: Do NOT let poor helpfulness automatically drag down professional tone. A response can be unhelpful but polite, or appropriately escalated and professional.

[SCORING RUBRIC (Integer 1 to 5 only)]
RELEVANCE:
5 = Directly addresses customer's intent OR appropriately routes the issue to human support.
4 = Clearly relevant but somewhat incomplete.
3 = Partially relevant.
2 = Weak connection to the customer's intent.
1 = Completely unrelated/off-topic.

HELPFULNESS:
5 = Provides an appropriate resolution or appropriate next step.
4 = Provides useful guidance or appropriate escalation.
3 = Somewhat useful but incomplete.
2 = Minimal usefulness.
1 = No meaningful assistance.

GROUNDEDNESS:
5 = Fully supported by the customer message, intent, and available context.
4 = Mostly supported with minor gaps.
3 = Partially supported.
2 = Contains questionable assumptions.
1 = Contradicts or ignores the available context.

NO_UNSUPPORTED_CLAIMS:
5 = No unsupported or fabricated claims.
4 = Minor questionable statement.
3 = Some unsupported content.
2 = Significant unsupported claims.
1 = Multiple fabricated or clearly unsupported claims.

PROFESSIONAL_TONE (Evaluate tone independently from helpfulness):
5 = Polite, empathetic, and professional.
4 = Professional with minor stylistic issues.
3 = Acceptable but could be improved.
2 = Noticeably poor tone.
1 = Rude, hostile, inappropriate, or unprofessional.

OVERALL:
Evaluate the response as a whole, considering the agent's operational triage role. (1 to 5, not a simple average).

Respond strictly with a valid JSON object matching this schema:
{{
  "relevance": <integer 1 to 5>,
  "helpfulness": <integer 1 to 5>,
  "groundedness": <integer 1 to 5>,
  "no_unsupported_claims": <integer 1 to 5>,
  "professional_tone": <integer 1 to 5>,
  "overall": <integer 1 to 5>,
  "reason": "<two or three sentences explaining the strengths, weaknesses, and rationale for your scores>"
}}
"""


def call_ollama(prompt, temperature=0.1):
    """
    Sends evaluation request to local Ollama API.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")


def validate_and_parse_scores(raw_text):
    """
    Validates that the returned string is valid JSON and all 6 scores
    are integers in the range [1, 5].
    Returns (parsed_dict, error_message).
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty response from Ollama"

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)}"

    validated = {}
    for field in SCORE_FIELDS:
        if field not in data:
            return None, f"Missing required field: '{field}'"

        try:
            val = int(data[field])
        except (ValueError, TypeError):
            return None, f"Field '{field}' cannot be converted to integer (got {data[field]})"

        if not (1 <= val <= 5):
            return None, f"Field '{field}' score {val} is outside allowed range [1, 5]"

        validated[field] = val

    reason = str(data.get("reason", "")).strip()
    if not reason:
        reason = "No explanation provided by LLM."
    validated["reason"] = reason

    return validated, None


def evaluate_single_interaction(sample_id, customer_message, gold_intent, agent_status, agent_response, is_first_sample=False):
    """
    Evaluates one interaction with retry capability, detailed logging,
    and strict error tracking.
    """
    prompt = build_evaluation_prompt(customer_message, gold_intent, agent_status, agent_response)

    raw_response = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            temp = 0.1 if attempt == 1 else 0.2
            raw_response = call_ollama(prompt, temperature=temp)

            if is_first_sample and attempt == 1:
                print("\n" + "-" * 50)
                print("[DEBUG] RAW OLLAMA RESPONSE (SAMPLE 1):")
                print(raw_response)
                print("-" * 50)

            parsed_scores, error_msg = validate_and_parse_scores(raw_response)

            if parsed_scores is not None:
                if is_first_sample and attempt == 1:
                    print("\n[DEBUG] FIRST PARSED JSON RESULT (SAMPLE 1):")
                    print(json.dumps(parsed_scores, indent=2))
                    print("-" * 50 + "\n")

                return {
                    "scores": parsed_scores,
                    "status": "SUCCESS",
                    "error": None,
                    "raw": raw_response
                }
            else:
                last_error = error_msg
                print(f"  [Attempt {attempt}] Validation failed: {error_msg}. Retrying...")
                time.sleep(1)

        except requests.exceptions.RequestException as req_err:
            last_error = f"Ollama HTTP error: {str(req_err)}"
            print(f"  [Attempt {attempt}] Request error: {last_error}. Retrying...")
            time.sleep(2)

    return {
        "scores": None,
        "status": f"FAILED: {last_error}",
        "error": last_error,
        "raw": raw_response
    }


def main():
    print("=" * 70)
    print("OLLAMA LLM-AS-A-JUDGE EVALUATION PIPELINE (TRIAGE-AWARE)")
    print(f"Model: {MODEL} | Endpoint: {OLLAMA_URL}")
    print("=" * 70)

    # 1. Verify Ollama Connection
    print("\n[1/4] Verifying Ollama connection and model availability...")
    try:
        tag_res = requests.get("http://localhost:11434/api/tags", timeout=10)
        tag_res.raise_for_status()
        installed_models = [m.get("name") for m in tag_res.json().get("models", [])]
        if not any(MODEL in m for m in installed_models):
            print(f"WARNING: Model '{MODEL}' not found in installed models: {installed_models}")
        else:
            print(f"Confirmed: Model '{MODEL}' is available and ready on Ollama.")
    except Exception as e:
        print(f"ERROR connecting to Ollama: {e}")
        sys.exit(1)

    # 2. Ingest Existing Evaluation Dataset
    print("\n[2/4] Loading agent evaluation dataset...")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Evaluation file '{INPUT_FILE}' does not exist.")
        sys.exit(1)

    df_in = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {len(df_in)} evaluation interactions from {INPUT_FILE}.")

    # 3. Perform Genuine LLM Evaluation
    print("\n[3/4] Evaluating all 20 interactions with local Qwen 2.5 (3B)...")
    results = []
    successful_count = 0
    failed_count = 0

    for idx, row in df_in.iterrows():
        sample_num = idx + 1
        total_samples = len(df_in)
        print(f"\nEvaluating response {sample_num}/{total_samples} (Sample ID: {row['sample_id']})...")

        eval_result = evaluate_single_interaction(
            sample_id=row["sample_id"],
            customer_message=row["customer_message"],
            gold_intent=row["gold_intent"],
            agent_status=row["agent_status"],
            agent_response=row["agent_response"],
            is_first_sample=(idx == 0)
        )

        status = eval_result["status"]
        scores = eval_result["scores"]

        if status == "SUCCESS":
            successful_count += 1
            print(f"  Ollama evaluation received.")
            print(f"  Scores: Rel={scores['relevance']} | Help={scores['helpfulness']} | "
                  f"Grd={scores['groundedness']} | Safe={scores['no_unsupported_claims']} | "
                  f"Tone={scores['professional_tone']} | Overall={scores['overall']}")

            results.append({
                "sample_id": row["sample_id"],
                "customer_message": row["customer_message"],
                "gold_intent": row["gold_intent"],
                "agent_status": row["agent_status"],
                "agent_response": row["agent_response"],
                "overall_confidence": row["overall_confidence"],
                "llm_relevance": scores["relevance"],
                "llm_helpfulness": scores["helpfulness"],
                "llm_groundedness": scores["groundedness"],
                "llm_no_unsupported_claims": scores["no_unsupported_claims"],
                "llm_professional_tone": scores["professional_tone"],
                "llm_overall": scores["overall"],
                "llm_reason": scores["reason"],
                "llm_evaluation_status": "SUCCESS"
            })
        else:
            failed_count += 1
            print(f"  FAILED to evaluate sample {row['sample_id']}: {eval_result['error']}")
            results.append({
                "sample_id": row["sample_id"],
                "customer_message": row["customer_message"],
                "gold_intent": row["gold_intent"],
                "agent_status": row["agent_status"],
                "agent_response": row["agent_response"],
                "overall_confidence": row["overall_confidence"],
                "llm_relevance": np.nan,
                "llm_helpfulness": np.nan,
                "llm_groundedness": np.nan,
                "llm_no_unsupported_claims": np.nan,
                "llm_professional_tone": np.nan,
                "llm_overall": np.nan,
                "llm_reason": eval_result["error"],
                "llm_evaluation_status": status
            })

    # 4. Save and Summarize
    print("\n[4/4] Generating summary metrics and exporting results...")
    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Calculate averages strictly over successful evaluations
    df_success = df_out[df_out["llm_evaluation_status"] == "SUCCESS"]

    print("\n" + "=" * 60)
    print("OLLAMA LLM JUDGE RESULTS")
    print("=" * 60)
    if len(df_success) > 0:
        print(f"Average Relevance:             {df_success['llm_relevance'].mean():.2f}/5")
        print(f"Average Helpfulness:           {df_success['llm_helpfulness'].mean():.2f}/5")
        print(f"Average Groundedness:          {df_success['llm_groundedness'].mean():.2f}/5")
        print(f"Average No Unsupported Claims: {df_success['llm_no_unsupported_claims'].mean():.2f}/5")
        print(f"Average Professional Tone:     {df_success['llm_professional_tone'].mean():.2f}/5")
        print(f"Average Overall Quality:       {df_success['llm_overall'].mean():.2f}/5")
    else:
        print("No successful evaluations to compute averages.")

    print(f"\nSuccessfully evaluated: {successful_count}/{len(df_in)}")
    print(f"Failed evaluations:     {failed_count}/{len(df_in)}")
    print(f"\nResults saved to:")
    print(OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()