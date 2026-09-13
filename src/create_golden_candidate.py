import pandas as pd

INPUT_PATH = "data/processed/experiment_data.csv"
OUTPUT_PATH = "golden_set/golden_candidates.csv"

print("Loading experiment dataset...")

df = pd.read_csv(INPUT_PATH)

print("Total English conversations:", len(df))


intent_keywords = {

    "Delivery & Tracking": [
        "delivery",
        "delivered",
        "shipping",
        "package",
        "parcel",
        "tracking",
        "arrive",
        "arrived",
        "late",
        "delay"
    ],

    "Order Management": [
        "order",
        "ordered",
        "cancel order",
        "cancel my order",
        "change order"
    ],

    "Payment & Charges": [
        "payment",
        "charged",
        "charge",
        "credit card",
        "debit card",
        "billing",
        "pay",
        "paid"
    ],

    "Returns & Refunds": [
        "refund",
        "return",
        "replacement",
        "money back"
    ],

    "Account & Login": [
        "account",
        "sign in",
        "signin",
        "login",
        "logged in",
        "password"
    ],

    "Prime Membership & Benefits": [
        "prime membership",
        "prime member",
        "prime subscription",
        "prime benefits"
    ],

    "Digital Services & Devices": [
        "prime video",
        "fire tv",
        "firestick",
        "alexa",
        "streaming",
        "video",
        "echo"
    ],

    "Product Problems": [
        "damaged",
        "broken",
        "defective",
        "wrong item",
        "wrong product",
        "product"
    ]
}


all_candidates = []

for intent, keywords in intent_keywords.items():

    print(f"\nFinding candidates for: {intent}")

    mask = pd.Series(False, index=df.index)

    for keyword in keywords:

        mask = mask | df["customer_text"].str.lower().str.contains(
            keyword,
            regex=False,
            na=False
        )

    candidates = df[mask].copy()

    # Remove duplicates
    candidates = candidates.drop_duplicates(
        subset=["customer_text"]
    )

    # Take up to 60 candidates per intent
    candidates = candidates.head(60)

    candidates["candidate_intent"] = intent

    all_candidates.append(candidates)


result = pd.concat(
    all_candidates,
    ignore_index=True
)


# Keep only useful columns

result = result[
    [
        "customer_tweet_id",
        "customer_text",
        "brand_reply",
        "candidate_intent"
    ]
]


result.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n--------------------------------")
print("Candidate generation completed!")
print("--------------------------------")

print("Total candidates:", len(result))

print("Saved to:")
print(OUTPUT_PATH)

print("\nCandidates per intent:")

print(
    result["candidate_intent"]
    .value_counts()
)