import pandas as pd

INPUT_PATH = "data/processed/clean_pairs.csv"

df = pd.read_csv(INPUT_PATH)

print("Total pairs:", len(df))


intent_keywords = {
    "Delivery Issues": [
        "delivery",
        "delivered",
        "delivery date",
        "package",
        "parcel",
        "shipping"
    ],

    "Order Issues": [
        "order",
        "ordered",
        "cancel order",
        "cancel my order"
    ],

    "Payment & Billing": [
        "payment",
        "charged",
        "charge",
        "credit card",
        "debit card",
        "pay",
        "billing"
    ],

    "Returns & Refunds": [
        "refund",
        "return",
        "money back"
    ],

    "Account & Login": [
        "account",
        "sign in",
        "signin",
        "login",
        "password",
        "logged in"
    ],

    "Prime Membership": [
        "prime membership",
        "prime member",
        "prime subscription",
        "prime membership"
    ],

    "Digital Content & Devices": [
        "prime video",
        "fire tv",
        "firestick",
        "alexa",
        "video",
        "streaming"
    ],

    "Product Issues": [
        "damaged",
        "broken",
        "defective",
        "wrong item",
        "product"
    ]
}


for intent, keywords in intent_keywords.items():

    print("\n")
    print("=" * 70)
    print(intent)
    print("=" * 70)

    mask = pd.Series(False, index=df.index)

    for keyword in keywords:
        mask = mask | df["customer_text"].str.lower().str.contains(
            keyword,
            regex=False,
            na=False
        )

    examples = df[mask].sample(
        n=min(5, mask.sum()),
        random_state=42
    )

    for i, (_, row) in enumerate(examples.iterrows(), start=1):

        print(f"\nExample {i}:")
        print("Customer:", row["customer_text"])
        print("AmazonHelp:", row["brand_reply"])