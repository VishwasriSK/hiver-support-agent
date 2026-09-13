import pandas as pd

INPUT_FILE = "golden_set/golden_200_final.xlsx"

ALLOWED_INTENTS = [
    "Delivery & Tracking",
    "Order Management",
    "Payment & Charges",
    "Returns & Refunds",
    "Account & Login",
    "Prime Membership & Benefits",
    "Digital Services & Devices",
    "Product Problems"
]

print("Loading final golden set...")

df = pd.read_excel(INPUT_FILE)

print(f"Total rows: {len(df)}")

errors = []

# 1. Check row count
if len(df) != 200:
    errors.append(f"Expected 200 rows, found {len(df)}")

# 2. Check required columns
required_columns = [
    "gold_id",
    "customer_text",
    "brand_reply",
    "candidate_intent",
    "gold_intent"
]

for column in required_columns:
    if column not in df.columns:
        errors.append(f"Missing column: {column}")

# Stop if required columns are missing
if not errors:

    # 3. Missing labels
    missing = df["gold_intent"].isna().sum()

    if missing > 0:
        errors.append(f"{missing} rows have missing gold_intent")

    # 4. Invalid labels
    invalid = df[
        ~df["gold_intent"].isin(ALLOWED_INTENTS)
    ]

    if len(invalid) > 0:
        errors.append(
            f"{len(invalid)} rows contain invalid intents"
        )

    # 5. Missing customer messages
    missing_customer = df["customer_text"].isna().sum()

    if missing_customer > 0:
        errors.append(
            f"{missing_customer} rows have missing customer_text"
        )

print()
print("--------------------------------")
print("GOLDEN SET VALIDATION")
print("--------------------------------")

if errors:

    print("❌ Validation failed!\n")

    for error in errors:
        print(" -", error)

else:

    print("✅ Row count: 200")
    print("✅ All required columns present")
    print("✅ No missing gold_intent values")
    print("✅ All intents are valid")
    print("✅ No missing customer messages")

    print("\nIntent distribution:")
    print(df["gold_intent"].value_counts())

    # Save final CSV
    output_file = "golden_set/golden_200_final.csv"
    df.to_csv(output_file, index=False)

    print("\n--------------------------------")
    print("Golden set finalized!")
    print("--------------------------------")
    print(f"Saved to: {output_file}")