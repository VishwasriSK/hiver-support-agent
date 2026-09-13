import pandas as pd
import re

INPUT_PATH = "data/processed/pairs.csv"
OUTPUT_PATH = "data/processed/clean_pairs.csv"

print("Loading conversation pairs...")

pairs = pd.read_csv(INPUT_PATH)

print("Original pairs:", len(pairs))


# --------------------------------------------------
# 1. Remove missing values
# --------------------------------------------------

pairs = pairs.dropna(subset=["customer_text", "brand_reply"])

print("After removing missing values:", len(pairs))


# --------------------------------------------------
# 2. Remove very short messages
# --------------------------------------------------

pairs = pairs[
    (pairs["customer_text"].str.len() >= 20) &
    (pairs["brand_reply"].str.len() >= 20)
].copy()

print("After removing short messages:", len(pairs))


# --------------------------------------------------
# 3. Keep English-looking customer messages
# --------------------------------------------------

def english_ratio(text):
    letters = re.findall(r"[A-Za-z]", str(text))
    total_letters = len(re.findall(r"[A-Za-zÀ-ÿ]", str(text)))

    if total_letters == 0:
        return 0

    return len(letters) / total_letters


pairs["english_ratio"] = pairs["customer_text"].apply(english_ratio)

pairs = pairs[pairs["english_ratio"] >= 0.70].copy()

pairs.drop(columns=["english_ratio"], inplace=True)

print("After English filtering:", len(pairs))


# --------------------------------------------------
# 4. Remove duplicate customer messages
# --------------------------------------------------

pairs = pairs.drop_duplicates(
    subset=["customer_text"]
)

print("After removing duplicates:", len(pairs))


# --------------------------------------------------
# 5. Save cleaned dataset
# --------------------------------------------------

pairs.to_csv(OUTPUT_PATH, index=False)

print("\nClean dataset saved successfully!")

print("Path:", OUTPUT_PATH)

print("Final pairs:", len(pairs))


# --------------------------------------------------
# 6. Show sample conversations
# --------------------------------------------------

print("\nSample cleaned conversations:\n")

print(
    pairs[
        [
            "customer_text",
            "brand_reply"
        ]
    ].head(10).to_string(index=False)
)