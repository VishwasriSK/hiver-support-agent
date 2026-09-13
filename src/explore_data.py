import pandas as pd

INPUT_PATH = "data/processed/clean_pairs.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

print("Total pairs:", len(df))

# Take a random sample
sample = df.sample(
    n=1000,
    random_state=42
)

print("\nSample size:", len(sample))

# Save sample for inspection
sample.to_csv(
    "data/processed/sample_1000.csv",
    index=False
)

print("\nSample saved to:")
print("data/processed/sample_1000.csv")

print("\nFirst 20 customer messages:\n")

for i, text in enumerate(sample["customer_text"].head(20), start=1):
    print(f"{i}. {text}")