import pandas as pd

INPUT_PATH = "golden_set/golden_candidates.csv"
OUTPUT_PATH = "golden_set/golden_200_to_label.csv"

print("Loading candidates...")

df = pd.read_csv(INPUT_PATH)

print("Total candidates:", len(df))


# --------------------------------------------------
# Select 25 candidates per candidate intent
# --------------------------------------------------

labeled_parts = []

for intent in df["candidate_intent"].unique():

    subset = df[
        df["candidate_intent"] == intent
    ].sample(
        n=25,
        random_state=42
    )

    labeled_parts.append(subset)


golden = pd.concat(
    labeled_parts,
    ignore_index=True
)


# --------------------------------------------------
# Shuffle so the labels aren't grouped by intent
# --------------------------------------------------

golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Add manual gold label
# --------------------------------------------------

golden.insert(
    0,
    "gold_id",
    range(1, len(golden) + 1)
)

golden["gold_intent"] = ""


# --------------------------------------------------
# Add review column
# --------------------------------------------------

golden["review_notes"] = ""


# --------------------------------------------------
# Save
# --------------------------------------------------

golden.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n--------------------------------")
print("Golden labeling file created!")
print("--------------------------------")

print("Total examples:", len(golden))

print("\nExamples per candidate intent:")

print(
    golden["candidate_intent"]
    .value_counts()
)

print("\nSaved to:")
print(OUTPUT_PATH)