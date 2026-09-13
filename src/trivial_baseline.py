import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "golden_set/golden_200_final.csv"

print("Loading golden dataset...")

df = pd.read_csv(INPUT_FILE)

X = df["customer_text"].astype(str)
y = df["gold_intent"].astype(str)

print(f"Total examples: {len(df)}")

# Use the exact same split as the simple baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

# Trivial baseline:
# Always predict the most frequent intent in the training set
majority_intent = y_train.value_counts().idxmax()

print(f"\nMost frequent training intent: {majority_intent}")

predictions = [majority_intent] * len(y_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n--------------------------------")
print("TRIVIAL BASELINE RESULTS")
print("--------------------------------")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)