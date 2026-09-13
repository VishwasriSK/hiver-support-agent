import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "golden_set/golden_200_final.csv"

print("Loading golden dataset...")

df = pd.read_csv(INPUT_FILE)

X = df["customer_text"].astype(str)
y = df["gold_intent"].astype(str)

print(f"Total examples: {len(df)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

# TF-IDF + Logistic Regression
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

print("\nTraining baseline model...")

model.fit(X_train, y_train)

print("Training completed!")

# Predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n--------------------------------")
print("BASELINE RESULTS")
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

# Save model
output_file = "evaluation/baseline_classifier.joblib"

joblib.dump(model, output_file)

print("\nBaseline model saved to:")
print(output_file)