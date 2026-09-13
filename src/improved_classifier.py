import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "golden_set/golden_200_final.csv"

print("Loading golden dataset...")
df = pd.read_csv(INPUT_FILE)

X = df["customer_text"].astype(str)
y = df["gold_intent"].astype(str)

print(f"Total examples: {len(df)}")

# Maintain identical split methodology as baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

# Combined Word + Character TF-IDF representation
features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=20000
        )
    )
])

# Classifier with class_weight='balanced'
model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
    )
])

print("\nTraining improved classifier model...")
model.fit(X_train, y_train)
print("Training completed!")

# Predictions on Test set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--------------------------------")
print("IMPROVED CLASSIFIER TEST RESULTS")
print("--------------------------------")
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model as separate artifact
output_file = "evaluation/improved_classifier.joblib"
joblib.dump(model, output_file)
print(f"\nImproved model saved to: {output_file}")
