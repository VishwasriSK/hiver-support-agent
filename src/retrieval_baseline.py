import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "golden_set/golden_200_final.csv"


print("Loading golden dataset...")

df = pd.read_csv(INPUT_FILE)

X = df["customer_text"].astype(str)
y = df["gold_intent"].astype(str)

print(f"Total examples: {len(df)}")


# Same split as our classifier baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Reference examples: {len(X_train)}")
print(f"Evaluation examples: {len(X_test)}")


# Convert reference and test messages into TF-IDF vectors
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=20000
)

print("\nBuilding TF-IDF vectors...")

train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)


print("Finding nearest examples...")

# Similarity between every test message and every reference message
similarities = cosine_similarity(
    test_vectors,
    train_vectors
)

predictions = []

for row in similarities:

    # Index of most similar training example
    best_index = row.argmax()

    # Get its intent
    predicted_intent = y_train.iloc[best_index]

    predictions.append(predicted_intent)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n--------------------------------")
print("RETRIEVAL BASELINE RESULTS")
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