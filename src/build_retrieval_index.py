import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


CLEAN_DATA = "data/processed/clean_pairs.csv"
GOLDEN_DATA = "golden_set/golden_200_final.csv"

OUTPUT_VECTORIZER = "evaluation/tfidf_vectorizer.joblib"
OUTPUT_MATRIX = "evaluation/tfidf_matrix.joblib"
OUTPUT_DATA = "evaluation/retrieval_data.csv"


print("Loading cleaned conversations...")

df = pd.read_csv(CLEAN_DATA)

print(f"Original conversations: {len(df)}")


# Load golden examples so we can exclude them
golden = pd.read_csv(GOLDEN_DATA)

golden_ids = set(
    golden["customer_text"]
    .astype(str)
    .str.strip()
)


print(f"Golden examples to exclude: {len(golden_ids)}")


# Remove golden examples from retrieval corpus
df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

before = len(df)

df = df[
    ~df["customer_text"].isin(golden_ids)
].copy()

removed = before - len(df)

print(f"Golden examples removed: {removed}")
print(f"Retrieval conversations: {len(df)}")


# Keep required columns
df = df[
    ["customer_text", "brand_reply"]
].copy()

df["brand_reply"] = (
    df["brand_reply"]
    .fillna("")
    .astype(str)
)


# Remove empty messages
df = df[
    df["customer_text"].str.strip() != ""
].copy()


print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

matrix = vectorizer.fit_transform(
    df["customer_text"]
)


print("TF-IDF index created!")
print(f"Matrix shape: {matrix.shape}")


print("\nSaving files...")

joblib.dump(
    vectorizer,
    OUTPUT_VECTORIZER
)

joblib.dump(
    matrix,
    OUTPUT_MATRIX
)

df.to_csv(
    OUTPUT_DATA,
    index=False
)


print("\n--------------------------------")
print("RETRIEVAL INDEX CREATED")
print("--------------------------------")

print(f"Vectorizer: {OUTPUT_VECTORIZER}")
print(f"Matrix:     {OUTPUT_MATRIX}")
print(f"Data:       {OUTPUT_DATA}")