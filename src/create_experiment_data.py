import pandas as pd
from langdetect import detect, DetectorFactory

INPUT_PATH = "data/processed/clean_pairs.csv"
OUTPUT_PATH = "data/processed/experiment_data.csv"

DetectorFactory.seed = 42

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

print("Available pairs:", len(df))


# --------------------------------------------------
# Sample manageable dataset
# --------------------------------------------------

sample_size = min(5000, len(df))

df = df.sample(
    n=sample_size,
    random_state=42
).reset_index(drop=True)

print("Sample selected:", len(df))


# --------------------------------------------------
# Language detection
# --------------------------------------------------

def is_english(text):

    try:
        return detect(str(text)) == "en"
    except:
        return False


print("\nDetecting language...")

df["is_english"] = df["customer_text"].apply(is_english)

df = df[df["is_english"]].copy()

df.drop(columns=["is_english"], inplace=True)

print("English conversations:", len(df))


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nExperiment dataset saved!")
print("Path:", OUTPUT_PATH)
print("Final conversations:", len(df))