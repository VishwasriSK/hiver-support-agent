import pandas as pd
import re
from collections import Counter

INPUT_PATH = "data/processed/clean_pairs.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print("Total pairs:", len(df))


# --------------------------------------------------
# Take a representative sample
# --------------------------------------------------

sample = df.sample(
    n=min(10000, len(df)),
    random_state=42
)


# --------------------------------------------------
# Combine customer messages
# --------------------------------------------------

text = " ".join(
    sample["customer_text"]
    .astype(str)
)


# --------------------------------------------------
# Clean text
# --------------------------------------------------

text = text.lower()

words = re.findall(
    r"\b[a-z]{3,}\b",
    text
)


# --------------------------------------------------
# Common words
# --------------------------------------------------

stop_words = {
    "the", "and", "for", "that", "this", "with",
    "have", "you", "your", "are", "was", "but",
    "not", "from", "they", "what", "can", "just",
    "please", "amazon", "help", "http", "https",
    "www", "com", "has", "had", "did", "got",
    "get", "about", "would", "could", "there",
    "been", "will", "its", "our", "out", "now",
    "all", "when", "who", "how", "why", "too",
    "really", "like", "still", "want", "need",
    "thanks", "thank"
}

filtered_words = [
    word for word in words
    if word not in stop_words
]


counter = Counter(filtered_words)


print("\nTop 100 meaningful words:\n")

for word, count in counter.most_common(100):
    print(f"{word:25} {count}")


# --------------------------------------------------
# Common two-word phrases
# --------------------------------------------------

phrases = []

for i in range(len(filtered_words) - 1):
    phrases.append(
        filtered_words[i] + " " + filtered_words[i + 1]
    )

phrase_counter = Counter(phrases)


print("\nTop 50 two-word phrases:\n")

for phrase, count in phrase_counter.most_common(50):
    print(f"{phrase:35} {count}")