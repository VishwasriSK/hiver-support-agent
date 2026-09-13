import pandas as pd

FILE_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/pairs.csv"

print("Loading dataset...")

df = pd.read_csv(FILE_PATH)

print("Dataset loaded!")
print("Total tweets:", len(df))

# Keep only AmazonHelp replies
amazon_replies = df[
    (df["author_id"] == "AmazonHelp") &
    (df["inbound"] == False) &
    (df["in_response_to_tweet_id"].notna())
].copy()

print("\nAmazonHelp replies with a parent tweet:",
      len(amazon_replies))

# Create a lookup table:
# tweet_id -> tweet text
tweet_lookup = df.set_index("tweet_id")["text"]

# Find the customer tweet that each AmazonHelp reply responds to
amazon_replies["customer_text"] = (
    amazon_replies["in_response_to_tweet_id"]
    .map(tweet_lookup)
)

# Keep only rows where we successfully found the customer tweet
pairs = amazon_replies[
    amazon_replies["customer_text"].notna()
].copy()

# Select the columns we need
pairs = pairs[
    [
        "in_response_to_tweet_id",
        "customer_text",
        "tweet_id",
        "text"
    ]
]

# Rename them to make the meaning clear
pairs.columns = [
    "customer_tweet_id",
    "customer_text",
    "brand_tweet_id",
    "brand_reply"
]

# Remove empty messages
pairs = pairs[
    pairs["customer_text"].str.strip().ne("") &
    pairs["brand_reply"].str.strip().ne("")
]

# Save the result
pairs.to_csv(OUTPUT_PATH, index=False)

print("\nConversation pairs created!")
print("Number of pairs:", len(pairs))
print("Saved to:", OUTPUT_PATH)

print("\nSample pairs:\n")
print(pairs.head(10).to_string(index=False))