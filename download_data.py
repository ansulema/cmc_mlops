import pandas as pd

df = pd.read_parquet("hf://datasets/darkQibit/russian-spam-detection/processed_combined.parquet")
print(f"Downloaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

df.to_csv("russian_spam.csv", index=False)
print("Saved to russian_spam.csv")
