import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Download spam dataset")
parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_parquet("hf://datasets/darkQibit/russian-spam-detection/processed_combined.parquet")
print(f"Downloaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

output_path = os.path.join(args.output_dir, "russian_spam.csv")
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
