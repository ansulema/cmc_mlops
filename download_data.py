import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Download spam dataset")
parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
parser.add_argument("--sample-frac", type=float, default=0.125, help="Fraction of data to keep (1/8 by default)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print("Downloading dataset...")
df = pd.read_parquet("hf://datasets/darkQibit/russian-spam-detection/processed_combined.parquet")
print(f"Full dataset: {len(df)} samples")

# Stratified sampling to keep class ratio
if args.sample_frac < 1.0:
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=args.sample_frac, random_state=42)
    )
    print(f"Sampled to: {len(df)} samples ({args.sample_frac:.1%})")

print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

output_path = os.path.join(args.output_dir, "russian_spam.csv")
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
