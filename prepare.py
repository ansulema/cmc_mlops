"""Prepare data: load, preprocess, split into train/val/test."""
import argparse
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loading data from {cfg['data_path']}")
    df = pd.read_csv(cfg["data_path"])
    df = df[[cfg["label_column"], cfg["text_column"]]].dropna().drop_duplicates()

    if df[cfg["label_column"]].dtype == object:
        df[cfg["label_column"]] = df[cfg["label_column"]].map({"ham": 0, "spam": 1})

    # Sample if needed
    if cfg.get("max_train_samples") and len(df) > cfg["max_train_samples"]:
        frac = cfg["max_train_samples"] / len(df)
        df = df.sample(frac=frac, random_state=cfg["random_seed"])
        logger.info(f"Sampled to {len(df)} samples")

    # Split
    train_df, temp_df = train_test_split(
        df, test_size=cfg["test_size"], stratify=df[cfg["label_column"]], random_state=cfg["random_seed"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=cfg["val_ratio"], stratify=temp_df[cfg["label_column"]], random_state=cfg["random_seed"]
    )

    # Save
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    logger.info(f"Saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    main()
