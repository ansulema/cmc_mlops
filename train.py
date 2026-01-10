import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_data_stats(df: pd.DataFrame, cfg: dict, logger: logging.Logger):
    """Log basic data statistics for validation."""
    label_col = cfg["label_column"]
    text_col = cfg["text_column"]
    
    logger.info("Data statistics:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Label distribution: {df[label_col].value_counts().to_dict()}")
    logger.info(f"  Spam ratio: {df[label_col].mean():.2%}")
    logger.info(f"  Avg text length: {df[text_col].str.len().mean():.0f} chars")
    logger.info(f"  Min text length: {df[text_col].str.len().min()}")
    logger.info(f"  Max text length: {df[text_col].str.len().max()}")


def load_and_preprocess_data(cfg: dict, logger: logging.Logger):
    logger.info(f"Loading data from {cfg['data_path']}")
    df = pd.read_csv(cfg["data_path"])
    
    # Validate required columns
    for col in [cfg["label_column"], cfg["text_column"]]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df = df[[cfg["label_column"], cfg["text_column"]]].dropna().drop_duplicates()
    
    # Map string labels to int if needed
    if df[cfg["label_column"]].dtype == object:
        df[cfg["label_column"]] = df[cfg["label_column"]].map({"ham": 0, "spam": 1})
    
    # Validate labels
    unique_labels = set(df[cfg["label_column"]].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1, got: {unique_labels}")
    
    # Log statistics before sampling
    log_data_stats(df, cfg, logger)
    
    # Sample if max_train_samples specified
    if cfg.get("max_train_samples") and len(df) > cfg["max_train_samples"]:
        df = df.groupby(cfg["label_column"], group_keys=False).apply(
            lambda x: x.sample(frac=cfg["max_train_samples"] / len(df), random_state=cfg["random_seed"])
        )
        logger.info(f"Sampled to {len(df)} samples")
    
    logger.info(f"Loaded {len(df)} samples")
    return df


def split_data(df: pd.DataFrame, cfg: dict, logger: logging.Logger):
    train_df, temp_df = train_test_split(
        df,
        test_size=cfg["test_size"],
        stratify=df[cfg["label_column"]],
        random_state=cfg["random_seed"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=cfg["val_ratio"],
        stratify=temp_df[cfg["label_column"]],
        random_state=cfg["random_seed"],
    )
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def tokenize_datasets(train_df, val_df, test_df, tokenizer, cfg: dict):
    def tokenize(batch):
        return tokenizer(
            batch[cfg["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=cfg["max_length"],
        )

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False).map(tokenize, batched=True)
    return train_ds, val_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, pos_label=1)
    recall = recall_score(labels, preds, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {"accuracy": acc, "f1_spam": f1, "recall_spam": recall, "fpr": fpr}


def train(cfg: dict, logger: logging.Logger):
    # Load data
    df = load_and_preprocess_data(cfg, logger)
    train_df, val_df, test_df = split_data(df, cfg, logger)

    # Tokenizer and model
    logger.info(f"Loading model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=cfg["num_labels"]
    )

    # Tokenize
    logger.info("Tokenizing datasets")
    train_ds, val_ds, test_ds = tokenize_datasets(train_df, val_df, test_df, tokenizer, cfg)

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1_spam",
        logging_steps=cfg["logging_steps"],
        seed=cfg["random_seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training")
    trainer.train()

    # Evaluate on test
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_ds)
    logger.info("Test results:")
    for k, v in test_results.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save model
    logger.info(f"Saving model to {cfg['output_dir']}")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logger.info("Done")

    return test_results


def main():
    parser = argparse.ArgumentParser(description="Train spam classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    logger.info(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    logger.debug(f"Config: {cfg}")

    train(cfg, logger)


if __name__ == "__main__":
    main()
