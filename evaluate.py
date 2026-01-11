"""Evaluate trained model on test set."""
import argparse
import json
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, pos_label=1)
    recall = recall_score(labels, preds, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {"accuracy": acc, "f1_spam": f1, "recall_spam": recall, "fpr": fpr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info("Loading model and test data")
    tokenizer = AutoTokenizer.from_pretrained(cfg["output_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["output_dir"])

    test_df = pd.read_csv("data/test.csv")

    def tokenize(batch):
        return tokenizer(batch[cfg["text_column"]], truncation=True, padding="max_length", max_length=cfg["max_length"])

    test_ds = Dataset.from_pandas(test_df, preserve_index=False).map(tokenize, batched=True)

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    results = trainer.evaluate(test_ds)

    logger.info("Test results:")
    metrics = {}
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
            metrics[k] = round(v, 4)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to metrics.json")


if __name__ == "__main__":
    main()
