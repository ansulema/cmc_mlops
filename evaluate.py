"""Evaluate trained model on test set with MLflow tracking."""
import argparse
import json
import logging
import yaml
import pandas as pd
import numpy as np
import mlflow
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
    parser.add_argument("--experiment", default="spam-classifier")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup MLflow
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="evaluate"):
        logger.info("Loading model and test data")
        tokenizer = AutoTokenizer.from_pretrained(cfg["output_dir"])
        model = AutoModelForSequenceClassification.from_pretrained(cfg["output_dir"])

        test_df = pd.read_csv("data/test.csv")
        mlflow.log_param("test_size", len(test_df))

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

        # Log metrics to MLflow
        mlflow.log_metrics({
            "test_accuracy": metrics.get("eval_accuracy", 0),
            "test_f1_spam": metrics.get("eval_f1_spam", 0),
            "test_recall_spam": metrics.get("eval_recall_spam", 0),
            "test_fpr": metrics.get("eval_fpr", 0),
        })

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact("metrics.json")
        logger.info("Saved metrics to metrics.json")


if __name__ == "__main__":
    main()
