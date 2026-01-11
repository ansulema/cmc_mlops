"""Train spam classifier model with MLflow tracking."""
import argparse
import logging
import os
import yaml
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

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


def get_dvc_hash(dvc_file: str) -> str:
    """Get hash from .dvc file."""
    try:
        with open(dvc_file) as f:
            content = yaml.safe_load(f)
        return content["outs"][0]["md5"]
    except:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment", default="spam-classifier")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup MLflow
    mlflow.set_experiment(args.experiment)
    mlflow.autolog(log_models=False)  # autolog for transformers

    with mlflow.start_run():
        # Log config parameters
        mlflow.log_params({
            "model_name": cfg["model_name"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "num_epochs": cfg["num_epochs"],
            "max_length": cfg["max_length"],
            "weight_decay": cfg["weight_decay"],
            "random_seed": cfg["random_seed"],
        })

        # Log DVC data hash
        data_hash = get_dvc_hash("data/russian_spam.csv.dvc")
        mlflow.set_tag("dvc_data_hash", data_hash)

        # Load prepared data
        logger.info("Loading prepared data")
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")

        mlflow.log_params({
            "train_size": len(train_df),
            "val_size": len(val_df),
        })

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

        # Tokenizer and model
        logger.info(f"Loading model: {cfg['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=cfg["num_labels"]
        )

        def tokenize(batch):
            return tokenizer(
                batch[cfg["text_column"]], truncation=True, padding="max_length", max_length=cfg["max_length"]
            )

        train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
        val_ds = Dataset.from_pandas(val_df, preserve_index=False).map(tokenize, batched=True)

        # Training
        os.makedirs(cfg["output_dir"], exist_ok=True)

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

        logger.info("Starting training")
        trainer.train()

        # Evaluate on validation
        eval_results = trainer.evaluate()
        logger.info(f"Eval results: {eval_results}")

        # Log final metrics
        mlflow.log_metrics({
            "final_accuracy": eval_results.get("eval_accuracy", 0),
            "final_f1_spam": eval_results.get("eval_f1_spam", 0),
            "final_recall_spam": eval_results.get("eval_recall_spam", 0),
            "final_fpr": eval_results.get("eval_fpr", 0),
        })

        # Save model
        logger.info(f"Saving model to {cfg['output_dir']}")
        trainer.save_model(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])

        # Log artifacts
        mlflow.log_artifact(args.config)
        if os.path.exists("dvc.lock"):
            mlflow.log_artifact("dvc.lock")

        logger.info("Done")


if __name__ == "__main__":
    main()
