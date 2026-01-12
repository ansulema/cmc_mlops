"""Batch prediction script for Docker container."""
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LABELS = {0: "ham", 1: "spam"}
MODEL_PATH = "models/spam_classifier"


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def probs_to_class(probs: np.ndarray, threshold: float = 0.5) -> int:
    """Convert probabilities to class label."""
    return 1 if probs[1] >= threshold else 0


def class_to_label(class_id: int) -> str:
    """Convert class id to string label."""
    if class_id not in LABELS:
        raise ValueError(f"Invalid class_id: {class_id}")
    return LABELS[class_id]


def predict_single(text: str, tokenizer, model, threshold: float = 0.5) -> dict:
    """Predict single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        logits = model(**inputs).logits.numpy()[0]
    
    probs = logits_to_probs(logits)
    class_id = probs_to_class(probs, threshold)
    label = class_to_label(class_id)
    
    return {
        "label": label,
        "class_id": class_id,
        "confidence": float(probs[class_id]),
        "prob_ham": float(probs[0]),
        "prob_spam": float(probs[1]),
    }


def predict_batch(texts: list, tokenizer, model, threshold: float = 0.5) -> list:
    """Predict batch of texts."""
    results = []
    for text in texts:
        result = predict_single(str(text), tokenizer, model, threshold)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch prediction for spam classifier")
    parser.add_argument("--input_path", type=str, required=True, help="Input CSV file with 'text' column")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV file for predictions")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to model directory")
    parser.add_argument("--text_column", type=str, default="text", help="Name of text column in input")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    logger.info(f"Reading input from {args.input_path}")
    df = pd.read_csv(args.input_path)
    
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found. Available: {df.columns.tolist()}")

    texts = df[args.text_column].tolist()
    logger.info(f"Processing {len(texts)} samples")

    predictions = predict_batch(texts, tokenizer, model, args.threshold)

    result_df = pd.DataFrame(predictions)
    result_df.to_csv(args.output_path, index=False)
    logger.info(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
