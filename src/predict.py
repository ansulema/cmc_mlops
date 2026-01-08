import numpy as np

LABELS = {0: "ham", 1: "spam"}


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


def predict(text: str, tokenizer, model, threshold: float = 0.5) -> dict:
    """Full prediction pipeline."""
    import torch
    
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
        "probs": {"ham": float(probs[0]), "spam": float(probs[1])},
    }
