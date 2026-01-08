import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "ham", 1: "spam"}


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model) -> tuple[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = probs.argmax().item()
    confidence = probs[0, pred_id].item()
    return LABELS[pred_id], confidence


def show_examples(data_path: str, tokenizer, model, text_col: str, label_col: str, n_per_class: int = 3):
    df = pd.read_csv(data_path, nrows=100000)  # limit for speed
    
    # Normalize labels to int
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].map({"ham": 0, "spam": 1})

    print("=" * 60)
    print("EXAMPLES FROM DATASET")
    print("=" * 60)

    for label_id, label_name in LABELS.items():
        print(f"\n--- {label_name.upper()} examples ---")
        subset = df[df[label_col] == label_id]
        if len(subset) == 0:
            print("    No examples found")
            continue
        samples = subset.sample(n=min(n_per_class, len(subset)), random_state=42)
        for _, row in samples.iterrows():
            text = str(row[text_col])
            text_display = text[:100] + "..." if len(text) > 100 else text
            pred, conf = predict(text, tokenizer, model)
            status = "OK" if pred == label_name else "WRONG"
            print(f"[{status}] pred={pred} ({conf:.2%})")
            print(f"    {text_display}\n")


def interactive_mode(tokenizer, model):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)

    while True:
        text = input("\nEnter text: ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue
        pred, conf = predict(text, tokenizer, model)
        print(f"Prediction: {pred} ({conf:.2%})")


def main():
    parser = argparse.ArgumentParser(description="Spam classifier inference")
    parser.add_argument("--model", type=str, default="models/spam_classifier", help="Path to model")
    parser.add_argument("--data", type=str, default="data/russian_spam.csv", help="Path to data for examples")
    parser.add_argument("--text-col", type=str, default="message", help="Text column name")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument("--examples", type=int, default=3, help="Number of examples per class")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive mode")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    show_examples(args.data, tokenizer, model, args.text_col, args.label_col, args.examples)

    if not args.no_interactive:
        interactive_mode(tokenizer, model)


if __name__ == "__main__":
    main()
