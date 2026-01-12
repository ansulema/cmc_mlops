"""Custom TorchServe handler for DistilBERT spam classifier."""
import json
import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

LABELS = {0: "ham", 1: "spam"}


class SpamClassifierHandler:
    """Handler for spam classification model."""

    def __init__(self):
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = 128

    def initialize(self, context):
        """Load model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get model directory from context
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        logger.info(f"Loading model from {model_dir}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True
        logger.info("Model loaded successfully")

    def preprocess(self, requests):
        """Tokenize input texts."""
        texts = []
        for request in requests:
            data = request.get("data") or request.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    # Plain text input
                    data = {"text": data}
            
            text = data.get("text", data.get("message", str(data)))
            texts.append(text)
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def inference(self, inputs):
        """Run model inference."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs

    def postprocess(self, probs):
        """Convert probabilities to response."""
        results = []
        for prob in probs:
            pred_id = prob.argmax().item()
            results.append({
                "label": LABELS[pred_id],
                "class_id": pred_id,
                "confidence": round(prob[pred_id].item(), 4),
                "prob_ham": round(prob[0].item(), 4),
                "prob_spam": round(prob[1].item(), 4)
            })
        return results

    def handle(self, requests, context):
        """Main entry point for TorchServe."""
        if not self.initialized:
            self.initialize(context)
        
        # Handle initialization call (requests is None)
        if requests is None:
            return None
        
        inputs = self.preprocess(requests)
        probs = self.inference(inputs)
        return self.postprocess(probs)


_service = SpamClassifierHandler()


def handle(data, context):
    """Entry point for TorchServe."""
    return _service.handle(data, context)
