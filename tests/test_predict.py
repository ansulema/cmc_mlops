import pytest
import numpy as np
import sys
sys.path.insert(0, "..")

from src.predict import (
    logits_to_probs,
    probs_to_class,
    class_to_label,
    LABELS,
)


class TestLogitsToProbs:
    def test_sums_to_one(self):
        logits = np.array([1.0, 2.0])
        probs = logits_to_probs(logits)
        assert np.isclose(probs.sum(), 1.0)

    def test_higher_logit_higher_prob(self):
        logits = np.array([1.0, 3.0])
        probs = logits_to_probs(logits)
        assert probs[1] > probs[0]

    def test_batch_processing(self):
        logits = np.array([[1.0, 2.0], [3.0, 1.0]])
        probs = logits_to_probs(logits)
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), [1.0, 1.0])


class TestProbsToClass:
    def test_threshold_default(self):
        assert probs_to_class(np.array([0.3, 0.7])) == 1
        assert probs_to_class(np.array([0.6, 0.4])) == 0

    def test_threshold_boundary(self):
        assert probs_to_class(np.array([0.5, 0.5])) == 1  # >= threshold
        assert probs_to_class(np.array([0.51, 0.49])) == 0

    def test_custom_threshold(self):
        probs = np.array([0.2, 0.8])
        assert probs_to_class(probs, threshold=0.9) == 0
        assert probs_to_class(probs, threshold=0.7) == 1


class TestClassToLabel:
    def test_valid_classes(self):
        assert class_to_label(0) == "ham"
        assert class_to_label(1) == "spam"

    def test_invalid_class(self):
        with pytest.raises(ValueError, match="Invalid class_id"):
            class_to_label(2)

    def test_labels_mapping(self):
        assert LABELS == {0: "ham", 1: "spam"}
