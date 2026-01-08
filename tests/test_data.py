import pytest
import pandas as pd
import sys
sys.path.insert(0, "..")

from src.data import (
    validate_dataframe,
    validate_labels,
    preprocess_data,
    sample_data,
    split_data,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text": ["hello", "spam message", "normal text", "buy now"],
        "label": [0, 1, 0, 1],
    })


@pytest.fixture
def string_labels_df():
    return pd.DataFrame({
        "text": ["hello", "spam"],
        "label": ["ham", "spam"],
    })


class TestValidateDataframe:
    def test_valid_df(self, sample_df):
        validate_dataframe(sample_df, "text", "label")  # should not raise

    def test_missing_text_column(self, sample_df):
        with pytest.raises(ValueError, match="Missing text column"):
            validate_dataframe(sample_df, "missing", "label")

    def test_missing_label_column(self, sample_df):
        with pytest.raises(ValueError, match="Missing label column"):
            validate_dataframe(sample_df, "text", "missing")

    def test_null_text(self):
        df = pd.DataFrame({"text": ["a", None], "label": [0, 1]})
        with pytest.raises(ValueError, match="null values"):
            validate_dataframe(df, "text", "label")


class TestValidateLabels:
    def test_valid_labels(self):
        labels = pd.Series([0, 1, 0, 1])
        validate_labels(labels)  # should not raise

    def test_invalid_labels(self):
        labels = pd.Series([0, 1, 2])
        with pytest.raises(ValueError, match="Labels must be 0 or 1"):
            validate_labels(labels)


class TestPreprocessData:
    def test_keeps_columns(self, sample_df):
        result = preprocess_data(sample_df, "text", "label")
        assert list(result.columns) == ["label", "text"]

    def test_maps_string_labels(self, string_labels_df):
        result = preprocess_data(string_labels_df, "text", "label")
        assert set(result["label"].unique()) == {0, 1}

    def test_drops_duplicates(self):
        df = pd.DataFrame({
            "text": ["a", "a", "b"],
            "label": [0, 0, 1],
        })
        result = preprocess_data(df, "text", "label")
        assert len(result) == 2


class TestSampleData:
    def test_no_sampling_if_small(self, sample_df):
        result = sample_data(sample_df, "label", max_samples=100, seed=42)
        assert len(result) == len(sample_df)

    def test_sampling_reduces_size(self):
        df = pd.DataFrame({
            "text": [f"text{i}" for i in range(1000)],
            "label": [0] * 500 + [1] * 500,
        })
        result = sample_data(df, "label", max_samples=100, seed=42)
        assert len(result) < len(df)


class TestSplitData:
    def test_split_sizes(self, sample_df):
        # Need more samples for stratified split
        df = pd.DataFrame({
            "text": [f"text{i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50,
        })
        train, val, test = split_data(df, "label", test_size=0.2, val_ratio=0.5, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_stratification(self):
        df = pd.DataFrame({
            "text": [f"text{i}" for i in range(100)],
            "label": [0] * 80 + [1] * 20,
        })
        train, val, test = split_data(df, "label", test_size=0.2, val_ratio=0.5, seed=42)
        # Check proportions are roughly maintained
        train_ratio = train["label"].mean()
        assert 0.15 < train_ratio < 0.25
