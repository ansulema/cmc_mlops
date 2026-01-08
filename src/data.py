import pandas as pd
from sklearn.model_selection import train_test_split


def validate_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> None:
    """Validate dataframe structure and types."""
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    if df[text_col].isnull().any():
        raise ValueError("Text column contains null values")
    if df[label_col].isnull().any():
        raise ValueError("Label column contains null values")


def validate_labels(labels: pd.Series) -> None:
    """Validate label values are 0 or 1."""
    unique = set(labels.unique())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1, got: {unique}")


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Clean and preprocess data."""
    df = df[[label_col, text_col]].dropna().drop_duplicates()
    
    # Map string labels to int if needed
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].map({"ham": 0, "spam": 1})
    
    return df


def sample_data(df: pd.DataFrame, label_col: str, max_samples: int, seed: int) -> pd.DataFrame:
    """Sample data proportionally."""
    if len(df) <= max_samples:
        return df
    frac = max_samples / len(df)
    return df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=seed), include_groups=False
    ).reset_index(level=0, drop=False)


def split_data(df: pd.DataFrame, label_col: str, test_size: float, val_ratio: float, seed: int):
    """Split data into train/val/test."""
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio, stratify=temp_df[label_col], random_state=seed
    )
    return train_df, val_df, test_df
