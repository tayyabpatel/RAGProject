import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert_datetime_column(df, column_name, unit="ms"):
    """
    Converts timestamp columns into readable datetime format.
    """
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
        df[column_name] = df[column_name].where(df[column_name] > 0)  # Ignore invalid timestamps
        df[column_name] = pd.to_datetime(df[column_name], unit=unit, errors="coerce")
    return df

def preprocess_dataframe(df):
    """
    Prepares DataFrame for embedding:
    - Converts timestamps
    - Merges text fields
    - Cleans missing values
    """
    df.fillna("", inplace=True)

    # Convert timestamps
    date_columns = ["publication_datetime", "modification_datetime"]
    for col in date_columns:
        df = convert_datetime_column(df, col, unit="ms")

    # Merge text fields
    df["full_text"] = df["title"] + " " + df["snippet"] + " " + df["body"]

    return df
