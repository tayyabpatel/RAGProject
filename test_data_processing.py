import pytest
import pandas as pd
from data_processing import load_avro_to_dataframe, preprocess_dataframe  # Removed generate_embeddings

# Define a sample AVRO file path for testing
AVRO_FILE_PATH = "/Users/pateltayyab/Downloads/news_data.avro"

def test_load_avro_to_dataframe():
    df = load_avro_to_dataframe(AVRO_FILE_PATH)
    assert isinstance(df, pd.DataFrame), "The output should be a Pandas DataFrame"
    assert not df.empty, "The DataFrame should not be empty"

def test_preprocess_dataframe():
    df = load_avro_to_dataframe(AVRO_FILE_PATH)
    df = preprocess_dataframe(df)

    assert "content_text" in df.columns, "Column 'content_text' should be present after preprocessing"
    assert df["content_text"].isna().sum() == 0, "There should be no null values in 'content_text'"
    assert "snippet" not in df.columns, "'snippet' column should be dropped"
    assert "body" not in df.columns, "'body' column should be dropped"

if __name__ == "__main__":
    pytest.main()
