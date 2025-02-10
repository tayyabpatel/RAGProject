import pytest
import pandas as pd
from data_processing import load_avro_to_dataframe, preprocess_dataframe  

AVRO_FILE_PATH = "/Users/pateltayyab/Downloads/news_data5000.avro"

def test_load_avro_to_dataframe():
    df = load_avro_to_dataframe(AVRO_FILE_PATH)
    assert isinstance(df, pd.DataFrame), "Output should be a Pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"

def test_preprocess_dataframe():
    df = load_avro_to_dataframe(AVRO_FILE_PATH)
    df = preprocess_dataframe(df)

    assert "content_text" in df.columns, "Column 'content_text' should be present"
    assert df["content_text"].isna().sum() == 0, "No null values in 'content_text'"
    assert "snippet" not in df.columns, "'snippet' should be dropped"
    assert "body" not in df.columns, "'body' should be dropped"

if __name__ == "__main__":
    pytest.main()
