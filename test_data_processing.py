import pytest
import pandas as pd
import numpy as np
from data_processing import load_avro_to_dataframe, preprocess_dataframe, generate_embeddings

# Sample AVRO file for testing
# Update this line in `test_data_processing.py`
TEST_AVRO_FILE = "/Users/pateltayyab/Downloads/news_data.avro"

@pytest.fixture
def sample_dataframe():
    """Load and preprocess the AVRO dataset for testing."""
    df = load_avro_to_dataframe(TEST_AVRO_FILE)
    df = preprocess_dataframe(df)
    return df

def test_load_avro(sample_dataframe):
    """Test if AVRO file is correctly loaded into a DataFrame."""
    assert isinstance(sample_dataframe, pd.DataFrame)
    assert not sample_dataframe.empty

def test_data_types(sample_dataframe):
    """Test if important fields have the correct data types."""
    assert sample_dataframe["publication_datetime"].dtype == "datetime64[ns]"
    assert sample_dataframe["word_count"].dtype == np.int64

def test_null_handling(sample_dataframe):
    """Ensure no null values exist after processing."""
    assert sample_dataframe.isnull().sum().sum() == 0

def test_content_text_column(sample_dataframe):
    """Check if 'content_text' correctly combines 'snippet' and 'body'."""
    assert "content_text" in sample_dataframe.columns
    for index, row in sample_dataframe.iterrows():
        expected_text = (row["snippet"] or "") + " " + (row["body"] or "")
        assert row["content_text"].strip() == expected_text.strip()

def test_embedding_generation(sample_dataframe):
    """Check if embeddings are generated correctly."""
    df_with_embeddings = generate_embeddings(sample_dataframe)
    
    assert "embedding" in df_with_embeddings.columns
    assert df_with_embeddings["embedding"].apply(lambda x: isinstance(x, list) and len(x) > 0).all()

if __name__ == "__main__":
    pytest.main()
