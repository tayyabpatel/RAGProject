import pandas as pd
import numpy as np
from tqdm import tqdm

# Commented out JinaAI Model (for now)
# import torch
# from transformers import AutoModel, AutoTokenizer

# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

def generate_dummy_embedding():
    """Returns a random array of 1024 float numbers to simulate embeddings."""
    return np.random.rand(1024).tolist()

def generate_article_embeddings(df, batch_size=8, max_length=512):
    """
    Generates dummy embeddings for news articles instead of JinaAI model.
    """
    if "full_text" not in df.columns:
        print("❌ Error: 'full_text' column not found.")
        return df

    embeddings = []
    for _ in tqdm(range(len(df)), desc="Generating dummy embeddings"):
        embeddings.append(generate_dummy_embedding())

    df["embedding"] = embeddings
    return df

def generate_query_embedding(query, max_length=512):
    """
    Generates a dummy embedding for an incoming search query.
    """
    return generate_dummy_embedding()
