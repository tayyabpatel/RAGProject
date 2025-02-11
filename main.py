import os
from data_processing import load_avro_to_dataframe, preprocess_dataframe  # Updated function imports
from embeddings import generate_article_embeddings
from vector_database import create_collection, insert_vectors
import uvicorn
from fastapi import FastAPI
from api import app  # Import FastAPI app from api.py

AVRO_FILE = "/Users/pateltayyab/Downloads/news_data100.avro"

# Ensure Qdrant collection exists
create_collection()

# Check if AVRO file exists before proceeding
if not os.path.exists(AVRO_FILE):
    print(f"❌ Error: AVRO file '{AVRO_FILE}' not found.")
    exit(1)

print("📥 Processing AVRO file...")
df = load_avro_to_dataframe(AVRO_FILE)  # Use correct function
if df is None or df.empty:
    print("❌ Error: Data loading failed or returned an empty DataFrame. Check AVRO file.")
    exit(1)

df = preprocess_dataframe(df)  # Preprocess the DataFrame

print("🧠 Generating embeddings in batches...")
df = generate_article_embeddings(df, batch_size=16)  # Lower batch size for safety

if "embedding" not in df.columns:
    print("❌ Error: Embeddings generation failed.")
    exit(1)

print("📤 Inserting embeddings into Qdrant...")
insert_vectors(df)

print("✅ Pipeline execution complete!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
