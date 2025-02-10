import pandas as pd
from data_processing import load_avro_to_dataframe, preprocess_dataframe
from embeddings import generate_article_embeddings, generate_query_embedding
from vector_database import create_collection, insert_embeddings, search_vectors


# Path to your AVRO data file
AVRO_FILE_PATH = "/Users/pateltayyab/Downloads/news_data.avro"

# 1️⃣ Create Qdrant Collection
create_collection()

# 2️⃣ Load and Process Data
print("📥 Loading AVRO file into DataFrame...")
df = load_avro_to_dataframe(AVRO_FILE_PATH)
df = preprocess_dataframe(df)
print(f"✅ DataFrame loaded with {len(df)} records.")

# 3️⃣ Generate Embeddings
print("🧠 Generating embeddings for news articles...")
df = generate_article_embeddings(df)

# 4️⃣ Insert Embeddings into Qdrant
print("📡 Inserting data into Qdrant...")
insert_embeddings(df)

# 5️⃣ Perform a Similarity Search
query = "latest renewable energy policies"
query_embedding = generate_query_embedding(query)
results = search_articles(query_embedding, top_k=5)

# 6️⃣ Display Search Results
print("\n🔍 **Search Results:**")
for i, res in enumerate(results):
    print(f"{i+1}. {res['title']} (Score: {res['score']:.4f})")
    print(f"   {res['content'][:300]}...\n")  # Show first 300 chars of content
