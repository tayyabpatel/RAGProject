from data_processing import process_avro
from embeddings import generate_article_embeddings, generate_query_embedding
from vector_database import create_collection, insert_vectors, search_vectors

AVRO_FILE_PATH = "/Users/pateltayyab/Downloads/news_data.avro"

# Create Qdrant collection
create_collection()

print("Processing AVRO file...")
df = process_avro(AVRO_FILE_PATH)

if df is not None:
    print("Generating embeddings...")
    df = generate_article_embeddings(df)
    
    print("Inserting into Qdrant...")
    insert_vectors(df)

    print("✅ Process completed successfully.")
else:
    print("❌ Error: Data processing failed. Ensure the AVRO file is valid.")
