import pandas as pd
import fastavro

def avro_to_dataframe(avro_file_path):
    # Open the AVRO file and read records
    with open(avro_file_path, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]  # Convert to list of dictionaries

    # Convert to Pandas DataFrame
    df = pd.DataFrame(records)
    
    return df

# Example usage
avro_file = "sample.avro"  # Replace with your actual AVRO file path
df = avro_to_dataframe(avro_file)

# Display DataFrame Schema
print("Schema of DataFrame:")
print(df.dtypes)
