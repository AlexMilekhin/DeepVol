import pandas as pd
from sqlalchemy import create_engine
import pyarrow as pa
import pyarrow.parquet as pq
import os

# 1. Connection Settings
# Make sure 'dolt sql-server' is running in your VS Code terminal
DB_URL = "mysql+mysqlconnector://root@127.0.0.1:3306/options"
engine = create_engine(DB_URL)

# 2. Extraction Settings
# Change this to any ticker from your list (ADBE, NFLX, etc.)
TICKER = 'ADBE' 
OUTPUT_FILE = f"{TICKER}.lower()_full_history.parquet"
CHUNK_SIZE = 50000 

print(f"Checking database for {TICKER}...")

# Initialize tracking variables
writer = None
total_rows = 0

try:
    # Pre-check: Verify row count before starting loop
    count_df = pd.read_sql(f"SELECT COUNT(*) as cnt FROM option_chain WHERE act_symbol = '{TICKER}'", engine)
    expected_rows = count_df['cnt'].iloc[0]

    if expected_rows == 0:
        print(f"Error: No data found for {TICKER}. Check if it needs to be uppercase/lowercase.")
    else:
        print(f"Found {expected_rows} rows. Starting chunked extraction...")

        # SQL Query
        query = f"SELECT * FROM option_chain WHERE act_symbol = '{TICKER}' ORDER BY date ASC"
        
        # Stream the data in chunks from SQL
        query_chunks = pd.read_sql(query, engine, chunksize=CHUNK_SIZE)

        for i, chunk in enumerate(query_chunks):
            # Clean/Standardize types for the Parquet schema
            chunk['date'] = pd.to_datetime(chunk['date'])
            chunk['expiration'] = pd.to_datetime(chunk['expiration'])
            
            # Optional: Calculate mid-price or other Greeks locally if missing
            if 'bid' in chunk.columns and 'ask' in chunk.columns:
                chunk['mid'] = (chunk['bid'] + chunk['ask']) / 2

            # Convert Pandas chunk to PyArrow Table
            table = pa.Table.from_pandas(chunk)

            # Initialize the writer on the very first chunk to lock the schema
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression='snappy')

            # Write this chunk to the file
            writer.write_table(table)
            
            total_rows += len(chunk)
            print(f"Written chunk {i+1} ({total_rows}/{expected_rows} rows)...")

        print(f"\nSuccess! Saved {total_rows} rows to {OUTPUT_FILE}")

except Exception as e:
    print(f"Extraction failed: {e}")

finally:
    # Safely close the writer even if the loop fails half-way
    if writer:
        writer.close()
        print("File handle closed.")