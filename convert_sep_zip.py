"""SEP.parquet.zip → SEP.parquet (chunked, memory-safe)."""
import zipfile
import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
ZIP_PATH = DATA_DIR / 'SEP.parquet.zip'
OUT_PATH = DATA_DIR / 'SEP.parquet'
CHUNK_SIZE = 1_000_000  # rows per chunk

print(f'Source: {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.1f}MB)', flush=True)
t0 = time.time()

with zipfile.ZipFile(ZIP_PATH) as z:
    csv_name = z.namelist()[0]
    print(f'CSV inside zip: {csv_name}', flush=True)

    # Stream CSV in chunks, write parquet in append mode via pyarrow
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    total_rows = 0
    with z.open(csv_name) as f:
        for i, chunk in enumerate(pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False)):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT_PATH, table.schema, compression='snappy')
            writer.write_table(table)
            total_rows += len(chunk)
            print(f'  chunk {i+1}: cumulative rows {total_rows:,} ({(time.time()-t0):.0f}s)', flush=True)
    if writer:
        writer.close()

elapsed = time.time() - t0
size_mb = OUT_PATH.stat().st_size / 1e6
print(f'\nDone: {total_rows:,} rows in {elapsed:.0f}s, {size_mb:.1f}MB', flush=True)

# Cleanup zip
ZIP_PATH.unlink()
print('Removed zip.', flush=True)
