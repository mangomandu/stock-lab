"""
Sharadar Bundle bulk download — one-shot ingestion for v0.8.0 reset.

Downloads 5 datasets:
  SF1     — Fundamentals (PIT, 30y, ~16k tickers)
  SEP     — Stock prices (incl delisted)
  TICKERS — Metadata (permaticker, isdelisted)
  SP500   — Historical S&P 500 changes
  DAILY   — Pre-computed daily ratios

Saves as parquet to data/sharadar/.
Idempotent: skips datasets already downloaded.

Runtime: ~10-30 min depending on connection (mostly SF1 + SEP).
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import nasdaqdatalink as ndl
import pandas as pd

load_dotenv()
ndl.ApiConfig.api_key = os.getenv('NASDAQ_DATA_LINK_API_KEY')

OUT_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # (table_code, out_name, filters, use_bulk_export)
    ('SHARADAR/TICKERS',  'TICKERS.parquet',  None,                    False),
    ('SHARADAR/SP500',    'SP500.parquet',    None,                    False),
    ('SHARADAR/ACTIONS',  'ACTIONS.parquet',  None,                    False),
    ('SHARADAR/SF1',      'SF1.parquet',      {'dimension': 'ARQ'},    False),
    ('SHARADAR/DAILY',    'DAILY.parquet',    None,                    True),   # Too large for get_table
    ('SHARADAR/SEP',      'SEP.parquet',      None,                    True),   # Too large for get_table
]


def download_one(table_code, out_name, filters=None, use_bulk=False):
    out_path = OUT_DIR / out_name
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f'  [SKIP] {table_code} already exists: {len(df):,} rows', flush=True)
        return df

    print(f'  [START] {table_code} -> {out_name} (bulk={use_bulk})', flush=True)
    t0 = time.time()

    if use_bulk:
        # Use export_table for large datasets — downloads zip → extract CSV → save as parquet
        zip_path = OUT_DIR / f'{out_name}.zip'
        ndl.export_table(table_code, filename=str(zip_path))

        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False)
        df.to_parquet(out_path)
        zip_path.unlink()  # remove zip after parquet saved
    else:
        kwargs = filters.copy() if filters else {}
        kwargs['paginate'] = True
        df = ndl.get_table(table_code, **kwargs)
        df.to_parquet(out_path)

    elapsed = time.time() - t0
    print(f'  [DONE]  {table_code}: {len(df):,} rows, {elapsed:.0f}s, {out_path.stat().st_size/1e6:.1f}MB',
          flush=True)
    return df


def verify_sentinels(loaded):
    """Sentinel checks — confirm critical assumptions."""
    print('\n=== Sentinel checks ===', flush=True)

    # 1. SEP must have LEHMQ (Lehman delisted)
    sep = loaded.get('SEP.parquet')
    if sep is not None:
        lh = sep[sep['ticker'] == 'LEHMQ']
        bankruptcy_day = lh[lh['date'] == '2008-09-15']
        ok = len(bankruptcy_day) == 1
        print(f'  SEP delisted (LEHMQ on 2008-09-15): {"OK" if ok else "FAIL"} ({len(lh)} total LEHMQ rows)')

    # 2. SP500 must have WAMUQ removal
    sp500 = loaded.get('SP500.parquet')
    if sp500 is not None:
        wamu_removed = sp500[(sp500['ticker'] == 'WAMUQ') & (sp500['action'] == 'removed')]
        ok = len(wamu_removed) >= 1
        print(f'  SP500 (WaMu removed): {"OK" if ok else "FAIL"} ({len(wamu_removed)} matching rows)')

    # 3. SF1 must have AAPL 2024 Q1 with datekey + pe
    sf1 = loaded.get('SF1.parquet')
    if sf1 is not None:
        aapl = sf1[(sf1['ticker'] == 'AAPL') & (sf1['calendardate'] == '2024-03-31')]
        ok = len(aapl) >= 1 and pd.notna(aapl.iloc[0].get('pe'))
        if ok:
            r = aapl.iloc[0]
            print(f'  SF1 PIT (AAPL 2024Q1 datekey={r["datekey"]}, pe={r["pe"]}): OK')
        else:
            print(f'  SF1 PIT: FAIL ({len(aapl)} rows)')

    # 4. TICKERS permaticker stable ID
    tickers = loaded.get('TICKERS.parquet')
    if tickers is not None:
        lehmq_perm = tickers[tickers['ticker'] == 'LEHMQ']
        if len(lehmq_perm) >= 1:
            print(f'  TICKERS (LEHMQ permaticker={lehmq_perm.iloc[0]["permaticker"]}): OK')


def main():
    print(f'Sharadar bulk download → {OUT_DIR}', flush=True)
    print(f'Datasets: {len(DATASETS)}', flush=True)
    print(f'API key: {ndl.ApiConfig.api_key[:10]}...', flush=True)
    print(flush=True)

    loaded = {}
    t_start = time.time()
    for table_code, out_name, filters, use_bulk in DATASETS:
        try:
            df = download_one(table_code, out_name, filters, use_bulk)
            loaded[out_name] = df
        except Exception as e:
            print(f'  [ERROR] {table_code}: {e}', flush=True)
            return 1

    print(f'\nTotal elapsed: {(time.time()-t_start)/60:.1f} min', flush=True)
    verify_sentinels(loaded)
    return 0


if __name__ == '__main__':
    sys.exit(main())
