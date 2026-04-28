"""Download S&P 500 + ETF lifetime daily data via yfinance.

Saves each ticker to data/master_sp500/{TICKER}.csv with columns:
    Datetime,Open,High,Low,Close,Volume

Uses ThreadPoolExecutor for parallel downloads but with rate limit awareness.
Skips tickers already downloaded successfully.
"""
import yfinance as yf
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
LIST_PATH = os.path.join(DATA_DIR, 'all_tickers.txt')

MAX_WORKERS = 8        # parallel downloads
RETRY_ATTEMPTS = 2     # retries on failure


def fetch_ticker(ticker):
    """Download and save one ticker. Returns (ticker, status, n_rows, first, last)."""
    out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(out_path):
        # Check if non-empty
        try:
            df = pd.read_csv(out_path, nrows=2)
            if len(df) > 0:
                return (ticker, 'CACHED', None, None, None)
        except Exception:
            pass

    for attempt in range(RETRY_ATTEMPTS):
        try:
            t_obj = yf.Ticker(ticker)
            df = t_obj.history(period="max", interval="1d", auto_adjust=True)
            if df.empty:
                return (ticker, 'EMPTY', 0, None, None)

            df = df.reset_index()
            # Keep only columns we need; ensure naive datetime
            keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[c for c in keep_cols if c in df.columns]]
            if 'Date' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df = df.drop(columns=['Date'])
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.to_csv(out_path, index=False)

            return (ticker, 'OK', len(df),
                    df['Datetime'].iloc[0].date(),
                    df['Datetime'].iloc[-1].date())
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(1.0 + attempt)
                continue
            return (ticker, f'ERROR: {str(e)[:60]}', None, None, None)


def main():
    with open(LIST_PATH, 'r') as f:
        tickers = [t.strip() for t in f if t.strip()]
    print(f"Total tickers to download: {len(tickers)}")

    t0 = time.time()
    counts = {'OK': 0, 'CACHED': 0, 'EMPTY': 0, 'ERROR': 0}
    progress = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, status, n, first, last = future.result()
                if status == 'OK':
                    counts['OK'] += 1
                    progress += 1
                    print(f"  [{progress}/{len(tickers)}] {ticker}: {n} rows ({first} ~ {last})", flush=True)
                elif status == 'CACHED':
                    counts['CACHED'] += 1
                    progress += 1
                elif status == 'EMPTY':
                    counts['EMPTY'] += 1
                    progress += 1
                    print(f"  [{progress}/{len(tickers)}] {ticker}: EMPTY (delisted?)", flush=True)
                else:
                    counts['ERROR'] += 1
                    progress += 1
                    print(f"  [{progress}/{len(tickers)}] {ticker}: {status}", flush=True)
            except Exception as e:
                counts['ERROR'] += 1
                print(f"  {ticker}: future exception {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  OK: {counts['OK']}")
    print(f"  CACHED: {counts['CACHED']}")
    print(f"  EMPTY: {counts['EMPTY']}")
    print(f"  ERROR: {counts['ERROR']}")
    print(f"  Total successful: {counts['OK'] + counts['CACHED']}/{len(tickers)}")


if __name__ == '__main__':
    main()
