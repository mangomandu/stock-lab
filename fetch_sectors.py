"""Fetch sector info for all S&P 500 + ETF tickers from yfinance."""
import yfinance as yf
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

DATA_DIR = '/home/dlfnek/stock_lab/data'
TICKERS_PATH = os.path.join(DATA_DIR, 'all_sp500_tickers.txt')
OUTPUT_PATH = os.path.join(DATA_DIR, 'sectors.csv')

# ETF sector mapping (yfinance often doesn't return sector for ETFs)
ETF_SECTORS = {
    'GLD': 'Defensive', 'TLT': 'Defensive', 'SHY': 'Defensive',
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Consumer Defensive',
    'XLY': 'Consumer Cyclical', 'XLU': 'Utilities', 'XLC': 'Communication',
    'XLB': 'Materials', 'XLRE': 'Real Estate', 'ITA': 'Industrials',
    'SPY': 'Index', 'VXX': 'Volatility',
}


def fetch_sector(ticker):
    if ticker in ETF_SECTORS:
        return (ticker, ETF_SECTORS[ticker])
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector') or info.get('quoteType', 'Unknown')
        return (ticker, sector)
    except Exception as e:
        return (ticker, f'ERROR')


def main():
    with open(TICKERS_PATH) as f:
        tickers = [t.strip() for t in f if t.strip()]
    print(f"Fetching sectors for {len(tickers)} tickers...")

    sectors = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_sector, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            ticker, sector = future.result()
            sectors[ticker] = sector
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tickers)}] elapsed {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame.from_dict(sectors, orient='index', columns=['Sector'])
    df.index.name = 'Ticker'
    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"\nSector distribution:")
    print(df['Sector'].value_counts().to_string())


if __name__ == '__main__':
    main()
