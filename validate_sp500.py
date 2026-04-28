"""Validate S&P 500 + ETF data after download."""
import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.csv'))
    print(f"[{datetime.now()}] S&P 500 + ETF Data Validation")
    print(f"  Files in directory: {len(files)}")

    # Read each file
    stats = []
    empty = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        ticker = f.replace('.csv', '')
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                empty.append(ticker)
                continue
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            df = df.dropna(subset=['Datetime'])
            stats.append({
                'ticker': ticker,
                'n_rows': len(df),
                'first_date': df['Datetime'].min(),
                'last_date': df['Datetime'].max(),
            })
        except Exception as e:
            empty.append(ticker)
            print(f"  Error reading {ticker}: {e}")

    df_stats = pd.DataFrame(stats)
    print(f"\n  Valid tickers: {len(df_stats)}")
    print(f"  Empty/error tickers: {len(empty)}")
    if empty:
        print(f"  Empty list: {sorted(empty)[:30]}{'...' if len(empty) > 30 else ''}")

    if len(df_stats) == 0:
        return

    # Date range stats
    df_stats['ipo_year'] = df_stats['first_date'].dt.year
    df_stats['last_year'] = df_stats['last_date'].dt.year

    print(f"\n## IPO year distribution")
    ipo_dist = df_stats['ipo_year'].value_counts().sort_index()
    decades = pd.cut(df_stats['ipo_year'],
                     bins=[1900, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
                     labels=['<1970', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'])
    print(decades.value_counts().sort_index().to_string())

    print(f"\n## Last date distribution")
    print(f"  Most recent: {df_stats['last_date'].max().date()}")
    print(f"  Oldest 'last': {df_stats['last_date'].min().date()}")
    stale_threshold = df_stats['last_date'].max() - pd.Timedelta(days=30)
    stale = df_stats[df_stats['last_date'] < stale_threshold]
    if len(stale) > 0:
        print(f"  Stale (>30 days old): {len(stale)} tickers")
        print(f"    {stale['ticker'].tolist()[:10]}")

    print(f"\n## Universe size by year (alive ticker count)")
    for year in [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]:
        cutoff = pd.Timestamp(f'{year}-01-01')
        alive = ((df_stats['first_date'] < cutoff) &
                 (df_stats['last_date'] >= cutoff)).sum()
        print(f"  {year}: {alive} tickers alive")

    print(f"\n## Row count distribution")
    print(f"  Mean rows: {df_stats['n_rows'].mean():.0f}")
    print(f"  Median rows: {df_stats['n_rows'].median():.0f}")
    print(f"  Min: {df_stats['n_rows'].min()} ({df_stats.loc[df_stats['n_rows'].idxmin(), 'ticker']})")
    print(f"  Max: {df_stats['n_rows'].max()} ({df_stats.loc[df_stats['n_rows'].idxmax(), 'ticker']})")

    # Save
    out_path = os.path.join(DATA_DIR, 'data_validation.csv')
    df_stats.to_csv(out_path, index=False)
    print(f"\n  Stats saved: {out_path}")


if __name__ == '__main__':
    main()
