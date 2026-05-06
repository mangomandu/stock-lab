"""
SEP → wide close/volume panel (S&P 500 historical members only).

Input:  data/sharadar/{SEP, SP500}.parquet
Output: data/panels/{sp500_close, sp500_volume}.parquet  (date × ticker wide)

Memory-safe: filter SEP to S&P 500 historical members FIRST (drops ~95% of rows),
then pivot. ~1500 tickers × 7000 days = manageable wide panel.

S&P 500 universe = union of all historical members (added/historical events).
Per-date membership filtering happens at backtest time, not here.
"""
import gc
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/data/panels')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_sp500_universe():
    """Union of all S&P 500 historical/added/removed tickers."""
    sp500 = pd.read_parquet(DATA_DIR / 'SP500.parquet')
    universe = set(sp500['ticker'].dropna().unique())
    universe.discard('N/A')
    return universe


def build_panels():
    print('=== Building S&P 500 wide panels ===', flush=True)
    universe = get_sp500_universe()
    print(f'S&P 500 historical universe: {len(universe):,} tickers', flush=True)

    # Filter SEP to universe (memory-safe — uses parquet filter pushdown)
    print('Loading filtered SEP...', flush=True)
    sep = pd.read_parquet(
        DATA_DIR / 'SEP.parquet',
        columns=['ticker', 'date', 'closeadj', 'volume'],
        filters=[('ticker', 'in', list(universe))],
    )
    sep['date'] = pd.to_datetime(sep['date'])
    print(f'SEP filtered: {len(sep):,} rows', flush=True)

    # Pivot wide
    print('Pivoting close...', flush=True)
    close = sep.pivot_table(index='date', columns='ticker', values='closeadj', aggfunc='first')
    close = close.sort_index()
    print(f'Close panel: {close.shape}', flush=True)

    print('Pivoting volume...', flush=True)
    volume = sep.pivot_table(index='date', columns='ticker', values='volume', aggfunc='first')
    volume = volume.sort_index()
    print(f'Volume panel: {volume.shape}', flush=True)

    del sep
    gc.collect()

    # Save
    close.to_parquet(OUT_DIR / 'sp500_close.parquet')
    volume.to_parquet(OUT_DIR / 'sp500_volume.parquet')
    print(f'Saved → {OUT_DIR}', flush=True)

    # Verify sentinels
    print('\n=== Sentinel checks ===', flush=True)
    print(f'Date range: {close.index.min().date()} ~ {close.index.max().date()}', flush=True)
    print(f'AAPL dates: {close["AAPL"].notna().sum():,}', flush=True)
    print(f'LEHMQ dates: {close["LEHMQ"].notna().sum() if "LEHMQ" in close else "MISSING"}', flush=True)
    leh_bk = close.loc['2008-09-15', 'LEHMQ'] if 'LEHMQ' in close else None
    print(f'LEHMQ on 2008-09-15: {leh_bk}', flush=True)


if __name__ == '__main__':
    build_panels()
