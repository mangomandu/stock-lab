"""Tier 1 integrity check: re-fetch sample tickers via yfinance and diff vs cached CSVs.

Detects:
- File corruption / truncation (cached_rows vs fresh_rows)
- Stale data (fresh has newer dates than cached)
- Anomaly: dates in cached but missing from fresh (possible ghost rows)
- Price mismatches on common dates (split/dividend re-adjustment or save errors)
"""
import yfinance as yf
import pandas as pd
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
SAMPLE_SIZE = 20
SEED = 42
RTOL = 1e-3            # 0.1% relative tolerance for OHLC
VOL_RTOL = 5e-3        # 0.5% for volume (yfinance can revise)
MAX_WORKERS = 4


def load_cached(ticker):
    p = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(p)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['Datetime']).set_index('Datetime').sort_index()
    df.index = df.index.normalize()  # date-only
    return df


def fetch_fresh(ticker):
    df = yf.Ticker(ticker).history(period='max', interval='1d', auto_adjust=True)
    if df.empty:
        return None
    df = df.reset_index()
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df['Datetime'] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df.set_index('Datetime').sort_index()


def compare_ticker(ticker):
    try:
        cached = load_cached(ticker)
    except Exception as e:
        return {'ticker': ticker, 'error': f'cached load: {e}'}
    try:
        fresh = fetch_fresh(ticker)
    except Exception as e:
        return {'ticker': ticker, 'error': f'fresh fetch: {e}'}
    if fresh is None:
        return {'ticker': ticker, 'error': 'fresh empty'}

    cached_dates = set(cached.index)
    fresh_dates = set(fresh.index)
    common = sorted(cached_dates & fresh_dates)
    only_cached = sorted(cached_dates - fresh_dates)
    only_fresh = sorted(fresh_dates - cached_dates)

    cc = cached.loc[common]
    ff = fresh.loc[common]

    mismatches = {}
    for col, tol in [('Open', RTOL), ('High', RTOL), ('Low', RTOL),
                     ('Close', RTOL), ('Volume', VOL_RTOL)]:
        denom = ff[col].abs().clip(lower=1e-9)
        rel = (cc[col] - ff[col]).abs() / denom
        bad = rel > tol
        if bad.any():
            mismatches[col] = {
                'n': int(bad.sum()),
                'max_rel': float(rel.max()),
                'first': str(cc.index[bad.values][0].date()),
                'last': str(cc.index[bad.values][-1].date()),
            }

    return {
        'ticker': ticker,
        'cached_rows': len(cached),
        'fresh_rows': len(fresh),
        'cached_first': str(cached.index.min().date()),
        'cached_last': str(cached.index.max().date()),
        'fresh_first': str(fresh.index.min().date()),
        'fresh_last': str(fresh.index.max().date()),
        'common': len(common),
        'only_cached_n': len(only_cached),
        'only_cached_sample': [str(d.date()) for d in only_cached[:5]],
        'only_fresh_n': len(only_fresh),
        'only_fresh_first': str(only_fresh[0].date()) if only_fresh else None,
        'only_fresh_last': str(only_fresh[-1].date()) if only_fresh else None,
        'mismatches': mismatches,
    }


def main():
    all_files = sorted(f for f in os.listdir(DATA_DIR)
                       if f.endswith('.csv') and not f.startswith('data_validation'))
    all_tickers = [f.replace('.csv', '') for f in all_files]

    random.seed(SEED)
    sample = random.sample(all_tickers, min(SAMPLE_SIZE, len(all_tickers)))
    must_include = ['SPY', 'AAPL', 'MSFT']
    for t in must_include:
        if t in all_tickers and t not in sample:
            sample.insert(0, t)

    print(f"Total tickers in cache: {len(all_tickers)}")
    print(f"Sample size: {len(sample)}")
    print(f"Tickers: {sample}")
    print('-' * 70)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(compare_ticker, t): t for t in sample}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            t = r['ticker']
            if 'error' in r:
                print(f"[{t:6}] ERROR: {r['error']}")
                continue
            stale = (pd.Timestamp(r['fresh_last']) - pd.Timestamp(r['cached_last'])).days
            tags = []
            if r['mismatches']:
                tags.append('PRICE_MISMATCH')
            if r['only_cached_n']:
                tags.append(f"GHOST({r['only_cached_n']})")
            if stale > 0:
                tags.append(f"STALE({stale}d,{r['only_fresh_n']}new)")
            status = ' '.join(tags) if tags else 'OK'
            print(f"[{t:6}] {status:30} cached={r['cached_rows']:>5} fresh={r['fresh_rows']:>5} "
                  f"common={r['common']:>5} cachedLast={r['cached_last']} freshLast={r['fresh_last']}")
            for col, m in r['mismatches'].items():
                print(f"          {col}: n={m['n']} maxRel={m['max_rel']:.4f} ({m['first']}~{m['last']})")
            if r['only_cached_n']:
                print(f"          ghosts (cached but not in fresh): sample={r['only_cached_sample']}")

    print('-' * 70)
    print('SUMMARY')
    n_clean = sum(1 for r in results if 'error' not in r and not r['mismatches']
                  and not r['only_cached_n'] and r['cached_last'] == r['fresh_last'])
    n_stale = sum(1 for r in results if 'error' not in r and r['only_fresh_n'])
    n_mismatch = sum(1 for r in results if r.get('mismatches'))
    n_ghost = sum(1 for r in results if r.get('only_cached_n'))
    n_err = sum(1 for r in results if 'error' in r)
    print(f"  Clean (perfect):        {n_clean}/{len(results)}")
    print(f"  Stale (newer available):{n_stale}/{len(results)}")
    print(f"  Price mismatches:       {n_mismatch}/{len(results)}")
    print(f"  Ghost rows (cached>fresh): {n_ghost}/{len(results)}")
    print(f"  Errors:                 {n_err}/{len(results)}")

    sys.exit(1 if (n_mismatch or n_ghost or n_err) else 0)


if __name__ == '__main__':
    main()
