"""Refresh the last N trading days for every cached ticker.

Re-fetches recent history from yfinance and overwrites overlapping rows in cache.
Older rows are preserved.

Use case: the last day's data was downloaded mid-trading and is stale.
"""
import yfinance as yf
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
LOOKBACK_DAYS = 7         # fetch recent N calendar days
MAX_WORKERS = 6


def refresh(ticker):
    cached_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(cached_path):
        return (ticker, 'NO_CACHE', 0, 0)
    try:
        cached = pd.read_csv(cached_path)
        cached['Datetime'] = pd.to_datetime(cached['Datetime']).dt.tz_localize(None)

        fresh = yf.Ticker(ticker).history(
            period=f'{LOOKBACK_DAYS}d', interval='1d', auto_adjust=True)
        if fresh.empty:
            return (ticker, 'FRESH_EMPTY', 0, 0)
        fresh = fresh.reset_index()
        date_col = 'Date' if 'Date' in fresh.columns else 'Datetime'
        fresh['Datetime'] = pd.to_datetime(fresh[date_col]).dt.tz_localize(None)
        fresh = fresh[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Drop cached rows that overlap with fresh window, then concat
        overlap_start = fresh['Datetime'].min()
        kept = cached[cached['Datetime'] < overlap_start]
        merged = pd.concat([kept, fresh], ignore_index=True)
        merged = merged.sort_values('Datetime').drop_duplicates(
            subset='Datetime', keep='last').reset_index(drop=True)

        # Sanity: merged must have >= cached rows
        if len(merged) < len(cached):
            return (ticker, f'SHRINK({len(cached)}->{len(merged)})', 0, 0)

        merged.to_csv(cached_path, index=False)
        n_new = len(merged) - len(cached)
        n_overwritten = len(fresh) - n_new
        return (ticker, 'OK', n_new, n_overwritten)
    except Exception as e:
        return (ticker, f'ERROR: {str(e)[:80]}', 0, 0)


def main():
    tickers = sorted(f.replace('.csv', '') for f in os.listdir(DATA_DIR)
                     if f.endswith('.csv') and not f.startswith('data_validation'))
    print(f'Refreshing last {LOOKBACK_DAYS} days for {len(tickers)} tickers...')
    t0 = time.time()
    counts = {'OK': 0, 'ERROR': 0, 'OTHER': 0}
    new_total = 0
    overwritten_total = 0
    issues = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(refresh, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            ticker, status, new, ovw = f.result()
            done += 1
            if status == 'OK':
                counts['OK'] += 1
                new_total += new
                overwritten_total += ovw
            elif status.startswith('ERROR'):
                counts['ERROR'] += 1
                issues.append((ticker, status))
            else:
                counts['OTHER'] += 1
                issues.append((ticker, status))
            if done % 50 == 0:
                print(f'  [{done}/{len(tickers)}]')
    print(f'\nDone in {time.time()-t0:.1f}s')
    print(f'  OK: {counts["OK"]}')
    print(f'  Errors: {counts["ERROR"]}')
    print(f'  Other: {counts["OTHER"]}')
    print(f'  Total new rows added: {new_total}')
    print(f'  Total rows overwritten: {overwritten_total}')
    if issues:
        print(f'\nIssues ({len(issues)}):')
        for t, s in issues[:30]:
            print(f'  {t}: {s}')
        if len(issues) > 30:
            print(f'  ... +{len(issues)-30} more')


if __name__ == '__main__':
    main()
