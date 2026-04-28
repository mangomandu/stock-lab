"""Full intra-file integrity scan for all 518 tickers.

Checks:
1. Value sanity: NaN, non-positive Close, negative Volume, OHLC consistency
2. Universe coverage: master_sp500/ vs sp500_tickers.txt + etf_tickers.txt
3. Date gaps: trading days where SPY traded but this ticker did not (within lifespan)
"""
import pandas as pd
import numpy as np
import os
from collections import defaultdict

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
SP500_LIST = '/home/dlfnek/stock_lab/data/sp500_tickers.txt'
ETF_LIST = '/home/dlfnek/stock_lab/data/etf_tickers.txt'

GAP_REPORT_THRESHOLD = 5   # report tickers with >this missing trading days vs SPY
TOP_N_TO_PRINT = 20


def load_ticker(path):
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['Datetime']).set_index('Datetime').sort_index()
    return df


def main():
    # --- Section 1: universe coverage ---
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.endswith('.csv') and not f.startswith('data_validation'))
    file_tickers = set(f.replace('.csv', '') for f in files)

    with open(SP500_LIST) as f:
        sp500 = set(t.strip() for t in f if t.strip())
    with open(ETF_LIST) as f:
        etfs = set(t.strip() for t in f if t.strip())
    expected = sp500 | etfs

    missing_files = expected - file_tickers   # in list but no CSV
    extra_files = file_tickers - expected     # CSV but not in list

    print('=' * 70)
    print('SECTION 1: UNIVERSE COVERAGE')
    print('=' * 70)
    print(f"Files in master_sp500/: {len(file_tickers)}")
    print(f"  sp500_tickers.txt:   {len(sp500)}")
    print(f"  etf_tickers.txt:     {len(etfs)}")
    print(f"  Expected total:      {len(expected)}")
    print(f"  Missing CSV files (in list, no data): {len(missing_files)}")
    if missing_files:
        for t in sorted(missing_files):
            print(f"    - {t}")
    print(f"  Extra CSV files (data, not in list):  {len(extra_files)}")
    if extra_files:
        for t in sorted(extra_files):
            print(f"    - {t}")

    # --- Load all tickers (used by section 2 and 3) ---
    print()
    print('Loading all tickers...')
    all_data = {}
    load_errors = []
    for f in files:
        ticker = f.replace('.csv', '')
        try:
            df = load_ticker(os.path.join(DATA_DIR, f))
            if len(df) == 0:
                load_errors.append((ticker, 'empty after parse'))
                continue
            all_data[ticker] = df
        except Exception as e:
            load_errors.append((ticker, str(e)[:80]))
    print(f'  Loaded: {len(all_data)}/{len(files)}')
    if load_errors:
        print(f'  Errors: {len(load_errors)}')
        for t, e in load_errors[:10]:
            print(f'    {t}: {e}')

    # --- Section 2: value sanity ---
    print()
    print('=' * 70)
    print('SECTION 2: VALUE SANITY')
    print('=' * 70)
    issues = defaultdict(list)   # issue_name -> [(ticker, count, sample)]
    for ticker, df in all_data.items():
        # NaN
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            n = int(df[col].isna().sum())
            if n > 0:
                issues[f'NaN_{col}'].append((ticker, n,
                    df.index[df[col].isna()][:3].strftime('%Y-%m-%d').tolist()))
        # Close <= 0
        bad = df[df['Close'] <= 0]
        if len(bad) > 0:
            issues['Close_le_0'].append((ticker, len(bad),
                bad.index[:3].strftime('%Y-%m-%d').tolist()))
        # Open <= 0
        bad = df[df['Open'] <= 0]
        if len(bad) > 0:
            issues['Open_le_0'].append((ticker, len(bad),
                bad.index[:3].strftime('%Y-%m-%d').tolist()))
        # Volume < 0
        bad = df[df['Volume'] < 0]
        if len(bad) > 0:
            issues['Volume_lt_0'].append((ticker, len(bad),
                bad.index[:3].strftime('%Y-%m-%d').tolist()))
        # OHLC consistency: Low <= min(O,C) and High >= max(O,C) and Low <= High
        # Use small epsilon to absorb float roundoff
        eps = 1e-6
        bad = df[df['Low'] > df['High'] + eps]
        if len(bad) > 0:
            issues['Low_gt_High'].append((ticker, len(bad),
                bad.index[:3].strftime('%Y-%m-%d').tolist()))
        min_oc = df[['Open', 'Close']].min(axis=1)
        max_oc = df[['Open', 'Close']].max(axis=1)
        bad = df[df['Low'] > min_oc + eps]
        if len(bad) > 0:
            issues['Low_gt_OpenOrClose'].append((ticker, len(bad),
                bad.index[:3].strftime('%Y-%m-%d').tolist()))
        bad = df[df['High'] < max_oc - eps]
        if len(bad) > 0:
            issues['High_lt_OpenOrClose'].append((ticker, len(bad),
                bad.index[:3].strftime('%Y-%m-%d').tolist()))

    if not issues:
        print('  CLEAN: no value sanity issues across all tickers.')
    else:
        for issue, items in sorted(issues.items()):
            total = sum(c for _, c, _ in items)
            print(f'  {issue}: {len(items)} tickers, {total} total rows')
            for ticker, n, sample in items[:TOP_N_TO_PRINT]:
                print(f'    {ticker}: {n} rows, sample={sample}')
            if len(items) > TOP_N_TO_PRINT:
                print(f'    ... +{len(items)-TOP_N_TO_PRINT} more')

    # --- Section 3: date gaps vs SPY calendar ---
    print()
    print('=' * 70)
    print('SECTION 3: DATE GAPS vs SPY TRADING CALENDAR')
    print('=' * 70)
    if 'SPY' not in all_data:
        print('  ERROR: SPY not loaded, skipping gap check')
    else:
        spy_dates = set(all_data['SPY'].index.normalize())
        spy_first = min(spy_dates)
        spy_last = max(spy_dates)
        print(f'  SPY calendar: {len(spy_dates)} trading days, {spy_first.date()} ~ {spy_last.date()}')
        gap_results = []
        for ticker, df in all_data.items():
            t_dates = set(df.index.normalize())
            t_first = min(t_dates)
            t_last = max(t_dates)
            # Trading days SPY has within [t_first, t_last] that ticker is missing
            spy_in_lifespan = {d for d in spy_dates if t_first <= d <= t_last}
            missing = spy_in_lifespan - t_dates
            if missing:
                gap_results.append((ticker, len(missing),
                    sorted([str(d.date()) for d in missing])[:5]))
        gap_results.sort(key=lambda x: -x[1])
        flagged = [r for r in gap_results if r[1] > GAP_REPORT_THRESHOLD]
        print(f'  Tickers with any missing trading days: {len(gap_results)}/{len(all_data)}')
        print(f'  Tickers with >{GAP_REPORT_THRESHOLD} missing days: {len(flagged)}')
        for ticker, n, sample in flagged[:TOP_N_TO_PRINT]:
            print(f'    {ticker}: {n} missing, sample={sample}')
        if len(flagged) > TOP_N_TO_PRINT:
            print(f'    ... +{len(flagged)-TOP_N_TO_PRINT} more')

    # --- Final verdict ---
    print()
    print('=' * 70)
    print('VERDICT')
    print('=' * 70)
    n_critical = (len(missing_files) + len(extra_files) + len(load_errors)
                  + sum(len(v) for k, v in issues.items()))
    n_minor_gap = sum(1 for r in gap_results if r[1] > GAP_REPORT_THRESHOLD) if 'SPY' in all_data else 0
    print(f'  Critical issues: {n_critical}')
    print(f'  Tickers w/ notable gaps (>{GAP_REPORT_THRESHOLD} missing): {n_minor_gap}')
    if n_critical == 0 and n_minor_gap == 0:
        print('  ALL CLEAN - safe to add new features')
    else:
        print('  Review issues above before adding features')


if __name__ == '__main__':
    main()
