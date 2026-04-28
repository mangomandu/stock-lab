"""
Event response analysis — sector winners/losers per event.

For each major event, compute:
  - Sector ETF returns over 1/5/10/30/60 days post-event
  - Best/worst sectors
  - Defensive ETF behavior (GLD, TLT, SHY)

Output: results/event_response.txt
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

# ETFs by category
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'ConsumerStaples',
    'XLY': 'ConsumerDiscretionary',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLC': 'Communications',
    'XLRE': 'RealEstate',
    'ITA': 'Aerospace/Defense',
}
DEFENSIVE_ETFS = {
    'GLD': 'Gold',
    'TLT': 'LongTreasury',
    'SHY': 'ShortTreasury',
}


def load_etf(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last()


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Event Response Analysis")
    w("=" * 100)

    # Load event catalog
    events = pd.read_csv(os.path.join(OUTPUT_DIR, 'event_catalog.csv'))
    events['date'] = pd.to_datetime(events['date'])

    # Load all ETFs
    all_etfs = {**SECTOR_ETFS, **DEFENSIVE_ETFS}
    etf_data = {}
    for ticker in all_etfs:
        s = load_etf(ticker)
        if s is not None:
            etf_data[ticker] = s
    w(f"\nLoaded {len(etf_data)}/{len(all_etfs)} ETFs")

    # For each event, compute returns of each ETF over horizons
    horizons = [5, 30, 60]
    rows = []
    for _, ev in events.iterrows():
        d = ev['date']
        for ticker, label in all_etfs.items():
            if ticker not in etf_data:
                continue
            s = etf_data[ticker]
            if d not in s.index and not (s.index > d).any():
                continue
            base_idx = s.index.get_indexer([d], method='nearest')[0]
            base_price = s.iloc[base_idx]
            for h in horizons:
                future_idx = base_idx + h
                if future_idx >= len(s):
                    continue
                future_price = s.iloc[future_idx]
                ret = future_price / base_price - 1
                rows.append({
                    'event_date': d, 'event_type': ev['type'],
                    'ticker': ticker, 'sector': label,
                    'horizon_days': h, 'return': ret,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        w("No event-ETF pairs found")
        return

    # Aggregate: average return per (sector, horizon) across events
    w(f"\n## Average sector return after major events (positive = recovery winner)")
    summary = df.groupby(['sector', 'horizon_days'])['return'].agg(['mean', 'count', 'std'])
    summary = summary.reset_index()

    for h in horizons:
        sub = summary[summary['horizon_days'] == h].copy()
        sub = sub.sort_values('mean', ascending=False)
        w(f"\n### {h}-day horizon ({sub['count'].iloc[0]} events with valid data)")
        w(f"{'Sector':<25} {'Mean Ret':>10} {'Std':>8} {'N':>4}")
        w("-" * 50)
        for _, r in sub.iterrows():
            w(f"  {r['sector']:<25} {r['mean']*100:>+9.2f}% {r['std']*100:>7.2f}% {r['count']:>4.0f}")

    # Compare to SPY (overall market)
    spy = load_etf('SPY')
    spy_rows = []
    for _, ev in events.iterrows():
        d = ev['date']
        if not (spy.index > d).any():
            continue
        base_idx = spy.index.get_indexer([d], method='nearest')[0]
        base_price = spy.iloc[base_idx]
        for h in horizons:
            future_idx = base_idx + h
            if future_idx >= len(spy):
                continue
            ret = spy.iloc[future_idx] / base_price - 1
            spy_rows.append({'horizon_days': h, 'return': ret})
    spy_df = pd.DataFrame(spy_rows)

    w(f"\n## Excess return vs SPY (sector outperformance)")
    for h in horizons:
        spy_avg = spy_df[spy_df['horizon_days'] == h]['return'].mean()
        sub = summary[summary['horizon_days'] == h].copy()
        sub['excess_vs_spy'] = sub['mean'] - spy_avg
        sub = sub.sort_values('excess_vs_spy', ascending=False)
        w(f"\n### {h}-day horizon (SPY avg: {spy_avg*100:+.2f}%)")
        w(f"{'Sector':<25} {'Excess':>10}")
        w("-" * 40)
        for _, r in sub.head(5).iterrows():
            w(f"  {r['sector']:<25} {r['excess_vs_spy']*100:>+9.2f}%p")
        w(f"  ...")
        for _, r in sub.tail(3).iterrows():
            w(f"  {r['sector']:<25} {r['excess_vs_spy']*100:>+9.2f}%p")

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'event_responses.csv'), index=False)
    out_path = os.path.join(OUTPUT_DIR, 'event_response.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
