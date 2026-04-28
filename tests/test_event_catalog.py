"""
Build event catalog from S&P 500 (SPY) historical data.

Events identified:
  1. Single-day crash: SPY 1-day return < -3%
  2. 1-week crash: SPY 5-day return < -8%
  3. 1-month crash: SPY 21-day return < -10%
  4. Bear market entry: SPY drops > 20% from peak

For each event: snapshot of features just before, prices in next 1/5/10/30/60 days.

Output: results/event_catalog.csv + .txt
"""
import core
import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date').agg({'Close': 'last', 'Volume': 'sum'})


def find_events(spy_df):
    """Identify major event days."""
    close = spy_df['Close']
    ret_1d = close.pct_change()
    ret_5d = close.pct_change(5)
    ret_21d = close.pct_change(21)

    # Drawdown from peak
    peak = close.cummax()
    drawdown = close / peak - 1

    events = []

    # 1-day crashes (≤ -3%)
    crash_1d = close.index[ret_1d <= -0.03]
    for d in crash_1d:
        events.append({
            'date': d, 'type': '1-day crash',
            'magnitude': ret_1d.loc[d],
            'spy_close': close.loc[d],
        })

    # 5-day crashes (≤ -8%)
    crash_5d = close.index[ret_5d <= -0.08]
    for d in crash_5d:
        events.append({
            'date': d, 'type': '5-day crash',
            'magnitude': ret_5d.loc[d],
            'spy_close': close.loc[d],
        })

    # 21-day crashes (≤ -15%)
    crash_21d = close.index[ret_21d <= -0.15]
    # Cluster: only first day of consecutive bad month
    crash_21d = crash_21d[~pd.Series(crash_21d.to_series().diff().dt.days < 30).fillna(False)]
    for d in crash_21d:
        events.append({
            'date': d, 'type': '21-day crash',
            'magnitude': ret_21d.loc[d],
            'spy_close': close.loc[d],
        })

    # Bear market entries (drawdown crosses -20%)
    bear_signal = (drawdown <= -0.20)
    bear_entries = close.index[bear_signal & (~bear_signal.shift(1, fill_value=False))]
    for d in bear_entries:
        events.append({
            'date': d, 'type': 'bear entry',
            'magnitude': drawdown.loc[d],
            'spy_close': close.loc[d],
        })

    return pd.DataFrame(events).sort_values('date').reset_index(drop=True)


def add_recovery_stats(events, spy_close):
    """For each event, compute SPY recovery over 1/5/10/30/60 days."""
    horizons = [1, 5, 10, 30, 60, 120, 252]
    for h in horizons:
        col = f'spy_ret_{h}d'
        rets = []
        for d in events['date']:
            future = spy_close[spy_close.index > d].head(h)
            if len(future) >= h:
                ret = future.iloc[-1] / spy_close.loc[d] - 1
                rets.append(ret)
            else:
                rets.append(np.nan)
        events[col] = rets
    return events


def cluster_events(events, gap_days=30):
    """Cluster events within gap_days into single 'major events'."""
    if events.empty:
        return events
    events = events.sort_values('date').reset_index(drop=True)
    events['cluster_id'] = 0
    current_cluster = 0
    last_date = events['date'].iloc[0]
    for i in range(len(events)):
        if (events['date'].iloc[i] - last_date).days > gap_days:
            current_cluster += 1
        events.at[i, 'cluster_id'] = current_cluster
        last_date = events['date'].iloc[i]

    # For each cluster, take the worst single-day or the first event
    representative = events.loc[events.groupby('cluster_id')['magnitude'].idxmin()]
    return representative.reset_index(drop=True)


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Event catalog construction")
    w("=" * 100)

    spy = load_spy()
    spy_close = spy['Close']
    w(f"\nSPY data: {spy.index.min().date()} ~ {spy.index.max().date()} ({len(spy)} days)")

    events = find_events(spy)
    w(f"\nRaw event detections: {len(events)}")
    w(f"  Type breakdown:")
    for t, cnt in events['type'].value_counts().items():
        w(f"    {t}: {cnt}")

    # Cluster
    major_events = cluster_events(events, gap_days=30)
    w(f"\nClustered (30-day gap): {len(major_events)} major events")

    # Add recovery
    major_events = add_recovery_stats(major_events, spy_close)

    w(f"\n## Major Events (clustered)")
    w(f"{'Date':<12} {'Type':<15} {'Mag':>8} {'+1d':>7} {'+5d':>7} {'+10d':>7} {'+30d':>7} {'+60d':>7} {'+120d':>7} {'+252d':>7}")
    w("-" * 100)
    for _, row in major_events.iterrows():
        w(f"{row['date'].date()!s:<12} {row['type']:<15} "
          f"{row['magnitude']*100:>7.2f}% "
          f"{row['spy_ret_1d']*100 if pd.notna(row['spy_ret_1d']) else 0:>6.2f}% "
          f"{row['spy_ret_5d']*100 if pd.notna(row['spy_ret_5d']) else 0:>6.2f}% "
          f"{row['spy_ret_10d']*100 if pd.notna(row['spy_ret_10d']) else 0:>6.2f}% "
          f"{row['spy_ret_30d']*100 if pd.notna(row['spy_ret_30d']) else 0:>6.2f}% "
          f"{row['spy_ret_60d']*100 if pd.notna(row['spy_ret_60d']) else 0:>6.2f}% "
          f"{row['spy_ret_120d']*100 if pd.notna(row['spy_ret_120d']) else 0:>6.2f}% "
          f"{row['spy_ret_252d']*100 if pd.notna(row['spy_ret_252d']) else 0:>6.2f}%")

    # Aggregate recovery stats
    w(f"\n## Average recovery (across all major events)")
    for h in [1, 5, 10, 30, 60, 120, 252]:
        col = f'spy_ret_{h}d'
        if col in major_events.columns:
            valid = major_events[col].dropna()
            if len(valid) > 0:
                avg = valid.mean()
                pos_rate = (valid > 0).mean()
                w(f"  {h:>3}d: avg {avg*100:+.2f}%, win rate {pos_rate*100:.0f}%")

    # Save
    csv_path = os.path.join(OUTPUT_DIR, 'event_catalog.csv')
    major_events.to_csv(csv_path, index=False)
    w(f"\nCSV: {csv_path}")

    out_path = os.path.join(OUTPUT_DIR, 'event_catalog.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
