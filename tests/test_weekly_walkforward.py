"""
Weekly walk-forward validation (2025-2026).

For each week starting Monday:
1. Train Ridge on 7 years of data up to (week_start - 1)
2. Score on week_start, take Top-20
3. Hold for next 5 trading days
4. Measure portfolio return + alpha vs SPY

Aggregate all weekly results.

Output: results/weekly_walkforward.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import ml_model
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
TOP_N = 20
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
FEATURES = ['lowvol', 'rsi', 'volsurge']
HOLD_DAYS = 5  # Weekly hold


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last()


def get_monday_schedule(close_index, start_date, end_date):
    """Return list of trading dates that are Mondays (or first day of week if Mon is holiday)."""
    dates = close_index[(close_index >= start_date) & (close_index < end_date)]
    # Find first trading day of each week
    schedule = []
    seen_weeks = set()
    for d in dates:
        # Use ISO week (year, week)
        iso_year, iso_week, _ = d.isocalendar()
        key = (iso_year, iso_week)
        if key not in seen_weeks:
            seen_weeks.add(key)
            schedule.append(d)
    return schedule


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Weekly walk-forward 2025-2026")
    w(f"  Features: {FEATURES} | Top-{TOP_N} | Train {TRAIN_YEARS}y")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_close = load_spy()
    spy_ret = spy_close.pct_change()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    start = pd.Timestamp('2025-01-01')
    end = pd.Timestamp('2026-04-28')
    schedule = get_monday_schedule(close.index, start, end)
    w(f"\nTotal weeks: {len(schedule)}")
    w(f"Range: {schedule[0].date()} ~ {schedule[-1].date()}\n")

    rows = []
    for week_idx, monday in enumerate(schedule):
        train_end = monday  # exclusive
        train_start = monday - pd.DateOffset(years=TRAIN_YEARS)
        train_mask = (close.index >= train_start) & (close.index < train_end)
        if train_mask.sum() < 252:
            continue

        # Valid tickers: enough train data
        valid = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
        if len(valid) < 100:
            continue
        close_sub = close[valid]
        vol_sub = vol[valid]

        # Get test mask: 5 trading days starting from monday
        future_idx = close_sub.index[close_sub.index >= monday]
        if len(future_idx) < HOLD_DAYS + 1:
            continue
        test_dates = future_idx[:HOLD_DAYS + 1]  # need close at start and end
        test_start_date = test_dates[0]
        test_end_date = test_dates[-1]

        # Train ML
        train_long, _, _, _ = ml_model.get_train_test_features(
            close_sub, vol_sub, train_mask,
            close_sub.index == monday,  # dummy test mask; we score on monday separately
            hp)
        train_long = train_long.dropna(subset=FEATURES)
        if len(train_long) < 1000:
            continue

        # Score on monday (use score features)
        feat_panels = ml_model.build_features_panel(close_sub, vol_sub)
        score_feats = {n: df.loc[[monday]] if monday in df.index else df.tail(0)
                       for n, df in feat_panels.items()}
        score_long = ml_model.stack_panel_to_long(score_feats)
        score_long = score_long.dropna(subset=FEATURES)
        if len(score_long) < TOP_N:
            continue

        scaler = StandardScaler()
        Xs = scaler.fit_transform(train_long[FEATURES].values)
        model = Ridge(alpha=1.0)
        model.fit(Xs, train_long['target'].values)
        preds = model.predict(scaler.transform(score_long[FEATURES].values))
        score_long['score'] = preds

        # Top-N
        score_long = score_long.sort_values('score', ascending=False)
        top = score_long.head(TOP_N)
        held_tickers = top['ticker'].tolist()

        # Compute portfolio return for next 5 days
        # Buy at test_start_date close, sell at test_end_date close
        try:
            start_prices = close_sub.loc[test_start_date, held_tickers]
            end_prices = close_sub.loc[test_end_date, held_tickers]
        except KeyError:
            continue

        # Equal weight
        rets_per_stock = (end_prices / start_prices - 1).dropna()
        if len(rets_per_stock) == 0:
            continue
        port_ret = rets_per_stock.mean()
        # Apply cost (full turnover = 100% out, 100% in = 200% trades = 2 * cost)
        cost_drag = 2 * COST_ONEWAY  # round trip on full portfolio
        port_ret_net = port_ret - cost_drag

        # SPY return
        try:
            spy_start = spy_close.loc[spy_close.index >= test_start_date].iloc[0]
            spy_end_dates = spy_close.index[spy_close.index >= test_end_date]
            if len(spy_end_dates) == 0:
                continue
            spy_end = spy_close.loc[spy_end_dates[0]]
            spy_ret_week = spy_end / spy_start - 1
        except (IndexError, KeyError):
            continue

        excess = (port_ret_net - spy_ret_week) * 100  # in %p

        rows.append({
            'week': week_idx + 1,
            'monday': monday.date(),
            'port_return': port_ret_net * 100,  # %
            'port_gross': port_ret * 100,       # %
            'spy_return': spy_ret_week * 100,
            'excess': excess,
            'top1': held_tickers[0] if held_tickers else None,
            'top5': held_tickers[:5],
        })

        if (week_idx + 1) % 5 == 0 or week_idx < 5:
            w(f"  Week {week_idx + 1:>2} ({monday.date()}): "
              f"Port {port_ret_net*100:+6.2f}% | SPY {spy_ret_week*100:+6.2f}% | "
              f"Excess {excess:+6.2f}%p | Top-3: {held_tickers[:3]}")

    if not rows:
        w("\nNo results")
        return

    # Aggregate
    df = pd.DataFrame(rows)
    n = len(df)
    avg_port = df['port_return'].mean()
    avg_spy = df['spy_return'].mean()
    avg_excess = df['excess'].mean()
    win_count = (df['excess'] > 0).sum()
    cum_port = ((df['port_return'] / 100 + 1).prod() - 1) * 100
    cum_spy = ((df['spy_return'] / 100 + 1).prod() - 1) * 100

    w(f"\n{'='*100}")
    w(f"## Aggregate ({n} weeks)")
    w(f"  Average weekly portfolio return:    {avg_port:+.2f}%")
    w(f"  Average weekly SPY return:          {avg_spy:+.2f}%")
    w(f"  Average weekly excess (alpha):      {avg_excess:+.2f}%p")
    w(f"  Annualized excess:                  {avg_excess * 52:+.2f}%p")
    w(f"  Cumulative portfolio:               {cum_port:+.2f}%")
    w(f"  Cumulative SPY:                     {cum_spy:+.2f}%")
    w(f"  Cumulative excess:                  {cum_port - cum_spy:+.2f}%p")
    w(f"  Win rate (port > SPY):              {win_count}/{n} ({win_count/n*100:.1f}%)")
    w(f"  Best week:                          +{df['excess'].max():.2f}%p ({df.loc[df['excess'].idxmax(), 'monday']})")
    w(f"  Worst week:                         {df['excess'].min():.2f}%p ({df.loc[df['excess'].idxmin(), 'monday']})")

    # Distribution
    w(f"\n## Weekly excess distribution")
    w(f"  P10: {df['excess'].quantile(0.10):+.2f}%p")
    w(f"  P25: {df['excess'].quantile(0.25):+.2f}%p")
    w(f"  P50: {df['excess'].quantile(0.50):+.2f}%p")
    w(f"  P75: {df['excess'].quantile(0.75):+.2f}%p")
    w(f"  P90: {df['excess'].quantile(0.90):+.2f}%p")

    # Per-month aggregation (rough)
    df['month'] = pd.to_datetime(df['monday']).dt.to_period('M')
    monthly = df.groupby('month').agg(
        weeks=('excess', 'count'),
        avg_excess=('excess', 'mean'),
        cum_port=('port_return', lambda x: ((x/100+1).prod()-1)*100),
        cum_spy=('spy_return', lambda x: ((x/100+1).prod()-1)*100),
    )
    w(f"\n## Per-month aggregate")
    w(f"{'Month':<10} {'Weeks':>6} {'Avg α':>8} {'Cum Port':>10} {'Cum SPY':>10} {'Diff':>8}")
    w("-" * 60)
    for m, row in monthly.iterrows():
        diff = row['cum_port'] - row['cum_spy']
        w(f"{str(m):<10} {row['weeks']:>6.0f} {row['avg_excess']:>+7.2f}p "
          f"{row['cum_port']:>+9.2f}% {row['cum_spy']:>+9.2f}% {diff:>+7.2f}p")

    # Save full per-week
    df_save = df.drop(columns=['top5'])
    df_save['top5'] = df['top5'].astype(str)
    df_save.to_csv(os.path.join(OUTPUT_DIR, 'weekly_walkforward.csv'), index=False)

    out_path = os.path.join(OUTPUT_DIR, 'weekly_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
