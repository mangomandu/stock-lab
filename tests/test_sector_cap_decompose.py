"""
Time-decomposed sector cap analysis.

Per-year alpha for No Cap vs Cap 25%. Identify Tech rally vs crash periods.

Output: results/sector_cap_decompose.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import ml_model
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
SECTORS_PATH = '/home/dlfnek/stock_lab/data/sectors.csv'

TOP_N = 20
REBAL_DAYS = 5
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last().pct_change()


def load_sectors():
    df = pd.read_csv(SECTORS_PATH, index_col='Ticker')
    return df['Sector'].to_dict()


def topn_with_sector_cap(scores_row, sectors, top_n, sector_cap):
    """Greedy: highest scored first, skip stocks beyond sector cap."""
    sorted_scores = scores_row.dropna().sort_values(ascending=False)
    held = []
    sector_count = defaultdict(int)
    max_per_sector = max(1, int(top_n * sector_cap))
    for ticker in sorted_scores.index:
        if len(held) >= top_n:
            break
        sec = sectors.get(ticker, 'Unknown')
        if sector_count[sec] >= max_per_sector:
            continue
        held.append(ticker)
        sector_count[sec] += 1
    return held, dict(sector_count)


def build_holdings_with_cap(score_wide, sectors, top_n, rebal_days, sector_cap):
    n_days, n_stocks = score_wide.shape
    holdings = np.zeros((n_days, n_stocks))
    sector_history = []  # sector composition at each rebalance
    last_holdings = np.zeros(n_stocks)
    last_rebal = -rebal_days

    cols = score_wide.columns.tolist()
    col_idx = {c: i for i, c in enumerate(cols)}

    for t in range(n_days):
        if (t - last_rebal) >= rebal_days:
            row = score_wide.iloc[t]
            held, sec_count = topn_with_sector_cap(row, sectors, top_n, sector_cap)
            new_holdings = np.zeros(n_stocks)
            for ticker in held:
                if ticker in col_idx:
                    new_holdings[col_idx[ticker]] = 1.0
            last_holdings = new_holdings
            last_rebal = t
            sector_history.append((score_wide.index[t], sec_count))
        holdings[t] = last_holdings

    return pd.DataFrame(holdings, index=score_wide.index, columns=cols), sector_history


def backtest_window(close, score_wide, sectors, top_n, sector_cap,
                    rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    in_top, sec_history = build_holdings_with_cap(score, sectors, top_n, rebal_days, sector_cap)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, sec_history


def run_one_window(close, vol, test_year, hp, spy_ret, sectors, sector_cap):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid_tickers) < MIN_UNIVERSE_SIZE:
        return None

    close_sub = close[valid_tickers]
    vol_sub = vol[valid_tickers]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    if len(train_long) < 1000:
        return None

    feat_cols = hp['feature_names']
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_long[feat_cols].values)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, train_long['target'].values)

    X_test_s = scaler.transform(test_long[feat_cols].values)
    preds = model.predict(X_test_s)

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    port_ret, sec_history = backtest_window(test_close, score_wide, sectors, TOP_N, sector_cap)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    # Average sector composition
    sector_avg = defaultdict(float)
    n_rebal = len(sec_history)
    for _, sec_count in sec_history:
        for sec, cnt in sec_count.items():
            sector_avg[sec] += cnt
    if n_rebal > 0:
        sector_avg = {sec: cnt/n_rebal for sec, cnt in sector_avg.items()}

    # Top sector
    top_sector = max(sector_avg.items(), key=lambda x: x[1]) if sector_avg else ('?', 0)

    return {
        'year': test_year, 'cagr': s['cagr'], 'sharpe': s['sharpe'],
        'mdd': s['mdd'], 'excess': excess,
        'top_sector': top_sector[0], 'top_sector_count': top_sector[1],
        'tech_count': sector_avg.get('Technology', 0),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Sector cap time decomposition")
    w(f"  No Cap vs Cap 25% per year")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    sectors = load_sectors()
    hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"Loaded {close.shape[1]} tickers\n")

    rows = []
    for test_year in range(1995, 2026):
        no_cap = run_one_window(close, vol, test_year, hp, spy_ret, sectors, 1.00)
        with_cap = run_one_window(close, vol, test_year, hp, spy_ret, sectors, 0.25)
        if no_cap is None or with_cap is None:
            continue

        diff = with_cap['excess'] - no_cap['excess']
        rows.append({
            'year': test_year,
            'no_cap_excess': no_cap['excess'],
            'cap25_excess': with_cap['excess'],
            'diff': diff,
            'no_cap_top_sec': no_cap['top_sector'],
            'no_cap_top_sec_n': no_cap['top_sector_count'],
            'no_cap_tech': no_cap['tech_count'],
        })
        w(f"  [{test_year}] NoCap {no_cap['excess']:+7.2f}%p | Cap25 {with_cap['excess']:+7.2f}%p | "
          f"Diff {diff:+6.2f}%p | TopSec={no_cap['top_sector']}({no_cap['top_sector_count']:.1f}) "
          f"Tech={no_cap['tech_count']:.1f}")

    if not rows:
        return

    df = pd.DataFrame(rows)
    w(f"\n{'='*100}")
    w(f"## Time period decomposition")

    # Define periods
    periods = {
        'Dot-com bubble (1995-1999)': (1995, 1999),
        'Dot-com crash (2000-2002)': (2000, 2002),
        'Recovery (2003-2007)': (2003, 2007),
        'Financial crisis (2008-2009)': (2008, 2009),
        'Recovery 2 (2010-2014)': (2010, 2014),
        'Mid-decade (2015-2019)': (2015, 2019),
        'Pandemic + recovery (2020-2022)': (2020, 2022),
        'Tech rally (2023-2025)': (2023, 2025),
    }

    w(f"\n{'Period':<35} {'NoCap':>8} {'Cap25':>8} {'Diff':>8} {'AvgTechN':>10}")
    w("-" * 80)
    for label, (y1, y2) in periods.items():
        sub = df[(df['year'] >= y1) & (df['year'] <= y2)]
        if len(sub) == 0:
            continue
        no_cap_avg = sub['no_cap_excess'].mean()
        cap_avg = sub['cap25_excess'].mean()
        diff_avg = sub['diff'].mean()
        tech_avg = sub['no_cap_tech'].mean()
        w(f"{label:<35} {no_cap_avg:>+7.2f}%p {cap_avg:>+7.2f}%p "
          f"{diff_avg:>+7.2f}%p {tech_avg:>9.1f}")

    # Overall
    w(f"\n{'='*100}")
    w(f"## Overall")
    w(f"  No Cap avg:   {df['no_cap_excess'].mean():+.2f}%p")
    w(f"  Cap 25% avg:  {df['cap25_excess'].mean():+.2f}%p")
    w(f"  Diff avg:     {df['diff'].mean():+.2f}%p")
    w(f"  Years Cap > NoCap: {(df['diff'] > 0).sum()}/{len(df)}")
    w(f"  Years Cap < NoCap: {(df['diff'] < 0).sum()}/{len(df)}")

    # Show extreme years
    w(f"\n## Most differing years")
    df_sorted = df.sort_values('diff', ascending=False)
    w(f"  Cap helped most:")
    for _, r in df_sorted.head(5).iterrows():
        w(f"    {int(r['year'])}: Cap +{r['diff']:.2f}%p (NoCap {r['no_cap_excess']:+.2f} → Cap25 {r['cap25_excess']:+.2f}, Tech {r['no_cap_tech']:.1f})")
    w(f"  Cap hurt most:")
    for _, r in df_sorted.tail(5).iterrows():
        w(f"    {int(r['year'])}: Cap {r['diff']:+.2f}%p (NoCap {r['no_cap_excess']:+.2f} → Cap25 {r['cap25_excess']:+.2f}, Tech {r['no_cap_tech']:.1f})")

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'sector_cap_decompose.csv'), index=False)
    out_path = os.path.join(OUTPUT_DIR, 'sector_cap_decompose.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
