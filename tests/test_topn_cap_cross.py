"""
Cross validation: Top-N × Sector Cap matrix.

Top-N ∈ {10, 15, 20} × Cap ∈ {None, 30%, 25%, 20%, 15%}
= 15 configs × 31 windows.

v4 environment (S&P 500 + Ridge + 7y train + Weekly).

Output: results/topn_cap_cross.txt
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


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def topn_with_sector_cap(scores_row, sectors, top_n, sector_cap):
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
    return held


def build_holdings(score_wide, sectors, top_n, rebal_days, sector_cap):
    n_days, n_stocks = score_wide.shape
    holdings = np.zeros((n_days, n_stocks))
    last_holdings = np.zeros(n_stocks)
    last_rebal = -rebal_days

    cols = score_wide.columns.tolist()
    col_idx = {c: i for i, c in enumerate(cols)}

    for t in range(n_days):
        if (t - last_rebal) >= rebal_days:
            row = score_wide.iloc[t]
            held = topn_with_sector_cap(row, sectors, top_n, sector_cap)
            new_holdings = np.zeros(n_stocks)
            for ticker in held:
                if ticker in col_idx:
                    new_holdings[col_idx[ticker]] = 1.0
            last_holdings = new_holdings
            last_rebal = t
        holdings[t] = last_holdings

    return pd.DataFrame(holdings, index=score_wide.index, columns=cols)


def backtest(close, score_wide, sectors, top_n, sector_cap):
    score = score_wide.where(close.notna())
    in_top = build_holdings(score, sectors, top_n, REBAL_DAYS, sector_cap)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * COST_ONEWAY
    return port_gross - daily_cost


def run_one_window(close, vol, test_year, hp, spy_ret, sectors, top_n, sector_cap):
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
    port_ret = backtest(test_close, score_wide, sectors, top_n, sector_cap)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {
        'cagr': s['cagr'], 'sharpe': s['sharpe'],
        'mdd': s['mdd'], 'excess': excess,
    }


def run_config(close, vol, hp, spy_ret, sectors, top_n, sector_cap):
    rows = []
    for test_year in range(1995, 2026):
        r = run_one_window(close, vol, test_year, hp, spy_ret, sectors, top_n, sector_cap)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'top_n': top_n, 'cap': sector_cap,
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Top-N × Cap cross validation")
    w(f"  Ridge + 7y train + Weekly | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    sectors = load_sectors()
    hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"Loaded {close.shape[1]} tickers\n")

    top_ns = [10, 15, 20]
    caps = [1.00, 0.30, 0.25, 0.20, 0.15]

    all_results = []
    for tn in top_ns:
        for cap in caps:
            label = f"Top-{tn} Cap{int(cap*100)}%" if cap < 1.0 else f"Top-{tn} NoCap"
            max_per = max(1, int(tn * cap))
            w(f"\n[{label}] (max {max_per}/sector) running...")
            r = run_config(close, vol, hp, spy_ret, sectors, tn, cap)
            if r:
                r['label'] = label
                r['max_per_sec'] = max_per
                all_results.append(r)
                w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
                  f"MDD {r['avg_mdd']*100:.2f}% | "
                  f"vs SPY {r['avg_excess']:+.2f}%p | "
                  f"win {r['win']}/{r['n_windows']} | "
                  f"t={r['t_stat']:.2f}")

    # Build matrix
    w(f"\n{'='*120}")
    w(f"## Sharpe matrix (Top-N × Cap)")
    w(f"{'Top-N':<8} {'NoCap':>10} {'Cap30%':>10} {'Cap25%':>10} {'Cap20%':>10} {'Cap15%':>10}")
    w("-" * 70)
    by_key = {(r['top_n'], r['cap']): r for r in all_results}
    for tn in top_ns:
        line = f"Top-{tn:<5}"
        for cap in caps:
            r = by_key.get((tn, cap))
            line += f" {r['avg_sharpe']:>10.2f}" if r else f"{'?':>10}"
        w(line)

    w(f"\n## Alpha matrix (vs SPY %p)")
    w(f"{'Top-N':<8} {'NoCap':>10} {'Cap30%':>10} {'Cap25%':>10} {'Cap20%':>10} {'Cap15%':>10}")
    w("-" * 70)
    for tn in top_ns:
        line = f"Top-{tn:<5}"
        for cap in caps:
            r = by_key.get((tn, cap))
            line += f" {r['avg_excess']:>+9.2f}p" if r else f"{'?':>10}"
        w(line)

    w(f"\n## MDD matrix")
    w(f"{'Top-N':<8} {'NoCap':>10} {'Cap30%':>10} {'Cap25%':>10} {'Cap20%':>10} {'Cap15%':>10}")
    w("-" * 70)
    for tn in top_ns:
        line = f"Top-{tn:<5}"
        for cap in caps:
            r = by_key.get((tn, cap))
            line += f" {r['avg_mdd']*100:>9.2f}%" if r else f"{'?':>10}"
        w(line)

    w(f"\n## Sorted by Sharpe (top 10)")
    sorted_r = sorted(all_results, key=lambda r: r['avg_sharpe'], reverse=True)
    w(f"{'Config':<25} {'CAGR':>8} {'Sh':>5} {'MDD':>8} {'vs SPY':>9} {'Win':>8} {'t':>6}")
    w("-" * 90)
    for r in sorted_r[:10]:
        w(f"{r['label']:<25} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p "
          f"{r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'topn_cap_cross.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
