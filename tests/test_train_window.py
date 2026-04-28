"""
Train window length grid: 3y / 5y / 7y / 10y.

Walk-forward (test 1y per window). For each window, ML retrained with the
specified train window. Top-20 Biweekly evaluation.

Output: results/train_window_grid.txt
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 10
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last().pct_change()


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def backtest_with_score(close, score_wide, top_n=TOP_N, rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    hp = {'top_n': top_n, 'rebal_days': rebal_days, 'hysteresis': 0,
          'cost_oneway': cost}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost


def run_one_window(close, vol, test_year, hp, train_years, spy_ret):
    train_start = pd.Timestamp(f'{test_year - train_years}-01-01')
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

    model = ml_model.train_model(train_long, hp)
    if model is None:
        return None

    score_long = ml_model.score_with_model(model, test_long, hp)
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    port_ret = backtest_with_score(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {
        'year': test_year, 'cagr': s['cagr'], 'sharpe': s['sharpe'],
        'mdd': s['mdd'], 'excess': excess,
    }


def run_config(close, vol, hp, train_years, spy_ret):
    rows = []
    # Earliest start year depends on train window
    start_year = max(1995, 1962 + train_years)  # need data
    for test_year in range(start_year, 2026):
        r = run_one_window(close, vol, test_year, hp, train_years, spy_ret)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'train_years': train_years,
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

    w(f"[{datetime.now()}] Train window length grid")
    w("=" * 90)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"Loaded {close.shape[1]} tickers\n")

    configs = [3, 5, 7, 10]
    results = []
    for tw in configs:
        w(f"\n[Train window = {tw} years] running...")
        r = run_config(close, vol, hp, tw, spy_ret)
        if r:
            results.append(r)
            w(f"  N={r['n_windows']} | CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | t={r['t_stat']:.2f}")

    results.sort(key=lambda r: r['avg_sharpe'], reverse=True)

    w(f"\n{'='*90}")
    w(f"## Summary (sorted by Sharpe)")
    w(f"{'TrainY':<8} {'N':<5} {'CAGR':>8} {'Sh':>5} {'MDD':>8} {'vs SPY':>9} {'Win':>8} {'t':>6}")
    w("-" * 90)
    for r in results:
        w(f"{r['train_years']:<8} {r['n_windows']:<5} {r['avg_cagr']*100:>+7.2f}% "
          f"{r['avg_sharpe']:>5.2f} {r['mdd_avg']*100 if 'mdd_avg' in r else r['avg_mdd']*100:>7.2f}% "
          f"{r['avg_excess']:>+8.2f}%p {r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'train_window_grid.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
