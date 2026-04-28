"""
Test all 2-feature combinations from {lowvol, rsi, volsurge}.

Goal: find true minimum feature set.

Output: results/2feature_combos.txt
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
REBAL_DAYS = 5
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005


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


def backtest_with_score(close, score_wide):
    score = score_wide.where(close.notna())
    hp = {'top_n': TOP_N, 'rebal_days': REBAL_DAYS, 'hysteresis': 0,
          'cost_oneway': COST_ONEWAY}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * COST_ONEWAY
    return port_gross - daily_cost


def run_one_window(close, vol, test_year, hp, spy_ret, feature_names):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid) < 100:
        return None

    close_sub = close[valid]
    vol_sub = vol[valid]
    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    train_long = train_long.dropna(subset=feature_names)
    test_long = test_long.dropna(subset=feature_names)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(train_long[feature_names].values)
    model = Ridge(alpha=1.0)
    model.fit(Xs, train_long['target'].values)
    preds = model.predict(scaler.transform(test_long[feature_names].values))

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(score_long, close_sub.index[test_mask], close_sub.columns)
    test_close = close_sub[test_mask]
    port_ret = backtest_with_score(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None
    return {'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'], 'excess': excess}


def run_config(close, vol, hp, spy_ret, feats):
    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, spy_ret, feats)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
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

    w(f"[{datetime.now()}] 2-feature combinations")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"Loaded {close.shape[1]} tickers")

    configs = [
        ('lowvol+rsi+volsurge (3) baseline', ['lowvol', 'rsi', 'volsurge']),
        ('lowvol+rsi (2)',                    ['lowvol', 'rsi']),
        ('lowvol+volsurge (2)',               ['lowvol', 'volsurge']),
        ('rsi+volsurge (2)',                  ['rsi', 'volsurge']),
    ]

    results = []
    for label, feats in configs:
        hp_copy = dict(hp)
        hp_copy['feature_names'] = feats
        w(f"\n[{label}] {feats}")
        r = run_config(close, vol, hp_copy, spy_ret, feats)
        if r:
            r['label'] = label
            r['n'] = len(feats)
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | t={r['t_stat']:.2f}")

    results.sort(key=lambda r: r['avg_sharpe'], reverse=True)
    w(f"\n{'='*100}")
    w(f"## Sorted by Sharpe")
    w(f"{'Config':<35} {'#':>3} {'CAGR':>8} {'Sh':>5} {'MDD':>8} {'vs SPY':>9} {'t':>6}")
    w("-" * 80)
    for r in results:
        w(f"{r['label']:<35} {r['n']:>3} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p {r['t_stat']:>6.2f}")

    out_path = os.path.join(OUTPUT_DIR, '2feature_combos.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
