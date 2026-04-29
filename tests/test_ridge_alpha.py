"""
Ridge alpha (regularization) tuning.

Test alpha grid to determine:
- Best alpha = 1.0 → model is well-balanced (default OK)
- Best alpha < 1.0 → underfitting (more complexity needed)
- Best alpha > 1.0 → overfitting (more regularization needed)

Run on v5 best (3-feature: lowvol+rsi+volsurge), 31 windows.

Output: results/ridge_alpha.txt
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
FEATURES = ['lowvol', 'rsi', 'volsurge']

ALPHAS = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]


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


def backtest(close, score_wide):
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


def run_one_window(close, vol, test_year, hp, spy_ret, alpha):
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
    train_long = train_long.dropna(subset=FEATURES)
    test_long = test_long.dropna(subset=FEATURES)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(train_long[FEATURES].values)
    model = Ridge(alpha=alpha)
    model.fit(Xs, train_long['target'].values)

    # Train alpha (in-sample)
    train_preds = model.predict(Xs)
    # Test alpha (OOS)
    test_preds = model.predict(scaler.transform(test_long[FEATURES].values))

    score_long = pd.Series(test_preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(score_long, close_sub.index[test_mask], close_sub.columns)
    test_close = close_sub[test_mask]
    port_ret = backtest(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    # Also compute in-sample (training year backtest)
    train_score_long = pd.Series(train_preds, index=pd.MultiIndex.from_arrays(
        [train_long['date'], train_long['ticker']]))
    train_score_wide = ml_model.long_to_wide(
        train_score_long, close_sub.index[train_mask], close_sub.columns)
    train_close = close_sub[train_mask]
    train_port_ret = backtest(train_close, train_score_wide)
    train_s = core.stats(train_port_ret)
    spy_train = spy_ret[(spy_ret.index >= train_start) & (spy_ret.index < test_start)]
    spy_train_s = core.stats(spy_train)
    train_excess = (train_s['cagr'] - spy_train_s['cagr']) * 100 if spy_train_s and train_s else None

    return {
        'sharpe': s['sharpe'], 'mdd': s['mdd'], 'excess': excess,
        'train_excess': train_excess,
    }


def run_config(close, vol, hp, spy_ret, alpha):
    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, spy_ret, alpha)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    train_excesses = [r['train_excess'] for r in rows if r['train_excess'] is not None]
    return {
        'n_windows': len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'avg_train_excess': sum(train_excesses) / len(train_excesses) if train_excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Ridge alpha tuning on 3-feature model")
    w(f"  Features: {FEATURES}")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    results = []
    for alpha in ALPHAS:
        w(f"\n[Ridge alpha = {alpha}] running...")
        r = run_config(close, vol, hp, spy_ret, alpha)
        if r:
            r['alpha'] = alpha
            r['gap'] = r['avg_train_excess'] - r['avg_excess'] if r['avg_train_excess'] else 0
            results.append(r)
            w(f"  Train α: {r['avg_train_excess']:+.2f}%p | Test α: {r['avg_excess']:+.2f}%p | "
              f"Gap: {r['gap']:+.2f}%p | Sharpe {r['avg_sharpe']:.2f} | t={r['t_stat']:.2f}")

    # Summary
    w(f"\n{'='*100}")
    w(f"## Summary (sorted by Test alpha)")
    results.sort(key=lambda r: r['avg_excess'], reverse=True)
    w(f"{'Alpha':<10} {'Train α':>10} {'Test α':>10} {'Gap':>8} {'Sharpe':>7} {'t-stat':>7}")
    w("-" * 60)
    for r in results:
        marker = ' ⭐' if r['alpha'] == 1.0 else ''
        w(f"{r['alpha']:<10} {r['avg_train_excess']:>+9.2f}p {r['avg_excess']:>+9.2f}p "
          f"{r['gap']:>+7.2f}p {r['avg_sharpe']:>7.2f} {r['t_stat']:>7.2f}{marker}")

    # Diagnosis
    best = results[0]
    w(f"\n## Diagnosis")
    w(f"  Best alpha (by Test α): {best['alpha']}")
    w(f"  Default alpha:          1.0")
    w(f"  Default Test alpha:     {next(r['avg_excess'] for r in results if r['alpha'] == 1.0):+.2f}%p")
    w(f"  Best Test alpha:        {best['avg_excess']:+.2f}%p")
    w(f"  Difference:             {best['avg_excess'] - next(r['avg_excess'] for r in results if r['alpha'] == 1.0):+.2f}%p")

    if best['alpha'] < 0.5:
        w(f"  → Best alpha < 0.5: model could use MORE complexity (mild underfit)")
    elif best['alpha'] > 5:
        w(f"  → Best alpha > 5: model overfits (more regularization needed)")
    else:
        w(f"  → Best alpha near 1.0: default well-balanced ✅")

    out_path = os.path.join(OUTPUT_DIR, 'ridge_alpha.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
