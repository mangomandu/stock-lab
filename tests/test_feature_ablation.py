"""
Feature ablation walk-forward.

For each feature, remove it from model and measure alpha drop.
Identifies which features are the true alpha source.

Walk-forward 31 windows on v4 (Ridge + 7y + Weekly + Top-20).

Output: results/feature_ablation.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import ml_model
import factors
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
MIN_UNIVERSE_SIZE = 100

ALL_FEATURES = ['momentum', 'lowvol', 'trend', 'rsi', 'ma', 'volsurge']


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

    valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid_tickers) < MIN_UNIVERSE_SIZE:
        return None

    close_sub = close[valid_tickers]
    vol_sub = vol[valid_tickers]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    train_long = train_long.dropna(subset=feature_names)
    test_long = test_long.dropna(subset=feature_names)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_long[feature_names].values)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, train_long['target'].values)
    X_test_s = scaler.transform(test_long[feature_names].values)
    preds = model.predict(X_test_s)

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
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
        'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'],
        'excess': excess,
    }


def run_config(close, vol, hp, spy_ret, feature_names):
    rows = []
    for test_year in range(1995, 2026):
        r = run_one_window(close, vol, test_year, hp, spy_ret, feature_names)
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

    w(f"[{datetime.now()}] Feature Ablation Walk-Forward")
    w(f"  Top-{TOP_N} | Ridge + 7y + Weekly | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"\nLoaded {close.shape[1]} tickers")

    # Run baseline + 6 ablations
    configs = [('All 6 features (baseline)', list(ALL_FEATURES))]
    for f_to_remove in ALL_FEATURES:
        feats = [f for f in ALL_FEATURES if f != f_to_remove]
        configs.append((f'Without {f_to_remove}', feats))

    # Also run single-feature versions (just to see)
    for f in ALL_FEATURES:
        configs.append((f'Only {f}', [f]))

    results = []
    for label, feats in configs:
        hp_copy = dict(hp)
        hp_copy['feature_names'] = feats
        w(f"\n[{label}] features={feats}")
        r = run_config(close, vol, hp_copy, spy_ret, feats)
        if r:
            r['label'] = label
            r['features'] = feats
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | "
              f"t={r['t_stat']:.2f}")

    # Baseline alpha
    baseline = results[0]
    w(f"\n{'='*100}")
    w(f"## Ablation analysis (sorted by alpha drop)")
    w(f"  Baseline alpha: {baseline['avg_excess']:+.2f}%p\n")

    # Compute drops for ablations (exclude baseline + single-feat)
    ablation_results = [r for r in results if r['label'].startswith('Without ')]
    ablation_results.sort(key=lambda r: baseline['avg_excess'] - r['avg_excess'], reverse=True)

    w(f"{'Removed Feature':<30} {'New Alpha':>10} {'Drop':>10} {'Drop %':>8}")
    w("-" * 65)
    for r in ablation_results:
        drop = baseline['avg_excess'] - r['avg_excess']
        drop_pct = drop / abs(baseline['avg_excess']) * 100 if baseline['avg_excess'] else 0
        w(f"{r['label']:<30} {r['avg_excess']:>+9.2f}p {drop:>+9.2f}p {drop_pct:>+7.1f}%")

    # Single-feature versions
    single_results = [r for r in results if r['label'].startswith('Only ')]
    single_results.sort(key=lambda r: r['avg_excess'], reverse=True)
    w(f"\n## Single-feature performance (one feature alone)")
    w(f"{'Feature':<30} {'Alpha':>10} {'Sharpe':>8}")
    w("-" * 55)
    for r in single_results:
        w(f"{r['label']:<30} {r['avg_excess']:>+9.2f}p {r['avg_sharpe']:>7.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'feature_ablation.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
