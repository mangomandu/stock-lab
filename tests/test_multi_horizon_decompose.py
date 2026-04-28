"""
Time-decomposed comparison: Baseline (6 features) vs Multi-horizon (9 features).

Same 31 windows, per-year alpha difference.

Output: results/multi_horizon_decompose.txt
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


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last().pct_change()


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
    return excess


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Multi-horizon decomposition by year")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    base_features = ['momentum', 'lowvol', 'trend', 'rsi', 'ma', 'volsurge']
    multi_features = base_features + ['momentum_1m', 'momentum_3m', 'momentum_6m']

    hp_base = dict(ml_model.ML_HP_DEFAULT)
    hp_base['feature_names'] = base_features
    hp_base['include_multi_horizon'] = False

    hp_multi = dict(ml_model.ML_HP_DEFAULT)
    hp_multi['feature_names'] = multi_features
    hp_multi['include_multi_horizon'] = True

    rows = []
    for test_year in range(1995, 2026):
        base = run_one_window(close, vol, test_year, hp_base, spy_ret, base_features)
        multi = run_one_window(close, vol, test_year, hp_multi, spy_ret, multi_features)
        if base is None or multi is None:
            continue
        rows.append({
            'year': test_year,
            'baseline': base,
            'multi': multi,
            'diff': multi - base,
        })

    w(f"\n{'Year':<6} {'Baseline':>10} {'Multi-H':>10} {'Diff':>10}")
    w("-" * 50)
    for r in rows:
        marker = '↑' if r['diff'] > 0.5 else ('↓' if r['diff'] < -0.5 else '·')
        w(f"{r['year']:<6} {r['baseline']:>+9.2f}p {r['multi']:>+9.2f}p {r['diff']:>+9.2f}p {marker}")

    # Aggregate
    diffs = [r['diff'] for r in rows]
    w(f"\n{'='*60}")
    w(f"## Aggregate")
    w(f"  N windows:        {len(rows)}")
    w(f"  Avg diff:         {sum(diffs)/len(diffs):+.2f}%p")
    w(f"  Multi > Base:     {sum(1 for d in diffs if d > 0)}/{len(rows)}")
    w(f"  Multi < Base:     {sum(1 for d in diffs if d < 0)}/{len(rows)}")
    w(f"  Multi >> Base (>+2%p):  {sum(1 for d in diffs if d > 2)}/{len(rows)}")
    w(f"  Multi << Base (<-2%p):  {sum(1 for d in diffs if d < -2)}/{len(rows)}")

    # By period
    periods = {
        '1995-2004': (1995, 2004),
        '2005-2014': (2005, 2014),
        '2015-2025': (2015, 2025),
    }
    w(f"\n## By period")
    for label, (y1, y2) in periods.items():
        sub = [r for r in rows if y1 <= r['year'] <= y2]
        if not sub:
            continue
        sub_diffs = [r['diff'] for r in sub]
        w(f"  {label}: avg diff {sum(sub_diffs)/len(sub_diffs):+.2f}%p, "
          f"win {sum(1 for d in sub_diffs if d > 0)}/{len(sub_diffs)}")

    out_path = os.path.join(OUTPUT_DIR, 'multi_horizon_decompose.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
