"""
Multi-horizon momentum walk-forward.

Compare:
  - Baseline (6 features: momentum, lowvol, trend, rsi, ma, volsurge)
  - + Multi-horizon (9 features: + momentum_1m, momentum_3m, momentum_6m)

Walk-forward 31 windows on v4 (Ridge + 7y + Weekly + Top-20).

Output: results/multi_horizon_walkforward.txt
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
    if len(train_long) < 1000:
        return None

    # Drop rows with NaN in any feature (multi-horizon may have more NaN)
    train_long = train_long.dropna(subset=feature_names)
    test_long = test_long.dropna(subset=feature_names)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    X_train = train_long[feature_names].values
    y_train = train_long['target'].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

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

    coef_dict = {f: float(c) for f, c in zip(feature_names, model.coef_)}
    return {
        'year': test_year, 'cagr': s['cagr'], 'sharpe': s['sharpe'],
        'mdd': s['mdd'], 'excess': excess, 'coefs': coef_dict,
    }


def run_config(close, vol, hp, spy_ret, feature_names, label):
    rows = []
    for test_year in range(1995, 2026):
        r = run_one_window(close, vol, test_year, hp, spy_ret, feature_names)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'label': label,
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
        'rows': rows,
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Multi-horizon momentum walk-forward")
    w(f"  Top-{TOP_N} | Ridge + 7y + Weekly | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    base_features = ['momentum', 'lowvol', 'trend', 'rsi', 'ma', 'volsurge']
    multi_features = base_features + ['momentum_1m', 'momentum_3m', 'momentum_6m']

    w(f"\nLoaded {close.shape[1]} tickers")
    w(f"Baseline features: {base_features}")
    w(f"Multi-horizon features: {multi_features}\n")

    # Baseline
    hp_base = dict(ml_model.ML_HP_DEFAULT)
    hp_base['feature_names'] = base_features
    hp_base['include_multi_horizon'] = False

    # Multi-horizon
    hp_multi = dict(ml_model.ML_HP_DEFAULT)
    hp_multi['feature_names'] = multi_features
    hp_multi['include_multi_horizon'] = True

    configs = [
        ('Baseline (6 features)', hp_base, base_features),
        ('+ Multi-horizon (9 features)', hp_multi, multi_features),
    ]

    results = []
    for label, hp, feat_names in configs:
        w(f"\n[{label}] running...")
        r = run_config(close, vol, hp, spy_ret, feat_names, label)
        if r:
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | "
              f"t={r['t_stat']:.2f}")

    # Comparison
    if len(results) >= 2:
        baseline = results[0]
        multi = results[1]
        w(f"\n{'='*100}")
        w(f"## Comparison")
        w(f"{'Metric':<20} {'Baseline':>15} {'+ Multi-H':>15} {'Diff':>10}")
        w("-" * 60)
        w(f"{'CAGR':<20} {baseline['avg_cagr']*100:>+14.2f}% {multi['avg_cagr']*100:>+14.2f}% "
          f"{(multi['avg_cagr']-baseline['avg_cagr'])*100:>+9.2f}%p")
        w(f"{'Sharpe':<20} {baseline['avg_sharpe']:>15.2f} {multi['avg_sharpe']:>15.2f} "
          f"{multi['avg_sharpe']-baseline['avg_sharpe']:>+10.2f}")
        w(f"{'MDD':<20} {baseline['avg_mdd']*100:>+14.2f}% {multi['avg_mdd']*100:>+14.2f}% "
          f"{(multi['avg_mdd']-baseline['avg_mdd'])*100:>+9.2f}%p")
        w(f"{'vs SPY':<20} {baseline['avg_excess']:>+14.2f}p {multi['avg_excess']:>+14.2f}p "
          f"{multi['avg_excess']-baseline['avg_excess']:>+9.2f}p")
        w(f"{'t-stat':<20} {baseline['t_stat']:>15.2f} {multi['t_stat']:>15.2f} "
          f"{multi['t_stat']-baseline['t_stat']:>+10.2f}")
        w(f"{'Win rate':<20} {baseline['win']}/{baseline['n_windows']:<13} "
          f"{multi['win']}/{multi['n_windows']:<13}")

        # Average coefficients (multi)
        w(f"\n## Average Ridge coefficients (multi-horizon model, 31 windows)")
        avg_coefs = {}
        n = 0
        for r in multi['rows']:
            for k, v in r['coefs'].items():
                avg_coefs[k] = avg_coefs.get(k, 0) + v
            n += 1
        avg_coefs = {k: v/n for k, v in avg_coefs.items()}
        total_abs = sum(abs(v) for v in avg_coefs.values())
        w(f"{'Feature':<15} {'Avg Coef':>10} {'Abs':>10} {'Importance %':>13}")
        w("-" * 55)
        for f, c in sorted(avg_coefs.items(), key=lambda x: -abs(x[1])):
            imp = abs(c) / total_abs * 100 if total_abs > 0 else 0
            w(f"{f:<15} {c:>+10.4f} {abs(c):>10.4f} {imp:>11.1f}%")

    out_path = os.path.join(OUTPUT_DIR, 'multi_horizon_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
