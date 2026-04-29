"""
Ensemble test: Ridge + LightGBM 평균.

Logic:
- Train Ridge (linear) and LightGBM (non-linear) on same features
- Final score = (ridge_pred + lgbm_pred) / 2  (or rank-averaged)
- Top-N from ensemble score

Goal: combine two models' strengths (Ridge linear, LGBM non-linear).

v5.2 environment: 3 features + 7y + Weekly + Top-20.

Output: results/ensemble.txt
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

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. pip install lightgbm")
    sys.exit(1)

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 5
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
FEATURES = ['lowvol', 'rsi', 'volsurge']

LGB_PARAMS = {
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.9,
    'verbose': -1,
}


def load_spy_returns():
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


def run_one_window(close, vol, test_year, hp, spy_ret, mode):
    """mode = 'ridge', 'lgbm', or 'ensemble'."""
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

    X_train = train_long[FEATURES].values
    y_train = train_long['target'].values
    X_test = test_long[FEATURES].values

    preds_list = []
    if mode in ('ridge', 'ensemble'):
        scaler = StandardScaler()
        Xs_tr = scaler.fit_transform(X_train)
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xs_tr, y_train)
        Xs_te = scaler.transform(X_test)
        ridge_preds = ridge.predict(Xs_te)
        preds_list.append(ridge_preds)

    if mode in ('lgbm', 'ensemble'):
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_model = lgb.train(LGB_PARAMS, lgb_train, num_boost_round=200)
        lgb_preds = lgb_model.predict(X_test)
        preds_list.append(lgb_preds)

    # Combine
    if len(preds_list) == 1:
        preds = preds_list[0]
    else:
        # Rank-average for robustness
        from scipy.stats import rankdata
        ranks = [rankdata(p) for p in preds_list]
        preds = np.mean(ranks, axis=0)

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    port_ret = backtest(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_full = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_full)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {
        'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'],
        'excess': excess,
    }


def run_config(close, vol, hp, spy_ret, mode):
    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, spy_ret, mode)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'mode': mode,
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

    w(f"[{datetime.now()}] Ensemble test (Ridge + LightGBM 평균)")
    w(f"  Features: {FEATURES} | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy_returns()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    results = []
    for mode in ['ridge', 'lgbm', 'ensemble']:
        w(f"\n[{mode}] running...")
        r = run_config(close, vol, hp, spy_ret, mode)
        if r:
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | t={r['t_stat']:.2f}")

    w(f"\n{'='*100}")
    w(f"## Summary")
    w(f"{'Mode':<15} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'Alpha':>9} {'t':>6}")
    w("-" * 65)
    for r in results:
        w(f"{r['mode']:<15} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>7.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p {r['t_stat']:>6.2f}")

    # Diagnosis
    w(f"\n## Diagnosis")
    ridge_r = next(r for r in results if r['mode'] == 'ridge')
    ens_r = next(r for r in results if r['mode'] == 'ensemble')
    delta_alpha = ens_r['avg_excess'] - ridge_r['avg_excess']
    delta_sharpe = ens_r['avg_sharpe'] - ridge_r['avg_sharpe']
    w(f"  Ridge baseline: Sharpe {ridge_r['avg_sharpe']:.2f}, alpha {ridge_r['avg_excess']:+.2f}%p")
    w(f"  Ensemble: Sharpe {ens_r['avg_sharpe']:.2f}, alpha {ens_r['avg_excess']:+.2f}%p")
    w(f"  Δ vs Ridge: Sharpe {delta_sharpe:+.2f}, alpha {delta_alpha:+.2f}%p")
    if delta_alpha > 2.0 and delta_sharpe > 0.05:
        w(f"  ✅ Ensemble 강한 효과. v6 트리거 가능.")
    elif delta_alpha > 0.5:
        w(f"  ⚠ 약간 개선. v5.3로 채택 검토")
    else:
        w(f"  ❌ 효과 미미. Ridge 단독 유지 권장.")

    out_path = os.path.join(OUTPUT_DIR, 'ensemble.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
