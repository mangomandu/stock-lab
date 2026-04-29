"""
ETF buffer test: blend Top-N model with passive index ETF (SPY).

Test mix ratios: model% / SPY%
- 100/0  (current default)
- 90/10
- 80/20
- 70/30
- 60/40
- 50/50

Goal: see if Sharpe ↑ via SPY diversification, alpha trade-off.

Yearly walk-forward 31 windows (1995-2025), v5 (3-feature).

Output: results/etf_buffer.txt
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

BUFFER_RATIOS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]


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
    """Run model-only backtest, returns daily portfolio returns."""
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


def run_one_window(close, vol, test_year, hp, spy_ret, buffer_ratio):
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
    model = Ridge(alpha=1.0)
    model.fit(Xs, train_long['target'].values)
    preds = model.predict(scaler.transform(test_long[FEATURES].values))

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    model_ret = backtest(test_close, score_wide)

    # SPY ret aligned to test dates
    spy_t = spy_ret.reindex(model_ret.index).fillna(0)

    # Blend: buffer_ratio in SPY, rest in model
    # Subtract minor cost on SPY rebalance? Treat as buy & hold (no cost)
    blended = (1 - buffer_ratio) * model_ret + buffer_ratio * spy_t

    s = core.stats(blended)
    if s is None or s['days'] < 50:
        return None

    spy_full = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_full)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {
        'cagr': s['cagr'], 'sharpe': s['sharpe'],
        'mdd': s['mdd'], 'excess': excess,
    }


def run_config(close, vol, hp, spy_ret, buffer_ratio):
    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, spy_ret, buffer_ratio)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'buffer': buffer_ratio,
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

    w(f"[{datetime.now()}] ETF buffer test: model + SPY blend")
    w(f"  Features: {FEATURES} | Top-{TOP_N} | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy_returns()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    results = []
    for ratio in BUFFER_RATIOS:
        label = f"Model{int((1-ratio)*100)}% + SPY{int(ratio*100)}%"
        w(f"\n[{label}] running...")
        r = run_config(close, vol, hp, spy_ret, ratio)
        if r:
            r['label'] = label
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | "
              f"t={r['t_stat']:.2f}")

    # Summary table
    w(f"\n{'='*100}")
    w(f"## Summary table")
    w(f"{'Buffer':<12} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'vs SPY':>9} {'Win':>8} {'t':>6}")
    w("-" * 70)
    for r in results:
        w(f"{r['label']:<25} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>7.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p "
          f"{r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f}")

    # Best by metric
    w(f"\n## Best by metric")
    best_sharpe = max(results, key=lambda r: r['avg_sharpe'])
    best_alpha = max(results, key=lambda r: r['avg_excess'])
    best_mdd = max(results, key=lambda r: r['avg_mdd'])  # MDD is negative, max = least bad
    w(f"  Sharpe 1위:  {best_sharpe['label']} (Sh {best_sharpe['avg_sharpe']:.2f})")
    w(f"  Alpha 1위:   {best_alpha['label']} (Alpha {best_alpha['avg_excess']:+.2f}%p)")
    w(f"  MDD 1위:     {best_mdd['label']} (MDD {best_mdd['avg_mdd']*100:+.2f}%)")

    # Diagnosis
    w(f"\n## Diagnosis")
    no_buf = next(r for r in results if r['buffer'] == 0)
    sweet = max(results, key=lambda r: r['avg_sharpe'])
    if sweet['buffer'] == 0:
        w(f"  100% model이 Sharpe 1위 — buffer 효과 없음. 현재 default 유지.")
    else:
        delta_sharpe = sweet['avg_sharpe'] - no_buf['avg_sharpe']
        delta_alpha = sweet['avg_excess'] - no_buf['avg_excess']
        w(f"  Sweet spot: {sweet['label']}")
        w(f"  vs No buffer: Sharpe {delta_sharpe:+.2f}, Alpha {delta_alpha:+.2f}%p")
        if delta_sharpe > 0.05 and delta_alpha > -5:
            w(f"  → 추천: buffer 도입 검토 (Sharpe ↑, alpha 손실 작음)")
        elif delta_sharpe > 0.05:
            w(f"  → buffer 추가 시 Sharpe ↑, but alpha 손실 큼 ({delta_alpha:.1f}%p)")
        else:
            w(f"  → 추가 가치 미미. 현재 default 유지 권장.")

    out_path = os.path.join(OUTPUT_DIR, 'etf_buffer.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
