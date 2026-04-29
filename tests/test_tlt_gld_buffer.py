"""
TLT + GLD buffer test: blend Top-N model with defensive ETFs.

Different asset classes (bonds, commodities) for real diversification.

Test ratios: model% / ETF%
- 100/0  (current default — same as ETF buffer)
- 80/20
- 70/30
- 60/40

Yearly walk-forward 31 windows, v5 (3-feature).

Output: results/tlt_gld_buffer.txt
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

BUFFER_RATIOS = [0.0, 0.20, 0.30, 0.40]
BUFFER_ETFS = ['TLT', 'GLD']  # already in universe


def load_etf_returns(ticker):
    p = os.path.join(DATA_DIR, f'{ticker}.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last().pct_change()


def load_spy_returns():
    return load_etf_returns('SPY')


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


def run_one_window(close, vol, test_year, hp, etf_ret, buffer_ratio, spy_ret):
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

    etf_t = etf_ret.reindex(model_ret.index).fillna(0)
    blended = (1 - buffer_ratio) * model_ret + buffer_ratio * etf_t

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


def run_config(close, vol, hp, etf_ret, buffer_ratio, spy_ret):
    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, etf_ret, buffer_ratio, spy_ret)
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

    w(f"[{datetime.now()}] TLT/GLD buffer test (다른 자산군)")
    w(f"  ETFs: {BUFFER_ETFS} | Top-{TOP_N} | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy_returns()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    all_results = {}
    for etf in BUFFER_ETFS:
        etf_ret = load_etf_returns(etf)
        if etf_ret is None:
            w(f"\n[{etf}] 데이터 없음, skip")
            continue
        w(f"\n### {etf} buffer ###")
        all_results[etf] = []
        for ratio in BUFFER_RATIOS:
            label = f"Model{int((1-ratio)*100)}% + {etf}{int(ratio*100)}%"
            w(f"  [{label}] running...")
            r = run_config(close, vol, hp, etf_ret, ratio, spy_ret)
            if r:
                r['label'] = label
                r['etf'] = etf
                all_results[etf].append(r)
                w(f"    CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
                  f"MDD {r['avg_mdd']*100:.2f}% | "
                  f"vs SPY {r['avg_excess']:+.2f}%p | "
                  f"win {r['win']}/{r['n_windows']} | "
                  f"t={r['t_stat']:.2f}")

    # Summary
    w(f"\n{'='*100}")
    w(f"## Summary by ETF")
    for etf, results in all_results.items():
        if not results:
            continue
        w(f"\n### {etf}")
        w(f"{'Buffer':<25} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'vs SPY':>9} {'Win':>8} {'t':>6}")
        w("-" * 75)
        for r in results:
            w(f"{r['label']:<25} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>7.2f} "
              f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p "
              f"{r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f}")

    # Best by metric across all
    all_flat = [r for results in all_results.values() for r in results]
    if all_flat:
        w(f"\n## Best across all configs (포함 No buffer)")
        best_sharpe = max(all_flat, key=lambda r: r['avg_sharpe'])
        best_alpha = max(all_flat, key=lambda r: r['avg_excess'])
        best_mdd = max(all_flat, key=lambda r: r['avg_mdd'])
        w(f"  Sharpe 1위:  {best_sharpe['label']} (Sh {best_sharpe['avg_sharpe']:.2f})")
        w(f"  Alpha 1위:   {best_alpha['label']} (Alpha {best_alpha['avg_excess']:+.2f}%p)")
        w(f"  MDD 1위:     {best_mdd['label']} (MDD {best_mdd['avg_mdd']*100:+.2f}%)")

    # Diagnosis
    w(f"\n## Diagnosis")
    no_buf_results = [r for r in all_flat if r['buffer'] == 0]
    if no_buf_results:
        no_buf = no_buf_results[0]  # all same
        sweet = max(all_flat, key=lambda r: r['avg_sharpe'])
        delta_sharpe = sweet['avg_sharpe'] - no_buf['avg_sharpe']
        delta_alpha = sweet['avg_excess'] - no_buf['avg_excess']
        delta_mdd = sweet['avg_mdd'] - no_buf['avg_mdd']
        w(f"  No buffer baseline: Sharpe {no_buf['avg_sharpe']:.2f}, alpha {no_buf['avg_excess']:+.2f}%p, MDD {no_buf['avg_mdd']*100:.2f}%")
        w(f"  Best Sharpe config: {sweet['label']}")
        w(f"  Δ vs No buffer: Sharpe {delta_sharpe:+.2f}, alpha {delta_alpha:+.2f}%p, MDD {delta_mdd*100:+.2f}%")
        if delta_sharpe > 0.05:
            w(f"  → 안정형 옵션 가치 있음. 보수적 사용자에게 매력적.")
        else:
            w(f"  → buffer 가치 미미. 모든 사용자에게 100% model 권장.")

    out_path = os.path.join(OUTPUT_DIR, 'tlt_gld_buffer.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
