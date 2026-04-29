"""
Top-N 늘리기 vs Hysteresis 비교.

목적: 회전율 감소 효과를 두 가지 방식으로 비교
- A) Top-N 자체를 늘림 (Top-30, 40, 50): 자연스럽게 signal stability ↑ but 신호 희석
- B) Top-20 + Hysteresis (exit_n=30, 40, 50): 진입 conviction 유지하면서 종료 임계점 늘림

비교 대상:
- Top-20 (baseline) | Top-30 | Top-40 | Top-50 (모두 no hyst)
- Top-20 + exit_30 | Top-20 + exit_40 | Top-20 + exit_50

Output: results/topn_vs_hyst.txt
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

REBAL_DAYS = 5
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
FEATURES = ['lowvol', 'rsi', 'volsurge']

# (top_n, exit_n) configs
CONFIGS = [
    (20, 20),  # baseline
    (30, 30),
    (40, 40),
    (50, 50),
    (20, 30),
    (20, 40),
    (20, 50),
]


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


def build_holdings_hyst(score_wide, top_n, exit_n, rebal_days):
    n_days, n_stocks = score_wide.shape
    holdings = np.zeros((n_days, n_stocks))
    cols = score_wide.columns.tolist()
    last_holdings = set()
    last_rebal = -rebal_days

    for t in range(n_days):
        if (t - last_rebal) >= rebal_days:
            row = score_wide.iloc[t].dropna()
            ranked = row.sort_values(ascending=False)
            ranks = {ticker: i for i, ticker in enumerate(ranked.index, 1)}

            new_holdings = set()
            for h in last_holdings:
                if h in ranks and ranks[h] <= exit_n:
                    new_holdings.add(h)
            for ticker, rank in ranks.items():
                if len(new_holdings) >= top_n:
                    break
                if ticker not in new_holdings and rank <= top_n:
                    new_holdings.add(ticker)

            last_holdings = new_holdings
            last_rebal = t

        for h in last_holdings:
            if h in cols:
                holdings[t, cols.index(h)] = 1.0

    return pd.DataFrame(holdings, index=score_wide.index, columns=cols)


def backtest(close, score_wide, top_n, exit_n):
    score = score_wide.where(close.notna())
    in_top = build_holdings_hyst(score, top_n, exit_n, REBAL_DAYS)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * COST_ONEWAY
    turnover = held.diff().abs().sum(axis=1).fillna(0) / 2
    return port_gross - daily_cost, turnover.mean()


def run_one_window(close, vol, test_year, hp, spy_ret, top_n, exit_n):
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
    port_ret, turn = backtest(test_close, score_wide, top_n, exit_n)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_full = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_full)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {
        'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'],
        'excess': excess, 'turnover': turn,
    }


def run_config(close, vol, hp, spy_ret, top_n, exit_n):
    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, spy_ret, top_n, exit_n)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'top_n': top_n, 'exit_n': exit_n,
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'avg_turnover': sum(r['turnover'] for r in rows) / len(rows),
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Top-N 늘리기 vs Hysteresis 비교")
    w(f"  Configs: {CONFIGS}")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy_returns()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    results = []
    for top_n, exit_n in CONFIGS:
        if exit_n == top_n:
            label = f"Top-{top_n} (no hyst)"
        else:
            label = f"Top-{top_n} + exit_{exit_n}"
        w(f"\n[{label}] running...")
        r = run_config(close, vol, hp, spy_ret, top_n, exit_n)
        if r:
            r['label'] = label
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | t={r['t_stat']:.2f} | "
              f"turn {r['avg_turnover']*100:.2f}%/day")

    w(f"\n{'='*100}")
    w(f"## Summary")
    w(f"{'Config':<28} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'Alpha':>9} {'t':>6} {'Turn':>8}")
    w("-" * 80)
    for r in results:
        w(f"{r['label']:<28} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>7.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p {r['t_stat']:>6.2f} "
          f"{r['avg_turnover']*100:>7.2f}%")

    # Comparison: at similar turnover levels
    w(f"\n## 회전율 비슷한 수준끼리 비교")
    w(f"  ~7% turnover:")
    for r in results:
        if 5 < r['avg_turnover']*100 < 8:
            w(f"    {r['label']}: alpha {r['avg_excess']:+.2f}%p, Sharpe {r['avg_sharpe']:.2f}")
    w(f"  ~10% turnover (baseline):")
    for r in results:
        if 9 < r['avg_turnover']*100 < 12:
            w(f"    {r['label']}: alpha {r['avg_excess']:+.2f}%p, Sharpe {r['avg_sharpe']:.2f}")

    w(f"\n## Diagnosis")
    # Best alpha
    best_alpha = max(results, key=lambda r: r['avg_excess'])
    best_sharpe = max(results, key=lambda r: r['avg_sharpe'])
    w(f"  Best alpha: {best_alpha['label']} ({best_alpha['avg_excess']:+.2f}%p)")
    w(f"  Best Sharpe: {best_sharpe['label']} (Sh {best_sharpe['avg_sharpe']:.2f})")

    # Top-20+exit_50 vs Top-50 직접 비교
    hyst_50 = next((r for r in results if r['top_n']==20 and r['exit_n']==50), None)
    top_50 = next((r for r in results if r['top_n']==50 and r['exit_n']==50), None)
    if hyst_50 and top_50:
        w(f"\n  Top-20 + exit_50 vs Top-50 (no hyst):")
        w(f"    Alpha:    {hyst_50['avg_excess']:+.2f}%p vs {top_50['avg_excess']:+.2f}%p (Δ {hyst_50['avg_excess']-top_50['avg_excess']:+.2f}%p)")
        w(f"    Sharpe:   {hyst_50['avg_sharpe']:.2f} vs {top_50['avg_sharpe']:.2f}")
        w(f"    Turnover: {hyst_50['avg_turnover']*100:.2f}% vs {top_50['avg_turnover']*100:.2f}%/day")
        if hyst_50['avg_excess'] > top_50['avg_excess'] + 1:
            w(f"  → Hysteresis 가 Top-N 늘리는 것보다 우월 (자본 집중 효과)")
        else:
            w(f"  → 둘이 비슷 — Top-N 늘리는 게 더 단순")

    out_path = os.path.join(OUTPUT_DIR, 'topn_vs_hyst.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
