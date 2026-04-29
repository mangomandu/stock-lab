"""
Hysteresis 방지턱 깊이 탐색 — exit_n 더 높이 테스트.

이미 알려진: exit_n ∈ {20, 25, 30, 40, 50} → exit_50 best (alpha +37.28%p, turnover 6.74%)

이번 테스트: exit_n ∈ {50, 60, 75, 100, 150, 200} → sweet spot 찾기

극단 (exit_n → universe size)에선 portfolio = 거의 모든 종목 = alpha 사라짐 예상.

Output: results/hysteresis_deep.txt
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

EXIT_NS = [45, 50, 55, 60, 65, 75]  # finer grid around 50 (already known best)


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


def run_one_window(close, vol, test_year, hp, spy_ret, exit_n):
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
    port_ret, turn = backtest(test_close, score_wide, TOP_N, exit_n)
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


def run_config(close, vol, hp, spy_ret, exit_n):
    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, spy_ret, exit_n)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'exit_n': exit_n,
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

    w(f"[{datetime.now()}] Hysteresis 방지턱 깊이 탐색")
    w(f"  Top-{TOP_N}, exit_n ∈ {EXIT_NS} | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy_returns()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    results = []
    # Include known exit_50 first as anchor
    for exit_n in EXIT_NS:
        w(f"\n[exit_{exit_n}] running...")
        r = run_config(close, vol, hp, spy_ret, exit_n)
        if r:
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | t={r['t_stat']:.2f} | "
              f"turn {r['avg_turnover']*100:.2f}%/day")

    w(f"\n{'='*100}")
    w(f"## Summary (vs known exit_50: alpha +37.28%p, Sharpe 1.80, turn 6.74%)")
    w(f"{'exit_n':<10} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'Alpha':>9} {'t':>6} {'Turn':>8}")
    w("-" * 70)
    for r in results:
        w(f"{r['exit_n']:<10} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>7.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p {r['t_stat']:>6.2f} "
          f"{r['avg_turnover']*100:>7.2f}%")

    # Best
    w(f"\n## Diagnosis")
    best_alpha = max(results, key=lambda r: r['avg_excess'])
    best_sharpe = max(results, key=lambda r: r['avg_sharpe'])
    w(f"  Best alpha: exit_{best_alpha['exit_n']} ({best_alpha['avg_excess']:+.2f}%p)")
    w(f"  Best Sharpe: exit_{best_sharpe['exit_n']} (Sh {best_sharpe['avg_sharpe']:.2f})")

    # Marginal returns analysis
    w(f"\n## Marginal returns (각 단계별 추가 효과)")
    sorted_r = sorted(results, key=lambda r: r['exit_n'])
    prev = None
    for r in sorted_r:
        if prev is None:
            prev = r
            continue
        d_alpha = r['avg_excess'] - prev['avg_excess']
        d_turn = (r['avg_turnover'] - prev['avg_turnover']) * 100
        w(f"  exit_{prev['exit_n']} → exit_{r['exit_n']}: Δalpha {d_alpha:+.2f}%p, Δturn {d_turn:+.2f}%/day")
        prev = r

    out_path = os.path.join(OUTPUT_DIR, 'hysteresis_deep.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
