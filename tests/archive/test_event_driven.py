"""
Event-driven rebalancing: Turnover and Score Confidence triggers.

Compare against periodic Biweekly baseline.

Triggers:
  A) Turnover-based: rebalance only if |Δw| ≥ threshold
  C) Score confidence: rebalance only if (top_avg - rest_avg) ≥ threshold
  D) Hybrid: A AND C both satisfied
  Plus baselines: Daily Top-20, Biweekly Top-20

Walk-forward 21 windows (2005-2025), Top-20 equal weight, cost 0.10%.

Output: results/event_driven_walkforward.txt
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
COST_ONEWAY = 0.0005


def load_qqq():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'qqq_close.csv'),
                     parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def turnover(prev_w, new_w):
    """Sum |Δw| / 2 = fraction of portfolio that turns over."""
    return (new_w - prev_w).abs().sum() / 2


def score_confidence(scores, top_n=TOP_N):
    """Top-N average score - rest-of-universe average score.
    Higher = model has stronger signal about what's good vs bad."""
    valid = scores.dropna()
    if len(valid) < top_n + 1:
        return 0
    sorted_s = valid.sort_values(ascending=False)
    top_avg = sorted_s.iloc[:top_n].mean()
    rest_avg = sorted_s.iloc[top_n:].mean()
    return top_avg - rest_avg


def build_event_driven_holdings(score_wide, top_n, mode, params,
                                 max_hold_days=60, min_hold_days=2):
    """For each day, compute new Top-N holdings, but only adopt if trigger fires.

    mode: 'turnover' | 'confidence' | 'hybrid' | 'biweekly' | 'daily'
    params: {threshold, ...}
    max_hold_days: force rebalance after this many days (safety net)
    min_hold_days: don't rebalance more often than this
    """
    n_days, n_stocks = score_wide.shape
    score_arr = score_wide.values
    holdings = np.zeros((n_days, n_stocks), dtype=bool)
    current = np.zeros(n_stocks, dtype=bool)
    days_since_rebal = max_hold_days  # force first rebalance

    rebal_count = 0

    for t in range(n_days):
        row = score_arr[t]
        valid = ~np.isnan(row)
        if not valid.any():
            holdings[t] = current
            continue

        # New optimal Top-N for today
        ranked = np.argsort(np.where(valid, row, -np.inf))[::-1]
        new_top = np.zeros(n_stocks, dtype=bool)
        new_top[ranked[:top_n]] = True

        # Trigger check
        should_rebal = False

        if days_since_rebal >= max_hold_days:
            should_rebal = True
        elif days_since_rebal < min_hold_days:
            should_rebal = False
        elif mode == 'biweekly':
            should_rebal = (days_since_rebal >= 10)
        elif mode == 'daily':
            should_rebal = True
        elif mode == 'turnover':
            new_w = new_top.astype(float) / new_top.sum() if new_top.sum() else new_top.astype(float)
            cur_w = current.astype(float) / current.sum() if current.sum() else current.astype(float)
            tov = np.abs(new_w - cur_w).sum() / 2
            should_rebal = tov >= params['threshold']
        elif mode == 'confidence':
            valid_scores = pd.Series(np.where(valid, row, np.nan))
            conf = score_confidence(valid_scores, top_n)
            should_rebal = conf >= params['threshold']
        elif mode == 'hybrid':
            new_w = new_top.astype(float) / new_top.sum() if new_top.sum() else new_top.astype(float)
            cur_w = current.astype(float) / current.sum() if current.sum() else current.astype(float)
            tov = np.abs(new_w - cur_w).sum() / 2
            valid_scores = pd.Series(np.where(valid, row, np.nan))
            conf = score_confidence(valid_scores, top_n)
            should_rebal = (tov >= params.get('turnover_threshold', 0.3)) and \
                           (conf >= params.get('confidence_threshold', 0.005))

        if should_rebal:
            current = new_top
            days_since_rebal = 0
            rebal_count += 1
        else:
            days_since_rebal += 1

        holdings[t] = current

    holdings_df = pd.DataFrame(holdings.astype(float),
                               index=score_wide.index, columns=score_wide.columns)
    return holdings_df, rebal_count


def backtest_event_driven(close, score_wide, top_n, mode, params, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    in_top, n_rebal = build_event_driven_holdings(score, top_n, mode, params)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, daily_cost, n_rebal


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def run_one_window(close, vol, test_year, hp, qqq_ret, mode, params):
    train_start = pd.Timestamp(f'{test_year - 5}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close, vol, train_mask, test_mask, hp)
    if len(train_long) < 1000:
        return None

    model = ml_model.train_model(train_long, hp)
    if model is None:
        return None

    score_long = ml_model.score_with_model(model, test_long, hp)
    score_wide = ml_model.long_to_wide(
        score_long, close.index[test_mask], close.columns)

    test_close = close[test_mask]
    port_ret, dcost, n_rebal = backtest_event_driven(
        test_close, score_wide, TOP_N, mode, params)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
    qs = core.stats(qqq_t)
    excess = (s['cagr'] - qs['cagr']) * 100 if qs else None

    return {
        'year': test_year,
        'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'],
        'qqq_cagr': qs['cagr'] if qs else None,
        'excess': excess,
        'n_rebal': n_rebal,
    }


def run_config(close, vol, qqq_ret, hp, mode, params, label):
    rows = []
    for test_year in range(2005, 2026):
        r = run_one_window(close, vol, test_year, hp, qqq_ret, mode, params)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'label': label, 'mode': mode, 'params': params,
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
        'avg_n_rebal': sum(r['n_rebal'] for r in rows) / len(rows),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line)
        out_lines.append(line)

    w(f"[{datetime.now()}] Event-driven rebalancing walk-forward")
    w(f"  Universe: NASDAQ-100 current snapshot (98 tickers)")
    w(f"  Top-N: {TOP_N} equal weight | Cost: {COST_ONEWAY*200:.2f}% RT")
    w("=" * 100)

    close, vol = core.load_panel()
    qqq_ret = load_qqq()
    hp = dict(ml_model.ML_HP_DEFAULT)

    configs = [
        # Baselines
        ('Daily Top-20',           'daily',      {}),
        ('Biweekly Top-20',        'biweekly',   {}),

        # Turnover-based (A)
        ('Turnover ≥ 10%',         'turnover',   {'threshold': 0.10}),
        ('Turnover ≥ 20%',         'turnover',   {'threshold': 0.20}),
        ('Turnover ≥ 30%',         'turnover',   {'threshold': 0.30}),
        ('Turnover ≥ 40%',         'turnover',   {'threshold': 0.40}),
        ('Turnover ≥ 50%',         'turnover',   {'threshold': 0.50}),

        # Confidence-based (C)
        ('Confidence ≥ 0.005',     'confidence', {'threshold': 0.005}),
        ('Confidence ≥ 0.010',     'confidence', {'threshold': 0.010}),
        ('Confidence ≥ 0.015',     'confidence', {'threshold': 0.015}),
        ('Confidence ≥ 0.020',     'confidence', {'threshold': 0.020}),

        # Hybrid (D)
        ('Hybrid TO≥30 + C≥0.010', 'hybrid',     {'turnover_threshold': 0.30, 'confidence_threshold': 0.010}),
        ('Hybrid TO≥20 + C≥0.005', 'hybrid',     {'turnover_threshold': 0.20, 'confidence_threshold': 0.005}),
    ]

    results = []
    for label, mode, params in configs:
        w(f"\n[{label}] running...")
        r = run_config(close, vol, qqq_ret, hp, mode, params, label)
        if r:
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% | Sh {r['avg_sharpe']:.2f} | "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs QQQ {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | "
              f"t={r['t_stat']:.2f} | reb/yr {r['avg_n_rebal']:.0f}")

    # Sort by Sharpe
    results.sort(key=lambda r: r['avg_sharpe'], reverse=True)

    w(f"\n{'='*120}")
    w(f"## Summary (sorted by Sharpe)")
    w(f"{'Config':<28} {'CAGR':>8} {'Sh':>5} {'MDD':>7} {'vs QQQ':>9} {'Win':>8} {'t':>6} {'Rebal/yr':>9}")
    w("-" * 120)
    for r in results:
        w(f"{r['label']:<28} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_mdd']*100:>6.2f}% {r['avg_excess']:>+8.2f}%p "
          f"{r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f} {r['avg_n_rebal']:>8.0f}")

    best = results[0]
    w(f"\n★ Best Sharpe: {best['label']}")
    w(f"  CAGR {best['avg_cagr']*100:+.2f}%, Sharpe {best['avg_sharpe']:.2f}, "
      f"vs QQQ {best['avg_excess']:+.2f}%p")
    w(f"  Rebalances/year: {best['avg_n_rebal']:.0f} (vs daily ~252)")

    # Best by alpha
    by_alpha = sorted(results, key=lambda r: r['avg_excess'] or 0, reverse=True)[0]
    w(f"\n★ Best vs QQQ: {by_alpha['label']}")
    w(f"  vs QQQ {by_alpha['avg_excess']:+.2f}%p, Sharpe {by_alpha['avg_sharpe']:.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'event_driven_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
