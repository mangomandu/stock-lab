"""
Drift Tolerance walk-forward.

Hold while still in Top-K (K > N). Sell only when drops out.
New stocks added when entering Top-N AND we have a slot.

= "stale 종목만 정리" 룰. Whipsaw 회피.

Compare K = N (= Biweekly baseline implicitly), 25, 30, 35, 40, 50, 60 (= almost no churn).

Walk-forward 21 windows (2005-2025), Top-N=20, cost 0.10%.
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


def build_drift_holdings(score_wide, top_n, hold_until_k, max_hold_days=60):
    """Drift tolerance rule.

    For each day:
      1. Sell holdings that drop out of Top-K
      2. Fill empty slots with highest-scored stocks not yet held
      3. Force full rebalance every max_hold_days as safety

    hold_until_k = N means strict (= daily Top-N rebalance)
    hold_until_k > N means tolerance (drift allowed)
    """
    n_days, n_stocks = score_wide.shape
    score_arr = score_wide.values
    holdings = np.zeros((n_days, n_stocks), dtype=bool)
    current = np.zeros(n_stocks, dtype=bool)
    days_since_full = 0
    rebal_count = 0

    for t in range(n_days):
        row = score_arr[t]
        valid = ~np.isnan(row)
        if not valid.any():
            holdings[t] = current
            continue

        # Rank stocks
        sorted_scores = np.where(valid, row, -np.inf)
        ranks = (-sorted_scores).argsort().argsort() + 1  # rank 1 = highest

        if days_since_full >= max_hold_days:
            # Forced full rebalance
            new_top = (ranks <= top_n) & valid
            if not np.array_equal(new_top, current):
                rebal_count += 1
            current = new_top.copy()
            days_since_full = 0
        else:
            # Drift rule:
            # 1. Sell stocks that drifted out of Top-K
            in_top_k = (ranks <= hold_until_k) & valid
            kept = current & in_top_k

            # 2. Fill slots: take highest-rank stocks not yet held
            n_need = top_n - int(kept.sum())
            if n_need > 0:
                ranked_idx = (-sorted_scores).argsort()
                fill = []
                for idx in ranked_idx:
                    if not kept[idx] and valid[idx]:
                        fill.append(idx)
                        if len(fill) == n_need:
                            break
                kept[fill] = True

            if not np.array_equal(kept, current):
                rebal_count += 1
            current = kept
            days_since_full += 1

        holdings[t] = current

    holdings_df = pd.DataFrame(holdings.astype(float),
                               index=score_wide.index, columns=score_wide.columns)
    return holdings_df, rebal_count


def backtest_drift(close, score_wide, top_n, hold_until_k, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    in_top, n_rebal = build_drift_holdings(score, top_n, hold_until_k)
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


def biweekly_backtest(close, score_wide, top_n, rebal_days, cost=COST_ONEWAY):
    """Biweekly baseline using core.build_holdings."""
    score = score_wide.where(close.notna())
    hp = {'top_n': top_n, 'rebal_days': rebal_days, 'hysteresis': 0,
          'cost_oneway': cost}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, daily_cost


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
    if mode == 'drift':
        port_ret, dcost, n_rebal = backtest_drift(
            test_close, score_wide, TOP_N, params['hold_until_k'])
    elif mode == 'biweekly':
        port_ret, dcost = biweekly_backtest(
            test_close, score_wide, TOP_N, params['rebal_days'])
        n_rebal = int(252 / params['rebal_days'])
    else:
        return None

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
        'label': label,
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

    w(f"[{datetime.now()}] Drift Tolerance walk-forward")
    w(f"  Rule: hold while in Top-K, sell when drops out")
    w("=" * 100)

    close, vol = core.load_panel()
    qqq_ret = load_qqq()
    hp = dict(ml_model.ML_HP_DEFAULT)

    configs = [
        # Baselines
        ('Biweekly Top-20',           'biweekly', {'rebal_days': 10}),
        ('Weekly Top-20',             'biweekly', {'rebal_days': 5}),
        ('Daily Top-20',              'biweekly', {'rebal_days': 1}),

        # Drift tolerance
        ('Drift K=20 (strict)',       'drift', {'hold_until_k': 20}),
        ('Drift K=25',                'drift', {'hold_until_k': 25}),
        ('Drift K=30',                'drift', {'hold_until_k': 30}),
        ('Drift K=35',                'drift', {'hold_until_k': 35}),
        ('Drift K=40',                'drift', {'hold_until_k': 40}),
        ('Drift K=50',                'drift', {'hold_until_k': 50}),
        ('Drift K=60',                'drift', {'hold_until_k': 60}),
        ('Drift K=80',                'drift', {'hold_until_k': 80}),
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
              f"t={r['t_stat']:.2f} | avg reb {r['avg_n_rebal']:.0f}")

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

    best_sh = results[0]
    w(f"\n★ Best Sharpe: {best_sh['label']}")
    w(f"  CAGR {best_sh['avg_cagr']*100:+.2f}%, Sharpe {best_sh['avg_sharpe']:.2f}, "
      f"vs QQQ {best_sh['avg_excess']:+.2f}%p, t={best_sh['t_stat']:.2f}")

    by_alpha = sorted(results, key=lambda r: r['avg_excess'] or 0, reverse=True)[0]
    if by_alpha['label'] != best_sh['label']:
        w(f"\n★ Best alpha: {by_alpha['label']}")
        w(f"  vs QQQ {by_alpha['avg_excess']:+.2f}%p, Sharpe {by_alpha['avg_sharpe']:.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'drift_tolerance.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
