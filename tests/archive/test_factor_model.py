"""
Task 10: Factor model walk-forward.

Replace hand-crafted RSI/MA/VOL with academic factors:
- Momentum (12-1)
- Low Volatility
- Trend Filter

Cross-sectional z-score → weighted sum → Top-N portfolio. Biweekly Top-20.
Walk-forward 11 windows. Compare to QQQ.

Output: results/factor_model_walkforward.txt
"""
import core
import factors
import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACTOR_NAMES = ['momentum', 'lowvol', 'trend']
TOP_N = 20
REBAL_DAYS = 10  # biweekly
COST_ONEWAY = 0.0005


def load_qqq():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'qqq_close.csv'),
                     parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def backtest_factor(close, z_factors, weights, top_n=TOP_N,
                    rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    """Backtest a factor-weighted strategy. Returns (port_ret, daily_cost)."""
    score = factors.combine_factors(z_factors, weights)
    score = score.where(close.notna())

    hp = {'top_n': top_n, 'rebal_days': rebal_days, 'hysteresis': 0,
          'cost_oneway': cost}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, daily_cost


def eval_factor_strategy(close, z_factors, weights, mask):
    """Apply mask to all panels then backtest."""
    c = close[mask]
    zf = {name: df[mask] for name, df in z_factors.items()}
    port, dcost = backtest_factor(c, zf, weights)
    s = core.stats(port)
    if s:
        s['cost_drag'] = float(dcost.sum())
    return s


def grid_search_factors(close, z_factors, mask, factor_names=FACTOR_NAMES, step=0.25):
    """Search over weight combinations. Returns sorted list by Sharpe."""
    candidates = factors.factor_weight_grid(factor_names, step=step)
    results = []
    for w in candidates:
        s = eval_factor_strategy(close, z_factors, w, mask)
        if s is None:
            continue
        s['weights'] = w
        results.append(s)
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    return results


def fmt_w(weights, factor_names=FACTOR_NAMES):
    return ', '.join(f'{n}:{weights[n]:.2f}' for n in factor_names)


def main():
    out_lines = []
    def w_log(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w_log(f"[{datetime.now()}] Factor Model Walk-Forward (Biweekly Top-{TOP_N})")
    w_log(f"  Factors: {FACTOR_NAMES}")
    w_log(f"  Cost: {COST_ONEWAY*200:.2f}% round-trip")
    w_log("=" * 100)

    close, vol = core.load_panel()
    z_factors = factors.compute_zscored_factors(close, vol)
    qqq_ret = load_qqq()

    w_log(f"\nUniverse: {close.shape[1]} tickers × {len(close)} dates")
    n_cands = len(factors.factor_weight_grid(FACTOR_NAMES, 0.25))
    w_log(f"Weight grid: {n_cands} candidates per window\n")

    rows = []
    for test_year in range(2015, 2026):
        train_start = pd.Timestamp(f'{test_year - 5}-01-01')
        test_start = pd.Timestamp(f'{test_year}-01-01')
        test_end = pd.Timestamp(f'{test_year + 1}-01-01')
        if test_start > close.index.max():
            continue

        train_mask = (close.index >= train_start) & (close.index < test_start)
        test_mask = (close.index >= test_start) & (close.index < test_end)
        if train_mask.sum() < 252 or test_mask.sum() < 100:
            continue

        train_results = grid_search_factors(close, z_factors, train_mask)
        if not train_results:
            continue
        best = train_results[0]
        test_s = eval_factor_strategy(close, z_factors, best['weights'], test_mask)
        if test_s is None or test_s['days'] < 50:
            continue

        qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
        qs = core.stats(qqq_t)
        excess = (test_s['cagr'] - qs['cagr']) * 100 if qs else None

        rows.append({
            'year': test_year,
            'weights': best['weights'],
            'train_cagr': best['cagr'],
            'test_cagr': test_s['cagr'],
            'test_sharpe': test_s['sharpe'],
            'test_mdd': test_s['mdd'],
            'qqq_cagr': qs['cagr'] if qs else None,
            'excess': excess,
        })

        w_log(f"  [{test_year}] {fmt_w(best['weights'])} | "
              f"train CAGR {best['cagr']*100:+.2f}% | "
              f"test CAGR {test_s['cagr']*100:+.2f}% Sh {test_s['sharpe']:.2f} | "
              f"QQQ {qs['cagr']*100:+.2f}% | Excess {excess:+.2f}%p")

    if not rows:
        w_log("\n결과 없음")
        return

    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    avg_test_cagr = sum(r['test_cagr'] for r in rows) / len(rows)
    avg_test_sharpe = sum(r['test_sharpe'] for r in rows) / len(rows)
    avg_test_mdd = sum(r['test_mdd'] for r in rows) / len(rows)
    avg_excess = sum(excesses) / len(excesses) if excesses else None
    win_count = sum(1 for e in excesses if e > 0)

    w_log(f"\n{'='*100}")
    w_log(f"## 11-window aggregate")
    w_log(f"  Avg Test CAGR:   {avg_test_cagr*100:+.2f}%")
    w_log(f"  Avg Test Sharpe: {avg_test_sharpe:.2f}")
    w_log(f"  Avg Test MDD:    {avg_test_mdd*100:.2f}%")
    if avg_excess is not None:
        w_log(f"  Avg vs QQQ:      {avg_excess:+.2f}%p")
        w_log(f"  Win rate:        {win_count}/{len(excesses)} ({win_count/len(excesses)*100:.0f}%)")

    # Weight stability
    w_counts = Counter(tuple(sorted(r['weights'].items())) for r in rows)
    w_log(f"\n## Best factor weight 분포 ({len(w_counts)} unique)")
    for w_tup, cnt in sorted(w_counts.items(), key=lambda x: -x[1])[:10]:
        wd = dict(w_tup)
        w_log(f"  {fmt_w(wd)}: {cnt}회")

    # Save CSV
    df = pd.DataFrame(rows)
    df['weights'] = df['weights'].astype(str)
    csv_path = os.path.join(OUTPUT_DIR, 'factor_model_walkforward.csv')
    df.to_csv(csv_path, index=False)
    w_log(f"\nCSV: {csv_path}")

    out_path = os.path.join(OUTPUT_DIR, 'factor_model_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
