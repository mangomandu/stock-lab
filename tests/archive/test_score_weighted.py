"""
Score-weighted portfolio walk-forward.

User insight: Top-N is an arbitrary parameter. Better — convert scores to weights
across ALL stocks, cut off weights below a threshold (small slices), renormalize
the survivors. The threshold is the only knob and ties naturally to seed size.

Methods compared:
  A) Softmax(score × temperature): exp-weighted by score
  B) Linear (z-score positive shift): linearly proportional
Both followed by min_weight cutoff + renormalization.

Walk-forward: 21 windows (2005-2025), Biweekly rebalance, cost 0.10%.

Output: results/score_weighted_walkforward.txt
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

REBAL_DAYS = 10
COST_ONEWAY = 0.0005


def load_qqq():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'qqq_close.csv'),
                     parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def softmax_weights(scores, temperature=1.0):
    """Softmax over a row of scores. Higher score → higher weight."""
    z = (scores - scores.mean()) / (scores.std() + 1e-9)
    exp_z = np.exp(z * temperature)
    return exp_z / exp_z.sum()


def linear_weights(scores):
    """Linearly proportional to (score - min). Always non-negative."""
    shifted = scores - scores.min()
    s = shifted.sum()
    if s == 0:
        return pd.Series(1 / len(scores), index=scores.index)
    return shifted / s


def apply_cutoff(weights, min_weight):
    """Set weights < min_weight to 0, renormalize survivors."""
    if min_weight <= 0:
        return weights
    kept = weights >= min_weight
    if not kept.any():
        # Fall back: keep top-1
        kept = weights == weights.max()
    survivor_weights = weights.where(kept, 0)
    s = survivor_weights.sum()
    return survivor_weights / s if s > 0 else survivor_weights


def build_score_weighted_holdings(score_wide, method, temperature, min_weight):
    """For each row of score_wide, compute portfolio weights, then sample at rebalance."""
    weights_list = []
    for date, row in score_wide.iterrows():
        valid = row.dropna()
        if len(valid) == 0:
            weights_list.append(pd.Series(0, index=score_wide.columns))
            continue
        if method == 'softmax':
            w = softmax_weights(valid, temperature)
        elif method == 'linear':
            w = linear_weights(valid)
        else:
            raise ValueError(method)
        w = apply_cutoff(w, min_weight)
        full = pd.Series(0.0, index=score_wide.columns)
        full.loc[w.index] = w
        weights_list.append(full)
    return pd.DataFrame(weights_list, index=score_wide.index)


def backtest_score_weighted(close, score_wide, method, temperature, min_weight,
                            rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    full_weights = build_score_weighted_holdings(score, method, temperature, min_weight)

    # Apply biweekly rebalance: only update weights every rebal_days
    n_days = len(full_weights)
    rebal_mask = np.zeros(n_days, dtype=bool)
    rebal_mask[::rebal_days] = True
    weights_rebalanced = full_weights.copy()
    last_w = pd.Series(0.0, index=full_weights.columns)
    for t in range(n_days):
        if rebal_mask[t]:
            last_w = full_weights.iloc[t].copy()
        weights_rebalanced.iloc[t] = last_w

    held = weights_rebalanced.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, daily_cost, weights_rebalanced


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def run_one_window(close, vol, test_year, hp, qqq_ret, method, temperature, min_weight):
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
    port_ret, dcost, weights = backtest_score_weighted(
        test_close, score_wide, method, temperature, min_weight)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
    qs = core.stats(qqq_t)
    excess = (s['cagr'] - qs['cagr']) * 100 if qs else None

    # Effective number of stocks held (avg)
    n_held_per_day = (weights > 0).sum(axis=1)
    avg_n_held = n_held_per_day.mean()

    return {
        'year': test_year,
        'cagr': s['cagr'],
        'sharpe': s['sharpe'],
        'mdd': s['mdd'],
        'qqq_cagr': qs['cagr'] if qs else None,
        'excess': excess,
        'avg_n_held': avg_n_held,
    }


def run_config(close, vol, qqq_ret, hp, method, temperature, min_weight, label):
    rows = []
    for test_year in range(2005, 2026):
        r = run_one_window(close, vol, test_year, hp, qqq_ret, method, temperature, min_weight)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'label': label,
        'method': method, 'temperature': temperature, 'min_weight': min_weight,
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
        'avg_n_held': sum(r['avg_n_held'] for r in rows) / len(rows),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Score-weighted portfolio walk-forward")
    w(f"  Goal: replace Top-N with score-based weights + min-weight cutoff")
    w("=" * 110)

    close, vol = core.load_panel()
    qqq_ret = load_qqq()
    hp = dict(ml_model.ML_HP_DEFAULT)

    # Configurations to compare
    configs = [
        # Top-20 baseline (for reference)
        # equal-weight Top-20 ≈ linear with min_weight=0.05 (only 20 above)
        # Will run that explicitly

        # A) Linear weighting + various cutoffs
        ('Linear + cutoff 1%',  'linear',  None, 0.01),
        ('Linear + cutoff 2%',  'linear',  None, 0.02),
        ('Linear + cutoff 3%',  'linear',  None, 0.03),
        ('Linear + cutoff 5%',  'linear',  None, 0.05),
        ('Linear + cutoff 7%',  'linear',  None, 0.07),

        # B) Softmax low temp (more equal)
        ('Softmax T=1 + cut 2%', 'softmax', 1.0, 0.02),
        ('Softmax T=2 + cut 2%', 'softmax', 2.0, 0.02),
        ('Softmax T=3 + cut 2%', 'softmax', 3.0, 0.02),
        ('Softmax T=5 + cut 2%', 'softmax', 5.0, 0.02),
        ('Softmax T=2 + cut 5%', 'softmax', 2.0, 0.05),
        ('Softmax T=5 + cut 5%', 'softmax', 5.0, 0.05),
    ]

    all_results = []
    for label, method, temp, min_w in configs:
        w(f"\n[{label}] running...")
        res = run_config(close, vol, qqq_ret, hp, method, temp, min_w, label)
        if res:
            all_results.append(res)
            w(f"  CAGR {res['avg_cagr']*100:+.2f}% | Sh {res['avg_sharpe']:.2f} | "
              f"MDD {res['avg_mdd']*100:.2f}% | "
              f"vs QQQ {res['avg_excess']:+.2f}%p | "
              f"win {res['win']}/{res['n_windows']} | "
              f"t={res['t_stat']:.2f} | avg held {res['avg_n_held']:.1f}")

    # Add Top-20 baseline for comparison
    w(f"\n[Top-20 equal weight (baseline)] running...")
    from test_extended_walkforward import run_one_window as eqw_run, t_stat_summary
    eqw_rows = []
    for test_year in range(2005, 2026):
        r = eqw_run(close, vol, test_year, hp, qqq_ret)
        if r:
            eqw_rows.append(r)
    if eqw_rows:
        eqw_excesses = [r['excess'] for r in eqw_rows if r['excess'] is not None]
        m, _, _, t = t_stat_summary(eqw_excesses)
        eqw_summary = {
            'label': 'Top-20 equal (baseline)',
            'avg_cagr': sum(r['test_cagr'] for r in eqw_rows) / len(eqw_rows),
            'avg_sharpe': sum(r['test_sharpe'] for r in eqw_rows) / len(eqw_rows),
            'avg_mdd': sum(r['test_mdd'] for r in eqw_rows) / len(eqw_rows),
            'avg_excess': m,
            'win': sum(1 for e in eqw_excesses if e > 0),
            'n_windows': len(eqw_rows),
            't_stat': t,
            'avg_n_held': 20.0,
        }
        all_results.append(eqw_summary)

    # Sort by t-stat
    all_results.sort(key=lambda r: r['t_stat'], reverse=True)

    w(f"\n{'='*120}")
    w(f"## Summary (sorted by t-stat)")
    w(f"{'Config':<30} {'CAGR':>8} {'Sh':>5} {'MDD':>7} {'vs QQQ':>9} {'Win':>8} {'t':>6} {'avgN':>6}")
    w("-" * 120)
    for r in all_results:
        w(f"{r['label']:<30} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_mdd']*100:>6.2f}% {r['avg_excess']:>+8.2f}%p "
          f"{r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f} {r['avg_n_held']:>6.1f}")

    # Best
    best = all_results[0]
    w(f"\n★ Best: {best['label']}")
    w(f"  Avg held: {best['avg_n_held']:.1f} stocks (avg over time)")
    w(f"  Excess: {best['avg_excess']:+.2f}%p, t={best['t_stat']:.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'score_weighted_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
