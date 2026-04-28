"""
Bonus: monthly rebalance and Top-10 + CASH variants.

Walk-forward style aggregation for comparison.
"""
import core
import pandas as pd
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'


def load_qqq():
    qqq_path = os.path.join(OUTPUT_DIR, 'qqq_close.csv')
    df = pd.read_csv(qqq_path, parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def run_walk_forward(close, rsi_s, ma_s, vol_s, hp_overrides, qqq_ret, label):
    rows = []
    for test_start_year in range(2015, 2026):
        train_start = pd.Timestamp(f'{test_start_year - 5}-01-01')
        test_start = pd.Timestamp(f'{test_start_year}-01-01')
        test_end = pd.Timestamp(f'{test_start_year + 1}-01-01')
        if test_start > close.index.max():
            continue

        train_mask = (close.index >= train_start) & (close.index < test_start)
        test_mask = (close.index >= test_start) & (close.index < test_end)
        if train_mask.sum() < 252 or test_mask.sum() < 100:
            continue

        hp = core.merge_hp(hp_overrides)
        train_results = core.grid_search(close, rsi_s, ma_s, vol_s, train_mask, hp)
        if not train_results:
            continue
        best = train_results[0]
        test_s = core.eval_strategy(close, rsi_s, ma_s, vol_s, best['w'], test_mask, hp)
        if test_s is None or test_s['days'] < 50:
            continue

        qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
        qs = core.stats(qqq_t)
        excess = (test_s['cagr'] - qs['cagr']) * 100 if qs else None

        rows.append({
            'year': test_start_year, 'best_w': best['w'],
            'cagr': test_s['cagr'], 'sharpe': test_s['sharpe'],
            'mdd': test_s['mdd'], 'qqq_cagr': qs['cagr'] if qs else None,
            'excess': excess,
        })

    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    avg_cagr = sum(r['cagr'] for r in rows) / len(rows)
    avg_sharpe = sum(r['sharpe'] for r in rows) / len(rows)
    avg_mdd = sum(r['mdd'] for r in rows) / len(rows)
    avg_excess = sum(excesses) / len(excesses) if excesses else None
    win_count = sum(1 for e in excesses if e > 0)

    return {
        'label': label, 'rows': rows,
        'avg_cagr': avg_cagr, 'avg_sharpe': avg_sharpe, 'avg_mdd': avg_mdd,
        'avg_excess': avg_excess,
        'win_count': win_count, 'total': len(excesses),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line)
        out_lines.append(line)

    w(f"[{datetime.now()}] Bonus: rebalance frequency / Top-10 + CASH")
    w("=" * 80)

    close, vol = core.load_panel()
    base_hp = core.merge_hp({})
    rsi_s, ma_s, vol_s = core.compute_scores(close, vol, base_hp)
    qqq_ret = load_qqq()

    configs = [
        ('Daily Top-20',                      {'top_n': 20, 'rebal_days': 1}),
        ('Weekly Top-20 (baseline)',           {'top_n': 20, 'rebal_days': 5}),
        ('Biweekly Top-20',                   {'top_n': 20, 'rebal_days': 10}),
        ('Monthly Top-20',                    {'top_n': 20, 'rebal_days': 20}),
        ('Weekly Top-10',                     {'top_n': 10, 'rebal_days': 5}),
        ('Weekly Top-10 + CASH(70)',          {'top_n': 10, 'rebal_days': 5, 'cash_score': 70}),
        ('Weekly Top-10 + CASH(80)',          {'top_n': 10, 'rebal_days': 5, 'cash_score': 80}),
        ('Monthly Top-10 + CASH(70)',         {'top_n': 10, 'rebal_days': 20, 'cash_score': 70}),
    ]

    summary = []
    for label, ov in configs:
        w(f"\n[{label}] running walk-forward...", )
        res = run_walk_forward(close, rsi_s, ma_s, vol_s, ov, qqq_ret, label)
        if res:
            summary.append(res)
            w(f"  Avg CAGR {res['avg_cagr']*100:+.2f}%, Sharpe {res['avg_sharpe']:.2f}, "
              f"MDD {res['avg_mdd']*100:.2f}%, vs QQQ {res['avg_excess']:+.2f}%p, "
              f"win {res['win_count']}/{res['total']}")

    w("\n" + "=" * 90)
    w(f"{'Config':<32} {'Avg CAGR':>10} {'Avg Sh':>8} {'Avg MDD':>10} "
      f"{'vs QQQ':>10} {'Win':>8}")
    w("-" * 90)
    for r in summary:
        w(f"{r['label']:<32} {r['avg_cagr']*100:>+9.2f}% {r['avg_sharpe']:>8.2f} "
          f"{r['avg_mdd']*100:>9.2f}% {r['avg_excess']:>+9.2f}%p "
          f"{r['win_count']}/{r['total']:>3}")
    w("=" * 90)

    # Best config
    best = max(summary, key=lambda r: r['avg_excess'])
    w(f"\n★ vs QQQ excess 최고: {best['label']}")
    w(f"  Avg CAGR {best['avg_cagr']*100:.2f}%, Sharpe {best['avg_sharpe']:.2f}, "
      f"MDD {best['avg_mdd']*100:.2f}%")
    w(f"  vs QQQ excess: {best['avg_excess']:+.2f}%p, win {best['win_count']}/{best['total']}")

    out_path = os.path.join(OUTPUT_DIR, 'bonus_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
