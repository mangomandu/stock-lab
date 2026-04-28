"""
Task 4: Cost sensitivity test.

Re-run Weekly Top-20 with cost levels 0.05%, 0.1%, 0.2%, 0.5%, 1.0% round-trip.
Find break-even cost where strategy stops beating QQQ.

Output: results/cost_sensitivity.txt
"""
import core
import pandas as pd
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (round_trip_pct, label)
COST_LEVELS = [
    (0.0001, '0.01% (HFT)'),
    (0.0005, '0.05%'),
    (0.0010, '0.10% (default)'),
    (0.0020, '0.20%'),
    (0.0050, '0.50% (Korean broker)'),
    (0.0100, '1.00% (high)'),
    (0.0200, '2.00% (very high)'),
]


def load_qqq():
    qqq_path = os.path.join(OUTPUT_DIR, 'qqq_close.csv')
    if not os.path.exists(qqq_path):
        return None
    df = pd.read_csv(qqq_path, parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def main():
    out_lines = []
    def w(line=''):
        print(line)
        out_lines.append(line)

    w(f"[{datetime.now()}] Cost sensitivity (Weekly Top-20, split 2018)")
    w("=" * 100)

    close, vol = core.load_panel()
    base_hp = core.merge_hp({'top_n': 20, 'rebal_days': 5})
    rsi_s, ma_s, vol_s = core.compute_scores(close, vol, base_hp)
    train_mask, test_mask = core.get_period_masks(close, base_hp)
    qqq_ret = load_qqq()

    qqq_test_s = None
    if qqq_ret is not None:
        qqq_test = qqq_ret[qqq_ret.index >= pd.Timestamp(base_hp['split_date'])]
        qqq_test_s = core.stats(qqq_test)
        w(f"\nQQQ Test CAGR (reference): {qqq_test_s['cagr']*100:.2f}%, "
          f"Sharpe {qqq_test_s['sharpe']:.2f}\n")

    w(f"{'Round-trip cost':<22} {'One-way':>10} {'Best W':<22} "
      f"{'Train CAGR':>10} {'Test CAGR':>10} {'Test Sh':>8} "
      f"{'Cost drag':>10} {'vs QQQ':>8}")
    w("-" * 100)

    for round_trip, label in COST_LEVELS:
        one_way = round_trip / 2
        hp = core.merge_hp({**base_hp, 'cost_oneway': one_way})

        train_results = core.grid_search(close, rsi_s, ma_s, vol_s, train_mask, hp)
        if not train_results:
            continue
        best = train_results[0]
        test_s = core.eval_strategy(close, rsi_s, ma_s, vol_s, best['w'], test_mask, hp)
        if test_s is None:
            continue

        excess = ''
        if qqq_test_s:
            diff = (test_s['cagr'] - qqq_test_s['cagr']) * 100
            excess = f"{diff:+.2f}%p"

        w(f"{label:<22} {one_way*100:>9.3f}% {core.fmt_w(best['w']):<22} "
          f"{best['cagr']*100:>9.2f}% {test_s['cagr']*100:>9.2f}% "
          f"{test_s['sharpe']:>8.2f} {test_s['cost_drag']*100:>9.2f}% "
          f"{excess:>8}")

    w("-" * 100)
    w("\n해석:")
    w("- 회전율은 매번 grid search로 새로 찾으니 비용 따라 최적 w도 변함")
    w("- 비용이 올라갈수록 회전율 적은 w를 선호 → 신호 약해짐")
    w("- 'vs QQQ'가 양수인 비용 구간이 실전 가능한 범위")

    out_path = os.path.join(OUTPUT_DIR, 'cost_sensitivity.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
