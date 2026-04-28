"""
Task 5: Walk-forward validation.

Rolling 5y train / 1y test windows from 2015 onwards. For each window, run
grid search on train and evaluate best w on test. Aggregate OOS results.

Output: results/walk_forward.txt
"""
import core
import pandas as pd
import os
from datetime import datetime
from collections import Counter

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Walk-forward windows
TRAIN_YEARS = 5
TEST_YEARS = 1
TEST_START_YEARS = list(range(2015, 2026))  # 2015 to 2025


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

    w(f"[{datetime.now()}] Walk-forward validation (Weekly Top-20)")
    w(f"  Train window: {TRAIN_YEARS}y | Test window: {TEST_YEARS}y")
    w("=" * 100)

    close, vol = core.load_panel()
    base_hp = core.merge_hp({'top_n': 20, 'rebal_days': 5})
    rsi_s, ma_s, vol_s = core.compute_scores(close, vol, base_hp)
    qqq_ret = load_qqq()

    w(f"\n{'Train Period':<25} {'Test Period':<25} {'Best W':<22} "
      f"{'TestCAGR':>9} {'TestSh':>7} {'TestMDD':>8} {'QQQ':>8} {'Excess':>9}")
    w("-" * 120)

    rows = []
    for test_start_year in TEST_START_YEARS:
        train_start = pd.Timestamp(f'{test_start_year - TRAIN_YEARS}-01-01')
        test_start = pd.Timestamp(f'{test_start_year}-01-01')
        test_end = pd.Timestamp(f'{test_start_year + TEST_YEARS}-01-01')

        # Skip if test extends beyond data
        if test_start > close.index.max():
            continue

        train_mask = (close.index >= train_start) & (close.index < test_start)
        test_mask = (close.index >= test_start) & (close.index < test_end)

        if train_mask.sum() < 252 or test_mask.sum() < 100:
            continue

        # Custom HP for this window's eval (to make eval_strategy work properly)
        win_hp = dict(base_hp)
        train_results = core.grid_search(close, rsi_s, ma_s, vol_s, train_mask, win_hp)
        if not train_results:
            continue
        best = train_results[0]
        test_s = core.eval_strategy(close, rsi_s, ma_s, vol_s, best['w'], test_mask, win_hp)
        if test_s is None or test_s['days'] < 50:
            continue

        # QQQ on same test period
        qqq_cagr = None
        excess = None
        if qqq_ret is not None:
            qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
            qs = core.stats(qqq_t)
            if qs and qs['days'] > 50:
                qqq_cagr = qs['cagr']
                excess = (test_s['cagr'] - qqq_cagr) * 100

        train_period = f"{train_start.year}~{test_start.year - 1}"
        test_period = f"{test_start.year}"
        w(f"{train_period:<25} {test_period:<25} {core.fmt_w(best['w']):<22} "
          f"{test_s['cagr']*100:>8.2f}% {test_s['sharpe']:>7.2f} "
          f"{test_s['mdd']*100:>7.2f}% "
          f"{(qqq_cagr*100 if qqq_cagr is not None else 0):>7.2f}% "
          f"{(excess if excess is not None else 0):>+8.2f}%p")

        rows.append({
            'train_start': train_start, 'test_start': test_start,
            'best_w': best['w'], 'train_cagr': best['cagr'],
            'test_cagr': test_s['cagr'], 'test_sharpe': test_s['sharpe'],
            'test_mdd': test_s['mdd'],
            'qqq_cagr': qqq_cagr, 'excess': excess,
        })

    w("-" * 120)
    if not rows:
        w("\n결과 없음")
        return

    # Aggregate stats
    n = len(rows)
    test_cagrs = [r['test_cagr'] for r in rows]
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    win_count = sum(1 for e in excesses if e > 0)
    avg_test_cagr = sum(test_cagrs) / n
    avg_test_sharpe = sum(r['test_sharpe'] for r in rows) / n

    w(f"\n## 윈도우 집계 ({n}개)")
    w(f"  평균 Test CAGR: {avg_test_cagr*100:+.2f}%")
    w(f"  평균 Test Sharpe: {avg_test_sharpe:.2f}")
    if excesses:
        avg_excess = sum(excesses) / len(excesses)
        w(f"  평균 vs QQQ excess: {avg_excess:+.2f}%p")
        w(f"  최소: {min(excesses):+.2f}%p / 최대: {max(excesses):+.2f}%p")
        w(f"  QQQ 이긴 윈도우: {win_count}/{len(excesses)} ({win_count/len(excesses)*100:.0f}%)")

    # Weight stability
    w_counts = Counter(r['best_w'] for r in rows)
    w(f"\n## Best weight 분포 (출현 빈도)")
    for w_tup, cnt in sorted(w_counts.items(), key=lambda x: -x[1]):
        w(f"  {core.fmt_w(w_tup)}: {cnt}회")

    # Save year-by-year for analysis
    df = pd.DataFrame(rows)
    df['best_w'] = df['best_w'].astype(str)
    csv_path = os.path.join(OUTPUT_DIR, 'walk_forward.csv')
    df.to_csv(csv_path, index=False)
    w(f"\nCSV: {csv_path}")

    out_path = os.path.join(OUTPUT_DIR, 'walk_forward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
