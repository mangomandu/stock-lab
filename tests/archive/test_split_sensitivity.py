"""
Task 3: Split date sensitivity test.

Re-run Weekly Top-20 with split dates 2015, 2017, 2019, 2021. Verify that the
result is robust, not 2018-cutoff specific.

Output: results/split_sensitivity.txt
"""
import core
import pandas as pd
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_DATES = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01',
               '2019-01-01', '2020-01-01', '2021-01-01']


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

    w(f"[{datetime.now()}] Split date sensitivity (Weekly Top-20)")
    w("=" * 90)

    # Load data once
    close, vol = core.load_panel()
    base_hp = core.merge_hp({'top_n': 20, 'rebal_days': 5})
    rsi_s, ma_s, vol_s = core.compute_scores(close, vol, base_hp)
    qqq_ret = load_qqq()

    w(f"\nUniverse: {close.shape[1]}개 종목 | Train start fixed at {base_hp['train_start']}\n")
    w(f"{'Split':<12} {'Best W':<22} {'Train CAGR':>10} {'Test CAGR':>10} "
      f"{'Test Sh':>8} {'Test MDD':>9} {'QQQ CAGR':>10} {'Excess':>8}")
    w("-" * 90)

    rows = []
    for split in SPLIT_DATES:
        hp = core.merge_hp({**base_hp, 'split_date': split})
        train_mask, test_mask = core.get_period_masks(close, hp)

        train_results = core.grid_search(close, rsi_s, ma_s, vol_s, train_mask, hp)
        if not train_results:
            w(f"{split:<12} no train data")
            continue
        best = train_results[0]
        test_s = core.eval_strategy(close, rsi_s, ma_s, vol_s, best['w'], test_mask, hp)
        if test_s is None:
            w(f"{split:<12} no test data")
            continue

        # QQQ on same test period
        qqq_test = None
        excess = None
        if qqq_ret is not None:
            tt = qqq_ret[qqq_ret.index >= pd.Timestamp(split)]
            qqq_test = core.stats(tt)
            if qqq_test:
                excess = (test_s['cagr'] - qqq_test['cagr']) * 100

        w(f"{split:<12} {core.fmt_w(best['w']):<22} "
          f"{best['cagr']*100:>9.2f}% {test_s['cagr']*100:>9.2f}% "
          f"{test_s['sharpe']:>8.2f} {test_s['mdd']*100:>8.2f}% "
          f"{(qqq_test['cagr']*100 if qqq_test else 0):>9.2f}% "
          f"{excess if excess is not None else 0:>+7.2f}%p")
        rows.append({
            'split': split, 'best_w': best['w'],
            'train_cagr': best['cagr'], 'test_cagr': test_s['cagr'],
            'test_sharpe': test_s['sharpe'], 'test_mdd': test_s['mdd'],
            'qqq_cagr': qqq_test['cagr'] if qqq_test else None,
            'excess': excess,
        })

    w("-" * 90)

    # Summary
    if rows:
        excesses = [r['excess'] for r in rows if r['excess'] is not None]
        cagrs_diff = [r['test_cagr'] - r['qqq_cagr']
                      for r in rows if r['qqq_cagr'] is not None]
        win_count = sum(1 for e in excesses if e > 0)
        w(f"\nQQQ 대비 알파 통계:")
        w(f"  평균 excess CAGR: {sum(excesses)/len(excesses):+.2f}%p")
        w(f"  최소: {min(excesses):+.2f}%p / 최대: {max(excesses):+.2f}%p")
        w(f"  QQQ 이긴 split: {win_count}/{len(excesses)}")

        # Best w stability
        ws_set = set(r['best_w'] for r in rows)
        w(f"\n최적 weight 분포: {len(ws_set)}개 unique 값")
        for r in rows:
            w(f"  {r['split']}: {core.fmt_w(r['best_w'])}")

    out_path = os.path.join(OUTPUT_DIR, 'split_sensitivity.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
