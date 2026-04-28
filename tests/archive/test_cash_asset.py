"""
Task 6: CASH asset integration.

Add synthetic CASH asset competing for Top-N slots. CASH gets constant score
(cash_score). Cost only on stock-side trades. Test cash_score in {None, 50, 60, 70, 80}.

Goal: reduce MDD without sacrificing too much CAGR. Cash should engage in
bear markets when most stocks score low.

Output: results/cash_asset.txt
"""
import core
import pandas as pd
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CASH_SCORES = [None, 40, 50, 60, 70, 80]


def load_qqq():
    qqq_path = os.path.join(OUTPUT_DIR, 'qqq_close.csv')
    if not os.path.exists(qqq_path):
        return None
    df = pd.read_csv(qqq_path, parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def cash_engagement(close, rsi_s, ma_s, vol_s, w, mask, hp):
    """Compute fraction of test days when CASH was held (any allocation)."""
    c, rs, ms, vs = core.slice_by_mask([close, rsi_s, ma_s, vol_s], mask)
    score = (rs * w[0] + ms * w[1] + vs * w[2]).where(c.notna())
    score = score.copy()
    c = c.copy()
    score['CASH'] = float(hp['cash_score'])
    c['CASH'] = 1.0
    in_top = core.build_holdings(score, hp)
    weights = in_top.div(in_top.sum(axis=1).replace(0, 1), axis=0)
    cash_held = weights['CASH'].fillna(0)
    return cash_held.mean()  # avg fraction of portfolio in cash


def main():
    out_lines = []
    def w(line=''):
        print(line)
        out_lines.append(line)

    w(f"[{datetime.now()}] CASH asset integration (Weekly Top-20, split 2018)")
    w("=" * 110)

    close, vol = core.load_panel()
    base_hp = core.merge_hp({'top_n': 20, 'rebal_days': 5})
    rsi_s, ma_s, vol_s = core.compute_scores(close, vol, base_hp)
    train_mask, test_mask = core.get_period_masks(close, base_hp)
    qqq_ret = load_qqq()

    qqq_test_s = None
    if qqq_ret is not None:
        qqq_test = qqq_ret[qqq_ret.index >= pd.Timestamp(base_hp['split_date'])]
        qqq_test_s = core.stats(qqq_test)

    if qqq_test_s:
        w(f"\nQQQ Test reference: CAGR {qqq_test_s['cagr']*100:.2f}%, "
          f"Sharpe {qqq_test_s['sharpe']:.2f}, MDD {qqq_test_s['mdd']*100:.2f}%\n")

    w(f"{'cash_score':<12} {'Best W':<22} {'Train CAGR':>10} {'Test CAGR':>10} "
      f"{'Test Sh':>8} {'Test MDD':>9} {'Avg %Cash':>10} {'vs QQQ':>9}")
    w("-" * 110)

    rows = []
    for cs in CASH_SCORES:
        hp = core.merge_hp({**base_hp, 'cash_score': cs})

        train_results = core.grid_search(close, rsi_s, ma_s, vol_s, train_mask, hp)
        if not train_results:
            continue
        best = train_results[0]
        test_s = core.eval_strategy(close, rsi_s, ma_s, vol_s, best['w'], test_mask, hp)
        if test_s is None:
            continue

        # Cash engagement on test period
        cash_pct = 0
        if cs is not None:
            cash_pct = cash_engagement(close, rsi_s, ma_s, vol_s,
                                       best['w'], test_mask, hp) * 100

        excess = ''
        if qqq_test_s:
            diff = (test_s['cagr'] - qqq_test_s['cagr']) * 100
            excess = f"{diff:+.2f}%p"

        cs_label = 'None (no cash)' if cs is None else str(cs)
        w(f"{cs_label:<12} {core.fmt_w(best['w']):<22} "
          f"{best['cagr']*100:>9.2f}% {test_s['cagr']*100:>9.2f}% "
          f"{test_s['sharpe']:>8.2f} {test_s['mdd']*100:>8.2f}% "
          f"{cash_pct:>9.1f}% {excess:>9}")

        rows.append({
            'cash_score': cs, 'best_w': best['w'],
            'test_cagr': test_s['cagr'], 'test_sharpe': test_s['sharpe'],
            'test_mdd': test_s['mdd'], 'cash_pct': cash_pct,
        })

    w("-" * 110)
    w("\n## 해석")
    w("- cash_score = None: 기존 100% stock 강제")
    w("- cash_score 클수록 cash가 더 자주 Top-N에 들어옴 → 방어적")
    w("- Test MDD 감소 / Sharpe 변화 / CAGR trade-off 비교")

    # Find best by test Sharpe
    if rows:
        best_sharpe = max(rows, key=lambda r: r['test_sharpe'])
        cs_label = 'None' if best_sharpe['cash_score'] is None else str(best_sharpe['cash_score'])
        w(f"\n## Test Sharpe 최고: cash_score = {cs_label}")
        w(f"  CAGR {best_sharpe['test_cagr']*100:.2f}% / Sharpe {best_sharpe['test_sharpe']:.2f} / "
          f"MDD {best_sharpe['test_mdd']*100:.2f}% / Cash {best_sharpe['cash_pct']:.1f}%")

    out_path = os.path.join(OUTPUT_DIR, 'cash_asset.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
