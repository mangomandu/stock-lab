"""
Task 12: ML model walk-forward.

For each test year:
  1. Train LightGBM on previous 5 years (features + 10-day forward target)
  2. Score all (date, ticker) on test year
  3. Build Top-N portfolio biweekly using ML scores
  4. Compare to QQQ, RSI/MA/VOL baseline, and academic factor model

Output: results/ml_model_walkforward.txt
"""
import core
import ml_model
import factors
import pandas as pd
import numpy as np
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N = 20
REBAL_DAYS = 10
COST_ONEWAY = 0.0005


def load_qqq():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'qqq_close.csv'),
                     parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def backtest_with_score(close, score_wide, top_n=TOP_N, rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
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


def run_one_window(close, vol, test_year, hp, qqq_ret):
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
    port_ret, dcost = backtest_with_score(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
    qs = core.stats(qqq_t)
    excess = (s['cagr'] - qs['cagr']) * 100 if qs else None

    fi = pd.Series(model.feature_importance('gain'), index=hp['feature_names'])
    top_feat = fi.nlargest(2).index.tolist()

    return {
        'year': test_year,
        'test_cagr': s['cagr'],
        'test_sharpe': s['sharpe'],
        'test_mdd': s['mdd'],
        'qqq_cagr': qs['cagr'] if qs else None,
        'excess': excess,
        'best_iter': model.best_iteration,
        'top_features': top_feat,
        'cost_drag': float(dcost.sum()),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] LightGBM ML Model Walk-Forward (Biweekly Top-{TOP_N})")
    w(f"  Forward target: 10-day cross-sectional rank")
    w(f"  Features: momentum, lowvol, trend, rsi, ma, volsurge")
    w("=" * 100)

    close, vol = core.load_panel()
    qqq_ret = load_qqq()
    hp = dict(ml_model.ML_HP_DEFAULT)

    w(f"\nUniverse: {close.shape[1]} tickers × {len(close)} dates\n")

    rows = []
    for test_year in range(2015, 2026):
        t0 = datetime.now()
        result = run_one_window(close, vol, test_year, hp, qqq_ret)
        elapsed = (datetime.now() - t0).total_seconds()
        if result is None:
            w(f"  [{test_year}] skipped")
            continue
        rows.append(result)
        w(f"  [{result['year']}] CAGR {result['test_cagr']*100:+6.2f}% Sh {result['test_sharpe']:.2f} "
          f"MDD {result['test_mdd']*100:.2f}% | QQQ {result['qqq_cagr']*100:+.2f}% | "
          f"Excess {result['excess']:+.2f}%p | top: {result['top_features']} ({elapsed:.0f}s)")

    if not rows:
        w("\n결과 없음")
        return

    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    avg_test_cagr = sum(r['test_cagr'] for r in rows) / len(rows)
    avg_test_sharpe = sum(r['test_sharpe'] for r in rows) / len(rows)
    avg_test_mdd = sum(r['test_mdd'] for r in rows) / len(rows)
    avg_excess = sum(excesses) / len(excesses) if excesses else None
    win_count = sum(1 for e in excesses if e > 0)

    w(f"\n{'='*100}")
    w(f"## 11-window aggregate")
    w(f"  Avg Test CAGR:   {avg_test_cagr*100:+.2f}%")
    w(f"  Avg Test Sharpe: {avg_test_sharpe:.2f}")
    w(f"  Avg Test MDD:    {avg_test_mdd*100:.2f}%")
    if avg_excess is not None:
        w(f"  Avg vs QQQ:      {avg_excess:+.2f}%p")
        w(f"  Win rate:        {win_count}/{len(excesses)} ({win_count/len(excesses)*100:.0f}%)")

    # Top features summary
    feat_count = {}
    for r in rows:
        for f in r['top_features']:
            feat_count[f] = feat_count.get(f, 0) + 1
    w(f"\n## Top-2 feature importance 출현 빈도 ({len(rows)} windows)")
    for f, c in sorted(feat_count.items(), key=lambda x: -x[1]):
        w(f"  {f}: {c}회")

    df = pd.DataFrame(rows)
    df['top_features'] = df['top_features'].astype(str)
    csv_path = os.path.join(OUTPUT_DIR, 'ml_model_walkforward.csv')
    df.to_csv(csv_path, index=False)
    w(f"\nCSV: {csv_path}")

    out_path = os.path.join(OUTPUT_DIR, 'ml_model_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
