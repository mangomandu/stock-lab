"""
Extended walk-forward: ML model with more windows.

Use data from 2000 onwards instead of 2010 → 21 windows instead of 11.
This boosts statistical power (t-stat).

Output: results/ml_extended_walkforward.txt
"""
import core
import ml_model
import factors
import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 10
COST_ONEWAY = 0.0005

# Run windows from this start year
TEST_START_YEARS = list(range(2005, 2026))  # 21 windows


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

    # Universe size on test_start
    universe_alive = close.loc[test_start:].iloc[0].notna().sum() if not close.loc[test_start:].empty else 0

    return {
        'year': test_year,
        'universe': int(universe_alive),
        'test_cagr': s['cagr'],
        'test_sharpe': s['sharpe'],
        'test_mdd': s['mdd'],
        'qqq_cagr': qs['cagr'] if qs else None,
        'excess': excess,
        'top_features': top_feat,
    }


def t_stat_summary(excesses):
    """Compute t-statistic for null hypothesis: alpha = 0."""
    n = len(excesses)
    mean = sum(excesses) / n
    std = (sum((e - mean) ** 2 for e in excesses) / (n - 1)) ** 0.5 if n > 1 else 0
    se = std / (n ** 0.5)
    t = mean / se if se > 0 else 0
    return mean, std, se, t


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Extended Walk-Forward: ML model")
    w(f"  Test years: {TEST_START_YEARS[0]} ~ {TEST_START_YEARS[-1]} ({len(TEST_START_YEARS)} windows)")
    w(f"  Train: 5y rolling | Test: 1y | Biweekly Top-{TOP_N}")
    w("=" * 100)

    close, vol = core.load_panel()
    qqq_ret = load_qqq()
    hp = dict(ml_model.ML_HP_DEFAULT)

    rows = []
    for test_year in TEST_START_YEARS:
        result = run_one_window(close, vol, test_year, hp, qqq_ret)
        if result is None:
            w(f"  [{test_year}] skipped")
            continue
        rows.append(result)
        w(f"  [{result['year']}] U={result['universe']:>3} | "
          f"CAGR {result['test_cagr']*100:+7.2f}% Sh {result['test_sharpe']:>5.2f} "
          f"MDD {result['test_mdd']*100:>6.2f}% | QQQ {result['qqq_cagr']*100:+7.2f}% | "
          f"Excess {result['excess']:+7.2f}%p | top: {result['top_features']}")

    if not rows:
        return

    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    avg_test_cagr = sum(r['test_cagr'] for r in rows) / len(rows)
    avg_test_sharpe = sum(r['test_sharpe'] for r in rows) / len(rows)
    avg_test_mdd = sum(r['test_mdd'] for r in rows) / len(rows)
    win_count = sum(1 for e in excesses if e > 0)

    mean_e, std_e, se_e, t_e = t_stat_summary(excesses)

    w(f"\n{'='*100}")
    w(f"## {len(rows)}-window aggregate")
    w(f"  Avg Test CAGR:    {avg_test_cagr*100:+.2f}%")
    w(f"  Avg Test Sharpe:  {avg_test_sharpe:.2f}")
    w(f"  Avg Test MDD:     {avg_test_mdd*100:.2f}%")
    w(f"  Avg vs QQQ:       {mean_e:+.2f}%p")
    w(f"  Win rate:         {win_count}/{len(excesses)} ({win_count/len(excesses)*100:.0f}%)")
    w(f"\n## Statistical significance (alpha vs 0)")
    w(f"  Mean alpha:       {mean_e:+.2f}%p")
    w(f"  Std (across win): {std_e:.2f}%p")
    w(f"  Standard error:   {se_e:.2f}%p")
    w(f"  t-statistic:      {t_e:.2f}")
    if abs(t_e) > 2.0:
        w(f"  → t > 2.0: 통계적으로 유의 (95% 신뢰도) ✅")
    elif abs(t_e) > 1.65:
        w(f"  → t > 1.65: 한계선 (90% 신뢰도) △")
    else:
        w(f"  → t < 1.65: 통계적으로 약함 ⚠")

    # Compare 11-window vs 21-window
    w(f"\n## 비교: 이전 11 윈도우 (2015-2025) vs 확장 {len(rows)} 윈도우")
    recent_rows = [r for r in rows if r['year'] >= 2015]
    if recent_rows and len(recent_rows) < len(rows):
        recent_excesses = [r['excess'] for r in recent_rows if r['excess'] is not None]
        m_r, _, se_r, t_r = t_stat_summary(recent_excesses)
        w(f"  11 win (2015~25):  alpha {m_r:+.2f}%p, t={t_r:.2f}")
        w(f"  {len(rows)} win ({TEST_START_YEARS[0]}~25):  alpha {mean_e:+.2f}%p, t={t_e:.2f}")

    # Feature importance
    feat_count = Counter()
    for r in rows:
        for f in r['top_features']:
            feat_count[f] += 1
    w(f"\n## Top-2 feature 출현 빈도")
    for f, c in feat_count.most_common():
        w(f"  {f}: {c}/{len(rows)}회")

    df = pd.DataFrame(rows)
    df['top_features'] = df['top_features'].astype(str)
    csv_path = os.path.join(OUTPUT_DIR, 'ml_extended_walkforward.csv')
    df.to_csv(csv_path, index=False)
    w(f"\nCSV: {csv_path}")

    out_path = os.path.join(OUTPUT_DIR, 'ml_extended_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
