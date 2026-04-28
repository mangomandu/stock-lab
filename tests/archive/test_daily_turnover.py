"""
Measure: how much does the ML Top-20 recommendation actually change day-to-day?

If daily turnover is small (~5%), Biweekly is fine — daily changes are noise.
If daily turnover is large (~30%+), event-driven might capture real info.

Procedure:
  1. Train ML model on 5 years up to year_start
  2. For every trading day in test_year, compute scores
  3. Take Top-20 each day
  4. Measure turnover vs previous day's Top-20
  5. Aggregate stats
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
TOP_N = 20


def load_qqq():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'qqq_close.csv'),
                     parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def measure_turnover_year(close, vol, hp, test_year):
    """For one test year, train + score every day + measure daily turnover."""
    train_start = pd.Timestamp(f'{test_year - 5}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')

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

    # For each day, compute Top-20
    top20_per_day = {}
    score_spreads = {}
    for date in score_wide.index:
        row = score_wide.loc[date].dropna()
        if len(row) < TOP_N:
            continue
        sorted_scores = row.sort_values(ascending=False)
        top20 = set(sorted_scores.head(TOP_N).index.tolist())
        top20_per_day[date] = top20
        # Spread
        top_avg = sorted_scores.head(TOP_N).mean()
        rest_avg = sorted_scores.iloc[TOP_N:].mean()
        score_spreads[date] = top_avg - rest_avg

    # Compute day-to-day turnover
    dates = sorted(top20_per_day.keys())
    turnovers = []
    score_changes = []
    for i in range(1, len(dates)):
        d_prev, d_curr = dates[i-1], dates[i]
        prev_set = top20_per_day[d_prev]
        curr_set = top20_per_day[d_curr]
        # Turnover = (changes / N)
        n_changes = len(curr_set - prev_set)  # new entries
        turnover = n_changes / TOP_N
        turnovers.append(turnover)
        # Score spread change
        score_changes.append(abs(score_spreads[d_curr] - score_spreads[d_prev]))

    if not turnovers:
        return None

    return {
        'year': test_year,
        'n_trading_days': len(dates),
        'turnovers': turnovers,
        'spreads': list(score_spreads.values()),
        'avg_turnover': np.mean(turnovers),
        'median_turnover': np.median(turnovers),
        'p25_turnover': np.percentile(turnovers, 25),
        'p75_turnover': np.percentile(turnovers, 75),
        'p95_turnover': np.percentile(turnovers, 95),
        'max_turnover': max(turnovers),
        'avg_spread': np.mean(list(score_spreads.values())),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line)
        out_lines.append(line)

    w(f"[{datetime.now()}] Daily ML Top-20 turnover measurement")
    w(f"  Question: do daily recommendations actually change a lot?")
    w("=" * 90)

    close, vol = core.load_panel()
    hp = dict(ml_model.ML_HP_DEFAULT)

    results = []
    for test_year in range(2018, 2026):  # recent 8 years
        w(f"\n[{test_year}] training + measuring...")
        r = measure_turnover_year(close, vol, hp, test_year)
        if r is None:
            continue
        results.append(r)
        w(f"  Avg daily turnover: {r['avg_turnover']*100:.2f}% "
          f"(median {r['median_turnover']*100:.1f}%, "
          f"p75 {r['p75_turnover']*100:.1f}%, "
          f"p95 {r['p95_turnover']*100:.1f}%, "
          f"max {r['max_turnover']*100:.1f}%)")
        w(f"  Avg score spread (top-rest): {r['avg_spread']:.4f}")

    if not results:
        return

    # Aggregate
    all_turnovers = []
    for r in results:
        all_turnovers.extend(r['turnovers'])

    w(f"\n{'='*90}")
    w(f"## Aggregate ({len(results)} years, {len(all_turnovers):,} day-to-day transitions)")
    w(f"  Mean daily turnover:   {np.mean(all_turnovers)*100:.2f}%")
    w(f"  Median daily turnover: {np.median(all_turnovers)*100:.2f}%")
    w(f"  25th percentile:       {np.percentile(all_turnovers, 25)*100:.2f}%")
    w(f"  75th percentile:       {np.percentile(all_turnovers, 75)*100:.2f}%")
    w(f"  95th percentile:       {np.percentile(all_turnovers, 95)*100:.2f}%")
    w(f"  Max:                   {max(all_turnovers)*100:.2f}%")
    w(f"  Days with 0 changes:   {sum(1 for t in all_turnovers if t == 0)}/{len(all_turnovers)} "
      f"({sum(1 for t in all_turnovers if t == 0)/len(all_turnovers)*100:.1f}%)")
    w(f"  Days with ≥30% changes: {sum(1 for t in all_turnovers if t >= 0.30)}/{len(all_turnovers)} "
      f"({sum(1 for t in all_turnovers if t >= 0.30)/len(all_turnovers)*100:.1f}%)")
    w(f"  Days with ≥50% changes: {sum(1 for t in all_turnovers if t >= 0.50)}/{len(all_turnovers)} "
      f"({sum(1 for t in all_turnovers if t >= 0.50)/len(all_turnovers)*100:.1f}%)")

    # Distribution by # of changes
    w(f"\n## Top-20 안에서 매일 종목 교체 수 분포")
    counts = pd.Series([int(t * TOP_N) for t in all_turnovers]).value_counts().sort_index()
    for n, cnt in counts.items():
        bar = '█' * int(cnt / max(counts) * 30)
        w(f"  {n:>2}개 교체: {cnt:>4}일 ({cnt/len(all_turnovers)*100:>5.1f}%) {bar}")

    w(f"\n## 해석")
    avg_n_changes = np.mean(all_turnovers) * TOP_N
    w(f"  하루 평균 Top-20 중 약 {avg_n_changes:.1f}개 종목 교체")
    if np.mean(all_turnovers) < 0.10:
        w(f"  → 일별 변동 작음. Biweekly가 최적 (며칠 변동은 노이즈)")
    elif np.mean(all_turnovers) < 0.25:
        w(f"  → 일별 변동 중간. Biweekly 합리적이나 weekly 검토 가능")
    else:
        w(f"  → 일별 변동 큼. 매매 빈도 늘릴 가치 있음")

    # 2주 vs 1주 turnover 비교 추정
    w(f"\n## Biweekly stale 정도 추정")
    if all_turnovers:
        avg = np.mean(all_turnovers)
        # 10일 동안 누적 변경 수 (independent assumption)
        # 실제로는 종목 들락날락 — overlap 있음, 그래서 단순 곱셈 X
        # 정확히 측정하려면 10일 간격으로 비교
        w(f"  매일 평균 {avg*100:.1f}% turnover")
        w(f"  10일 후 추정 누적 turnover: ~{min(avg * 10, 1.0)*100:.0f}% (independent 가정 상한)")
        w(f"  실측은 walk-forward 결과로 대체 (overlap 효과)")

    out_path = os.path.join(OUTPUT_DIR, 'daily_turnover.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
