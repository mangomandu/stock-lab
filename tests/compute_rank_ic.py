"""
Rank IC (Information Coefficient) — model 신호의 SNR 측정.

매 backtest 날짜마다 모델 예측 score와 실제 forward return의 Spearman correlation 계산.
30년 평균 / IR / per-year breakdown 출력.

기준 (학계):
  IC = 0.00     → 무작위 (예측 의미 X)
  IC = 0.02     → 의미 있음 (학계 minimum)
  IC = 0.05     → 강한 신호 (대부분 fund target)
  IC > 0.10     → overfit 의심

사용 예:
  from tests.compute_rank_ic import compute_rank_ic
  result = compute_rank_ic(scores_df, returns_df, horizon=21)
  print(f"IC: {result['ic_mean']:.4f}, IR: {result['ic_ir']:.2f}")
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats


def compute_rank_ic(scores_df, forward_returns_df, min_stocks_per_date=20):
    """
    scores_df:           wide DataFrame (date × ticker) of model predicted scores
    forward_returns_df:  wide DataFrame (date × ticker) of forward returns (already shifted)
    min_stocks_per_date: skip dates with too few stocks (insufficient cross-section)

    Returns dict with:
      ic_series       — pd.Series of daily IC (date indexed)
      ic_mean         — annualized mean
      ic_std          — daily std
      ic_ir           — IR = mean/std × sqrt(252)
      ic_t_stat       — t-stat on mean ≠ 0
      ic_by_year      — pd.Series, per-year mean IC
      n_dates         — total dates included
    """
    # Align dates and tickers
    common_dates = scores_df.index.intersection(forward_returns_df.index)
    common_tickers = scores_df.columns.intersection(forward_returns_df.columns)
    s = scores_df.loc[common_dates, common_tickers]
    r = forward_returns_df.loc[common_dates, common_tickers]

    ics = {}
    for date in common_dates:
        s_row = s.loc[date].dropna()
        r_row = r.loc[date].dropna()
        common = s_row.index.intersection(r_row.index)
        if len(common) < min_stocks_per_date:
            continue
        # Spearman rank correlation
        ic, _ = stats.spearmanr(s_row[common].values, r_row[common].values)
        if not np.isnan(ic):
            ics[date] = ic

    if not ics:
        raise ValueError("No valid dates with enough stocks for IC computation")

    ic_series = pd.Series(ics).sort_index()

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else 0
    ic_t_stat, _ = stats.ttest_1samp(ic_series.values, 0)

    ic_series.index = pd.to_datetime(ic_series.index)
    ic_by_year = ic_series.groupby(ic_series.index.year).mean()

    return {
        'ic_series': ic_series,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_t_stat': ic_t_stat,
        'ic_by_year': ic_by_year,
        'n_dates': len(ic_series),
    }


def make_forward_returns(close_df, horizon=21):
    """Wide close panel → forward returns over horizon days.

    forward_return[t, ticker] = close[t + horizon] / close[t] - 1
    """
    return close_df.shift(-horizon) / close_df - 1


def print_ic_report(result, label=''):
    """Pretty-print IC report."""
    print(f"\n=== Rank IC Report{' — ' + label if label else ''} ===")
    print(f"  N dates:     {result['n_dates']:,}")
    print(f"  Mean IC:     {result['ic_mean']:+.4f}")
    print(f"  Daily std:   {result['ic_std']:.4f}")
    print(f"  IR:          {result['ic_ir']:+.2f}")
    print(f"  t-stat:      {result['ic_t_stat']:+.2f}")
    print(f"\n  Per-year IC:")
    for year, ic in result['ic_by_year'].items():
        flag = '  '
        if ic > 0.05:
            flag = '✓✓'
        elif ic > 0.02:
            flag = '✓ '
        elif ic < 0:
            flag = '✗ '
        print(f"    {year}: {ic:+.4f}  {flag}")


# Self-test with synthetic data
def _self_test():
    """Synthetic test: random scores + perfect-correlation scores."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252)
    tickers = [f'T{i:03d}' for i in range(50)]

    # Random scores → IC ≈ 0
    rand_scores = pd.DataFrame(np.random.randn(252, 50), index=dates, columns=tickers)
    # Forward returns proportional to scores (perfect signal) + noise
    fwd_returns = rand_scores * 0.5 + np.random.randn(252, 50) * 0.5

    # Random vs random
    rand2 = pd.DataFrame(np.random.randn(252, 50), index=dates, columns=tickers)
    r1 = compute_rank_ic(rand2, fwd_returns)
    print(f"Random vs returns: IC = {r1['ic_mean']:+.4f} (expect ~0)")

    # Score vs returns (positive correlation)
    r2 = compute_rank_ic(rand_scores, fwd_returns)
    print(f"Score vs returns:  IC = {r2['ic_mean']:+.4f} (expect > 0)")

    assert abs(r1['ic_mean']) < 0.05, "Random IC should be near 0"
    assert r2['ic_mean'] > 0.4, "Perfect-signal IC should be high"
    print("\n✓ Self-test passed")


if __name__ == '__main__':
    _self_test()
