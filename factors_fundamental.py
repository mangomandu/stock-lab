"""
Fundamental factors from Sharadar SF1 (ARQ dimension) — PIT-enforced.

8 factors:
  pe, pb         — valuation (직접 ARQ)
  roe_ttm        — trailing 4-quarter ROE (직접 계산)
  roa_ttm        — trailing 4-quarter ROA (직접 계산)
  netmargin      — net income / revenue (직접 ARQ)
  de             — debt / equity (직접 ARQ)
  eps_growth_yoy — eps_q / eps_q_minus_4 - 1
  rev_growth_yoy — revenue_q / revenue_q_minus_4 - 1

PIT enforcement:
  available_date = datekey + lag_days (default 60)
  매 backtest 날짜 t에 사용 가능한 latest fundamental = 가장 최근 available_date ≤ t
"""
import pandas as pd
import numpy as np


def apply_pit_lag(sf1_df, lag_days=60):
    """Add 'available_date' column = datekey + lag_days. PIT 강제 위해 필수."""
    df = sf1_df.copy()
    df['datekey'] = pd.to_datetime(df['datekey'])
    df['available_date'] = df['datekey'] + pd.Timedelta(days=lag_days)
    return df


def compute_ttm_factors(sf1_df):
    """Compute trailing-4-quarter factors per ticker.

    Returns sf1_df with extra columns: roe_ttm, roa_ttm, eps_growth_yoy, rev_growth_yoy.

    Strategy: per ticker, sort by calendardate, then rolling sum/avg over 4 quarters.
    """
    df = sf1_df.copy()
    df['calendardate'] = pd.to_datetime(df['calendardate'])
    df = df.sort_values(['ticker', 'calendardate'])

    g = df.groupby('ticker', sort=False)

    # TTM net income (sum of last 4 quarters)
    df['netinc_ttm'] = g['netinc'].transform(
        lambda x: x.rolling(window=4, min_periods=4).sum()
    )

    # 5-period average equity & assets (start-of-period to end, approximating avg over the year)
    df['equity_avg5'] = g['equity'].transform(
        lambda x: x.rolling(window=5, min_periods=4).mean()
    )
    df['assets_avg5'] = g['assets'].transform(
        lambda x: x.rolling(window=5, min_periods=4).mean()
    )

    # TTM ratios
    df['roe_ttm'] = df['netinc_ttm'] / df['equity_avg5'].replace(0, np.nan)
    df['roa_ttm'] = df['netinc_ttm'] / df['assets_avg5'].replace(0, np.nan)

    # YoY growth (q / q-4 - 1)
    df['eps_growth_yoy'] = g['eps'].transform(
        lambda x: x / x.shift(4) - 1
    )
    df['rev_growth_yoy'] = g['revenue'].transform(
        lambda x: x / x.shift(4) - 1
    )

    return df


# 8 final fundamental factor columns (all from ARQ + computed TTM)
FUNDAMENTAL_FACTORS = [
    'pe', 'pb',                                  # valuation
    'roe_ttm', 'roa_ttm',                        # profitability TTM
    'netmargin', 'de',                           # quality
    'eps_growth_yoy', 'rev_growth_yoy',          # growth
]


def build_fundamental_panel(sf1_df, lag_days=60):
    """Pipeline: SF1 ARQ → TTM factors + PIT lag.

    Returns DataFrame with columns: permaticker, ticker, available_date, + 8 factor cols.
    """
    df = compute_ttm_factors(sf1_df)
    df = apply_pit_lag(df, lag_days)

    # Note: Sharadar SF1 ARQ doesn't have permaticker column; we'll join via ticker
    # in actual use (TICKERS table provides ticker→permaticker)
    keep_cols = ['ticker', 'available_date', 'datekey', 'calendardate'] + FUNDAMENTAL_FACTORS
    return df[keep_cols].copy()


def latest_fundamentals_at(panel_df, as_of_date):
    """For each ticker, get the latest fundamental row available as-of as_of_date.

    panel_df: output of build_fundamental_panel()
    as_of_date: pd.Timestamp or string
    Returns DataFrame indexed by ticker with the 8 factor columns.
    """
    as_of = pd.to_datetime(as_of_date)
    df = panel_df[panel_df['available_date'] <= as_of].copy()
    if df.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_FACTORS)

    df = df.sort_values('available_date')
    latest = df.groupby('ticker', as_index=True).last()
    return latest[FUNDAMENTAL_FACTORS]


# Self-test using AAPL 4 quarters of 2024
def _self_test():
    import os
    sf1_path = '/home/dlfnek/stock_lab/data/sharadar/SF1.parquet'
    if not os.path.exists(sf1_path):
        print('SF1.parquet not found — skipping self-test')
        return

    sf1 = pd.read_parquet(sf1_path)
    sf1 = sf1[sf1['dimension'] == 'ARQ']
    aapl = sf1[sf1['ticker'] == 'AAPL'].copy()
    panel = build_fundamental_panel(aapl)

    print(f'AAPL ARQ: {len(aapl)} quarters loaded')
    print(f'Panel shape: {panel.shape}')
    print('\nLatest 5 quarters:')
    cols_show = ['ticker', 'calendardate', 'datekey', 'available_date'] + FUNDAMENTAL_FACTORS
    print(panel.sort_values('calendardate').tail()[cols_show].to_string())

    # Sentinel: 2024-12-31 → available_date should be ~2025-04-01
    q4 = panel[panel['calendardate'] == pd.Timestamp('2024-12-31')]
    if not q4.empty:
        r = q4.iloc[0]
        print(f'\nAAPL 2024 Q4:')
        print(f'  filed (datekey): {r["datekey"].date()}')
        print(f'  available (datekey+60d): {r["available_date"].date()}')
        print(f'  pe={r["pe"]:.2f}, pb={r["pb"]:.2f}, roe_ttm={r["roe_ttm"]:.3f}')

    # As-of test
    as_of = '2025-06-01'
    latest = latest_fundamentals_at(panel, as_of)
    print(f'\nAAPL latest fundamentals as-of {as_of}:')
    print(latest.to_string())


if __name__ == '__main__':
    _self_test()
