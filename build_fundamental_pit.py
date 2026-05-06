"""
SF1 → fundamental panel with PIT lag (60일) + TTM 8 features.

Input:  data/sharadar/SF1.parquet (ARQ dimension)
Output: data/panels/fundamental_pit.parquet
        — long format with columns: ticker, available_date, calendardate, 8 features

8 features (factors_fundamental.FUNDAMENTAL_FACTORS):
  pe, pb, roe_ttm, roa_ttm, netmargin, de, eps_growth_yoy, rev_growth_yoy

Backtest 사용 시:
  매 날짜 t, 각 ticker 마다 freshness ≤ 90일 인 latest 행 선택.
  freshness = t - available_date.
"""
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from factors_fundamental import (
    apply_pit_lag,
    compute_ttm_factors,
    FUNDAMENTAL_FACTORS,
)

DATA_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/data/panels')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build():
    print('=== Building fundamental PIT panel ===', flush=True)
    print('Loading SF1 ARQ...', flush=True)
    sf1 = pd.read_parquet(
        DATA_DIR / 'SF1.parquet',
        filters=[('dimension', '=', 'ARQ')],
    )
    print(f'SF1 ARQ: {len(sf1):,} rows', flush=True)

    print('Computing TTM factors...', flush=True)
    sf1 = compute_ttm_factors(sf1)

    print('Applying PIT lag (60일)...', flush=True)
    sf1 = apply_pit_lag(sf1, lag_days=60)

    keep_cols = ['ticker', 'available_date', 'datekey', 'calendardate'] + FUNDAMENTAL_FACTORS
    out = sf1[keep_cols].copy()
    out['available_date'] = pd.to_datetime(out['available_date'])
    out['calendardate'] = pd.to_datetime(out['calendardate'])

    out.to_parquet(OUT_DIR / 'fundamental_pit.parquet')
    print(f'Saved {len(out):,} rows → {OUT_DIR / "fundamental_pit.parquet"}', flush=True)

    # Sentinel
    print('\n=== Sentinels ===', flush=True)
    aapl_q1 = out[(out['ticker'] == 'AAPL') &
                  (out['calendardate'] == pd.Timestamp('2024-03-31'))]
    if not aapl_q1.empty:
        r = aapl_q1.iloc[0]
        print(f'AAPL 2024 Q1:', flush=True)
        print(f'  calendardate={r["calendardate"].date()}, datekey={r["datekey"].date()}',
              flush=True)
        print(f'  available_date={r["available_date"].date()}', flush=True)
        print(f'  pe={r["pe"]:.2f}, roe_ttm={r["roe_ttm"]:.3f}', flush=True)


if __name__ == '__main__':
    build()
