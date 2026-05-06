"""
Sharadar 데이터 integrity 검증 — memory-safe 버전.

문제: SEP(1GB) + DAILY(750MB) 한꺼번에 pandas DataFrame으로 로딩 → 11GB+ RSS → OOM kill.

해결: 데이터셋 하나씩 처리 + 검증 후 즉시 del + gc. 큰 거는 column subset만 읽음.
"""
import gc
import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')

EXPECTED = {
    'TICKERS.parquet': {'min_rows': 50_000,    'max_rows': 100_000},
    'SP500.parquet':   {'min_rows': 30_000,    'max_rows': 100_000},
    'ACTIONS.parquet': {'min_rows': 500_000,   'max_rows': 1_500_000},
    'SF1.parquet':     {'min_rows': 500_000,   'max_rows': 1_000_000},
    'DAILY.parquet':   {'min_rows': 30_000_000,'max_rows': 60_000_000},
    'SEP.parquet':     {'min_rows': 30_000_000,'max_rows': 100_000_000},
}

PASS = 0
FAIL = 0


def check(label, condition, details=''):
    global PASS, FAIL
    flag = '✓' if condition else '✗'
    print(f'  [{flag}] {label}{": " + details if details else ""}', flush=True)
    if condition:
        PASS += 1
    else:
        FAIL += 1
    return condition


def parquet_info(path):
    """Get row count + columns without loading full table."""
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, pf.schema.names


def verify_small_table(name):
    """Small tables (TICKERS/SP500/ACTIONS) — load fully."""
    p = DATA_DIR / name
    if not p.exists():
        check(f'{name} exists', False, 'MISSING')
        return None
    n_rows, cols = parquet_info(p)
    bounds = EXPECTED.get(name, {})
    ok = bounds.get('min_rows', 0) <= n_rows <= bounds.get('max_rows', float('inf'))
    size_mb = p.stat().st_size / 1e6
    check(f'{name}', ok, f'{n_rows:,} rows, {size_mb:.1f}MB, {len(cols)} cols')
    return pd.read_parquet(p)


def verify_huge_table_metadata(name):
    """Huge tables (SEP/DAILY) — only check row count + size, no load."""
    p = DATA_DIR / name
    if not p.exists():
        check(f'{name} exists', False, 'MISSING')
        return False
    n_rows, cols = parquet_info(p)
    bounds = EXPECTED.get(name, {})
    ok = bounds.get('min_rows', 0) <= n_rows <= bounds.get('max_rows', float('inf'))
    size_mb = p.stat().st_size / 1e6
    check(f'{name}', ok, f'{n_rows:,} rows, {size_mb:.1f}MB, {len(cols)} cols')
    return True


def main():
    print('=' * 70, flush=True)
    print('Sharadar Data Integrity Check (memory-safe)', flush=True)
    print('=' * 70, flush=True)

    # ===== 1. File metadata only (no full load) =====
    print('\n## 1. File metadata + row count', flush=True)
    tickers = verify_small_table('TICKERS.parquet')
    sp500   = verify_small_table('SP500.parquet')
    actions = verify_small_table('ACTIONS.parquet')
    verify_huge_table_metadata('SF1.parquet')
    verify_huge_table_metadata('DAILY.parquet')
    verify_huge_table_metadata('SEP.parquet')
    del actions  # free immediately
    gc.collect()

    # ===== 2. SF1 — load ARQ AAPL subset only =====
    print('\n## 2. SF1 PIT (AAPL subset only)', flush=True)
    sf1_aapl = pd.read_parquet(
        DATA_DIR / 'SF1.parquet',
        filters=[('ticker', '=', 'AAPL'), ('dimension', '=', 'ARQ')],
    )
    check('SF1 AAPL ARQ ≥ 80 quarters', len(sf1_aapl) >= 80, f'{len(sf1_aapl)} quarters')

    aapl_q1 = sf1_aapl[sf1_aapl['calendardate'] == '2024-03-31']
    if len(aapl_q1) >= 1:
        r = aapl_q1.iloc[0]
        check('AAPL 2024Q1 datekey present', pd.notna(r.get('datekey')),
              f'datekey={r.get("datekey")}, pe={r.get("pe")}')
    else:
        check('AAPL 2024Q1 row exists', False)

    dk_min = pd.to_datetime(sf1_aapl['datekey']).min()
    check('SF1 AAPL coverage ≥ 1996', dk_min < pd.Timestamp('1996-01-01'),
          f'min datekey={dk_min.date()}')
    del sf1_aapl
    gc.collect()

    # ===== 3. SEP — Lehman delisted sentinel only =====
    print('\n## 3. SEP delisted sentinel (LEHMQ subset only)', flush=True)
    sep_lehmq = pd.read_parquet(
        DATA_DIR / 'SEP.parquet',
        filters=[('ticker', '=', 'LEHMQ')],
    )
    check('SEP LEHMQ rows ≥ 100', len(sep_lehmq) >= 100, f'{len(sep_lehmq)} rows')

    bk = sep_lehmq[pd.to_datetime(sep_lehmq['date']) == pd.Timestamp('2008-09-15')]
    check('Lehman bankruptcy day 2008-09-15 in SEP', len(bk) == 1,
          f'{len(bk)} rows on 2008-09-15')
    del sep_lehmq
    gc.collect()

    # ===== 4. SEP AAPL — date range coverage =====
    print('\n## 4. SEP AAPL coverage', flush=True)
    sep_aapl = pd.read_parquet(
        DATA_DIR / 'SEP.parquet',
        filters=[('ticker', '=', 'AAPL')],
        columns=['ticker', 'date', 'close'],
    )
    check('AAPL SEP ≥ 5000 days', len(sep_aapl) >= 5000, f'{len(sep_aapl)} days')
    d_min = pd.to_datetime(sep_aapl['date']).min()
    d_max = pd.to_datetime(sep_aapl['date']).max()
    check('AAPL SEP starts before 2000', d_min < pd.Timestamp('2000-01-01'),
          f'min={d_min.date()}')
    check('AAPL SEP latest ≥ 2026', d_max >= pd.Timestamp('2026-01-01'),
          f'max={d_max.date()}')
    del sep_aapl
    gc.collect()

    # ===== 5. SP500 + TICKERS sentinels (already loaded) =====
    print('\n## 5. SP500 + TICKERS sentinel', flush=True)
    if sp500 is not None:
        wamu = sp500[(sp500['ticker'] == 'WAMUQ') & (sp500['action'] == 'removed')]
        check('WaMu removed event in SP500', len(wamu) >= 1, f'{len(wamu)} matching')

    if tickers is not None:
        leh = tickers[tickers['ticker'] == 'LEHMQ']['permaticker']
        check('LEHMQ has permaticker', len(leh) >= 1,
              f'permaticker={list(leh)[0] if len(leh) else "MISSING"}')

    # ===== 6. Cross-table integrity (memory-safe) =====
    print('\n## 6. Cross-table integrity', flush=True)
    if tickers is not None:
        # Get SF1 unique tickers via parquet column read only
        sf1_ticks = pd.read_parquet(DATA_DIR / 'SF1.parquet', columns=['ticker'])['ticker'].unique()
        tickers_sf1 = set(tickers[tickers['table'] == 'SF1']['ticker'])
        missing = set(sf1_ticks) - tickers_sf1
        check('SF1 tickers ⊆ TICKERS', len(missing) == 0,
              f'missing in TICKERS: {len(missing)}' if missing else 'all matched')
        del sf1_ticks, tickers_sf1
        gc.collect()

    # ===== Summary =====
    print('\n' + '=' * 70, flush=True)
    print(f'PASSED: {PASS}, FAILED: {FAIL}', flush=True)
    print('=' * 70, flush=True)
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
