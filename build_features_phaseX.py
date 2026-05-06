"""
Phase X — 모든 feature panels 빌드 + 저장.

산출물 (data/panels/features/):
  Price/volume (10):
    mom_1m, mom_3m, mom_6m, mom_12m_skip1, mom_accel
    reversal_1w
    lowvol, idio_vol, volsurge
    high_52w_distance
  Sector-relative (3):
    sec_rel_mom_12m, sec_rel_mom_3m, sec_rel_lowvol
  Cross-sectional (2):
    rsi, beta_to_spy
  Fundamental (8 - 이미 fundamental_pit.parquet에 있음 + 추가 4):
    pe, pb, roe_ttm, roa_ttm, netmargin, de, eps_growth_yoy, rev_growth_yoy
    + ps, evebitda, roic, fcfps  (SF1에서 직접)
  Macro (2):
    vix_level, yield_curve

각 feature: cross-sectional z-score per date (factors.py 정합).
저장 format: wide DataFrame parquet (date × ticker).
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request

sys.path.insert(0, str(Path(__file__).parent))

PANEL_DIR = Path('/home/dlfnek/stock_lab/data/panels')
SHARADAR_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
FEATURE_DIR = Path('/home/dlfnek/stock_lab/data/panels/features')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def cross_sectional_zscore(df):
    """매 row (date) 별로 z-score (NaN-safe)."""
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, 1e-9)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def save(df, name):
    out_path = FEATURE_DIR / f'{name}.parquet'
    df.to_parquet(out_path)
    nan_pct = df.isna().sum().sum() / df.size * 100
    print(f'  ✓ {name:<25} shape={df.shape}, NaN={nan_pct:.1f}%, '
          f'size={out_path.stat().st_size/1e6:.1f}MB', flush=True)


# ─────────────────────────────────────────────────────────────────────
# Price-based features
# ─────────────────────────────────────────────────────────────────────
def build_price_features(close, vol):
    print('\n[Price/volume features]', flush=True)
    daily_ret = close.pct_change()

    # Multi-horizon momentum
    save(cross_sectional_zscore(close.pct_change(21)), 'mom_1m')
    save(cross_sectional_zscore(close.pct_change(63)), 'mom_3m')
    save(cross_sectional_zscore(close.pct_change(126)), 'mom_6m')
    mom_12m_skip1 = close.shift(21) / close.shift(252) - 1
    save(cross_sectional_zscore(mom_12m_skip1), 'mom_12m_skip1')

    # Momentum acceleration
    mom_3m = close.pct_change(63)
    mom_6m = close.pct_change(126)
    save(cross_sectional_zscore(mom_3m - mom_6m), 'mom_accel')

    # Reversal
    save(cross_sectional_zscore(-close.pct_change(5)), 'reversal_1w')  # negative = high score for big drops

    # Volatility
    lowvol_raw = -daily_ret.rolling(252).std()  # higher = lower vol
    save(cross_sectional_zscore(lowvol_raw), 'lowvol')

    # Volume surge
    vma = vol.rolling(20).mean()
    vol_ratio = (vol / vma.replace(0, 1e-9))
    save(cross_sectional_zscore(vol_ratio), 'volsurge')

    # 52w-high distance
    high_52w = close.rolling(252).max()
    distance = close / high_52w - 1  # 0 = at high, -0.3 = 30% drawdown from high
    save(cross_sectional_zscore(distance), 'high_52w_distance')

    # RSI (mean reversion)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    rsi_score = (70 - rsi.fillna(50)) / 40 * 100  # higher = oversold
    save(cross_sectional_zscore(rsi_score), 'rsi')

    return daily_ret


def build_market_relative(close, daily_ret, spy_ret):
    """Beta to SPY + idiosyncratic vol."""
    print('\n[Market-relative]', flush=True)
    spy_aligned = spy_ret.reindex(close.index).fillna(0)

    # Rolling beta (252d)
    cov = daily_ret.rolling(252).cov(spy_aligned)
    var = spy_aligned.rolling(252).var()
    beta = cov.div(var.replace(0, 1e-9), axis=0)
    save(cross_sectional_zscore(-beta), 'beta_to_spy')  # negative beta = lower correlation = good?

    # Idiosyncratic vol = vol of (ticker - beta×SPY)
    excess_ret = daily_ret.sub(beta.mul(spy_aligned, axis=0))
    idio_vol_raw = -excess_ret.rolling(252).std()
    save(cross_sectional_zscore(idio_vol_raw), 'idio_vol')


def sector_demean(raw_df, sector_map):
    """For each date, subtract sector mean from each ticker (axis=1 grouping)."""
    ticker_to_sector = pd.Series({tk: sector_map.get(tk, 'Unknown') for tk in raw_df.columns})
    result = raw_df.copy()
    for sector in ticker_to_sector.unique():
        tk_in_sec = ticker_to_sector[ticker_to_sector == sector].index.tolist()
        if len(tk_in_sec) < 2:
            continue
        sector_mean = raw_df[tk_in_sec].mean(axis=1)
        result[tk_in_sec] = raw_df[tk_in_sec].sub(sector_mean, axis=0)
    return result


def build_sector_relative(close, daily_ret, sector_map):
    """Sector-relative momentum / lowvol."""
    print('\n[Sector-relative]', flush=True)

    mom_12m = close.shift(21) / close.shift(252) - 1
    save(cross_sectional_zscore(sector_demean(mom_12m, sector_map)), 'sec_rel_mom_12m')

    mom_3m = close.pct_change(63)
    save(cross_sectional_zscore(sector_demean(mom_3m, sector_map)), 'sec_rel_mom_3m')

    lowvol = -daily_ret.rolling(252).std()
    save(cross_sectional_zscore(sector_demean(lowvol, sector_map)), 'sec_rel_lowvol')


# ─────────────────────────────────────────────────────────────────────
# Fundamental — already built in fundamental_pit.parquet, build wide format
# ─────────────────────────────────────────────────────────────────────
def build_fundamental_features(close):
    print('\n[Fundamental — extending fundamental_pit.parquet]', flush=True)

    # Load existing fundamental panel + add ps, evebitda, roic, fcfps directly from SF1
    sf1 = pd.read_parquet(SHARADAR_DIR / 'SF1.parquet')
    sf1 = sf1[sf1['dimension'] == 'ARQ'].copy()
    sf1['datekey'] = pd.to_datetime(sf1['datekey'])
    sf1['available_date'] = sf1['datekey'] + pd.Timedelta(days=60)

    # Existing 8 features from fundamental_pit.parquet
    fund = pd.read_parquet(PANEL_DIR / 'fundamental_pit.parquet')
    fund['available_date'] = pd.to_datetime(fund['available_date'])

    existing_features = ['pe', 'pb', 'roe_ttm', 'roa_ttm', 'netmargin', 'de',
                          'eps_growth_yoy', 'rev_growth_yoy']
    new_sf1_features = ['ps', 'evebitda', 'roic', 'fcfps']

    # Add new features from SF1
    sf1_join_cols = ['ticker', 'available_date'] + new_sf1_features
    new_panel = sf1[['ticker', 'available_date'] + new_sf1_features].copy()
    new_panel.columns = sf1_join_cols
    fund = fund.merge(new_panel, on=['ticker', 'available_date'], how='left')

    print(f'  Long fund panel: {len(fund):,} rows', flush=True)

    # Convert to wide per feature: for each date, pick latest fundamental ≤ date with freshness ≤ 90d
    # Use merge_asof per ticker (slow but already-tested logic)
    common_tickers = sorted(set(fund['ticker'].unique()) & set(close.columns))
    dates_df = pd.DataFrame({'date': close.index}).sort_values('date').reset_index(drop=True)

    all_feats = existing_features + new_sf1_features
    out = {f: pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
           for f in all_feats}

    print(f'  Building wide panels for {len(all_feats)} fundamental features ({len(common_tickers)} tickers)...',
          flush=True)
    for i, ticker in enumerate(common_tickers):
        if i % 200 == 0:
            print(f'    {i}/{len(common_tickers)}...', flush=True)
        sub = fund[fund['ticker'] == ticker][['available_date'] + all_feats].copy()
        sub = sub.sort_values('available_date').rename(columns={'available_date': 'avail'})
        sub['avail_match'] = sub['avail']
        merged = pd.merge_asof(
            dates_df, sub,
            left_on='date', right_on='avail',
            direction='backward',
        )
        freshness = (merged['date'] - merged['avail_match']).dt.days
        ok = freshness <= 90
        for f in all_feats:
            vals = merged[f].values.copy()
            vals[~ok.values] = np.nan
            out[f][ticker] = vals

    # Save each (apply cross-sectional z-score)
    print(f'\n  Saving fundamental panels...', flush=True)
    for f in all_feats:
        save(cross_sectional_zscore(out[f]), f)


# ─────────────────────────────────────────────────────────────────────
# Macro — VIX from FRED
# ─────────────────────────────────────────────────────────────────────
def build_macro(close):
    print('\n[Macro]', flush=True)
    cache_dir = Path('/home/dlfnek/stock_lab/data/macro')
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_fred(series_id):
        cache_path = cache_dir / f'{series_id}.csv'
        if cache_path.exists():
            print(f'  cached {series_id}', flush=True)
            return pd.read_csv(cache_path, parse_dates=['observation_date']).set_index('observation_date').iloc[:, 0]
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        urllib.request.urlretrieve(url, cache_path)
        print(f'  downloaded {series_id}', flush=True)
        df = pd.read_csv(cache_path, parse_dates=['observation_date'])
        return df.set_index('observation_date').iloc[:, 0]

    # VIX
    vix = fetch_fred('VIXCLS')
    vix = pd.to_numeric(vix, errors='coerce').dropna()
    vix = vix.reindex(close.index).ffill()

    # 10Y - 2Y yield curve
    t10 = fetch_fred('DGS10')
    t10 = pd.to_numeric(t10, errors='coerce').dropna()
    t2 = fetch_fred('DGS2')
    t2 = pd.to_numeric(t2, errors='coerce').dropna()
    yc = (t10 - t2).reindex(close.index).ffill()

    # Broadcast to all tickers (same value per date)
    n_tickers = len(close.columns)
    vix_panel = pd.DataFrame(
        np.tile(vix.values.reshape(-1, 1), (1, n_tickers)),
        index=close.index, columns=close.columns,
    )
    yc_panel = pd.DataFrame(
        np.tile(yc.values.reshape(-1, 1), (1, n_tickers)),
        index=close.index, columns=close.columns,
    )

    # NOTE: Cross-sectional z-score is meaningless for macro (same value per ticker per day).
    # Save raw — Phase Y will normalize differently if needed.
    save(vix_panel, 'vix_level')
    save(yc_panel, 'yield_curve')


def main():
    print('=== Phase X: Building all feature panels ===', flush=True)

    print('Loading panels + sector mapping...', flush=True)
    close = pd.read_parquet(PANEL_DIR / 'sp500_close.parquet')
    vol = pd.read_parquet(PANEL_DIR / 'sp500_volume.parquet')
    print(f'Close: {close.shape}', flush=True)

    # Sector mapping (Sharadar TICKERS)
    tickers_df = pd.read_parquet(SHARADAR_DIR / 'TICKERS.parquet')
    sf1_tk = tickers_df[tickers_df['table'] == 'SF1']
    sector_map = sf1_tk.set_index('ticker')['famaindustry'].to_dict()
    sicsector_map = sf1_tk.set_index('ticker')['sicsector'].to_dict()
    # Fallback to sicsector
    for tk in close.columns:
        if tk not in sector_map or pd.isna(sector_map.get(tk)):
            sector_map[tk] = sicsector_map.get(tk, 'Unknown')
    print(f'Sector mapping: {len(sector_map):,} tickers', flush=True)

    # SPY for market beta
    from phase1_compare import load_spy_returns
    spy_ret = load_spy_returns(close.index.min(), close.index.max())

    # 1. Price/volume features
    daily_ret = build_price_features(close, vol)

    # 2. Market-relative
    build_market_relative(close, daily_ret, spy_ret)

    # 3. Sector-relative
    build_sector_relative(close, daily_ret, sector_map)

    # 4. Fundamental
    build_fundamental_features(close)

    # 5. Macro (VIX, yield curve)
    build_macro(close)

    # Summary
    print('\n=== Phase X Complete ===', flush=True)
    files = sorted(FEATURE_DIR.glob('*.parquet'))
    total_size = sum(f.stat().st_size for f in files) / 1e6
    print(f'Total features: {len(files)}', flush=True)
    print(f'Total size: {total_size:.1f}MB', flush=True)
    print(f'\nFeature list:', flush=True)
    for f in files:
        print(f'  {f.stem}', flush=True)


if __name__ == '__main__':
    main()
