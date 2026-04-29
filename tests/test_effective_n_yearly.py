"""
Effective N diagnostic — Yearly walk-forward (31 windows, 1995-2025).

For each test year, run Ridge model and at each rebalance date (every 5 days)
compute the realized portfolio's avg pairwise correlation + Effective N.

Aggregate per-window + per-year time series.

Output: results/effective_n_yearly.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import ml_model
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
SECTORS_PATH = '/home/dlfnek/stock_lab/data/sectors.csv'
TOP_N = 20
TRAIN_YEARS = 7
REBAL_DAYS = 5
FEATURES = ['lowvol', 'rsi', 'volsurge']
CORR_LOOKBACK = 60


def load_sectors():
    df = pd.read_csv(SECTORS_PATH, index_col='Ticker')
    return df['Sector'].to_dict()


def avg_pairwise_corr(returns_df):
    if returns_df.shape[1] < 2:
        return np.nan
    corr = returns_df.corr().values
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    pairs = corr[iu]
    pairs = pairs[~np.isnan(pairs)]
    if len(pairs) == 0:
        return np.nan
    return pairs.mean()


def effective_n(n, avg_corr):
    if avg_corr is None or np.isnan(avg_corr):
        return np.nan
    return n / (1 + (n - 1) * max(avg_corr, 0))


def run_one_year(close, vol, sectors, test_year, hp):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid) < 100:
        return None
    close_sub = close[valid]
    vol_sub = vol[valid]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    train_long = train_long.dropna(subset=FEATURES)
    test_long = test_long.dropna(subset=FEATURES)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(train_long[FEATURES].values)
    model = Ridge(alpha=1.0)
    model.fit(Xs, train_long['target'].values)
    preds = model.predict(scaler.transform(test_long[FEATURES].values))

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_dates = close_sub.index[test_mask]
    rebal_dates = test_dates[::REBAL_DAYS]

    rows = []
    for rebal_date in rebal_dates:
        if rebal_date not in score_wide.index:
            continue
        row = score_wide.loc[rebal_date]
        if row.dropna().shape[0] < TOP_N:
            continue
        held = row.dropna().sort_values(ascending=False).head(TOP_N).index.tolist()

        # Lookback corr
        lb_start = rebal_date - pd.Timedelta(days=int(CORR_LOOKBACK * 1.5))
        lb_mask = (close_sub.index >= lb_start) & (close_sub.index < rebal_date)
        rets = close_sub.loc[lb_mask, held].pct_change().dropna(how='all')
        if len(rets) < 30:
            continue

        avg_corr = avg_pairwise_corr(rets)
        eff_n = effective_n(TOP_N, avg_corr)

        sec_counts = defaultdict(int)
        for t in held:
            sec_counts[sectors.get(t, 'Unknown')] += 1
        n_sectors = len(sec_counts)
        n_total = len(held)
        hhi_sec = sum((c / n_total) ** 2 for c in sec_counts.values())
        eff_sectors = 1 / hhi_sec if hhi_sec > 0 else np.nan
        top_sec = max(sec_counts, key=sec_counts.get)
        top_share = sec_counts[top_sec] / n_total

        rows.append({
            'year': test_year,
            'date': rebal_date.date(),
            'avg_corr': avg_corr,
            'eff_n': eff_n,
            'n_sectors': n_sectors,
            'eff_sectors': eff_sectors,
            'top_sector': top_sec,
            'top_sec_share': top_share,
        })

    return rows


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Effective N — Yearly walk-forward 1995-2025")
    w(f"  Top-{TOP_N} | rebal every {REBAL_DAYS}d | corr lookback {CORR_LOOKBACK}d")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    sectors = load_sectors()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    all_rows = []
    yearly_summary = []
    for test_year in range(1995, 2026):
        rows = run_one_year(close, vol, sectors, test_year, hp)
        if not rows:
            continue
        all_rows.extend(rows)
        df_y = pd.DataFrame(rows)
        yearly_summary.append({
            'year': test_year,
            'n_rebal': len(rows),
            'avg_corr': df_y['avg_corr'].mean(),
            'avg_eff_n': df_y['eff_n'].mean(),
            'avg_n_sec': df_y['n_sectors'].mean(),
            'avg_eff_sec': df_y['eff_sectors'].mean(),
            'avg_top_share': df_y['top_sec_share'].mean(),
            'top_sec_dom': df_y['top_sector'].mode()[0] if not df_y['top_sector'].empty else 'N/A',
        })
        w(f"  Year {test_year}: rebals {len(rows):>3} | "
          f"corr {df_y['avg_corr'].mean():+.2f} | "
          f"EffN {df_y['eff_n'].mean():.1f} | "
          f"sectors {df_y['n_sectors'].mean():.1f} (eff {df_y['eff_sectors'].mean():.1f}) | "
          f"top sec: {df_y['top_sector'].mode()[0] if not df_y['top_sector'].empty else 'N/A'}")

    if not all_rows:
        w("\nNo results")
        return

    df = pd.DataFrame(all_rows)
    n = len(df)

    w(f"\n{'='*100}")
    w(f"## Aggregate ({n} rebal events across 31 years)")
    w(f"  Average pairwise correlation:    {df['avg_corr'].mean():+.3f}")
    w(f"  Average Effective N:             {df['eff_n'].mean():.2f}")
    w(f"  Median Effective N:              {df['eff_n'].median():.2f}")
    w(f"  Min Eff N:                       {df['eff_n'].min():.2f}")
    w(f"  Max Eff N:                       {df['eff_n'].max():.2f}")
    w(f"")
    w(f"  Average # distinct sectors:      {df['n_sectors'].mean():.1f}")
    w(f"  Average effective sectors:       {df['eff_sectors'].mean():.2f}")
    w(f"  Average top-sector share:        {df['top_sec_share'].mean()*100:.1f}%")

    # By era
    w(f"\n## By era (Effective N)")
    df['era'] = pd.cut(df['year'], bins=[1994, 2000, 2010, 2020, 2026],
                       labels=['1995-2000', '2001-2010', '2011-2020', '2021-2025'])
    era_stats = df.groupby('era', observed=True).agg(
        n=('eff_n', 'count'),
        avg_corr=('avg_corr', 'mean'),
        avg_eff_n=('eff_n', 'mean'),
        avg_n_sec=('n_sectors', 'mean'),
        avg_eff_sec=('eff_sectors', 'mean'),
        avg_top_share=('top_sec_share', 'mean'),
    )
    w(f"{'Era':<12} {'N':>5} {'Corr':>7} {'EffN':>6} {'#Sec':>5} {'EffSec':>7} {'TopShare':>9}")
    w("-" * 60)
    for era, row in era_stats.iterrows():
        w(f"{str(era):<12} {row['n']:>5.0f} {row['avg_corr']:>+6.2f} "
          f"{row['avg_eff_n']:>6.1f} {row['avg_n_sec']:>5.1f} "
          f"{row['avg_eff_sec']:>7.1f} {row['avg_top_share']*100:>+8.1f}%")

    # Distribution
    w(f"\n## Effective N distribution (all 31y)")
    w(f"  P10: {df['eff_n'].quantile(0.10):.2f}")
    w(f"  P25: {df['eff_n'].quantile(0.25):.2f}")
    w(f"  P50: {df['eff_n'].quantile(0.50):.2f}")
    w(f"  P75: {df['eff_n'].quantile(0.75):.2f}")
    w(f"  P90: {df['eff_n'].quantile(0.90):.2f}")

    # Sector dominance
    w(f"\n## Sector dominance frequency (top sector when Eff N low)")
    low_eff = df[df['eff_n'] < df['eff_n'].quantile(0.25)]
    sec_freq = low_eff['top_sector'].value_counts()
    w(f"  (rebal events with bottom 25% Eff N — most concentrated)")
    for sec, cnt in sec_freq.head(8).items():
        w(f"  {sec:<25} {cnt:>3}/{len(low_eff)} ({cnt/len(low_eff)*100:.1f}%)")

    # Diagnosis
    w(f"\n## Diagnosis")
    avg_eff = df['eff_n'].mean()
    avg_corr = df['avg_corr'].mean()
    if avg_eff > 12:
        w(f"  ✅ 진짜 분산 잘 됨 (Eff N {avg_eff:.1f}) — 추가 작업 불필요")
    elif avg_eff > 6:
        w(f"  ⚠ 보통 (Eff N {avg_eff:.1f}, corr {avg_corr:.2f}) — ETF buffer 검토 가치 있음")
    else:
        w(f"  ❌ 심각한 몰빵 (Eff N {avg_eff:.1f}, corr {avg_corr:.2f}) — factor cap / ETF buffer 필요")

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'effective_n_yearly.csv'), index=False)
    pd.DataFrame(yearly_summary).to_csv(os.path.join(OUTPUT_DIR, 'effective_n_yearly_summary.csv'), index=False)

    out_path = os.path.join(OUTPUT_DIR, 'effective_n_yearly.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
