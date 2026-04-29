"""
Effective N (correlation-adjusted diversification) diagnostic.

For each weekly walk-forward window, compute:
- Realized portfolio's average pairwise correlation
- Effective N = N / (1 + (N-1) * avg_corr)
- Per-window time series of Eff N

Compare with sector diversity to see if sector cap actually achieves real diversification.

Output: results/effective_n.txt
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
HOLD_DAYS = 5
FEATURES = ['lowvol', 'rsi', 'volsurge']
CORR_LOOKBACK = 60  # days for computing correlation


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last()


def load_sectors():
    df = pd.read_csv(SECTORS_PATH, index_col='Ticker')
    return df['Sector'].to_dict()


def get_monday_schedule(close_index, start_date, end_date):
    dates = close_index[(close_index >= start_date) & (close_index < end_date)]
    schedule = []
    seen = set()
    for d in dates:
        iy, iw, _ = d.isocalendar()
        if (iy, iw) not in seen:
            seen.add((iy, iw))
            schedule.append(d)
    return schedule


def avg_pairwise_corr(returns_df):
    """Avg of all pairwise correlations (excluding diagonal)."""
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


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Effective N diagnostic on v5 weekly walk-forward")
    w(f"  Features: {FEATURES} | Top-{TOP_N} | corr lookback {CORR_LOOKBACK}d")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    sectors = load_sectors()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    start = pd.Timestamp('2025-01-01')
    end = pd.Timestamp('2026-04-28')
    schedule = get_monday_schedule(close.index, start, end)
    w(f"\nTotal weeks: {len(schedule)}\n")

    rows = []
    for week_idx, monday in enumerate(schedule):
        train_end = monday
        train_start = monday - pd.DateOffset(years=TRAIN_YEARS)
        train_mask = (close.index >= train_start) & (close.index < train_end)
        if train_mask.sum() < 252:
            continue

        valid = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
        if len(valid) < 100:
            continue
        close_sub = close[valid]
        vol_sub = vol[valid]

        train_long, _, _, _ = ml_model.get_train_test_features(
            close_sub, vol_sub, train_mask,
            close_sub.index == monday, hp)
        train_long = train_long.dropna(subset=FEATURES)
        if len(train_long) < 1000:
            continue

        feat_panels = ml_model.build_features_panel(close_sub, vol_sub)
        score_feats = {n: df.loc[[monday]] if monday in df.index else df.tail(0)
                       for n, df in feat_panels.items()}
        score_long = ml_model.stack_panel_to_long(score_feats)
        score_long = score_long.dropna(subset=FEATURES)
        if len(score_long) < TOP_N:
            continue

        scaler = StandardScaler()
        Xs = scaler.fit_transform(train_long[FEATURES].values)
        model = Ridge(alpha=1.0)
        model.fit(Xs, train_long['target'].values)
        preds = model.predict(scaler.transform(score_long[FEATURES].values))
        score_long['score'] = preds
        top = score_long.sort_values('score', ascending=False).head(TOP_N)
        held_tickers = top['ticker'].tolist()

        # Compute correlation: lookback 60d before monday
        lookback_start = monday - pd.Timedelta(days=int(CORR_LOOKBACK * 1.5))
        lookback_mask = (close.index >= lookback_start) & (close.index < monday)
        rets = close_sub.loc[lookback_mask, held_tickers].pct_change().dropna(how='all')
        if len(rets) < 30:
            continue

        avg_corr = avg_pairwise_corr(rets)
        eff_n = effective_n(TOP_N, avg_corr)

        # Sector diversity
        sec_counts = defaultdict(int)
        for t in held_tickers:
            sec_counts[sectors.get(t, 'Unknown')] += 1
        n_sectors = len(sec_counts)
        max_sec_count = max(sec_counts.values())
        # Herfindahl on sector
        n_total = len(held_tickers)
        hhi_sec = sum((c / n_total) ** 2 for c in sec_counts.values())
        eff_sectors = 1 / hhi_sec if hhi_sec > 0 else np.nan

        # Top sector + share
        top_sec = max(sec_counts, key=sec_counts.get)
        top_sec_share = sec_counts[top_sec] / n_total

        rows.append({
            'week': week_idx + 1,
            'monday': monday.date(),
            'avg_corr': avg_corr,
            'eff_n': eff_n,
            'n_sectors': n_sectors,
            'eff_sectors': eff_sectors,
            'top_sector': top_sec,
            'top_sec_share': top_sec_share,
            'top3': held_tickers[:3],
        })

        if (week_idx + 1) % 10 == 0 or week_idx < 3:
            w(f"  Week {week_idx+1:>2} ({monday.date()}): "
              f"corr {avg_corr:+.2f} | EffN {eff_n:.1f} | "
              f"sectors {n_sectors} (eff {eff_sectors:.1f}) | "
              f"top sec: {top_sec} ({top_sec_share*100:.0f}%) | "
              f"top3: {held_tickers[:3]}")

    if not rows:
        w("\nNo results")
        return

    df = pd.DataFrame(rows)
    n = len(df)

    w(f"\n{'='*100}")
    w(f"## Aggregate ({n} weeks)")
    w(f"  Average pairwise correlation:    {df['avg_corr'].mean():+.3f}")
    w(f"  Average Effective N:             {df['eff_n'].mean():.2f}")
    w(f"  Median Effective N:              {df['eff_n'].median():.2f}")
    w(f"  Min Eff N:                       {df['eff_n'].min():.2f} ({df.loc[df['eff_n'].idxmin(), 'monday']})")
    w(f"  Max Eff N:                       {df['eff_n'].max():.2f} ({df.loc[df['eff_n'].idxmax(), 'monday']})")
    w(f"")
    w(f"  Average # distinct sectors:      {df['n_sectors'].mean():.1f}")
    w(f"  Average effective sectors:       {df['eff_sectors'].mean():.2f}")
    w(f"  Average top-sector share:        {df['top_sec_share'].mean()*100:.1f}%")
    w(f"  Max top-sector share:            {df['top_sec_share'].max()*100:.1f}%")

    # Distribution
    w(f"\n## Effective N distribution")
    w(f"  P10: {df['eff_n'].quantile(0.10):.2f}")
    w(f"  P25: {df['eff_n'].quantile(0.25):.2f}")
    w(f"  P50: {df['eff_n'].quantile(0.50):.2f}")
    w(f"  P75: {df['eff_n'].quantile(0.75):.2f}")
    w(f"  P90: {df['eff_n'].quantile(0.90):.2f}")

    # Top sector breakdown
    w(f"\n## Top sector frequency (which sector dominates)")
    sec_freq = df['top_sector'].value_counts()
    for sec, cnt in sec_freq.head(8).items():
        w(f"  {sec:<25} {cnt:>3}/{n} ({cnt/n*100:.1f}%)")

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
    df_save = df.drop(columns=['top3'])
    df_save['top3'] = df['top3'].astype(str)
    df_save.to_csv(os.path.join(OUTPUT_DIR, 'effective_n.csv'), index=False)

    out_path = os.path.join(OUTPUT_DIR, 'effective_n.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
