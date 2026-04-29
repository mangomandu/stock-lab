"""
Effective N baseline comparison.

For each rebalance date in 2015-2025:
- A. Our model's Top-20 (from Ridge)
- B. Random Top-20 from same valid universe (10 random draws, averaged)
- C. SPY underlying — entire S&P 500 valid universe at that time

Compare avg pairwise corr + Effective N.

Question: is our model's concentration WORSE than random/full-universe?

Output: results/effective_n_baseline.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import ml_model
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
TRAIN_YEARS = 7
REBAL_DAYS = 5
FEATURES = ['lowvol', 'rsi', 'volsurge']
CORR_LOOKBACK = 60
N_RANDOM_DRAWS = 10

np.random.seed(42)


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


def measure(close_sub, tickers, ref_date):
    lb_start = ref_date - pd.Timedelta(days=int(CORR_LOOKBACK * 1.5))
    lb_mask = (close_sub.index >= lb_start) & (close_sub.index < ref_date)
    rets = close_sub.loc[lb_mask, tickers].pct_change().dropna(how='all')
    if len(rets) < 30:
        return None, None
    ac = avg_pairwise_corr(rets)
    n_actual = rets.shape[1]
    return ac, effective_n(n_actual, ac)


def run_one_year(close, vol, test_year, hp):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return []

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return []

    valid = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid) < 100:
        return []
    close_sub = close[valid]
    vol_sub = vol[valid]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    train_long = train_long.dropna(subset=FEATURES)
    test_long = test_long.dropna(subset=FEATURES)
    if len(train_long) < 1000 or len(test_long) == 0:
        return []

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
        valid_now = row.dropna().index.tolist()
        if len(valid_now) < TOP_N:
            continue
        # A. Our model's Top-20
        held_a = row.dropna().sort_values(ascending=False).head(TOP_N).index.tolist()
        ac_a, en_a = measure(close_sub, held_a, rebal_date)

        # B. Random Top-20, averaged over multiple draws
        random_corrs = []
        random_effs = []
        for _ in range(N_RANDOM_DRAWS):
            held_b = list(np.random.choice(valid_now, TOP_N, replace=False))
            ac_b, en_b = measure(close_sub, held_b, rebal_date)
            if ac_b is not None and not np.isnan(ac_b):
                random_corrs.append(ac_b)
                random_effs.append(en_b)
        ac_b_avg = np.mean(random_corrs) if random_corrs else np.nan
        en_b_avg = np.mean(random_effs) if random_effs else np.nan

        # C. Full universe (SPY underlying — all valid stocks)
        ac_c, en_c = measure(close_sub, valid_now, rebal_date)

        rows.append({
            'year': test_year,
            'date': rebal_date.date(),
            'n_universe': len(valid_now),
            'corr_model': ac_a, 'effn_model': en_a,
            'corr_random': ac_b_avg, 'effn_random': en_b_avg,
            'corr_full': ac_c, 'effn_full': en_c,
        })
    return rows


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Effective N — baseline comparison")
    w(f"  Years: 2015-2025 (11 years) | Top-{TOP_N}")
    w(f"  A. Ridge model | B. Random Top-20 (avg of {N_RANDOM_DRAWS}) | C. Full universe")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    all_rows = []
    for test_year in range(2015, 2026):
        rows = run_one_year(close, vol, test_year, hp)
        all_rows.extend(rows)
        if rows:
            df_y = pd.DataFrame(rows)
            w(f"  Year {test_year}: rebals {len(rows):>3} | "
              f"Model corr {df_y['corr_model'].mean():+.2f} EffN {df_y['effn_model'].mean():.1f} | "
              f"Random corr {df_y['corr_random'].mean():+.2f} EffN {df_y['effn_random'].mean():.1f} | "
              f"Full corr {df_y['corr_full'].mean():+.2f} EffN {df_y['effn_full'].mean():.1f}")

    if not all_rows:
        w("\nNo results")
        return

    df = pd.DataFrame(all_rows)
    n = len(df)

    w(f"\n{'='*100}")
    w(f"## Aggregate ({n} rebal events, 2015-2025)")
    w(f"")
    w(f"{'Group':<25} {'Avg corr':>10} {'Avg EffN':>10} {'Median EffN':>13}")
    w("-" * 65)
    w(f"{'A. Our Ridge model':<25} {df['corr_model'].mean():>+9.3f} "
      f"{df['effn_model'].mean():>10.2f} {df['effn_model'].median():>13.2f}")
    w(f"{'B. Random Top-20':<25} {df['corr_random'].mean():>+9.3f} "
      f"{df['effn_random'].mean():>10.2f} {df['effn_random'].median():>13.2f}")
    w(f"{'C. Full universe':<25} {df['corr_full'].mean():>+9.3f} "
      f"{df['effn_full'].mean():>10.2f} {df['effn_full'].median():>13.2f}")

    # Difference analysis
    w(f"\n## Model vs Random (positive = we are MORE concentrated)")
    w(f"  Δ corr:  {df['corr_model'].mean() - df['corr_random'].mean():+.3f}")
    w(f"  Δ EffN:  {df['effn_model'].mean() - df['effn_random'].mean():+.2f}")
    w(f"  Win rate (model > random EffN): "
      f"{(df['effn_model'] > df['effn_random']).sum()}/{n} "
      f"({(df['effn_model'] > df['effn_random']).sum()/n*100:.1f}%)")

    # Stress periods
    w(f"\n## Stress periods (top 10% corr_full = market under stress)")
    stress_threshold = df['corr_full'].quantile(0.90)
    stress = df[df['corr_full'] >= stress_threshold]
    w(f"  Stress threshold: corr_full >= {stress_threshold:.2f}")
    w(f"  N events: {len(stress)}")
    w(f"  Stress avg EffN — Model:  {stress['effn_model'].mean():.2f}")
    w(f"  Stress avg EffN — Random: {stress['effn_random'].mean():.2f}")
    w(f"  Stress avg EffN — Full:   {stress['effn_full'].mean():.2f}")

    # Calm periods
    w(f"\n## Calm periods (bottom 10% corr_full)")
    calm_threshold = df['corr_full'].quantile(0.10)
    calm = df[df['corr_full'] <= calm_threshold]
    w(f"  Calm threshold: corr_full <= {calm_threshold:.2f}")
    w(f"  N events: {len(calm)}")
    w(f"  Calm avg EffN — Model:  {calm['effn_model'].mean():.2f}")
    w(f"  Calm avg EffN — Random: {calm['effn_random'].mean():.2f}")
    w(f"  Calm avg EffN — Full:   {calm['effn_full'].mean():.2f}")

    # Diagnosis
    w(f"\n## Diagnosis")
    delta_eff = df['effn_model'].mean() - df['effn_random'].mean()
    delta_corr = df['corr_model'].mean() - df['corr_random'].mean()
    if delta_eff < -1.0:
        w(f"  ❌ 우리 모델 명백한 몰빵 (랜덤 baseline 대비 EffN {delta_eff:.1f})")
        w(f"     → Factor cap / ETF buffer 가치 큼")
    elif delta_eff < -0.3:
        w(f"  ⚠ 약간 더 몰빵 (랜덤 baseline 대비 EffN {delta_eff:.1f})")
        w(f"     → ETF buffer로 베타 안정 시도 가치 있음")
    else:
        w(f"  ✅ 랜덤 baseline 수준 (EffN diff {delta_eff:+.1f})")
        w(f"     → 우리 모델 특유 몰빵 X. 시장 자체 corr 한계 — 폭락장은 어차피 다 떨어짐")

    out_path = os.path.join(OUTPUT_DIR, 'effective_n_baseline.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    df.to_csv(os.path.join(OUTPUT_DIR, 'effective_n_baseline.csv'), index=False)
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
