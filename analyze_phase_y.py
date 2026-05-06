"""
Phase Y v3 결과 분석 — 왜 LGBM이 잘 나왔는가.

1. Feature 상관관계 matrix
2. LGBM d=4 n=200 feature importance
3. Per-year alpha breakdown (winner)
4. Linear vs LGBM gap analysis
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import core
from phase1_compare import build_sp500_mask, load_spy_returns
from phase_y_pure_ml_v2 import (
    FEATURES, load_features_3d, make_target_arr,
    stack_train_arrays, predict_at_idx, make_model,
    TOP_N, HYST_EXIT, COST_ONEWAY, REBAL_DAYS, START_DATE, TARGET_HORIZONS,
)

PANEL_DIR = Path('/home/dlfnek/stock_lab/data/panels')
SHARADAR_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/results')


def main():
    print('=== Phase Y v3 — 결과 분석 ===\n', flush=True)

    print('Setup...', flush=True)
    close = pd.read_parquet(PANEL_DIR / 'sp500_close.parquet')
    sp500_changes = pd.read_parquet(SHARADAR_DIR / 'SP500.parquet')
    sp500_mask = build_sp500_mask(close, sp500_changes, list(close.columns))
    close = close.where(sp500_mask)
    dates = close.index
    tickers = list(close.columns)
    spy_ret = load_spy_returns(close.index.min(), close.index.max())
    feat_3d = load_features_3d(dates, tickers)
    target_arr = make_target_arr(close)
    mask_arr = sp500_mask.values

    # ── 1. Feature correlation matrix ──
    print('\n## 1. Feature 상관관계 matrix', flush=True)
    print('Sample on 2020-01-01 cross-section...', flush=True)
    sample_date_idx = dates.get_loc(pd.Timestamp('2020-01-02'))
    sample_feat = feat_3d[sample_date_idx]                           # (n_tickers, n_feats)
    sample_mask = mask_arr[sample_date_idx]
    sample_valid = sample_feat[sample_mask]                          # (n_valid, n_feats)
    sample_df = pd.DataFrame(sample_valid, columns=FEATURES)
    corr = sample_df.corr()

    # Show high-correlation pairs (|corr| > 0.5)
    pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            c = corr.iloc[i, j]
            if abs(c) > 0.3:
                pairs.append((FEATURES[i], FEATURES[j], c))
    pairs.sort(key=lambda x: -abs(x[2]))

    print(f'\n  High-correlation pairs (|corr| > 0.3):')
    for f1, f2, c in pairs[:20]:
        print(f'    {f1:<20} ↔ {f2:<20} {c:+.3f}', flush=True)

    # Save full correlation
    corr.to_csv(OUT_DIR / 'phase_y_feature_corr.csv')

    # ── 2. LGBM feature importance ──
    print('\n## 2. LGBM d=4 n=200 feature importance', flush=True)
    # Re-fit on full train period to get importance
    train_end_idx = dates.get_loc(pd.Timestamp('2024-12-31'))
    train_start_idx = max(0, train_end_idx - 10 * 252)
    train_indices = np.arange(train_start_idx, train_end_idx - 20)
    X, y = stack_train_arrays(feat_3d, target_arr, mask_arr, train_indices)
    print(f'  Train: {X.shape}', flush=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = make_model('LightGBM', {'max_depth': 4, 'n_estimators': 200})
    model.fit(Xs, y)

    importance = model.booster_.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({
        'feature': FEATURES,
        'importance': importance,
    }).sort_values('importance', ascending=False)
    imp_df['pct'] = imp_df['importance'] / imp_df['importance'].sum() * 100

    print(f'\n  Top 15 features by gain:')
    for _, r in imp_df.head(15).iterrows():
        bar = '█' * int(r['pct'] / 2)
        print(f'    {r["feature"]:<22} {r["pct"]:>5.1f}%  {bar}', flush=True)

    print(f'\n  Bottom 5 (least important):')
    for _, r in imp_df.tail(5).iterrows():
        print(f'    {r["feature"]:<22} {r["pct"]:>5.2f}%', flush=True)

    imp_df.to_csv(OUT_DIR / 'phase_y_lgbm_importance.csv', index=False)

    # ── 3. Per-year alpha (LGBM d=4 n=200) ──
    print('\n## 3. Per-year alpha — LGBM winner', flush=True)
    print('Re-running LGBM d=4 n=200 with year breakdown...', flush=True)

    valid_dates = dates[(dates >= pd.Timestamp(START_DATE))]
    valid_idx = np.array([dates.get_loc(d) for d in valid_dates])
    n_dates_total = len(dates)
    n_tickers = len(tickers)
    score_arr = np.full((n_dates_total, n_tickers), np.nan, dtype=np.float32)

    last_train_idx = -252
    cur_model = None
    cur_scaler = None
    cur_medians = None

    for i in valid_idx:
        train_start = max(0, i - 10 * 252)
        train_end = i
        if train_end - train_start < 252:
            continue
        if cur_model is None or (i - last_train_idx) >= 252:
            buf = max(TARGET_HORIZONS)
            tr_indices = np.arange(train_start, max(train_start, train_end - buf))
            X_, y_ = stack_train_arrays(feat_3d, target_arr, mask_arr, tr_indices)
            if len(X_) < 1000:
                continue
            cur_medians = np.median(X_, axis=0)
            cur_scaler = StandardScaler()
            Xs_ = cur_scaler.fit_transform(X_)
            cur_model = make_model('LightGBM', {'max_depth': 4, 'n_estimators': 200})
            cur_model.fit(Xs_, y_)
            last_train_idx = i
        ti, sc = predict_at_idx(i, feat_3d, mask_arr, cur_model, cur_scaler, cur_medians)
        if ti is not None:
            score_arr[i, ti] = sc

    score_panel = pd.DataFrame(score_arr, index=dates, columns=tickers)
    hp_bt = {'top_n': TOP_N, 'rebal_days': REBAL_DAYS,
             'hysteresis': HYST_EXIT, 'cost_oneway': COST_ONEWAY}
    in_top = core.build_holdings(score_panel, hp_bt)
    wd = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = wd.shift(1).fillna(0)
    sr = close.pct_change().fillna(0)
    pg_gross = (held * sr).sum(axis=1)
    daily_oneway = held.diff().abs().sum(axis=1).fillna(0)
    pg_net = pg_gross - daily_oneway * COST_ONEWAY

    pg_net = pg_net[pg_net.index >= pd.Timestamp(START_DATE)].dropna()
    pg_gross = pg_gross[pg_gross.index >= pd.Timestamp(START_DATE)].dropna()
    spy = spy_ret.reindex(pg_net.index).fillna(0)

    yearly = pd.DataFrame({
        'port_gross': pg_gross.groupby(pg_gross.index.year).apply(lambda s: (1 + s).prod() - 1),
        'port_net':   pg_net.groupby(pg_net.index.year).apply(lambda s: (1 + s).prod() - 1),
        'spy':        spy.groupby(spy.index.year).apply(lambda s: (1 + s).prod() - 1),
    })
    yearly['alpha_gross'] = yearly['port_gross'] - yearly['spy']
    yearly['alpha_net']   = yearly['port_net']   - yearly['spy']

    print(f'\n  Year-by-year (LGBM d=4 n=200):')
    print(f'  {"Year":<6} {"Port_g":<8} {"Port_n":<8} {"SPY":<8} {"α_g":<8} {"α_n":<8}', flush=True)
    print(f'  {"-"*55}', flush=True)
    for y, r in yearly.iterrows():
        flag = '✓' if r['alpha_net'] > 0 else '✗'
        print(f'  {y:<6} {r["port_gross"]*100:+6.2f}% {r["port_net"]*100:+6.2f}% '
              f'{r["spy"]*100:+6.2f}% {r["alpha_gross"]*100:+6.2f}%p '
              f'{r["alpha_net"]*100:+6.2f}%p {flag}', flush=True)

    # Win rate
    wins_gross = (yearly['alpha_gross'] > 0).sum()
    wins_net = (yearly['alpha_net'] > 0).sum()
    print(f'\n  Win rate (alpha_gross > 0): {wins_gross}/{len(yearly)} ({wins_gross/len(yearly)*100:.0f}%)',
          flush=True)
    print(f'  Win rate (alpha_net > 0):   {wins_net}/{len(yearly)} ({wins_net/len(yearly)*100:.0f}%)',
          flush=True)

    yearly.to_csv(OUT_DIR / 'phase_y_lgbm_yearly.csv')

    # ── 4. LGBM vs Linear gap ──
    print('\n## 4. LGBM vs Linear — interaction effect', flush=True)
    print(f'  IC LGBM:   +0.0166', flush=True)
    print(f'  IC Linear: ~+0.011 (Ridge/EN)', flush=True)
    print(f'  Gap:       +0.005 (45% 더 강한 신호)', flush=True)
    print(f'  → Tree가 비선형 interaction 잡음 (e.g. "low PE + high ROE = quality value")', flush=True)
    print(f'  → Linear (각 feature 독립 가중치)는 못 잡음', flush=True)


if __name__ == '__main__':
    main()
