"""
Phase Y v2 — Vectorized predict (100x faster).

Optimization:
  - Pre-stack all 27 features into 3D numpy array (n_dates, n_tickers, n_feats)
  - predict_at_date 가 numpy slicing 만 사용
  - 메모리 ~900MB

Setup (v0.7.2 정합):
  Universe: Sharadar historical S&P 500
  Cost: 0.25% per side (한국투자증권 평시)
  Configs B1 (fast/short) + B2 (slow/long)
  Models: 24 (Ridge 5 + Lasso 4 + EN 9 + LGBM 6)
  Total: 48 cells × ~3-5 min = ~3 hours
"""
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import core
from phase1_compare import build_sp500_mask, load_spy_returns

PANEL_DIR = Path('/home/dlfnek/stock_lab/data/panels')
FEATURE_DIR = PANEL_DIR / 'features'
SHARADAR_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/results')

# HP
TOP_N = 10
HYST_EXIT = 40
TARGET_HORIZONS = [5, 10, 20]
COST_ONEWAY = 0.0025  # 한국투자증권 평시 0.25%
START_DATE = pd.Timestamp('1998-01-01')

REBAL_DAYS = 5  # Weekly (Gemini 권장: 0.25% cost 환경 필수)
CONFIGS = [
    {'name': 'B2_slow_long',  'retrain_days': 252, 'train_years': 10},
]

FEATURES = [
    'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m_skip1', 'mom_accel',
    'reversal_1w',
    'lowvol', 'idio_vol', 'volsurge', 'high_52w_distance',
    'rsi', 'beta_to_spy',
    'sec_rel_mom_12m', 'sec_rel_mom_3m', 'sec_rel_lowvol',
    'pe', 'pb', 'ps', 'evebitda',
    'roe_ttm', 'roa_ttm', 'netmargin', 'de',
    'rev_growth_yoy', 'fcfps',
    'vix_level', 'yield_curve',
]


def load_features_3d(dates, tickers):
    """Load all features and stack into 3D numpy array (n_dates, n_tickers, n_feats)."""
    n_dates = len(dates)
    n_tickers = len(tickers)
    n_feats = len(FEATURES)
    print(f'Allocating 3D array: ({n_dates}, {n_tickers}, {n_feats}) = '
          f'{n_dates * n_tickers * n_feats * 4 / 1e9:.2f} GB', flush=True)

    arr = np.full((n_dates, n_tickers, n_feats), np.nan, dtype=np.float32)
    for fi, fn in enumerate(FEATURES):
        path = FEATURE_DIR / f'{fn}.parquet'
        if not path.exists():
            print(f'  WARN: {fn} not found', flush=True)
            continue
        df = pd.read_parquet(path)
        df = df.reindex(index=dates, columns=tickers)
        arr[:, :, fi] = df.values.astype(np.float32)
        print(f'  [{fi+1}/{n_feats}] {fn}', flush=True)
    return arr


def make_target_arr(close):
    rank_dfs = []
    for h in TARGET_HORIZONS:
        fwd = close.shift(-h) / close - 1
        rank_dfs.append(fwd.rank(axis=1, pct=True))
    return (sum(rank_dfs) / len(rank_dfs)).values.astype(np.float32)


def stack_train_arrays(feat_3d, target_arr, mask_arr, train_indices):
    """feat_3d 사용해서 (X, y) 빌드. NaN은 feature-별 median으로 fillna (Gemini 권장)."""
    feat_train = feat_3d[train_indices]                # (n_train, n_tickers, n_feats)
    target_train = target_arr[train_indices]
    mask_train = mask_arr[train_indices]

    target_valid = ~np.isnan(target_train)
    valid = mask_train & target_valid  # SP500 member + target valid (feature는 fillna)

    n_valid = int(valid.sum())
    X = np.empty((n_valid, len(FEATURES)), dtype=np.float32)
    for fi in range(len(FEATURES)):
        feat_col = feat_train[:, :, fi]
        # Median fill (per feature, on valid mask)
        valid_vals = feat_col[mask_train & ~np.isnan(feat_col)]
        median = np.median(valid_vals) if len(valid_vals) > 0 else 0.0
        col = feat_col.copy()
        col[np.isnan(col)] = median
        X[:, fi] = col[valid]
    y = target_train[valid]
    return X, y


def predict_at_idx(date_idx, feat_3d, mask_arr, model, scaler, feat_medians):
    """Fast vectorized predict. NaN은 feat_medians (학습 시 통계)로 fill."""
    feat_row = feat_3d[date_idx].copy()                # (n_tickers, n_feats)
    mask_row = mask_arr[date_idx]                      # (n_tickers,)

    if not mask_row.any():
        return None, None

    # Fill NaN with train-time medians per feature
    for fi in range(feat_row.shape[1]):
        col = feat_row[:, fi]
        col[np.isnan(col)] = feat_medians[fi]

    X = feat_row[mask_row]
    Xs = scaler.transform(X)
    scores = model.predict(Xs)
    return np.where(mask_row)[0], scores


def make_model(model_type, params):
    if model_type == 'Ridge':
        return Ridge(alpha=params['alpha'], random_state=42)
    elif model_type == 'Lasso':
        return Lasso(alpha=params['alpha'], random_state=42, max_iter=10000)
    elif model_type == 'ElasticNet':
        return ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'],
                          random_state=42, max_iter=10000)
    elif model_type == 'LightGBM':
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            num_leaves=2 ** params['max_depth'],
            random_state=42,
            verbose=-1,
        )


def run_cell(feat_3d, target_arr, mask_arr, dates, tickers, close,
             spy_ret, model_type, params, retrain_days, train_years, label):
    """Vectorized walk-forward."""
    print(f'\n=== {label} ===', flush=True)
    t0 = time.time()

    valid_dates = dates[(dates >= START_DATE)]
    valid_idx = np.array([dates.get_loc(d) for d in valid_dates])

    n_dates = len(dates)
    n_tickers = len(tickers)
    score_arr = np.full((n_dates, n_tickers), np.nan, dtype=np.float32)

    last_train_idx = -retrain_days
    model = None
    scaler = None
    feat_medians = None
    n_retrains = 0

    for i in valid_idx:
        train_start = max(0, i - train_years * 252)
        train_end = i
        if train_end - train_start < 252:
            continue
        if model is None or (i - last_train_idx) >= retrain_days:
            buf = max(TARGET_HORIZONS)
            train_indices = np.arange(train_start, max(train_start, train_end - buf))
            X, y = stack_train_arrays(feat_3d, target_arr, mask_arr, train_indices)
            if len(X) < 1000:
                continue
            # Save train-time median per feature (for predict-time fillna)
            feat_medians = np.median(X, axis=0)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = make_model(model_type, params)
            try:
                model.fit(Xs, y)
            except Exception as e:
                print(f'  fit failed at idx={i}: {e}', flush=True)
                continue
            last_train_idx = i
            n_retrains += 1

        ticker_idx, scores = predict_at_idx(i, feat_3d, mask_arr, model, scaler, feat_medians)
        if ticker_idx is not None:
            score_arr[i, ticker_idx] = scores

    t_pred = time.time() - t0

    # Portfolio sim (pandas로 변환). Weekly rebal (rebal_days=5).
    score_panel = pd.DataFrame(score_arr, index=dates, columns=tickers)
    hp_bt = {'top_n': TOP_N, 'rebal_days': REBAL_DAYS,
             'hysteresis': HYST_EXIT, 'cost_oneway': COST_ONEWAY}
    in_top = core.build_holdings(score_panel, hp_bt)
    wd = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = wd.shift(1).fillna(0)
    sr = close.pct_change().fillna(0)
    pg_gross = (held * sr).sum(axis=1)
    daily_oneway = held.diff().abs().sum(axis=1).fillna(0)
    cost_drag = daily_oneway * COST_ONEWAY
    pg_net = pg_gross - cost_drag

    pg_net = pg_net[pg_net.index >= START_DATE].dropna()
    pg_gross = pg_gross[pg_gross.index >= START_DATE].dropna()
    daily_oneway = daily_oneway[daily_oneway.index >= START_DATE]

    spy = spy_ret.reindex(pg_net.index).fillna(0)

    def cagr(s):
        cum = (1 + s).prod() - 1
        ny = len(s) / 252
        return (1 + cum) ** (1 / ny) - 1 if ny > 0 else 0

    p_gross = float(cagr(pg_gross))
    p_net = float(cagr(pg_net))
    spy_c = float(cagr(spy))
    sharpe = float(pg_net.mean() / pg_net.std() * np.sqrt(252)) if pg_net.std() > 0 else 0
    cum_curve = (1 + pg_net).cumprod()
    mdd = float((cum_curve / cum_curve.cummax() - 1).min())
    annual_turnover = float(daily_oneway.sum() / (len(daily_oneway) / 252))

    # Rank IC
    fwd_21 = close.shift(-21) / close - 1
    ic_per_day = []
    for d in score_panel.dropna(how='all').index:
        s = score_panel.loc[d].dropna()
        r = fwd_21.loc[d].reindex(s.index).dropna() if d in fwd_21.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)
        if len(common) < 20:
            continue
        s_vals, r_vals = s[common].values, r[common].values
        if np.std(s_vals) == 0 or np.std(r_vals) == 0:
            continue
        ic, _ = spearmanr(s_vals, r_vals)
        if not np.isnan(ic):
            ic_per_day.append(ic)
    ic_mean = float(np.mean(ic_per_day)) if ic_per_day else 0.0
    ic_std = float(np.std(ic_per_day)) if ic_per_day else 1e-9
    ic_ir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else 0

    elapsed = time.time() - t0
    print(f'  α_gross={(p_gross - spy_c)*100:+.2f}%p  α_net={(p_net - spy_c)*100:+.2f}%p  '
          f'Sharpe={sharpe:.2f}  MDD={mdd*100:.1f}%  IC={ic_mean:+.4f}  '
          f'TO={annual_turnover:.1f}x  ({elapsed:.0f}s, predict={t_pred:.0f}s, retrains={n_retrains})',
          flush=True)

    return {
        'label': label,
        'model': model_type,
        'params': params,
        'gross_alpha': float(p_gross - spy_c),
        'net_alpha': float(p_net - spy_c),
        'cagr_gross': p_gross,
        'cagr_net': p_net,
        'cagr_spy': spy_c,
        'sharpe': sharpe,
        'mdd': mdd,
        'ic_mean': ic_mean,
        'ic_ir': float(ic_ir),
        'annual_turnover': annual_turnover,
        'n_days': len(pg_net),
        'n_retrains': n_retrains,
        'elapsed_sec': elapsed,
    }


def main():
    print('=== Phase Y v2: Vectorized bake-off ===', flush=True)

    print('Loading panels...', flush=True)
    close = pd.read_parquet(PANEL_DIR / 'sp500_close.parquet')
    sp500_changes = pd.read_parquet(SHARADAR_DIR / 'SP500.parquet')
    sp500_mask = build_sp500_mask(close, sp500_changes, list(close.columns))
    close = close.where(sp500_mask)

    dates = close.index
    tickers = list(close.columns)

    spy_ret = load_spy_returns(close.index.min(), close.index.max())

    print('\nLoading + stacking features (3D)...', flush=True)
    t0 = time.time()
    feat_3d = load_features_3d(dates, tickers)
    print(f'  Done: {(time.time()-t0):.0f}s, '
          f'memory ≈ {feat_3d.nbytes/1e9:.2f}GB', flush=True)

    print('\nBuilding target...', flush=True)
    target_arr = make_target_arr(close)
    mask_arr = sp500_mask.values

    # Models
    models = []
    for alpha in [0.1, 1, 10, 100, 1000]:
        models.append(('Ridge', {'alpha': alpha}, f'Ridge α={alpha}'))
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        models.append(('Lasso', {'alpha': alpha}, f'Lasso α={alpha}'))
    for alpha in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            models.append(('ElasticNet',
                           {'alpha': alpha, 'l1_ratio': l1_ratio},
                           f'EN α={alpha} l1={l1_ratio}'))
    for max_depth in [3, 4]:
        for n_est in [50, 100, 200]:
            models.append(('LightGBM',
                           {'max_depth': max_depth, 'n_estimators': n_est},
                           f'LGBM d={max_depth} n={n_est}'))

    total = len(models) * len(CONFIGS)
    print(f'\nTotal cells: {total} ({len(models)} models × {len(CONFIGS)} configs)', flush=True)

    results = []
    cell_idx = 0
    for cfg in CONFIGS:
        for model_type, params, mlabel in models:
            cell_idx += 1
            label = f'[{cfg["name"]}] {mlabel}'
            print(f'\n[{cell_idx}/{total}]', flush=True)
            try:
                res = run_cell(feat_3d, target_arr, mask_arr, dates, tickers, close,
                               spy_ret, model_type, params,
                               retrain_days=cfg['retrain_days'],
                               train_years=cfg['train_years'],
                               label=label)
                res['config'] = cfg['name']
                res['retrain_days'] = cfg['retrain_days']
                res['train_years'] = cfg['train_years']
                results.append(res)
                with open(OUT_DIR / 'phase_y_v3_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f'  ERROR: {e}', flush=True)
                import traceback
                traceback.print_exc()

    # Summary
    print('\n' + '=' * 90, flush=True)
    print('PHASE Y v2 SUMMARY', flush=True)
    print('=' * 90, flush=True)
    df = pd.DataFrame(results).sort_values('net_alpha', ascending=False)
    print(df[['label', 'gross_alpha', 'net_alpha', 'sharpe', 'mdd', 'ic_mean',
              'annual_turnover']].to_string(), flush=True)
    print('\nTop 5 by net α:', flush=True)
    print(df.head(5)[['label', 'gross_alpha', 'net_alpha', 'sharpe', 'ic_mean']].to_string(),
          flush=True)


if __name__ == '__main__':
    main()
