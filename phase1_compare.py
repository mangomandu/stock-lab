"""
Phase 1: v0.7.1' (4 features) vs v0.8.0 (12 features) minimum comparison.

Same Sharadar SP500 historical data, same Ridge α=1.0, same HP.
Difference: features 4 vs 12.

Decision gate:
  Δ Rank IC > +0.005 OR Δ alpha > +1%p → Phase 2 진행
  미달 → 12 features 무의미.
"""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))

PANEL_DIR = Path('/home/dlfnek/stock_lab/data/panels')
SHARADAR_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/results')
OUT_DIR.mkdir(exist_ok=True)

# ── Lock된 HP ──
TRAIN_YEARS = 10
RETRAIN_FREQ = 252
TOP_N = 10
HYST_EXIT = 40
RIDGE_ALPHA = 1.0
TARGET_HORIZONS = [5, 10, 20]
FRESHNESS_DAYS = 90
COST_ONEWAY = 0.0005
START_DATE = pd.Timestamp('1998-01-01')

PRICE_FEATURES = ['lowvol', 'rsi', 'volsurge', 'momentum']
FUND_FEATURES = ['pe', 'pb', 'roe_ttm', 'roa_ttm',
                 'netmargin', 'de', 'eps_growth_yoy', 'rev_growth_yoy']


# ─────────────────────────────────────────────────────────────────────
# Price features (vectorized)
# ─────────────────────────────────────────────────────────────────────
def feat_lowvol(close, window=252):
    return -close.pct_change().rolling(window).std()


def feat_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    return ((70 - rsi.fillna(50)) / 40 * 100).clip(0, 100)


def feat_volsurge(close, vol, period=20):
    vma = vol.rolling(period).mean()
    vol_ratio = (vol / vma.replace(0, 1e-9)) * 100
    return ((vol_ratio - 50) / 150 * 100).clip(0, 100)


def feat_momentum(close, long_d=252, skip_d=21):
    return close.shift(skip_d) / close.shift(long_d) - 1


def build_price_features(close, vol):
    return {
        'lowvol':   feat_lowvol(close),
        'rsi':      feat_rsi(close),
        'volsurge': feat_volsurge(close, vol),
        'momentum': feat_momentum(close),
    }


# ─────────────────────────────────────────────────────────────────────
# Fundamental long → wide panels with freshness check (per-ticker merge_asof)
# ─────────────────────────────────────────────────────────────────────
def build_fundamental_wide(fund_df, all_dates, all_tickers, freshness_days=90):
    """For each ticker × date, pick latest fundamental row with freshness ≤ N days."""
    print(f'  Building wide fund panels (freshness ≤ {freshness_days}d)...', flush=True)
    out = {f: pd.DataFrame(np.nan, index=all_dates, columns=all_tickers, dtype=float)
           for f in FUND_FEATURES}

    fund_df = fund_df.sort_values(['ticker', 'available_date'])
    common = sorted(set(fund_df['ticker'].unique()) & set(all_tickers))
    dates_df = pd.DataFrame({'date': all_dates}).sort_values('date').reset_index(drop=True)

    for i, ticker in enumerate(common):
        if i % 200 == 0:
            print(f'    {i}/{len(common)} tickers...', flush=True)
        sub = fund_df[fund_df['ticker'] == ticker][['available_date'] + FUND_FEATURES].copy()
        sub = sub.sort_values('available_date').rename(columns={'available_date': 'avail'})
        sub['avail_match'] = sub['avail']
        merged = pd.merge_asof(
            dates_df, sub,
            left_on='date', right_on='avail',
            direction='backward',
        )
        freshness = (merged['date'] - merged['avail_match']).dt.days
        ok = freshness <= freshness_days
        for f in FUND_FEATURES:
            vals = merged[f].values.copy()
            vals[~ok.values] = np.nan
            out[f][ticker] = vals
    return out


# ─────────────────────────────────────────────────────────────────────
# Cross-sectional Gauss-rank (fast — scipy rankdata)
# ─────────────────────────────────────────────────────────────────────
def gauss_rank_panel(panel_df):
    arr = panel_df.values.astype(float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        row = arr[i]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        ranks = rankdata(row[valid], method='average')
        n = valid.sum()
        uniform = np.clip((ranks - 0.5) / n, 0.001, 0.999)
        out[i, valid] = norm.ppf(uniform)
    return pd.DataFrame(out, index=panel_df.index, columns=panel_df.columns)


# ─────────────────────────────────────────────────────────────────────
# Multi-horizon rank target
# ─────────────────────────────────────────────────────────────────────
def make_multi_horizon_target(close, horizons=TARGET_HORIZONS):
    rank_dfs = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1
        rank_dfs.append(fwd.rank(axis=1, pct=True))
    return sum(rank_dfs) / len(rank_dfs)


# ─────────────────────────────────────────────────────────────────────
# SP500 membership mask (date × ticker → bool)
# ─────────────────────────────────────────────────────────────────────
def build_sp500_mask(close, sp500_changes, all_tickers):
    print('  Building SP500 membership mask...', flush=True)
    snaps = sp500_changes[sp500_changes['action'] == 'historical'].copy()
    snaps['date'] = pd.to_datetime(snaps['date'])
    snap_dates = sorted(snaps['date'].unique())
    snap_lookup = {d: set(g['ticker']) for d, g in snaps.groupby('date')}

    mask = pd.DataFrame(False, index=close.index, columns=all_tickers)
    cur_set = set()
    cur_idx = 0
    for d in close.index:
        while cur_idx < len(snap_dates) and snap_dates[cur_idx] <= d:
            cur_set = snap_lookup[snap_dates[cur_idx]]
            cur_idx += 1
        members = list(cur_set & set(all_tickers))
        if members:
            mask.loc[d, members] = True
    return mask


# ─────────────────────────────────────────────────────────────────────
# Walk-forward backtest (vectorized predict)
# ─────────────────────────────────────────────────────────────────────
def stack_long_for_train(norm_feats, target, sp500_mask, dates):
    """Numpy-based stacking — memory efficient (no MultiIndex)."""
    feat_names = list(norm_feats.keys())
    n_feats = len(feat_names)

    # Get values as float32 arrays
    target_arr = target.loc[dates].values.astype(np.float32)
    mask_arr = sp500_mask.loc[dates].values  # bool

    # Stack features (n_feats, n_dates, n_tickers)
    n_dates, n_tickers = target_arr.shape
    feat_arr = np.empty((n_feats, n_dates, n_tickers), dtype=np.float32)
    for i, n in enumerate(feat_names):
        feat_arr[i] = norm_feats[n].loc[dates].values.astype(np.float32)

    # Validity: SP500 member + all features valid + target valid
    feat_valid = ~np.any(np.isnan(feat_arr), axis=0)  # (n_dates, n_tickers)
    target_valid = ~np.isnan(target_arr)
    valid = mask_arr & feat_valid & target_valid

    n_valid = int(valid.sum())
    X = np.empty((n_valid, n_feats), dtype=np.float32)
    for i in range(n_feats):
        X[:, i] = feat_arr[i][valid]
    y = target_arr[valid]
    return X, y, None


def predict_panel_at(t, norm_feats, feat_cols, sp500_mask_row, model):
    """Vectorized: build feature matrix at date t, predict scores."""
    members = sp500_mask_row[sp500_mask_row].index
    if len(members) == 0:
        return pd.Series(dtype=float)
    mat = pd.concat([norm_feats[fn].loc[t, members] for fn in feat_cols], axis=1)
    mat.columns = feat_cols
    mat = mat.dropna()
    if len(mat) == 0:
        return pd.Series(dtype=float)
    scores = model.predict(mat.values)
    return pd.Series(scores, index=mat.index)


def run_backtest(close, vol, fund_panels, feature_names, label, sp500_mask):
    print(f'\n=== Run: {label} ({len(feature_names)} features) ===', flush=True)

    raw_feats = {}
    if any(f in feature_names for f in PRICE_FEATURES):
        raw_feats.update(build_price_features(close, vol))
    raw_feats.update(fund_panels)
    selected = {n: raw_feats[n] for n in feature_names}

    print(f'  Gauss-rank normalize {len(selected)} features...', flush=True)
    norm_feats = {}
    for n, df in selected.items():
        norm_feats[n] = gauss_rank_panel(df)
        print(f'    {n} done', flush=True)

    print('  Build multi-horizon target...', flush=True)
    target = make_multi_horizon_target(close)

    dates = close.index[close.index >= START_DATE]
    train_idx_min = TRAIN_YEARS * 252

    score_panel = pd.DataFrame(np.nan, index=dates, columns=close.columns)

    last_train_idx = -RETRAIN_FREQ
    model = None
    feat_cols = feature_names

    for i, t in enumerate(dates):
        if i < train_idx_min:
            continue
        # Retrain
        if model is None or (i - last_train_idx) >= RETRAIN_FREQ:
            train_dates = dates[max(0, i - TRAIN_YEARS * 252):i]
            buf = max(TARGET_HORIZONS)
            train_dates_safe = train_dates[:-buf] if len(train_dates) > buf else train_dates
            X, y, _ = stack_long_for_train(norm_feats, target, sp500_mask, train_dates_safe)
            if len(X) < 1000:
                continue
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X, y)
            last_train_idx = i
            print(f'  {t.date()}: retrained on {len(X):,} rows  '
                  f'coef=[{",".join(f"{c:+.3f}" for c in model.coef_)}]', flush=True)

        # Vectorized predict
        scores = predict_panel_at(t, norm_feats, feat_cols, sp500_mask.loc[t], model)
        if len(scores) > 0:
            score_panel.loc[t, scores.index] = scores.values

    return score_panel


def load_spy_returns(start, end):
    """SPY ETF 가격 — SHARADAR Equities Bundle에 ETF는 미포함.
    yfinance 캐시 (master_sp500/SPY.csv) 사용.
    """
    spy_path = Path('/home/dlfnek/stock_lab/data/master_sp500/SPY.csv')
    spy = pd.read_csv(spy_path)
    spy['Datetime'] = pd.to_datetime(spy['Datetime']).dt.tz_localize(None)
    spy = spy.set_index('Datetime').sort_index()
    spy = spy.loc[start:end]['Close']
    return spy.pct_change()


def evaluate_score_panel(score_panel, close, label, spy_returns=None):
    print(f'\n  Evaluating {label}...', flush=True)

    fwd_21 = close.shift(-21) / close - 1
    ic_per_day = []
    for d in score_panel.dropna(how='all').index:
        s = score_panel.loc[d].dropna()
        r = fwd_21.loc[d].reindex(s.index).dropna()
        common = s.index.intersection(r.index)
        if len(common) < 20:
            continue
        ic, _ = spearmanr(s[common].values, r[common].values)
        if not np.isnan(ic):
            ic_per_day.append((d, ic))
    ic_series = pd.Series(dict(ic_per_day))

    daily_ret = close.pct_change()
    held = []
    port_returns = []
    port_dates = []
    valid_dates = score_panel.dropna(how='all').index
    for d in valid_dates[:-1]:
        scores = score_panel.loc[d].dropna()
        if len(scores) < TOP_N:
            continue
        ranked = scores.sort_values(ascending=False)
        ranks = {t: i for i, t in enumerate(ranked.index, 1)}

        new_held = []
        for h in held:
            if h in ranks and ranks[h] <= HYST_EXIT and len(new_held) < TOP_N:
                new_held.append(h)
        for tk in ranked.index:
            if len(new_held) >= TOP_N:
                break
            if tk in new_held:
                continue
            new_held.append(tk)

        next_d_idx = close.index.get_indexer([d])[0] + 1
        if next_d_idx >= len(close.index):
            break
        next_d = close.index[next_d_idx]
        rets = [daily_ret.at[next_d, tk] for tk in new_held if tk in daily_ret.columns]
        rets = [r for r in rets if not pd.isna(r)]
        if not rets:
            continue
        port_ret = np.mean(rets)
        n_changed = len([t for t in new_held if t not in held])
        cost = (n_changed / TOP_N) * COST_ONEWAY * 2
        port_ret -= cost
        port_returns.append(port_ret)
        port_dates.append(next_d)
        held = new_held

    port = pd.Series(port_returns, index=port_dates)
    if spy_returns is not None:
        spy_ret = spy_returns.reindex(port.index).fillna(0)
        cum_spy = (1 + spy_ret).cumprod().iloc[-1]
        n_years = (port.index[-1] - port.index[0]).days / 365.25
        cagr_spy = cum_spy ** (1 / n_years) - 1
    else:
        cagr_spy = 0

    cum_port = (1 + port).cumprod().iloc[-1]
    n_years = (port.index[-1] - port.index[0]).days / 365.25
    cagr_port = cum_port ** (1 / n_years) - 1
    alpha = cagr_port - cagr_spy
    sharpe = port.mean() / port.std() * np.sqrt(252) if port.std() > 0 else 0

    return {
        'label': label,
        'n_days': len(port),
        'cagr_port': float(cagr_port),
        'cagr_spy': float(cagr_spy),
        'alpha': float(alpha),
        'sharpe': float(sharpe),
        'ic_mean': float(ic_series.mean()),
        'ic_std': float(ic_series.std()),
        'ic_ir': float(ic_series.mean() / ic_series.std() * np.sqrt(252)) if ic_series.std() > 0 else 0,
        'ic_n': int(len(ic_series)),
    }


def main():
    print('=== Phase 1: 4 vs 12 features comparison ===', flush=True)

    print('Loading panels...', flush=True)
    close = pd.read_parquet(PANEL_DIR / 'sp500_close.parquet')
    vol = pd.read_parquet(PANEL_DIR / 'sp500_volume.parquet')
    fund_long = pd.read_parquet(PANEL_DIR / 'fundamental_pit.parquet')
    sp500_changes = pd.read_parquet(SHARADAR_DIR / 'SP500.parquet')

    print(f'Close: {close.shape}, Volume: {vol.shape}', flush=True)
    print(f'Fundamental long: {len(fund_long):,} rows', flush=True)

    all_tickers = list(close.columns)
    sp500_mask = build_sp500_mask(close, sp500_changes, all_tickers)
    print(f'Mask True: {sp500_mask.values.sum():,}', flush=True)

    fund_panels = build_fundamental_wide(fund_long, close.index, all_tickers,
                                         freshness_days=FRESHNESS_DAYS)
    del fund_long
    gc.collect()

    spy_ret = load_spy_returns(close.index.min(), close.index.max())
    print(f'  SPY returns: {len(spy_ret)} days', flush=True)

    score_4f = run_backtest(close, vol, fund_panels, PRICE_FEATURES,
                            'v0.7.1\' (4 features)', sp500_mask)
    res_4f = evaluate_score_panel(score_4f, close, 'v0.7.1\' (4f)', spy_returns=spy_ret)
    print(json.dumps(res_4f, indent=2), flush=True)

    score_12f = run_backtest(close, vol, fund_panels,
                             PRICE_FEATURES + FUND_FEATURES,
                             'v0.8.0 (12 features)', sp500_mask)
    res_12f = evaluate_score_panel(score_12f, close, 'v0.8.0 (12f)', spy_returns=spy_ret)
    print(json.dumps(res_12f, indent=2), flush=True)

    delta_alpha = res_12f['alpha'] - res_4f['alpha']
    delta_ic = res_12f['ic_mean'] - res_4f['ic_mean']
    delta_sharpe = res_12f['sharpe'] - res_4f['sharpe']

    print('\n' + '=' * 60, flush=True)
    print('PHASE 1 COMPARISON', flush=True)
    print('=' * 60, flush=True)
    print(f'                    4f         12f        Δ', flush=True)
    print(f'  α vs SPY:      {res_4f["alpha"]*100:+6.2f}%p   '
          f'{res_12f["alpha"]*100:+6.2f}%p   {delta_alpha*100:+6.2f}%p', flush=True)
    print(f'  Sharpe:        {res_4f["sharpe"]:6.2f}      '
          f'{res_12f["sharpe"]:6.2f}      {delta_sharpe:+6.2f}', flush=True)
    print(f'  Rank IC:       {res_4f["ic_mean"]:+6.4f}    '
          f'{res_12f["ic_mean"]:+6.4f}    {delta_ic:+6.4f}', flush=True)

    pass_alpha = delta_alpha > 0.01
    pass_ic = delta_ic > 0.005
    print('\n  Decision gate (Δα>+1%p OR ΔIC>+0.005):', flush=True)
    if pass_alpha or pass_ic:
        print(f'  ✓ PASS — Phase 2 진행', flush=True)
    else:
        print(f'  ✗ FAIL — 12 features 의미 X', flush=True)

    out = {'4f': res_4f, '12f': res_12f,
           'delta_alpha': delta_alpha, 'delta_ic': delta_ic, 'delta_sharpe': delta_sharpe,
           'pass': bool(pass_alpha or pass_ic)}
    with open(OUT_DIR / 'phase1_comparison.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {OUT_DIR / "phase1_comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
