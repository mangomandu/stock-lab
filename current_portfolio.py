"""
Current portfolio recommendation — v5.2 (3-feature minimum model + risk profiles).

Run weekly to get Top-N stocks for the week.

2 Profiles (model 비중):
  - 'standard'   : 100% model         → 알파 추구, 현재 라이브
  - 'low_risk'   : 60% model + 40% TLT → 안정 추구, MDD 절반

같은 model 내 변형 (TOP_N + SECTOR_CAP) — 검증된 옵션:
  - Top-20 No cap   (default, 단순)         Sharpe 1.79, alpha +34.6%p
  - Top-15 Cap 20%  (Sharpe peak)            Sharpe 1.84, alpha +37.7%p
  - Top-10 Cap 30%  (최대 알파)              Sharpe 1.78, alpha +45.2%p
  - Top-20 Cap 15%  (낮은 MDD)               Sharpe 1.82, alpha +31.7%p, MDD -19.7%

자세한 매트릭스 + 검증 결과 README 참조.
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIG
# =============================================================================
# [1] Profile — model 비중 (위험도)
PROFILE      = 'standard'   # 'standard' (100% model) / 'low_risk' (60% model + 40% TLT)

# [2] Parameters — 같은 model 내 변형 (개별 조정 가능)
TOP_N        = 20           # 10 / 15 / 20 (검증된 매트릭스 README 참조)
SECTOR_CAP   = None         # None / 0.15 / 0.20 / 0.25 / 0.30
HYST_EXIT    = 50           # exit_n: TOP_N=20 + exit_n=50 → 50등 밖되면 매도 (회전율 ↓ + alpha ↑)
                            # 옵션: 20 (no hyst), 25, 30, 40, 50 (best)

# [3] Common
SEED_USD     = 400          # your seed in USD
TRAIN_YEARS  = 7            # 7 is sweet spot (validated)
FEATURES     = 'minimum'    # 'minimum' (3-feat v5 best) or 'full' (6-feat v4 baseline)
# =============================================================================

PROFILES = {
    'standard': {'tlt_buffer': 0.0,
                 'desc': '100% model (알파 추구, 라이브)'},
    'low_risk': {'tlt_buffer': 0.40,
                 'desc': '60% model + 40% TLT (안정 추구, MDD 절반)'},
}

FEATURE_SETS = {
    'minimum': ['lowvol', 'rsi', 'volsurge'],          # v5 best (Sharpe 1.77)
    'full':    ['momentum', 'lowvol', 'trend',          # v4 baseline
                'rsi', 'ma', 'volsurge'],
}

# Resolve profile
if PROFILE not in PROFILES:
    raise ValueError(f"Unknown profile: {PROFILE}. Choose from {list(PROFILES.keys())}")
TLT_BUFFER = PROFILES[PROFILE]['tlt_buffer']

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
SECTORS_PATH = '/home/dlfnek/stock_lab/data/sectors.csv'


def load_sectors():
    if not os.path.exists(SECTORS_PATH):
        return None
    df = pd.read_csv(SECTORS_PATH, index_col='Ticker')
    return df['Sector'].to_dict()


def topn_with_sector_cap(score_series, sectors, top_n, sector_cap, prev_held=None, hyst_exit=None):
    """Greedy: take highest scored, skip if sector cap violated.

    Hysteresis: prev_held 종목이 hyst_exit등 안에 있으면 우선 유지 (회전율 ↓).
    """
    sorted_scores = score_series.dropna().sort_values(ascending=False)
    ranks = {ticker: i for i, ticker in enumerate(sorted_scores.index, 1)}
    held = []
    sector_count = defaultdict(int)
    max_per_sector = max(1, int(top_n * sector_cap)) if sector_cap else top_n

    # Hysteresis: keep prev_held if still ranked within hyst_exit
    if prev_held and hyst_exit and hyst_exit > top_n:
        for h in prev_held:
            if h in ranks and ranks[h] <= hyst_exit and len(held) < top_n:
                sec = sectors.get(h, 'Unknown') if sectors else 'Unknown'
                if sector_cap is None or sector_count[sec] < max_per_sector:
                    held.append(h)
                    sector_count[sec] += 1

    # Fill up with new top entries
    for ticker in sorted_scores.index:
        if len(held) >= top_n:
            break
        if ticker in held:
            continue
        sec = sectors.get(ticker, 'Unknown') if sectors else 'Unknown'
        if sector_cap and sector_count[sec] >= max_per_sector:
            continue
        held.append(ticker)
        sector_count[sec] += 1

    return held


def main():
    cap_label = "No cap" if SECTOR_CAP is None else f"Cap {int(SECTOR_CAP*100)}%"
    feat_list = FEATURE_SETS[FEATURES]
    profile_desc = PROFILES[PROFILE]['desc']
    print(f"[{datetime.now()}] ML Portfolio v5.2 — Ridge + 7y + Weekly")
    print(f"  Profile: {PROFILE} ({profile_desc})")
    print(f"  Seed: ${SEED_USD} | Top-{TOP_N} | {cap_label} | Features: {FEATURES} ({len(feat_list)})")
    if TLT_BUFFER > 0:
        model_usd = SEED_USD * (1 - TLT_BUFFER)
        tlt_usd = SEED_USD * TLT_BUFFER
        print(f"  Buffer: model ${model_usd:.0f} ({(1-TLT_BUFFER)*100:.0f}%) + TLT ${tlt_usd:.0f} ({TLT_BUFFER*100:.0f}%)")
    print("=" * 80)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    sectors = load_sectors()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = feat_list

    # Load previous portfolio for hysteresis (if exists)
    prev_held = []
    prev_csv = os.path.join(OUTPUT_DIR, 'current_portfolio.csv')
    if HYST_EXIT > TOP_N and os.path.exists(prev_csv):
        try:
            prev_df = pd.read_csv(prev_csv)
            prev_held = [t for t in prev_df['ticker'].tolist() if t != 'TLT']
            print(f"  Hysteresis: prev portfolio loaded ({len(prev_held)} stocks), exit_n={HYST_EXIT}")
        except Exception as e:
            print(f"  Hysteresis: prev portfolio load failed ({e}), 처음 실행 또는 fresh start")

    latest_date = close.index.max()
    train_start = latest_date - pd.DateOffset(years=TRAIN_YEARS)
    train_mask = (close.index >= train_start) & (close.index < latest_date)
    score_mask = close.index == latest_date

    valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    close_sub = close[valid_tickers]
    vol_sub = vol[valid_tickers]

    print(f"\nTrain: {train_start.date()} ~ {latest_date.date()} ({TRAIN_YEARS} years)")
    print(f"Universe: {len(valid_tickers)} tickers\n")

    # Train Ridge
    feat_panels = ml_model.build_features_panel(close_sub, vol_sub)
    target = ml_model.make_target(close_sub, hp['forward_days'], hp['target_type'])
    train_feats = {n: df[train_mask] for n, df in feat_panels.items()}
    train_target = target[train_mask]
    train_long = ml_model.stack_panel_to_long(train_feats, train_target)

    score_feats = {n: df[score_mask] for n, df in feat_panels.items()}
    score_long = ml_model.stack_panel_to_long(score_feats)

    feat_cols = hp['feature_names']
    # Drop rows with NaN in selected features
    train_long = train_long.dropna(subset=feat_cols)
    score_long = score_long.dropna(subset=feat_cols)
    print(f"Training Ridge regression on {len(train_long):,} rows × {len(feat_cols)} features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_long[feat_cols].values)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, train_long['target'].values)
    print(f"  Coefficients:")
    for f, c in zip(feat_cols, model.coef_):
        print(f"    {f:<12} {c:>+.4f}")

    X_score_s = scaler.transform(score_long[feat_cols].values)
    preds = model.predict(X_score_s)
    score_long['score'] = preds
    score_series = score_long.set_index('ticker')['score']

    # Apply Top-N + sector cap + hysteresis
    held_tickers = topn_with_sector_cap(score_series, sectors, TOP_N, SECTOR_CAP,
                                          prev_held=prev_held, hyst_exit=HYST_EXIT)
    if prev_held:
        new_buys = [t for t in held_tickers if t not in prev_held]
        sells = [t for t in prev_held if t not in held_tickers]
        print(f"\n  Changes from prev: +{len(new_buys)} buys ({new_buys[:5]}{'...' if len(new_buys)>5 else ''}), "
              f"-{len(sells)} sells ({sells[:5]}{'...' if len(sells)>5 else ''})")

    latest_close = close_sub.loc[latest_date]
    # Allocate model portion of seed (1 - TLT_BUFFER) to Top-N stocks equally
    model_seed = SEED_USD * (1 - TLT_BUFFER)
    tlt_seed = SEED_USD * TLT_BUFFER
    per_stock = model_seed / TOP_N

    rows = []
    for i, ticker in enumerate(held_tickers, 1):
        rows.append({
            'rank': i,
            'ticker': ticker,
            'score': float(score_series[ticker]),
            'price': float(latest_close[ticker]),
            'sector': sectors.get(ticker, 'Unknown') if sectors else '-',
            'target_usd': per_stock,
            'shares': per_stock / float(latest_close[ticker]),
        })
    portfolio = pd.DataFrame(rows)

    # Print
    print(f"\n{'='*80}")
    print(f"★ Top-{TOP_N} Portfolio ({cap_label}, ${per_stock:.2f}/stock)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Ticker':<8} {'Score':>7} {'Price':>10} {'Shares':>10} {'USD':>8} {'Sector':<22}")
    print("-" * 90)
    for _, r in portfolio.iterrows():
        print(f"{r['rank']:<5} {r['ticker']:<8} {r['score']:>7.3f} "
              f"${r['price']:>8.2f} {r['shares']:>10.6f} ${r['target_usd']:>6.2f} "
              f"{r['sector']:<22}")
    print("-" * 90)
    print(f"{'Model total':<24} {'':<10} {'':<10} ${portfolio['target_usd'].sum():>7.2f}")
    if TLT_BUFFER > 0:
        # Compute TLT shares
        tlt_path = os.path.join(DATA_DIR, 'TLT.csv')
        if os.path.exists(tlt_path):
            tlt_df = pd.read_csv(tlt_path)
            tlt_df['Datetime'] = pd.to_datetime(tlt_df['Datetime'], errors='coerce')
            tlt_df = tlt_df.dropna(subset=['Datetime']).sort_values('Datetime')
            tlt_price = float(tlt_df['Close'].iloc[-1])
            tlt_shares = tlt_seed / tlt_price
            print(f"{'TLT (buffer)':<24} {'-':<10} ${tlt_price:>8.2f} {tlt_shares:>10.6f} ${tlt_seed:>6.2f} {'Bonds':<22}")
            # Add TLT to portfolio CSV
            portfolio = pd.concat([portfolio, pd.DataFrame([{
                'rank': len(portfolio) + 1, 'ticker': 'TLT',
                'score': float('nan'), 'price': tlt_price,
                'sector': 'Bonds', 'target_usd': tlt_seed,
                'shares': tlt_shares,
            }])], ignore_index=True)
            print(f"{'Grand total':<24} {'':<10} {'':<10} ${portfolio['target_usd'].sum():>7.2f}")

    if SECTOR_CAP and sectors:
        print(f"\n## Sector breakdown (Cap {int(SECTOR_CAP*100)}%, max {max(1, int(TOP_N*SECTOR_CAP))}/sector)")
        sec_count = portfolio['sector'].value_counts()
        for sec, cnt in sec_count.items():
            print(f"  {sec:<25} {cnt}")

    out_path = os.path.join(OUTPUT_DIR, 'current_portfolio.csv')
    portfolio.to_csv(out_path, index=False)
    print(f"\n매수 리스트 CSV: {out_path}")


if __name__ == '__main__':
    main()
