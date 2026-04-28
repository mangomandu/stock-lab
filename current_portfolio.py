"""
Current portfolio recommendation — v4 (final tuning).

Best config (validated):
  - Universe: S&P 500 + ETF (518 tickers)
  - Model: Ridge regression
  - Train years: 7
  - Forward days: 10
  - Rebal: Weekly (rebal_days=5)
  - Top-N: 20 equal weight
"""
import core
import ml_model
import factors
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
SEED_USD = 400
TOP_N = 20
TRAIN_YEARS = 7


def main():
    print(f"[{datetime.now()}] ML Portfolio v4 (Ridge + 7y train + Weekly)")
    print(f"  Seed: ${SEED_USD} | Top-N: {TOP_N} | Equal weight | Weekly")
    print("=" * 80)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    hp = dict(ml_model.ML_HP_DEFAULT)

    latest_date = close.index.max()
    train_start = latest_date - pd.DateOffset(years=TRAIN_YEARS)
    train_mask = (close.index >= train_start) & (close.index < latest_date)
    score_mask = close.index == latest_date

    valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    close_sub = close[valid_tickers]
    vol_sub = vol[valid_tickers]

    print(f"\nTrain: {train_start.date()} ~ {latest_date.date()} ({TRAIN_YEARS} years)")
    print(f"Valid universe: {len(valid_tickers)} tickers\n")

    feat_panels = ml_model.build_features_panel(close_sub, vol_sub)
    target = ml_model.make_target(close_sub, hp['forward_days'], hp['target_type'])

    train_feats = {n: df[train_mask] for n, df in feat_panels.items()}
    train_target = target[train_mask]
    train_long = ml_model.stack_panel_to_long(train_feats, train_target)

    score_feats = {n: df[score_mask] for n, df in feat_panels.items()}
    score_long = ml_model.stack_panel_to_long(score_feats)

    feat_cols = hp['feature_names']
    X_train = train_long[feat_cols].values
    y_train = train_long['target'].values
    X_score = score_long[feat_cols].values

    print("Training Ridge regression...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_score_s = scaler.transform(X_score)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    print(f"  Coefficients (per feature):")
    for f, c in zip(feat_cols, model.coef_):
        print(f"    {f:<12} {c:>+.4f}")

    preds = model.predict(X_score_s)
    score_long_results = score_long.copy()
    score_long_results['score'] = preds
    score_long_results = score_long_results.sort_values('score', ascending=False).reset_index(drop=True)

    latest_close = close_sub.loc[latest_date]
    top = score_long_results.head(TOP_N).copy()
    top['price'] = top['ticker'].map(latest_close)
    per_stock = SEED_USD / TOP_N
    top['target_usd'] = per_stock
    top['shares'] = top['target_usd'] / top['price']

    print(f"\n{'='*80}")
    print(f"★ Top-{TOP_N} Portfolio (v4 — Ridge, Equal weight, ${per_stock:.2f}/stock)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Ticker':<8} {'Score':>8} {'Price':>10} {'Shares':>10} {'USD':>9}")
    print("-" * 60)
    for i, row in top.iterrows():
        print(f"{i+1:<5} {row['ticker']:<8} {row['score']:>8.3f} "
              f"${row['price']:>8.2f} {row['shares']:>10.6f} ${row['target_usd']:>7.2f}")
    print("-" * 60)
    print(f"{'Total':<24} {'':<10} {'':<10} ${top['target_usd'].sum():>8.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'current_portfolio.csv')
    top[['ticker','score','price','target_usd','shares']].to_csv(out_path, index=False)
    print(f"\n매수 리스트 CSV: {out_path}")


if __name__ == '__main__':
    main()
