"""
Compute average Ridge coefficients across 31 walk-forward windows.

Output: results/avg_coefs.txt
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
TRAIN_YEARS = 7


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Average Ridge coefficients across 31 walk-forward windows")
    w("=" * 80)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    hp = dict(ml_model.ML_HP_DEFAULT)
    feat_cols = hp['feature_names']

    coefs_per_window = []
    years = []
    for test_year in range(1995, 2026):
        train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
        test_start = pd.Timestamp(f'{test_year}-01-01')
        train_mask = (close.index >= train_start) & (close.index < test_start)
        if train_mask.sum() < 252:
            continue

        valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
        if len(valid_tickers) < 100:
            continue

        close_sub = close[valid_tickers]
        vol_sub = vol[valid_tickers]

        feat_panels = ml_model.build_features_panel(close_sub, vol_sub)
        target = ml_model.make_target(close_sub, hp['forward_days'], hp['target_type'])
        train_feats = {n: df[train_mask] for n, df in feat_panels.items()}
        train_target = target[train_mask]
        train_long = ml_model.stack_panel_to_long(train_feats, train_target)
        if len(train_long) < 1000:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train_long[feat_cols].values)
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, train_long['target'].values)
        coefs_per_window.append(model.coef_)
        years.append(test_year)

    if not coefs_per_window:
        w("No data")
        return

    coefs = np.array(coefs_per_window)  # (n_windows, n_features)
    avg_coef = coefs.mean(axis=0)
    std_coef = coefs.std(axis=0)
    abs_coef = np.abs(coefs)
    avg_abs = abs_coef.mean(axis=0)

    # Sign consistency
    sign_pos = (coefs > 0).sum(axis=0)
    sign_neg = (coefs < 0).sum(axis=0)

    n = len(coefs_per_window)
    w(f"\n## {n} windows, average coefficients")
    w(f"{'Feature':<12} {'Mean':>10} {'Std':>10} {'AbsMean':>10} {'Pos':>5} {'Neg':>5} {'Importance %':>13}")
    w("-" * 75)
    total_abs = avg_abs.sum()
    sorted_idx = np.argsort(-avg_abs)
    for i in sorted_idx:
        f = feat_cols[i]
        importance = avg_abs[i] / total_abs * 100 if total_abs > 0 else 0
        w(f"{f:<12} {avg_coef[i]:>+10.4f} {std_coef[i]:>10.4f} {avg_abs[i]:>10.4f} "
          f"{sign_pos[i]:>5} {sign_neg[i]:>5} {importance:>11.1f}%")

    w(f"\n## Per-window coefficients (showing trend over time)")
    w(f"{'Year':<6} " + " ".join(f"{f:>10}" for f in feat_cols))
    for i, year in enumerate(years):
        w(f"{year:<6} " + " ".join(f"{coefs[i][j]:>+10.4f}" for j in range(len(feat_cols))))

    out_path = os.path.join(OUTPUT_DIR, 'avg_coefs.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
