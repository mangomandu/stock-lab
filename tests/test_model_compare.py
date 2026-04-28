"""
Compare ML models: LightGBM vs XGBoost vs RandomForest vs Ridge vs MLP.

Same features (6 z-scored factors), same Top-20 Biweekly evaluation.
21-window walk-forward (2005-2025 to keep computation manageable).

Output: results/model_comparison.txt
"""
import core
import ml_model
import factors
import pandas as pd
import numpy as np
import os
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 5  # Weekly (new sweet spot)
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last().pct_change()


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def backtest_with_score(close, score_wide, top_n=TOP_N, rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    hp = {'top_n': top_n, 'rebal_days': rebal_days, 'hysteresis': 0,
          'cost_oneway': cost}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost


def make_features_target(close, vol, train_mask, test_mask, hp):
    return ml_model.get_train_test_features(close, vol, train_mask, test_mask, hp)


def train_predict(model_name, train_long, test_long, hp):
    feat_cols = hp['feature_names']
    X_train = train_long[feat_cols].values
    y_train = train_long['target'].values
    X_test = test_long[feat_cols].values

    if model_name == 'lightgbm':
        train_set = lgb.Dataset(X_train, label=y_train)
        # Use last 10% for val
        n = len(X_train)
        split = int(n * 0.9)
        val_set = lgb.Dataset(X_train[split:], label=y_train[split:], reference=train_set)
        train_set2 = lgb.Dataset(X_train[:split], label=y_train[:split])
        model = lgb.train(hp['lgb_params'], train_set2, num_boost_round=200,
                          valid_sets=[val_set], callbacks=[lgb.early_stopping(20, verbose=False),
                                                            lgb.log_evaluation(0)])
        preds = model.predict(X_test)

    elif model_name == 'xgboost':
        # Sort by date for time-aware split
        sorted_idx = train_long.sort_values('date').index
        X_train_sorted = X_train[train_long.index.get_indexer(sorted_idx)]
        y_train_sorted = y_train[train_long.index.get_indexer(sorted_idx)]
        n = len(X_train_sorted)
        split = int(n * 0.9)
        dtrain = xgb.DMatrix(X_train_sorted[:split], label=y_train_sorted[:split])
        dval = xgb.DMatrix(X_train_sorted[split:], label=y_train_sorted[split:])
        params = {'objective': 'reg:squarederror', 'eta': 0.05, 'max_depth': 6,
                  'subsample': 0.8, 'colsample_bytree': 0.9, 'verbosity': 0}
        model = xgb.train(params, dtrain, num_boost_round=200,
                          evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=0)
        preds = model.predict(xgb.DMatrix(X_test))

    elif model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=8,
                                       min_samples_leaf=50, n_jobs=4, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    elif model_name == 'ridge':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

    elif model_name == 'mlp':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=100,
                             random_state=42, early_stopping=True,
                             validation_fraction=0.1)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

    else:
        raise ValueError(model_name)

    return preds


def run_one_window(close, vol, test_year, hp, spy_ret, model_name):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid_tickers) < MIN_UNIVERSE_SIZE:
        return None

    close_sub = close[valid_tickers]
    vol_sub = vol[valid_tickers]

    train_long, test_long, _, _ = make_features_target(
        close_sub, vol_sub, train_mask, test_mask, hp)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    try:
        preds = train_predict(model_name, train_long, test_long, hp)
    except Exception as e:
        print(f"  [{test_year}] {model_name} ERROR: {e}")
        return None

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    port_ret = backtest_with_score(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {'cagr': s['cagr'], 'sharpe': s['sharpe'], 'excess': excess}


def evaluate_model(close, vol, hp, spy_ret, model_name):
    rows = []
    for test_year in range(2005, 2026):  # 21 windows for speed
        r = run_one_window(close, vol, test_year, hp, spy_ret, model_name)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] ML model comparison")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"Loaded {close.shape[1]} tickers\n")

    models = ['lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp']
    results = []
    for m in models:
        w(f"\n[{m}] running...")
        r = evaluate_model(close, vol, hp, spy_ret, m)
        if r:
            r['model'] = m
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"vs SPY {r['avg_excess']:+.2f}%p win {r['win']}/{r['n_windows']} t={r['t_stat']:.2f}")

    results.sort(key=lambda r: r['avg_sharpe'], reverse=True)

    w(f"\n{'='*100}")
    w(f"## Summary (sorted by Sharpe)")
    w(f"{'Model':<20} {'CAGR':>8} {'Sh':>5} {'vs SPY':>9} {'Win':>8} {'t':>6}")
    w("-" * 80)
    for r in results:
        w(f"{r['model']:<20} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_excess']:>+8.2f}%p {r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'model_comparison.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
