"""
자양동 Stock Lab — LightGBM ML model.

Predicts forward 10-day return for each (ticker, date) given factor features.
At inference time, score all stocks on a given date, take Top-N predicted.
Biweekly rebalancing (10 trading days, matching the prediction horizon).

Features used: momentum, lowvol, trend, rsi, ma, volsurge (all from factors.py).
Target: forward 10-day cross-sectional rank (or raw return — configurable).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import factors

ML_HP_DEFAULT = {
    'forward_days':       10,         # match biweekly rebalance
    'feature_names':      ['momentum', 'lowvol', 'trend', 'rsi', 'ma', 'volsurge'],
    'target_type':        'rank',     # 'rank' or 'return'
    'lgb_params': {
        'objective':       'regression',
        'metric':          'rmse',
        'learning_rate':   0.05,
        'num_leaves':      31,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq':    5,
        'verbose':         -1,
        'seed':            42,
    },
    'num_rounds':         200,
    'early_stopping':     20,
}


def stack_panel_to_long(panel_dict, target=None):
    """Convert {feature: wide_df} into long-format DataFrame.

    Returns DataFrame with columns: date, ticker, <features>, [target].
    Drops rows where any feature or target is NaN.
    """
    frames = []
    for name, df in panel_dict.items():
        long = df.stack(future_stack=True).rename(name)
        frames.append(long)
    if target is not None:
        target_long = target.stack(future_stack=True).rename('target')
        frames.append(target_long)
    out = pd.concat(frames, axis=1).dropna()
    out.index.names = ['date', 'ticker']
    return out.reset_index()


def make_target(close, forward_days, target_type='rank'):
    """Compute forward target for each (date, ticker).

    target_type='return': raw forward N-day return
    target_type='rank':   cross-sectional rank of forward return (uniform [0,1])
    """
    forward_ret = close.shift(-forward_days) / close - 1
    if target_type == 'return':
        return forward_ret
    elif target_type == 'rank':
        # Per-row rank, normalized to [0, 1]
        ranks = forward_ret.rank(axis=1, pct=True)
        return ranks
    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def train_model(features_long, hp):
    """Train LightGBM on a long-format DataFrame with features and target.

    Splits last 10% as validation for early stopping.
    """
    feat_cols = hp['feature_names']
    n = len(features_long)
    if n < 1000:
        return None

    # Time-based train/val split (last 10% as val)
    sorted_df = features_long.sort_values('date').reset_index(drop=True)
    split = int(n * 0.9)
    train_df = sorted_df.iloc[:split]
    val_df = sorted_df.iloc[split:]

    train_set = lgb.Dataset(train_df[feat_cols].values, label=train_df['target'].values)
    val_set = lgb.Dataset(val_df[feat_cols].values, label=val_df['target'].values,
                          reference=train_set)

    model = lgb.train(
        hp['lgb_params'],
        train_set,
        num_boost_round=hp['num_rounds'],
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(hp['early_stopping'], verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


def score_with_model(model, features_long, hp):
    """Predict on long-format DataFrame.

    Returns Series indexed by (date, ticker) of predictions.
    """
    feat_cols = hp['feature_names']
    preds = model.predict(features_long[feat_cols].values)
    out = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [features_long['date'], features_long['ticker']]))
    return out


def long_to_wide(score_long, all_dates, all_tickers):
    """Pivot long-format scores back to wide DataFrame (date × ticker)."""
    df = score_long.reset_index()
    df.columns = ['date', 'ticker', 'score']
    wide = df.pivot(index='date', columns='ticker', values='score')
    wide = wide.reindex(index=all_dates, columns=all_tickers)
    return wide


def build_features_panel(close, vol, include_multi_horizon=False):
    """Compute feature panels (z-scored where applicable).

    Default: 6 features (momentum, lowvol, trend, rsi, ma, volsurge).
    If include_multi_horizon: + momentum_1m, momentum_3m, momentum_6m (9 total).
    """
    return factors.compute_zscored_factors(
        close, vol, include_multi_horizon=include_multi_horizon)


def get_train_test_features(close, vol, train_mask, test_mask, hp):
    """Build train/test long-format DataFrames.

    Train: features + target on train_mask dates (with target leakage buffer).
    Test:  features only (no target needed) on test_mask dates.

    If hp['include_multi_horizon'] is True, uses 9 features.

    Target leakage 방지:
      Train target = price[t + forward_days] / price[t] - 1
      만약 t가 train 끝이면 t+forward_days가 test로 침범 → leakage.
      따라서 train의 마지막 forward_days만큼 drop.
    """
    include_mh = hp.get('include_multi_horizon', False)
    feat_panels = build_features_panel(close, vol, include_multi_horizon=include_mh)
    target = make_target(close, hp['forward_days'], hp['target_type'])

    # Buffer: train mask에서 마지막 forward_days만큼 drop
    forward_days = hp['forward_days']
    train_dates = close.index[train_mask]
    if len(train_dates) > forward_days:
        effective_train_end = train_dates[-forward_days]
        train_mask_buffered = train_mask & (close.index < effective_train_end)
    else:
        train_mask_buffered = train_mask

    # Restrict each to its mask
    train_feats = {n: df[train_mask_buffered] for n, df in feat_panels.items()}
    train_target = target[train_mask_buffered]

    test_feats = {n: df[test_mask] for n, df in feat_panels.items()}

    train_long = stack_panel_to_long(train_feats, train_target)
    test_long = stack_panel_to_long(test_feats)
    return train_long, test_long, feat_panels, target
