"""
Add market regime features to ML model.

New features:
  - vix_level: VIX index value (^VIX from yfinance)
  - spy_vol_20: SPY 20-day rolling daily-return std
  - spy_drawdown: SPY current / cummax - 1

These are time-only features (same value for all tickers on a given date).
Model learns regime-dependent behavior.

Walk-forward 31 windows on v4 (Ridge + 7y + Weekly + Top-20).

Output: results/regime_features_walkforward.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import ml_model
import factors
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yfinance as yf

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 5
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last()


def fetch_vix():
    """Fetch ^VIX historical data from yfinance."""
    vix = yf.Ticker('^VIX').history(period='max', interval='1d').reset_index()
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None).dt.normalize()
    return vix.set_index('Date')['Close']


def build_regime_features(close_index, spy_close, vix):
    """Build per-date regime features.

    Returns DataFrame indexed by date with columns:
      - vix_level
      - spy_vol_20 (annualized %)
      - spy_drawdown
    """
    spy_aligned = spy_close.reindex(close_index, method='ffill')
    vix_aligned = vix.reindex(close_index, method='ffill')

    spy_ret = spy_aligned.pct_change()
    spy_vol_20 = spy_ret.rolling(20).std() * np.sqrt(252)  # annualized
    spy_peak = spy_aligned.cummax()
    spy_dd = spy_aligned / spy_peak - 1

    df = pd.DataFrame({
        'vix_level': vix_aligned,
        'spy_vol_20': spy_vol_20,
        'spy_drawdown': spy_dd,
    }, index=close_index)
    return df.ffill().fillna(0)


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


def get_features_with_regime(close, vol, hp, regime_df, use_regime=True):
    """Compute features. If use_regime, append regime cols."""
    feat_panels = ml_model.build_features_panel(close, vol)
    target = ml_model.make_target(close, hp['forward_days'], hp['target_type'])
    return feat_panels, target


def train_eval(close, vol, train_mask, test_mask, hp, regime_df, use_regime,
               feature_names):
    """Train Ridge on features (+ regime) and return test scores."""
    feat_panels = ml_model.build_features_panel(close, vol)
    target = ml_model.make_target(close, hp['forward_days'], hp['target_type'])

    train_feats = {n: df[train_mask] for n, df in feat_panels.items()}
    train_target = target[train_mask]
    train_long = ml_model.stack_panel_to_long(train_feats, train_target)

    test_feats = {n: df[test_mask] for n, df in feat_panels.items()}
    test_long = ml_model.stack_panel_to_long(test_feats)

    if use_regime:
        # Merge regime features by date (broadcast across tickers)
        regime_train = regime_df[regime_df.index.isin(train_long['date'].unique())]
        regime_test = regime_df[regime_df.index.isin(test_long['date'].unique())]
        train_long = train_long.merge(regime_train, left_on='date', right_index=True, how='left')
        test_long = test_long.merge(regime_test, left_on='date', right_index=True, how='left')

    if len(train_long) < 1000:
        return None

    feat_cols = list(feature_names)
    train_long = train_long.dropna(subset=feat_cols)
    test_long = test_long.dropna(subset=feat_cols)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_long[feat_cols].values)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, train_long['target'].values)

    X_test_s = scaler.transform(test_long[feat_cols].values)
    preds = model.predict(X_test_s)

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    return score_long, model.coef_, feat_cols


def run_one_window(close, vol, test_year, hp, regime_df, spy_ret_full,
                   use_regime, feature_names):
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

    result = train_eval(close_sub, vol_sub, train_mask, test_mask, hp,
                        regime_df, use_regime, feature_names)
    if result is None:
        return None
    score_long, coefs, feat_cols = result

    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)
    test_close = close_sub[test_mask]
    port_ret = backtest_with_score(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret_full[(spy_ret_full.index >= test_start) & (spy_ret_full.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    coef_dict = {f: float(c) for f, c in zip(feat_cols, coefs)}
    return {
        'year': test_year, 'cagr': s['cagr'], 'sharpe': s['sharpe'],
        'mdd': s['mdd'], 'excess': excess, 'coefs': coef_dict,
    }


def run_config(close, vol, hp, regime_df, spy_ret_full, use_regime, feature_names, label):
    rows = []
    for test_year in range(1995, 2026):
        r = run_one_window(close, vol, test_year, hp, regime_df, spy_ret_full,
                           use_regime, feature_names)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'label': label,
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
        'rows': rows,
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Market regime features walk-forward")
    w(f"  Top-{TOP_N} | Ridge + 7y + Weekly | 31 windows")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_close = load_spy()
    spy_ret_full = spy_close.pct_change()

    print("Fetching VIX...")
    try:
        vix = fetch_vix()
        print(f"  VIX: {vix.index.min().date()} ~ {vix.index.max().date()} ({len(vix)} days)")
    except Exception as e:
        print(f"  VIX fetch failed: {e}")
        return

    regime_df = build_regime_features(close.index, spy_close, vix)
    w(f"\nRegime features built: {regime_df.shape}")
    w(f"  Latest VIX: {regime_df['vix_level'].iloc[-1]:.2f}")
    w(f"  Latest SPY 20d vol (annualized): {regime_df['spy_vol_20'].iloc[-1]*100:.2f}%")
    w(f"  Latest SPY drawdown: {regime_df['spy_drawdown'].iloc[-1]*100:.2f}%")

    hp = dict(ml_model.ML_HP_DEFAULT)
    base_features = list(hp['feature_names'])
    regime_features = base_features + ['vix_level', 'spy_vol_20', 'spy_drawdown']

    configs = [
        ('Baseline (no regime)',  False, base_features),
        ('+ Regime features',     True,  regime_features),
    ]

    results = []
    for label, use_regime, feat_names in configs:
        w(f"\n[{label}] running...")
        r = run_config(close, vol, hp, regime_df, spy_ret_full, use_regime, feat_names, label)
        if r:
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | "
              f"t={r['t_stat']:.2f}")

    # Comparison
    if len(results) >= 2:
        baseline = results[0]
        regime = results[1]
        w(f"\n{'='*100}")
        w(f"## Comparison")
        w(f"{'Metric':<20} {'Baseline':>15} {'+ Regime':>15} {'Diff':>10}")
        w("-" * 60)
        w(f"{'CAGR':<20} {baseline['avg_cagr']*100:>+14.2f}% {regime['avg_cagr']*100:>+14.2f}% "
          f"{(regime['avg_cagr']-baseline['avg_cagr'])*100:>+9.2f}%p")
        w(f"{'Sharpe':<20} {baseline['avg_sharpe']:>15.2f} {regime['avg_sharpe']:>15.2f} "
          f"{regime['avg_sharpe']-baseline['avg_sharpe']:>+10.2f}")
        w(f"{'MDD':<20} {baseline['avg_mdd']*100:>+14.2f}% {regime['avg_mdd']*100:>+14.2f}% "
          f"{(regime['avg_mdd']-baseline['avg_mdd'])*100:>+9.2f}%p")
        w(f"{'vs SPY':<20} {baseline['avg_excess']:>+14.2f}p {regime['avg_excess']:>+14.2f}p "
          f"{regime['avg_excess']-baseline['avg_excess']:>+9.2f}p")
        w(f"{'t-stat':<20} {baseline['t_stat']:>15.2f} {regime['t_stat']:>15.2f} "
          f"{regime['t_stat']-baseline['t_stat']:>+10.2f}")

        # Average regime feature coefficients
        if regime.get('rows'):
            avg_coefs = {}
            n = 0
            for r in regime['rows']:
                for k, v in r['coefs'].items():
                    avg_coefs[k] = avg_coefs.get(k, 0) + v
                n += 1
            avg_coefs = {k: v/n for k, v in avg_coefs.items()}
            w(f"\n## Average Ridge coefficients (regime model)")
            for f, c in sorted(avg_coefs.items(), key=lambda x: -abs(x[1])):
                w(f"  {f:<15} {c:>+.4f}")

    out_path = os.path.join(OUTPUT_DIR, 'regime_features_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
