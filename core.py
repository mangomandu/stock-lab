"""
자양동 Stock Lab — Backtest core engine.

Single HP dict holds all hyperparameters. Functions take HP and override fields
as needed. Test scripts import this module and run experiments on top.
"""
import pandas as pd
import numpy as np
import os

MASTER_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
TRADING_DAYS = 252

# -----------------------------------------------------------------------------
# Default hyperparameters. v4 best (validated).
# -----------------------------------------------------------------------------
DEFAULT_HP = {
    # Indicator lookbacks (legacy RSI/MA/VOL — kept for ML feature input)
    'rsi_period': 14,
    'ma_period': 20,
    'vol_period': 20,

    # Portfolio construction
    'top_n': 20,
    'rebal_days': 5,        # Weekly (5 trading days) — v4 sweet spot
    'hysteresis': 0,

    # Costs
    'cost_oneway': 0.0005,  # 0.05%/side -> 0.10% round-trip
    'cash_score': None,     # CASH option disabled (validated as ineffective)

    # Validation period (use 7y train walk-forward in tests)
    'train_start': '1995-01-01',
    'split_date':  '2005-01-01',

    # Weight grid (legacy — for academic factor model only)
    'weight_step': 0.1,
    'rsi_signed':  True,
}


def merge_hp(overrides=None):
    hp = dict(DEFAULT_HP)
    if overrides:
        hp.update(overrides)
    return hp


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_panel(master_dir=MASTER_DIR):
    """Load all tickers into wide-format Close/Volume DataFrames.

    Each ticker CSV must have at least Datetime, Close, Volume columns.
    Aggregates to daily granularity (handles both daily and intraday inputs).
    """
    files = sorted(f for f in os.listdir(master_dir) if f.endswith('.csv'))
    closes, vols = {}, {}
    for file in files:
        ticker = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(master_dir, file))
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime']).sort_values('Datetime')
        df['Date'] = df['Datetime'].dt.normalize()
        daily = df.groupby('Date').agg({'Close': 'last', 'Volume': 'sum'})
        closes[ticker] = daily['Close']
        vols[ticker] = daily['Volume']
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).sort_index()
    return close, vol


def compute_scores(close, vol, hp):
    """Compute RSI/MA/VOL score panels."""
    p_rsi = hp['rsi_period']
    p_ma = hp['ma_period']
    p_vol = hp['vol_period']

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(p_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(p_rsi).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    rsi_s = ((70 - rsi.fillna(50)) / 40 * 100).clip(0, 100)
    ma_s = (close > close.rolling(p_ma).mean()).astype(float) * 100
    vma = vol.rolling(p_vol).mean()
    vol_ratio = (vol / vma.replace(0, 1e-9)) * 100
    vol_s = ((vol_ratio - 50) / 150 * 100).clip(0, 100)
    return rsi_s, ma_s, vol_s


# -----------------------------------------------------------------------------
# Weight grid
# -----------------------------------------------------------------------------
def weight_grid(hp):
    """Generate weight candidates. |w_R| + w_M + w_V = 1, w_R signed (or 0+)."""
    step = hp['weight_step']
    n = int(round(1 / step))  # e.g., 10 for step=0.1
    candidates = []
    r_range = range(-n + 1, n) if hp['rsi_signed'] else range(0, n)
    for r in r_range:
        rw = round(r * step, 2)
        for m in range(1, n):
            mw = round(m * step, 2)
            vw = round(1 - abs(rw) - mw, 2)
            if step <= vw <= 1 - step:
                candidates.append((rw, mw, vw))
    return candidates


# -----------------------------------------------------------------------------
# Backtest engine
# -----------------------------------------------------------------------------
def build_holdings(score, hp):
    """Returns DataFrame of float holdings (1.0 if held, 0.0 otherwise)."""
    top_n = hp['top_n']
    rebal_days = hp['rebal_days']
    hysteresis = hp['hysteresis']

    if rebal_days == 1 and hysteresis == 0:
        ranks = score.rank(axis=1, ascending=False, method='first')
        return (ranks <= top_n).astype(float)

    n_days, n_stocks = score.shape
    score_arr = score.values
    holdings_arr = np.zeros((n_days, n_stocks), dtype=bool)
    current = np.zeros(n_stocks, dtype=bool)
    last_rebal = -rebal_days

    for t in range(n_days):
        if (t - last_rebal) < rebal_days:
            holdings_arr[t] = current
            continue

        row = score_arr[t]
        valid_mask = ~np.isnan(row)
        if not valid_mask.any():
            holdings_arr[t] = current
            continue

        sorted_scores = np.where(valid_mask, row, -np.inf)
        ranked_idx = np.argsort(-sorted_scores)

        if hysteresis > 0 and current.any():
            wider_mask = np.zeros(n_stocks, dtype=bool)
            wider_mask[ranked_idx[:top_n + hysteresis]] = True
            kept = current & wider_mask
            n_need = top_n - int(kept.sum())
            if n_need > 0:
                fill = ranked_idx[~kept[ranked_idx]]
                if len(fill) > 0:
                    kept[fill[:n_need]] = True
            current = kept
        else:
            current = np.zeros(n_stocks, dtype=bool)
            current[ranked_idx[:top_n]] = True

        holdings_arr[t] = current
        last_rebal = t

    return pd.DataFrame(holdings_arr.astype(float),
                        index=score.index, columns=score.columns)


def backtest(close, rsi_s, ma_s, vol_s, w, hp):
    """Run one backtest. Returns (port_net, daily_cost) Series.

    If hp['cash_score'] is not None, CASH is added as a synthetic asset that
    competes for Top-N slots. Cost is applied only to stock movements.
    """
    score = (rsi_s * w[0] + ma_s * w[1] + vol_s * w[2]).where(close.notna())

    cash_score = hp.get('cash_score')
    if cash_score is not None:
        score = score.copy()
        close = close.copy()
        score['CASH'] = float(cash_score)
        # Constant price — pct_change yields 0 for cash
        close['CASH'] = 1.0

    in_top = build_holdings(score, hp)
    weights = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)

    # Cost only applies to stock-side trades. Exclude CASH from cost.
    cost_cols = [c for c in held.columns if c != 'CASH']
    daily_cost = held[cost_cols].diff().abs().sum(axis=1).fillna(0) * hp['cost_oneway']

    return port_gross - daily_cost, daily_cost


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
def stats(port_ret):
    port_ret = port_ret.dropna()
    if len(port_ret) == 0:
        return None
    cum = (1 + port_ret).prod() - 1
    years = len(port_ret) / TRADING_DAYS
    cagr = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    std = port_ret.std()
    sharpe = port_ret.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else 0
    cum_curve = (1 + port_ret).cumprod()
    mdd = (cum_curve / cum_curve.cummax() - 1).min()
    return {
        'cum': cum, 'cagr': cagr, 'sharpe': sharpe, 'mdd': mdd,
        'days': len(port_ret),
    }


def benchmark_eq_weight(close, mask):
    return close[mask].pct_change().mean(axis=1)


# -----------------------------------------------------------------------------
# Strategy evaluation
# -----------------------------------------------------------------------------
def slice_by_mask(panels, mask):
    return [p[mask] for p in panels]


def eval_strategy(close, rsi_s, ma_s, vol_s, w, mask, hp):
    """Evaluate strategy w on the masked period. Returns stats dict + cost."""
    c, rs, ms, vs = slice_by_mask([close, rsi_s, ma_s, vol_s], mask)
    port, dcost = backtest(c, rs, ms, vs, w, hp)
    s = stats(port)
    if s:
        s['cost_drag'] = float(dcost.sum())
        # Avg per-day fraction of portfolio traded (one-way + reverse = total).
        s['avg_turnover'] = (s['cost_drag'] / hp['cost_oneway']) / s['days']
    return s


def grid_search(close, rsi_s, ma_s, vol_s, mask, hp):
    """Grid search over weight candidates. Returns list of dicts sorted by Sharpe."""
    results = []
    for w in weight_grid(hp):
        s = eval_strategy(close, rsi_s, ma_s, vol_s, w, mask, hp)
        if s is None:
            continue
        s['w'] = w
        results.append(s)
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    return results


def get_period_masks(close, hp):
    """Returns (train_mask, test_mask) based on hp's train_start/split_date."""
    train_start = pd.Timestamp(hp['train_start'])
    split = pd.Timestamp(hp['split_date'])
    train_mask = (close.index >= train_start) & (close.index < split)
    test_mask = close.index >= split
    return train_mask, test_mask


def fmt_w(w):
    """Format weight tuple for display."""
    return f"({w[0]:+.2f},{w[1]:.2f},{w[2]:.2f})"
