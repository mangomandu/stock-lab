# 자양동 Stock Lab

S&P 500 universe + LightGBM ML 기반 quant 투자 모델.

---

## 📊 Quick Status

```
Current lock:   v0.8.0 (2026-05-06)
Net α vs SPY:   +5.44%p (Sharpe 0.56, IC 0.0166)
Cost 환경:       한국투자증권 평시 0.25% per side
Backtest:       1998-2026 (28년 walk-forward)
Universe:       Sharadar historical S&P 500 (delisted 포함)
운용 상태:       Backtest only (paper trading 미시작)
```

---

## 🎯 v0.8.0 — Current Winner

### Setup

| Component | Value |
|---|---|
| **Features** | 27 (price 10 + cross 2 + sector 3 + fundamental 10 + macro 2) |
| **Model** | LightGBM (depth=4, n_estimators=200, learning_rate=0.1) |
| **Universe** | Sharadar historical S&P 500 (1998-2026) |
| **Train window** | 10y rolling |
| **Retrain freq** | 252d (yearly) |
| **Rebal freq** | 5d (weekly) |
| **NaN handling** | fillna(median) per feature |
| **PIT lag** | 60일 (fundamental) |
| **Target** | Multi-horizon rank (5d/10d/20d forward return mean) |
| **TOP_N** | 10 |
| **HYST_EXIT** | 40 |
| **Cost** | 0.25% per side (한투 평시) |

### 성능

| Metric | v0.7.2 baseline | v0.8.0 winner | Δ |
|---|---|---|---|
| α gross vs SPY | — | **+11.83%p** | — |
| α net vs SPY | +3.98%p | **+5.44%p** | **+1.46%p** |
| Sharpe | 0.60 | 0.56 | -0.04 |
| MDD | — | -59.6% | — |
| Rank IC | — | +0.0166 | — |
| Cost 환경 | 0.05% | 0.25% (5x) | — |
| Win rate (gross) | — | 18/29 (62%) | — |
| Win rate (net) | — | 14/29 (48%) | — |

→ Cost 5배 환경에서도 alpha 개선. 진짜 운용 가능.

### Feature importance top 5

```
1. yield_curve  20.3%   (10Y - 2Y treasury yield, regime detection)
2. vix_level     7.1%   (CBOE volatility, regime)
3. beta_to_spy   6.8%   (market sensitivity)
4. roe_ttm       5.0%   (quality factor)
5. pb            5.0%   (value factor)
```

→ 모델 alpha source = **"Macro regime detection + Regime-specific stock picking"** (비선형 interaction, Tree만 잡음).

### 검증 패턴

**Bear/Recovery 강함**:
- 2002 +51.7%p, 2008 +39.3%p, 2009 +95.7%p, 2020 +63.0%p

**Bull/AI rally 약함** (v0.8.1 fix 후보):
- 2023 -33.6%p, 2024 -12.5%p (AI rally 못 따라감)

→ 모델 = "**Contrarian, defensive, regime-aware**".

---

## 🗺️ Roadmap

```
✅ v0.5-v0.6   yfinance + 4 features baseline 시대 (legacy)
✅ v0.7.1     yfinance + 9000 cells grid (+45%p, survivorship 부풀림 입증)
✅ v0.7.2     Sharadar baseline (4f, +3.98%p) — 진짜 baseline
✅ v0.8.0     27f LightGBM lock (+5.44%p, current)

🟡 v0.8.1     Bull-catcher feature 추가 (sector momentum, growth-quality)
              → 2023-2024 -46%p AI rally 약점 fix 시도
              → 1-2시간 코딩

🟡 v0.8.2     HYST mechanism alternatives
              → fixed/dynamic/trailing/take_profit/stop_loss + cooldown
              → 운영 안정성 ↑

🟡 v0.8.3     Refresh meta-policy
              → fixed forever / numeric / regime-aware (VIX) / model drift
              → 적응형 retrain 정책

🔵 v0.9.0     Paper trading (Alpaca)
              → 매주 portfolio 자동 + 실시간 가격 시뮬
              → 4-8주 OOS 실시간 검증 (backtest 알파 70%+ 재현 확인)

🔵 v0.9.1     Slippage / market impact 측정
              → 실제 spread/timing 측정
              → backtest cost 가정 검증

🟢 v1.0       운용 시작 (시드 충분 시)
              → 시드 $5,000+ 도달 + paper trading 통과 후
              → 한국투자증권 (또는 Alpaca) 자동 매매

🟣 v2.0+      Multi-factor / BGR cascaded
              → User EE 직관 idea (regime-independent alpha)
              → Multi-strategy ensemble

🟣 v3.0+      단기 add-on (event-driven, sentiment)
              → 장기 모델 lock 후 평가
```

---

## 🚀 Usage

### 데이터 다운로드 (한 번만)

```bash
# Sharadar API key 필요 (data.nasdaq.com)
echo "NASDAQ_DATA_LINK_API_KEY=xxx" > .env

# Sharadar bulk download
python3 fetch_sharadar.py
```

### Feature panels 빌드 (한 번만)

```bash
# SP500 close + volume (wide panels)
python3 build_sp500_panel.py

# Fundamental panel (PIT lag 적용)
python3 build_fundamental_pit.py

# 29 features 빌드 (price + sector + fundamental + macro)
python3 build_features_phaseX.py
```

### v0.8.0 Walk-forward backtest

```bash
# Phase Y v3 — 24 cells × 2 configs (B1+B2)
python3 phase_y_pure_ml_v2.py

# LGBM HP sweep (depth/n_est/lr)
python3 phase_y_lgbm_sweep.py

# Full grid (24 models × 27 portfolio HP = 648 cells)
python3 v080_full_grid.py
```

### 분석

```bash
# Feature importance + correlation + per-year breakdown
python3 analyze_phase_y.py

# Portfolio HP sweep (TOP × HYST × REBAL)
python3 v080_portfolio_sweep.py
```

### 매수 추천 (현재 v0.7.1 기반, v0.8.0으로 마이그레이션 필요)

```bash
# 데이터 갱신
python3 refresh_recent.py

# Top-N portfolio 출력
python3 current_portfolio.py
```

---

## 📁 Project Structure

```
stock_lab/
├── README.md                       이 파일
├── CHANGELOG.md                    버전 history
├── .env                            API keys (git ignored)
│
├── core.py                         Backtest engine
├── factors.py                      Price-based factor 함수
├── factors_fundamental.py          SF1 fundamental factors (TTM 계산 + PIT lag)
├── ml_model.py                     Ridge ML pipeline (legacy v5)
│
├── phase1_compare.py               v0.7.2 baseline + 공통 유틸 (build_sp500_mask 등)
├── phase_y_pure_ml_v2.py           v0.8.0 main training (vectorized)
├── phase_y_lgbm_sweep.py           LGBM HP sweep (depth/n_est/lr)
├── v080_full_grid.py               24 models × 27 portfolio HP grid
├── v080_portfolio_sweep.py         Portfolio HP only sweep
├── analyze_phase_y.py              Feature importance + corr + per-year
│
├── build_features_phaseX.py        29 features 빌드
├── build_fundamental_pit.py        SF1 → fundamental panel + PIT
├── build_sp500_panel.py            SEP → wide close/volume panels
│
├── fetch_sharadar.py               Sharadar bulk download
├── refresh_recent.py               yfinance 최근 데이터 갱신
├── download_sp500.py               yfinance bulk
├── verify_sharadar_integrity.py    Data integrity check
├── current_portfolio.py            매수 추천 (v0.7.1 lock — 마이그레이션 필요)
│
├── data/
│   ├── sharadar/                   Sharadar parquet (raw, ~2GB)
│   ├── panels/
│   │   ├── sp500_close.parquet     Wide close (date × ticker)
│   │   ├── sp500_volume.parquet
│   │   ├── fundamental_pit.parquet
│   │   └── features/               29 feature panels (1.1GB)
│   ├── master_sp500/               yfinance 캐시 (legacy)
│   └── macro/                      FRED VIX / yield curve
│
├── results/
│   ├── v0_8_0_winner_config.json   v0.8.0 lock config
│   ├── v080_full_grid_results.json 648 cells results
│   ├── v080_portfolio_sweep.json   27 portfolio HP results
│   ├── lgbm_sweep_results.json     32 LGBM HP cells
│   ├── phase_y_v3_results.json     24 model bake-off
│   ├── phase_y_lgbm_*.csv          LGBM analysis (importance, yearly)
│   ├── v080_score_panels/          24 model score panels (cache)
│   ├── current_portfolio.csv       매수 list
│   └── archive/                    옛 reports
│
├── tests/
│   ├── compute_rank_ic.py          Rank IC (Spearman) infra
│   └── compute_vif.py              Multicollinearity check
│
├── archive/                        옛 코드 / results
│   ├── old_phases/                 19 outdated scripts
│   ├── old_tests/                  62 옛 실험 tests
│   ├── old_results/                ~50 옛 results
│   └── reports/                    옛 보고서
│
└── notes/                          temp notes / prompts
```

---

## 🔧 Reference

- **CHANGELOG.md**: 버전별 상세 변경 이력
- **memory** (Claude): `~/.claude/projects/-home-dlfnek-stock-lab/memory/` (auto-saved context)
- **Plans** (Claude): `~/.claude/plans/` (현재 비움)

---

## ⚠️ 운용 시 주의사항

```
1. v0.8.0 = backtest only. 실전 운용 검증 X.
2. Survivorship correction 적용했지만 100% 완벽 X (Sharadar 한계).
3. Cost 0.25% 가정 (한투 평시). 환전 (~0.5%) + 양도소득세 (22%) 별도.
4. Realistic α 추정 (보수): backtest +5.44%p → 실전 ~+3.8%p.
5. v1.0 운용 전 paper trading 4-8주 OOS 검증 필수.
6. 시드 작으면 (<$20k) cost drag 비중 큼 → 실효 alpha 작음.
```

---

## 📜 License + Source

Personal research project. Data: Sharadar (Nasdaq Data Link) + yfinance + FRED.
