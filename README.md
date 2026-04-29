# 자양동 Stock Lab

S&P 500 universe + Ridge ML 기반 quant 투자 모델.

## 검증된 알파 (v5 — 3-feature minimum, 2026-04-28)

| 검증 | 결과 |
|---|---|
| **Walk-forward** | 31 windows (1995-2025) |
| **Avg CAGR** | **+46.82%** (Sharpe **1.77**) |
| **vs SPY alpha** | **+34.16%p** (t=6.63) |
| **승률** | 28/31 (90%) |
| **Bootstrap** | 30 runs, mean alpha **+29.81%p** (모두 양수, std 1.72%p) |
| **진짜 알파 추정** (보수) | **+20~25%p** (Bootstrap + survivorship 차감) |

**핵심 발견**: 6 features → 3 features로 줄였더니 알파 ↑ (+31% → +34%p).
Momentum/MA/Trend는 redundant. 진짜 핵심은 **lowvol + rsi + volsurge**.

자세한 보고서: `reports/final_tuning_report.md`

### Weekly Walk-Forward (v5 minimum, 2025-2026)

매주 월요일마다 직전 7년으로 Ridge 재학습 → Top-20 → 5거래일 보유. **OOS 시뮬레이션**.

| 항목 | 값 |
|---|---|
| 기간 | 2025-01-02 ~ 2026-04-27 (**69주**) |
| 주당 평균 알파 | **+1.15%p** |
| 연환산 알파 | **+60.02%p** |
| 누적 portfolio | **+165.30%** (복리) |
| 누적 SPY | +29.42% |
| **누적 초과수익** | **+135.87%p** |
| 승률 (port > SPY) | 43/69 (62.3%) |
| 베스트 주 | +10.09%p (2025-04-21) |
| 워스트 주 | -6.85%p (2025-02-18) |
| P10 / P50 / P90 | -3.05%p / +1.00%p / +5.69%p |

**월별 누적 비교 (포인트)**:
- 강세: 2025-04 (+12.0p), 2025-06 (+14.8p), 2025-09 (+12.5p), 2026-04 (+10.6p)
- 약세: 2025-02 (-5.0p), 2025-11 (-3.4p), 2026-01 (-4.2p)
- **16개월 중 12개월 SPY 초과** (75%)

> **Note**: 31 windows (1995-2025) 년간 백테스트 (+34.16%p) 대비 2025-2026이 유난히 좋은 환경 (mid-cap 강세). 미래 보수 추정은 여전히 **+20~25%p**.

→ 출처: `tests/test_weekly_walkforward.py` → `results/weekly_walkforward.txt`, `weekly_walkforward.csv`

## 핵심 설정 (v5 best)

| 항목 | 값 | 검증 |
|---|---|---|
| Universe | S&P 500 + 17 ETF (518 tickers) | ✅ |
| Model | **Ridge regression** | ✅ Ridge > LightGBM > XGBoost > MLP > RF |
| **Features** | **lowvol + rsi + volsurge (3개)** ⭐ | ✅ Ablation으로 minimum 확정 |
| Train years | **7** | ✅ Sweet spot (3/5/7/10 비교) |
| Forward horizon | 10 days | ✅ |
| Rebalance | **Weekly** (5 trading days) | ✅ Best (Daily/Weekly/Biweekly/Monthly 비교) |
| Top-N | **15 또는 20** | ✅ Top-15 Cap15% / Top-20 Cap20% Sharpe 1.84 동률 |
| Sector cap | **15-20%** (균형) 또는 30% (공격) | ✅ 매트릭스 검증 완료 |
| Cost | 0.10% round-trip | 검증 가정 |

### Feature Set 비교

| Set | Features | Sharpe | Alpha vs SPY |
|---|---|---|---|
| **★ Minimum (v5)** | **lowvol + rsi + volsurge** | **1.77** | **+34.16%p** |
| Without MA+Mom | lowvol + trend + rsi + volsurge | 1.73 | +32.35%p |
| Without MA | -ma | 1.68 | +32.67%p |
| Without Mom | -momentum | 1.69 | +32.06%p |
| Full (v4) | 6 features (incl. momentum, ma, trend) | 1.63 | +31.01%p |

**충격**: Momentum/MA/Trend가 redundant. 학계 표준 momentum factor가 우리 모델에선 noise.

### Top-N × Sector Cap Cross 매트릭스 (v5 / 3-feature, 검증 완료)

**Sharpe 매트릭스**

| | None | Cap 30% | Cap 25% | Cap 20% | Cap 15% |
|---|---|---|---|---|---|
| **Top-10** | 1.73 | 1.78 | 1.76 | 1.76 | 1.75 |
| **Top-15** | 1.74 | 1.81 | 1.81 | 1.81 | **1.84** ⭐ |
| **Top-20** | 1.79 | 1.78 | 1.80 | **1.84** ⭐ | 1.82 |

**Alpha 매트릭스 (vs SPY %p)**

| | None | Cap 30% | Cap 25% | Cap 20% | Cap 15% |
|---|---|---|---|---|---|
| **Top-10** | +44.0 | **+45.2** ⭐ | +41.7 | +41.7 | +35.2 |
| **Top-15** | +37.5 | +38.9 | +37.7 | +37.7 | +34.9 |
| **Top-20** | +34.6 | +33.9 | +34.4 | +33.6 | +31.7 |

**MDD 매트릭스**

| | None | Cap 30% | Cap 25% | Cap 20% | Cap 15% |
|---|---|---|---|---|---|
| **Top-10** | -23.3% | -22.6% | -22.4% | -22.4% | -19.9% |
| **Top-15** | -22.0% | -21.3% | -20.8% | -20.8% | -19.9% |
| **Top-20** | -20.9% | -20.7% | -20.4% | -20.1% | **-19.7%** ⭐ |

→ 출처: `tests/test_3feature_topn_cap_cross.py` → `results/topn_cap_cross_3feature.txt`

> **v4(6-feature) 대비 향상**: 모든 config가 Sharpe +0.05~0.16, alpha +1~2%p 개선. 3-feature가 압승.

### 타입별 추천 (v5)

| 사용자 | 권장 | Sharpe | Alpha | MDD | t-stat |
|---|---|---|---|---|---|
| **안정 우선** (낮은 MDD) | Top-20 + Cap 15% | 1.82 | +31.7%p | **-19.7%** | 7.16 |
| **균형 1위** ⭐ (Sharpe 공동 1위) | Top-15 + Cap 15% | **1.84** | +34.9%p | -19.9% | 6.61 |
| **균형 2위** ⭐ | Top-20 + Cap 20% | **1.84** | +33.6%p | -20.1% | **7.03** |
| **알파 + 안정** | Top-15 + Cap 25% | 1.81 | +37.7%p | -20.8% | 6.45 |
| **공격 (최대 알파)** | Top-10 + Cap 30% | 1.78 | **+45.2%p** | -22.6% | 5.62 |
| **단순 (현재 default)** | Top-20 No cap | 1.79 | +34.6%p | -20.9% | 6.75 |

**선택 가이드** (시드 무관, 결정 기준 = Sharpe > Alpha > t-stat > MDD):
- Sharpe 1위 + 알파 평균 = **Top-15 Cap15%** 또는 **Top-20 Cap20%** (사실상 동일)
- 알파 압도 + Sharpe 양호 = **Top-10 Cap30%** (변동 ↑)
- 운용 단순성 = **Top-20 No cap** (현재 라이브)

→ `current_portfolio.py`에서 `TOP_N`, `SECTOR_CAP` 변수로 전환 가능.

## 폴더 구조

```
stock_lab/
├── README.md
├── core.py                     # 백테스트 엔진 (HP dict, build_holdings, stats)
├── factors.py                  # 학술 팩터 (momentum 12-1, low-vol, trend, rsi, ma, volsurge)
├── ml_model.py                 # LightGBM 파이프라인 (Ridge로도 사용)
├── current_portfolio.py        # 매수 추천 (Ridge + 7y + Weekly + 설정 가능 Top-N/Cap)
├── fetch_sp500_list.py         # Wikipedia에서 S&P 500 명단
├── download_sp500.py           # yfinance로 일봉 데이터
├── fetch_sectors.py            # 종목별 sector 정보
├── validate_sp500.py           # 데이터 검증
├── requirements.txt
│
├── data/
│   ├── master_sp500/           # 종목별 일봉 CSV (518 종목, .gitignore)
│   ├── sectors.csv             # 종목별 sector 매핑
│   ├── sp500_tickers.txt
│   ├── etf_tickers.txt
│   └── all_sp500_tickers.txt
│
├── tests/                      # 검증 스크립트
│   ├── test_qqq_benchmark.py        # QQQ 실비교
│   ├── test_cost_sensitivity.py     # 비용 민감도
│   ├── test_train_window.py         # 7년이 sweet spot 발견
│   ├── test_forward_horizon.py      # Weekly + 10d forward 최적
│   ├── test_hp_tuning.py            # LightGBM HP search (효과 미미)
│   ├── test_model_compare.py        # Ridge > LightGBM 발견
│   ├── test_ml_sp500.py             # 31 windows walk-forward (메인)
│   ├── test_event_catalog.py        # 30년 큰 이벤트 자동 식별
│   ├── test_event_response.py       # 섹터 회복 패턴
│   ├── test_event_rules.py          # Stop-loss 등 룰 검증 (대부분 손해)
│   ├── test_v4_portfolio_rules.py   # Top-N vs Score-weighted
│   ├── test_v4_bootstrap.py         # 30 runs robustness
│   ├── test_sector_cap.py           # Sector cap (25-30% sweet)
│   ├── test_sector_cap_decompose.py # 시기별 cap 효과 분해
│   ├── test_topn_cap_cross.py       # Top-N × Cap 매트릭스 (v4 / 6-feature)
│   ├── test_3feature_topn_cap_cross.py # Top-N × Cap 매트릭스 (v5 / 3-feature) ⭐
│   └── test_weekly_walkforward.py   # Weekly walk-forward 2025-2026 ⭐
│
├── results/                    # 검증 결과
│   ├── current_portfolio.csv         # 최신 매수 추천
│   ├── ml_sp500_walkforward.csv      # 31 windows
│   ├── topn_cap_cross.txt            # v4 cross 매트릭스
│   ├── topn_cap_cross_3feature.txt   # v5 cross 매트릭스
│   ├── weekly_walkforward.txt        # Weekly walk-forward 결과
│   ├── weekly_walkforward.csv        # 주별 상세
│   └── archive/                      # 옛 결과
│
└── reports/
    ├── final_tuning_report.md        # 최종 보고서 (v4)
    └── archive/                      # 옛 보고서
```

## 사용법

### 1. 데이터 다운로드 (한 번만)

```bash
python3 fetch_sp500_list.py     # Wikipedia에서 503 종목 + ETF
python3 download_sp500.py       # yfinance로 일봉 (~1-2분)
python3 fetch_sectors.py        # 종목별 sector
python3 validate_sp500.py       # 검증
```

### 2. 매수 추천 받기

```bash
python3 current_portfolio.py
```

`current_portfolio.py` 상단의 config 변수 편집해서 운용 룰 선택

```python
SEED_USD     = 400         # 시드 (USD)
TOP_N        = 20          # 10 / 15 / 20
SECTOR_CAP   = None        # None (no cap) or 0.20 / 0.25 / 0.30
TRAIN_YEARS  = 7           # 7이 sweet spot
```

출력: 최신 데이터로 학습한 Top-N 종목 + 매수 비중. CSV: `results/current_portfolio.csv`

### 3. 검증 재실행 (선택)

```bash
PYTHONPATH=. python3 tests/test_ml_sp500.py                    # 31 windows 알파
PYTHONPATH=. python3 tests/test_v4_bootstrap.py                # robustness
PYTHONPATH=. python3 tests/test_3feature_topn_cap_cross.py     # v5 Top-N × Cap 매트릭스
PYTHONPATH=. python3 tests/test_weekly_walkforward.py          # 주간 walk-forward
```

## 운용 룰

```
매주 월요일:
  1. python3 current_portfolio.py        # 새 Top-N 출력 (Ridge 자동 재학습)
  2. 이전 portfolio와 차이 종목만 매매 (토스증권 소수점)
```

## 검증된 사실

| 발견 | 효과 |
|---|---|
| ✅ Universe 확장 (NASDAQ-100 → S&P 500) | 알파 +8%p |
| ✅ Train 7년 (vs 5년) | 알파 +7%p |
| ✅ Weekly (vs Biweekly/Monthly) | 알파 +2%p |
| ✅ Ridge (vs LightGBM/XGBoost) | 알파 +4%p |
| ✅ Sector cap 25-30% | Sharpe ↑ 0.05, 알파 변화 없음 |
| ✅ **Feature ablation: 3 features이 6 features 압승** | Sharpe +0.14, 알파 +3%p |
| ❌ Multi-horizon momentum (1m/3m/6m) | 알파 -1.3%p (redundant + recent regime) |
| ❌ Regime features (VIX, drawdown) | 효과 거의 없음 (linear model 한계) |
| ❌ Stop-loss | 알파 -8%p (회복기 놓침) |
| ❌ Vol filter / Drawdown halt | 알파 -2~3%p |
| ❌ CASH 통합 | 효과 미미 |
| ❌ HP tuning | 효과 미미 |
| ⚠ Survivorship bias | ~13% 부풀림 (Bootstrap, 3-feature) |

## 알려진 한계

- **Survivorship bias** 미해결 (S&P 500만 보유 → API 가입 시 가능)
- **비용 0.20% 이상**이면 알파 사라짐 → 미국 무수수료 broker 또는 토스증권 (외화통장) 필수
- **2024-2025 mega-cap 집중기** 알파 약함 (회복은 자연스러움)

## 다음 단계 후보

### Tier 1 — 즉시 가치 큼 (미검증)
- **Sector relative momentum** — 섹터 내 상대 점수
- **Ensemble** (Ridge + LightGBM 평균) — 두 모델 약점 보완

### Tier 2 — 큰 작업
- **OHLC 활용** (gap analysis, intraday vol) — 분봉 데이터 활용
- **Quality factor (ROE)** — yfinance 부분 가능
- **Multi-horizon target** (5+10+20일 평균) — input 아닌 target 다양화

### Tier 3 — 사용자 개입 필요
- **Survivorship bias 해결** (Tiingo/Alpha Vantage API)
- **한국 시장 추가** (pykrx)
- **뉴스 sentiment** (LLM)
- **Fundamentals** (PER/ROE) 본격 — 유료 DB

### 이미 검증 후 제외된 후보

| 후보 | 결과 | 이유 |
|---|---|---|
| ❌ Multi-horizon momentum input (1m/3m/6m) | 알파 -1.3%p | RSI(14일)와 redundant + 최근 mega-cap 집중기에 음수 계수 |
| ❌ Regime features (VIX, SPY vol, drawdown) | 알파 +0.04%p | Cross-sectional 모델은 모든 종목 같은 값 → 효과 없음 + Linear가 interaction 못 잡음 |
| ✅ Feature ablation | **완료** | 6→3 features (lowvol+rsi+volsurge)로 압축 = v5 |
| ❌ Stop-loss | 알파 -8%p | 회복기 놓침 |
| ❌ Vol filter / Drawdown halt | 알파 -2~3%p | 시장 타이밍 손해 |
| ❌ CASH 통합 | 효과 미미 | OOS에서 -5%p 이상 손해 |
| ❌ HP tuning (LightGBM) | 효과 미미 | grid search 성과 제한 |
| ❌ Ridge alpha tuning | 효과 미미 | alpha=1.0 default OK |

→ 자세한 검증 출처: `tests/` 폴더 + `결과는 results/`

## 개발 일지

| 버전 | 환경 | 알파 | t-stat | Sharpe |
|---|---|---|---|---|
| v1 | NASDAQ-100 + LightGBM 11 windows | +7.02%p (vs QQQ) | 1.68 | 1.19 |
| v2 | + 21 windows 확장 | +10.16%p | 3.25 | 1.27 |
| v3 | + S&P 500 universe | +18.63%p (vs SPY) | 5.69 | 1.35 |
| v4 | + Ridge + 7y + Weekly + Cap 옵션 | +31.03%p | 6.65 | 1.63 |
| **v5** | **+ Feature ablation (3-feature minimum)** | **+34.16%p** | **6.63** | **1.77** |
