# 자양동 Stock Lab

S&P 500 universe + Ridge ML 기반 quant 투자 모델.

## 검증된 알파 (v5 — 3-feature minimum + 진단 + leakage fix)

> **2026-04-29 정정**: Target leakage 수정 후 모든 alpha 수치 ~1.7%p 하향. 상대 비교는 유효.

| 검증 | 결과 |
|---|---|
| **Walk-forward** | 31 windows (1995-2025) |
| **Avg CAGR** | **+45.51%** (Sharpe **1.74**) |
| **vs SPY alpha** | **+32.85%p** (t=6.52) |
| **승률** | 27/31 (87%) |
| **Bootstrap** | 30 runs, mean alpha **+29.81%p** (leakage 영향 ~28%p 추정) |
| **진짜 알파 추정** (보수) | **+18~23%p** (Bootstrap + survivorship 13% 차감) |

**핵심 발견**: 6 features → 3 features로 줄였더니 알파 ↑ (+31% → +34%p).
Momentum/MA/Trend는 redundant. 진짜 핵심은 **lowvol + rsi + volsurge**.

자세한 보고서: `reports/final_tuning_report.md`

### Weekly Walk-Forward (2025-2026)

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

### Effective N 진단

명목 Top-20 보유 시 실제 분산 효과 측정. Pairwise correlation 기반.

**Yearly walk-forward 31 windows (1577 rebal events, 1995-2025)**:
| 그룹 | Avg corr | Avg EffN | 의미 |
|---|---|---|---|
| 우리 Ridge 모델 | +0.249 | **4.02** | 명목 20개 → 실효 4 |
| (참고) Random Top-20 | +0.289 | 3.54 | 우리가 약간 더 분산 |
| (참고) S&P 500 전체 (~500) | +0.289 | 4.09 | 500개 들어도 비슷 |

**Stress periods (top 10% market corr, 2008/2020/2022 같은 위기)**:
- 우리: EffN 2.27 | Random: 1.87 | S&P 500 전체: 1.92
- → **위기 시엔 시장 자체가 corr 1로 수렴** (모든 미국 주식 동시 폭락). 우리만의 문제 X.

**Calm periods (bottom 10% market corr)**:
- 우리: 5.78 | Random: 6.06 | S&P 500 전체: 8.00
- → 평시엔 500 분산이 좋지만 우리도 baseline 수준

**시기별 일관**: 1995-2000 EffN 5.7, 2001-2010 3.4, 2011-2020 3.7, 2021-2025 3.8 — 모든 era 비슷

**Top sector dominance**: Technology 34.5% (low EffN 시기), Consumer Cyclical 18.8%, Financial Services 16.2%

**결론**:
- Eff N 4.0 = 절대값 낮아 보이지만 **랜덤 baseline 대비 더 좋음** (+0.4)
- "껍데기만 다른 몰빵" (블로그 글) 지적은 모든 미국 주식 portfolio의 한계, 우리 모델 특유 문제 아님
- → **Sector cap 강제 / Factor cap 추가 가치 X** (이미 baseline 이상)

→ 출처: `tests/test_effective_n_yearly.py`, `test_effective_n_baseline.py`

### ETF Buffer 검증 — 진짜 분산은 다른 자산군

**같은 자산군 (SPY) — 무가치**:
| Buffer | Sharpe | Alpha | MDD |
|---|---|---|---|
| 100% model | **1.79** | +34.6%p | -20.9% |
| 90% + SPY 10% | 1.76 | +30.7%p | -20.1% |
| 80% + SPY 20% | 1.72 | +26.9%p | -19.3% |
| 50% + SPY 50% | 1.55 | +16.1%p | -17.1% |

→ SPY는 우리 universe와 **같은 미국 주식** → 분산 효과 X. 모든 비율에서 Sharpe ↓, alpha ↓.

**다른 자산군 (TLT 장기채) — Sharpe ↑, 알파 ↓ 트레이드오프**:
| Buffer | CAGR | Sharpe | Alpha | MDD |
|---|---|---|---|---|
| **100% model (default)** | **+47.2%** | 1.79 | **+34.6%p** | -20.9% |
| 80% + TLT 20% | +38.0% | 1.85 | +25.3%p | -16.7% |
| 70% + TLT 30% | +33.5% | 1.88 | +20.8%p | -14.6% |
| 60% + TLT 40% | +29.0% | **1.91** | +16.4%p | **-12.7%** |

→ Sharpe **+0.12 개선** = "위험 대비 수익률"만 살짝 ↑. 다만:
- **CAGR -18%p 손해** (47% → 29%, 복리로 큰 손실)
- **Alpha -18.2%p 손해** (34.6 → 16.4, 시장 대비 초과수익 반토막)
- MDD -8.2%p 개선 (변동성 절반)

**핵심**: Sharpe 올라가는 건 **변동성이 더 빨리 줄어서**지 알파가 늘어서가 아님. 절대 부의 증가는 100% model이 압도.

**또 다른 자산군 (GLD 금) — 비슷한 효과**:
| Buffer | Sharpe | Alpha | MDD |
|---|---|---|---|
| 80% + GLD 20% | 1.83 | +26.5%p | -17.3% |
| 70% + GLD 30% | 1.85 | +22.5%p | -15.5% |
| 60% + GLD 40% | 1.86 | +18.6%p | -13.8% |

→ TLT와 비슷 (약간 낮음). 인플레 대비 헤지로 가치.

**결론**:
- 같은 자산군 buffer (SPY) = **무가치** ❌ (모든 비율 Sharpe ↓ alpha ↓)
- 다른 자산군 buffer (TLT/GLD) = **알파-안정 트레이드오프** (Sharpe 미세 ↑, 알파 큰 손해)
- **장기 자산 증식 = 100% model 유지가 정답** (복리로 알파 차이 어마무시)
- TLT/GLD buffer는 "MDD 못 견디는 사용자"용 별도 옵션 (예: 노후자금)
- 시드 작을 땐 buffer 사치 (수수료/소수점 매매 한계)

→ 출처: `tests/test_etf_buffer.py`, `test_tlt_gld_buffer.py`

### Hysteresis (회전율 ↓ + alpha ↑)

**아이디어**: Top-N 새 진입은 N등 이내, 기존 보유는 exit_n 등 안에 있으면 holding 유지. "진입 까다롭게, 퇴장 너그럽게".

**Full curve** (Top-20, 31 windows):
| exit_n | Alpha | Sharpe | Turnover/day |
|---|---|---|---|
| 20 (no hyst, baseline) | +34.6%p | 1.79 | 10.82% |
| 25 | +34.07%p | 1.76 | 9.63% |
| 30 | +35.99%p | 1.79 | 8.80% |
| 40 | +35.59%p | 1.79 | 7.62% |
| 45 | +36.91%p | **1.81** ⭐ | 7.14% |
| **50** ⭐ (default) | **+37.28%p** ⭐ | 1.80 | 6.74% |
| 55 | +36.61%p | 1.80 | 6.40% |
| 60 | +36.20%p | 1.79 | 6.08% |
| 65 | +34.82%p | 1.76 ↓ | 5.78% |
| 75 | +35.59%p | 1.77 | 5.31% |

→ **Sweet spot: exit_45~55**. 우리 default = exit_50 (alpha peak).
→ 65 이상은 신호 무시 시작 → alpha degradation.

**효과 (exit_50 기준)**:
- Alpha **+2.72%p** (34.56 → 37.28)
- Turnover **-37%** (10.82% → 6.74%/day)
- 동시 달성 = **공짜 점심**

> **메커니즘**: 매주 Top-20 강제 회전이 비용 + 변동성만 늘림. 대신 30~50등 약간 떨어진 종목 holding 유지하면:
> - **비용 ↓** (불필요한 매매 줄임)
> - **알파 ↑** (가짜 sell signal 무시, 진짜 신호만 반영)
> - **자본 집중** (Top-50 분산 X, 항상 Top-20 conviction 유지)

> **Top-N 늘리기 vs Hysteresis**: Top-50 (no hyst)도 turnover 낮지만 종목당 비중 1/50=2%로 신호 희석 → alpha ↓. Hysteresis는 Top-20 비중 1/20=5%를 유지해 자본 집중.

→ 출처: `tests/test_hysteresis.py`, `test_hysteresis_deep.py`

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
| **Hysteresis exit_n** | **50** (best, alpha +2.72%p, turnover -37%) | ✅ exit_50이 sweet |
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

### 운용 옵션 — 2축 구조

**축 1: PROFILE (model 비중, 위험도 결정)**

| Profile | model 비중 | TLT buffer | Sharpe | Alpha | MDD |
|---|---|---|---|---|---|
| **standard** ⭐ (default, 라이브) | 100% | 0% | 1.79 | **+34.6%p** | -20.9% |
| **low_risk** | 60% | 40% | **1.91** | +16.4%p | **-12.7%** |

→ `current_portfolio.py`의 `PROFILE` 변수 한 줄로 전환. 알파 추구냐 안정 추구냐만 결정.

**축 2: TOP_N + SECTOR_CAP (같은 model 내 변형, 선택사항)**

검증된 매트릭스에서 자유 선택. 같은 PROFILE 내에서 운용 변형 가능:

| 운용 변형 | Sharpe | Alpha | MDD | 메모 |
|---|---|---|---|---|
| **Top-20 No cap** ⭐ (default) | 1.79 | +34.6%p | -20.9% | 가장 단순, 라이브 |
| Top-15 Cap 20% | 1.84 | +37.7%p | -20.8% | Sharpe peak |
| Top-10 Cap 30% | 1.78 | +45.2%p | -22.6% | 최대 알파 |
| Top-20 Cap 15% | 1.82 | +31.7%p | -19.7% | 낮은 MDD |
| Top-15 Cap 15% | 1.84 | +34.9%p | -19.9% | Sharpe + MDD 양호 |

→ `current_portfolio.py`의 `TOP_N`, `SECTOR_CAP` 변수로 자유 변경.

**Fix 시점 (현재 default 변경 검토)**:
- 라이브 6개월 후 (실제 데이터 모이면)
- 매년 매트릭스 재검증 (universe/regime 변화 가능)

**선택 가이드 (시드 무관, 결정 기준 = Alpha > Sharpe > t-stat > MDD)**:
- 알파 우선 = **Standard + Top-20 No cap** (현재 라이브)
- 알파 + Sharpe 균형 = Standard + Top-15 Cap20%
- 안정 우선 = **Low-Risk** (단, alpha 절반 희생 — 장기 부 증식엔 부적합)

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
│   ├── test_weekly_walkforward.py   # Weekly walk-forward 2025-2026 ⭐
│   ├── test_effective_n.py          # Effective N 진단 (weekly)
│   ├── test_effective_n_yearly.py   # Effective N (yearly 31 windows) ⭐
│   ├── test_effective_n_baseline.py # vs random/full universe baseline ⭐
│   ├── test_etf_buffer.py           # SPY buffer (무가치 확정)
│   └── test_tlt_gld_buffer.py       # TLT/GLD buffer (안정형 옵션) ⭐
│
├── results/                    # 검증 결과
│   ├── current_portfolio.csv         # 최신 매수 추천
│   ├── ml_sp500_walkforward.csv      # 31 windows
│   ├── topn_cap_cross.txt            # v4 cross 매트릭스
│   ├── topn_cap_cross_3feature.txt   # v5 cross 매트릭스
│   ├── weekly_walkforward.txt        # Weekly walk-forward 결과
│   ├── weekly_walkforward.csv        # 주별 상세
│   ├── effective_n.txt / .csv        # Eff N (weekly)
│   ├── effective_n_yearly.txt / .csv # Eff N (yearly)
│   ├── effective_n_baseline.txt      # baseline 비교
│   ├── etf_buffer.txt                # SPY buffer
│   ├── tlt_gld_buffer.txt            # TLT/GLD buffer
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
PYTHONPATH=. python3 tests/test_effective_n_yearly.py          # Eff N 진단
PYTHONPATH=. python3 tests/test_effective_n_baseline.py        # vs baseline
PYTHONPATH=. python3 tests/test_tlt_gld_buffer.py              # TLT/GLD buffer
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
| ✅ **Hysteresis exit_50** | 알파 **+2.72%p**, turnover -37% (공짜 점심) |
| ⚠ **TLT/GLD buffer**: Sharpe ↑, alpha ↓ 트레이드오프 | 안정형 옵션. 장기 부 증식엔 부적합 (CAGR/alpha 큰 손실) |
| ✅ **Effective N 진단**: 우리 모델 baseline 이상 | 추가 cap 작업 불필요 확정 |
| ❌ Multi-horizon momentum (1m/3m/6m) | 알파 -1.3%p (redundant + recent regime) |
| ❌ Regime features (VIX, drawdown) | 효과 거의 없음 (linear model 한계) |
| ❌ **SPY buffer (같은 자산군)** | 모든 비율 Sharpe ↓ alpha ↓ |
| ❌ Stop-loss | 알파 -8%p (회복기 놓침) |
| ❌ Vol filter / Drawdown halt | 알파 -2~3%p |
| ❌ CASH 통합 | 효과 미미 |
| ❌ HP tuning | 효과 미미 |
| ❌ Ridge alpha tuning | alpha=1.0 default OK |
| ⚠ Survivorship bias | ~13% 부풀림 (Bootstrap, 3-feature) |

## 알려진 한계

- **Survivorship bias** 미해결 (S&P 500만 보유 → API 가입 시 가능)
- **비용 0.20% 이상**이면 알파 사라짐 → 미국 무수수료 broker 또는 토스증권 (외화통장) 필수
- **2024-2025 mega-cap 집중기** 알파 약함 (회복은 자연스러움)

## 다음 단계 후보

### Tier 1 — 미검증
- **Multi-horizon target** (5+10+20일 평균) — input 아닌 target 다양화 (~30분)

### Tier 2 — 큰 작업
- **OHLC 활용** (gap analysis, intraday vol) — 분봉 데이터 활용
- **Quality factor (ROE)** — yfinance 부분 가능

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
| ✅ Top-N × Cap cross | **완료** | Top-15 Cap15% / Top-20 Cap20% Sharpe 1.84 동률 |
| ✅ Effective N 진단 + baseline 비교 | **완료** | 우리 모델 baseline 이상, factor cap 추가 불필요 |
| ✅ **Hysteresis** | **채택** (exit_50) | 알파 +2.72%p, turnover -37% |
| ⚠ ETF buffer (TLT/GLD) | **트레이드오프 확인** | Sharpe 미세 ↑이지만 alpha -18%p. 장기 부 증식엔 부적합 |
| ❌ ETF buffer (SPY) | 모든 비율 손해 | 같은 자산군 (미국 주식) → 분산 효과 X |
| ❌ **Sector-relative score** | 알파 -4.5%p | Sharpe 1.85 ↑이지만 절대 알파 큰 손해 |
| ❌ **Ensemble (Ridge + LightGBM)** | 알파 -7%p | 3-feature에선 LGBM 약함 (Sharpe 1.43), 평균해도 Ridge 단독보다 ↓ |
| ❌ Stop-loss | 알파 -8%p | 회복기 놓침 |
| ❌ Vol filter / Drawdown halt | 알파 -2~3%p | 시장 타이밍 손해 |
| ❌ CASH 통합 | 효과 미미 | OOS에서 -5%p 이상 손해 |
| ❌ HP tuning (LightGBM) | 효과 미미 | grid search 성과 제한 |
| ❌ Ridge alpha tuning | 효과 미미 | alpha=1.0 default OK |

→ 자세한 검증 출처: `tests/` 폴더 + `결과는 results/`

## 개발 일지

> **버전 관리 규칙**: 메이저 정수만 표시 (rolling window: 현재 + 이전 1개). 소수점 minor 변경은 `CHANGELOG.md` 참조.

| 버전 | 환경 | 알파 | t-stat | Sharpe |
|---|---|---|---|---|
| v4 | Ridge + 7y + Weekly + Cap 옵션 (6-feature) | +31.03%p | 6.65 | 1.63 |
| **v5 (현재)** | **+ Feature ablation + Hysteresis + Leakage fix** | **+32.85%p** | **6.52** | **1.74** |

> v5는 **leakage 수정 반영**된 정정 수치. 이전 +34.16%p는 train target buffer 부재로 ~1.7%p 부풀림.

자세한 history (v1~v5.4): [CHANGELOG.md](CHANGELOG.md)
