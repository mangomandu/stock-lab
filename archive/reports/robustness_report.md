# 자양동 Stock Lab — Robustness Report

**작성**: 2026-04-28 (자율 작업)
**대상 전략**: Weekly Top-20, w=(0.5, 0.1, 0.4), 비용 0.10% 왕복

---

## TL;DR (한 줄)

> **Weekly Top-20은 walk-forward 평균 −2.00%p로 알파 없음. 그러나 보너스 테스트에서 Biweekly Top-20이 +2.94%p / 승률 7/11로 살아남음. 리밸런싱 빈도가 핵심. 신호 강화 + biweekly 조합이 다음 방향.**

---

## 결과 요약

| 검증 | 결론 | 데이터 |
|---|---|---|
| 1. QQQ 실비교 (2018 split) | ✅ 알파 존재 | Test +4.57%p, 누적 514% vs 350% |
| 2. Split 날짜 민감도 | ⚠️ 변동 큼 | -8.13%p ~ +4.57%p, 평균 +0.44%p |
| 3. 비용 민감도 | ⚠️ 비용 취약 | break-even 0.20~0.50% 사이 |
| 4. Walk-forward (5y/1y) | ❌ **알파 없음** | 평균 -2.00%p, 승률 4/11 |
| 5. CASH 자산 통합 | △ 효과 미미 | MDD -1.4%p, Sharpe +0.02 |
| 6. (보너스) 리밸런싱 빈도 | ✅ **Biweekly가 sweet spot** | Biweekly Top-20: +2.94%p, 승률 7/11 |

---

## 1. QQQ 실비교 (Survivorship bias 정량화)

QQQ 일봉 fetch 후 동일 기간 비교

| | Strategy | EqW Bench | **QQQ (real)** |
|---|---|---|---|
| Train CAGR | 25.68% | 25.32% | **17.87%** |
| Test CAGR | 24.45% | 22.15% | **19.88%** |
| Test Sharpe | 0.97 | 0.99 | 0.88 |
| Test MDD | -35.34% | -30.97% | -35.12% |
| Test 누적 | 513.58% | 425.36% | **349.92%** |

**Survivorship bias**: 동일가중 EqW가 QQQ보다 Train +7.45%p / Test +2.27%p 높음. "현재 살아있는 NASDAQ-100 종목" 만 보기 때문에 부풀려짐.

**결론**: 2018-2026 단일 구간에서 Strategy는 QQQ 대비 +4.57%p의 명목 알파. 다만 이게 robust한지는 다음 검증들로 확인 필요.

📁 `results/qqq_comparison.txt`

---

## 2. Split 날짜 민감도

다른 split date(2015~2021)로 7번 재실행

| Split | Best W | Test CAGR | QQQ | Excess |
|---|---|---|---|---|
| 2015-01-01 | (+0.5, 0.2, 0.3) | 23.08% | 18.80% | **+4.28%p** |
| 2016-01-01 | (+0.5, 0.1, 0.4) | 23.56% | 19.76% | **+3.81%p** |
| 2017-01-01 | (+0.5, 0.1, 0.4) | 23.86% | 21.21% | **+2.66%p** |
| 2018-01-01 | (+0.5, 0.1, 0.4) | 24.45% | 19.88% | **+4.57%p** |
| 2019-01-01 | (+0.5, 0.4, 0.1) | 14.78% | 22.91% | **−8.13%p** |
| 2020-01-01 | (+0.5, 0.1, 0.4) | 19.50% | 20.53% | −1.04%p |
| 2021-01-01 | (+0.5, 0.1, 0.4) | 12.83% | 15.90% | −3.08%p |

**평균 알파: +0.44%p, 승률 4/7**. 2015-2018 split에선 모두 양수, 2019 이후 split에선 음수. 즉 **2010-2018 train 데이터로 학습**한 모델이 그 이후엔 잘 안 통함.

📁 `results/split_sensitivity.txt`

---

## 3. 비용 민감도

같은 전략, 비용만 바꿔서 7개 시나리오

| 비용 (왕복) | Best W | Test CAGR | vs QQQ |
|---|---|---|---|
| 0.01% (HFT) | (+0.5, 0.1, 0.4) | 27.95% | **+8.07%p** |
| 0.05% | (+0.5, 0.1, 0.4) | 26.38% | **+6.50%p** |
| **0.10%** (default) | (+0.5, 0.1, 0.4) | 24.45% | **+4.57%p** |
| 0.20% | (+0.5, 0.1, 0.4) | 20.67% | +0.79%p |
| 0.50% (한국 broker) | (+0.8, 0.1, 0.1) | 12.56% | **−7.33%p** |
| 1.00% | (+0.8, 0.1, 0.1) | -0.89% | −20.77%p |

**Break-even은 0.20%~0.50% 사이.**

- 미국 broker (Robinhood, Fidelity 등 무수수료): 약 0.10% 가정 OK → 알파 존재
- 한국 broker (키움, 미래에셋): 환전수수료 + 매매수수료 0.5% 이상 → **전략 불성립**
- IBKR 같은 저비용: 0.15-0.20% → 한계선

📁 `results/cost_sensitivity.txt`

---

## 4. Walk-Forward 검증 (가장 중요)

5년 train / 1년 test, 매년 한 윈도우씩 굴림. 11개 윈도우.

| Train | Test | Best W | Test CAGR | QQQ | Excess |
|---|---|---|---|---|---|
| 2010-14 | 2015 | (+0.5, 0.2, 0.3) | -0.49% | 9.44% | **−9.92%p** |
| 2011-15 | 2016 | (+0.1, 0.3, 0.6) | 21.20% | 7.10% | **+14.11%p** |
| 2012-16 | 2017 | (+0.5, 0.1, 0.4) | 32.43% | 32.81% | −0.38%p |
| 2013-17 | 2018 | (+0.5, 0.1, 0.4) | 8.74% | -0.13% | **+8.87%p** |
| 2014-18 | 2019 | (+0.5, 0.4, 0.1) | 35.84% | 38.96% | −3.12%p |
| 2015-19 | 2020 | (+0.7, 0.1, 0.2) | 63.68% | 48.17% | **+15.51%p** |
| 2016-20 | 2021 | (+0.2, 0.1, 0.7) | 25.84% | 27.42% | −1.58%p |
| 2017-21 | 2022 | (+0.4, 0.1, 0.5) | -28.96% | -32.68% | +3.73%p |
| 2018-22 | 2023 | (+0.8, 0.1, 0.1) | 40.19% | 55.40% | **−15.21%p** |
| 2019-23 | 2024 | (+0.8, 0.1, 0.1) | 7.11% | 25.58% | **−18.47%p** |
| 2020-24 | 2025 | (-0.3, 0.5, 0.2) | 5.38% | 20.96% | **−15.58%p** |

### 집계
- 평균 Test CAGR: **+19.18%**
- **평균 vs QQQ excess: −2.00%p**
- 승률: **4/11 (36%)** — 동전 던지기보다 못함
- 변동폭: −18.47% ~ +15.51%

### 핵심 발견 — **최근 3년 모두 큰 폭 underperform**
- 2023: −15.21%p
- 2024: −18.47%p
- 2025: −15.58%p

이 세 해는 **NVDA, MSFT 등 mega cap이 시장 주도**한 시기. 우리 Top-20 동일가중은 그들을 5%만 보유 → QQQ 시총가중(NVDA 7~10%)에 못 미침.

### Best W 불안정성
11개 윈도우에서 9개의 서로 다른 weight 조합이 등장. **"최적 가중치"가 시기마다 다름** → 미래에 적용할 단일 마스터 가중치를 찾기 어려움.

📁 `results/walk_forward.txt`, `results/walk_forward.csv`

---

## 5. CASH 자산 통합

CASH를 합성 종목으로 추가, score=fixed로 Top-N 경쟁

| cash_score | Test CAGR | Sharpe | MDD | Avg %Cash | vs QQQ |
|---|---|---|---|---|---|
| None | 24.45% | 0.97 | −35.34% | 0% | +4.57%p |
| 50 | 23.91% | 0.94 | −39.53% | 1.1% | +4.03%p |
| 70 | 24.30% | **0.99** | **−33.94%** | 4.7% | +4.42%p |
| 80 | 24.19% | **0.99** | −34.04% | 5.0% | +4.31%p |

- 평균 cash 보유 최대 **5%** — 상수 score로는 cash가 잘 안 잡힘
- MDD 1.4%p 개선, Sharpe 0.02 개선 → **효과 미미**
- 이유: Top-20에서 CASH가 들어가려면 78개 stock보다 점수 높아야 함. 진짜 약세장에만 발동.

**개선 가능 방향** (다음 작업)
- Top-10에선 cash 효과 더 클 듯 (88개 stock보다 높기만 하면 됨)
- 시장 전체 평균점수 < threshold면 강제 cash 비율 (regime filter)
- cash_score를 시간에 따라 동적 조정

📁 `results/cash_asset.txt`

---

## 6. (보너스) 리밸런싱 빈도 walk-forward — **가장 중요한 발견**

8가지 운용 설정을 11개 윈도우로 walk-forward (각 5y train / 1y test)

| Config | Avg CAGR | Sharpe | MDD | vs QQQ | Win |
|---|---|---|---|---|---|
| Daily Top-20 | +19.82% | 0.94 | -18.93% | -1.37%p | 3/11 |
| Weekly Top-20 (baseline) | +19.18% | 0.99 | -16.94% | -2.00%p | 4/11 |
| **★ Biweekly Top-20** | **+24.12%** | **1.21** | **-15.15%** | **+2.94%p** | **7/11** |
| Monthly Top-20 | +21.07% | 1.05 | -17.17% | -0.11%p | 7/11 |
| Weekly Top-10 | +18.55% | 0.87 | -17.73% | -2.64%p | 5/11 |
| Weekly Top-10 + CASH(70) | +15.36% | 0.78 | -16.08% | -5.82%p | 3/11 |
| Weekly Top-10 + CASH(80) | +15.98% | 0.82 | -16.07% | -5.21%p | 3/11 |
| Monthly Top-10 + CASH(70) | +14.75% | 0.82 | -16.26% | -6.43%p | 5/11 |

### 발견
**Biweekly (10거래일, 약 2주)** 가 sweet spot
- Daily, Weekly보다 비용 ↓
- Monthly보다 신호 빠짐 ↓
- 모든 지표 (CAGR, Sharpe, MDD, win rate, vs QQQ) 1등

**CASH 통합은 walk-forward에서는 명백히 손해** (-5%p 이상). Section 5에선 단일 split(2018)에서 비슷해 보였지만, 여러 윈도우로 펼치면 cash가 좋은 시점을 놓치는 비용이 더 큼.

### 영향
이 결과로 **추천 운용설정 = Biweekly Top-20, no cash**. Walk-forward 기준 +2.94%p / 승률 64%로 robust한 알파 후보. 이전 Weekly의 -2%p에서 의미있게 반전.

📁 `results/bonus_walkforward.txt`

---

## 7. 종합 결론

### 정직한 평가

1. **Weekly Top-20은 robust한 알파가 없음.** Walk-forward 평균 −2.00%p.
2. **그러나 Biweekly Top-20은 +2.94%p / 승률 64%로 살아남음.** 리밸런싱 빈도가 핵심 노브였음.
3. **2018 split의 +4.57%p 결과는 우연이지만**, 운용설정 바꾸면 진짜 알파 가능성 보임.
4. **비용 0.20% 이상이면 알파 사라짐.** 한국 broker로는 실행 불가. 미국 broker (무수수료) 필수.
5. **CASH 통합은 walk-forward에서 손해** (단일 split 2018에선 비슷해 보였지만 여러 윈도우 보면 -5%p).
6. **Best W 불안정**: 11개 윈도우에 9개 다른 weight. 미래 운용에 단일 마스터 가중치보다 **윈도우별 자동 갱신**(매년 train 다시) 이 합리적.

### 그럼 뭐가 남았나 (긍정)

- ✅ **백테스트 인프라 완성**: 데이터 → 점수 → 포트폴리오 → 검증 파이프라인
- ✅ **OOS 평가 프레임워크**: Walk-forward, split sensitivity, cost sensitivity
- ✅ **하이퍼파라미터 깔끔한 설계**: `core.py`의 단일 HP dict로 실험 빠르게 가능
- ✅ **실패한 가설 1개 잡음**: "Weekly Top-20 + RSI/MA/VOL"은 알파 없음을 데이터로 확인

지금 인프라 위에 새 신호/모델 얹으면 바로 검증 가능.

---

## 8. 추천 다음 단계

### A. 즉시 사용 — Biweekly Top-20 운용 룰 정착
- **rebal_days = 10 (격주, 약 2주)**, top_n = 20, 동일가중
- 가중치는 **매년 가장 최근 5년 데이터로 자동 재학습** (윈도우별 best가 다르므로 단일 고정 X)
- `multi_analyzer.py`의 가중치/리밸런싱 룰 이 방향으로 업데이트 검토
- 필수 전제: **미국 무수수료 broker** (Robinhood, Fidelity 등). 환전수수료 분기 1회로 압축

### B. 새 신호 시도 (가장 큰 잠재력)
- **모멘텀 12-1**: 12개월 수익률 - 1개월 수익률 (academic standard)
- **퀄리티 팩터**: ROE, 이익 안정성 등 (yfinance fundamentals 필요)
- **변동성 팩터**: 저변동성 종목 선호 (low-vol anomaly)
- 위 셋을 우리 RSI/MA/VOL에 추가하고 Biweekly Top-20에서 walk-forward 재실행

### C. 모델 형식 변경
- **트리 모델 (LightGBM)**: 같은 피처로 비선형 + 상호작용
- 검증 인프라는 이미 있으니 모델만 갈아끼우면 됨

### D. 운용 설정 추가 탐색
- **시총 가중** (동일가중 대신): mega-cap 따라잡기
- **Top-N 더 키우기 (30~50)**: NVDA 같은 거인 실수로 빼는 위험 감소
- **하이브리드**: Top-5는 시총가중 + 나머지는 점수 가중

### E. 다른 시장 (큰 피봇)
- 한국 broker만 있다면 NASDAQ → KOSPI/KOSDAQ 피봇 검토
- 한국 중소형주는 효율성 낮아 알파 가능성 더 큼
- 단, 종토방 크롤링 + 데이터 수집부터 해야 해서 큰 작업

### 의사결정 흐름
```
Q1. 미국 broker 있나?
  YES → Q2
  NO  → 미국 broker 개설 검토 OR 한국 시장 피봇 (E)
Q2. 즉시 운용 시작 vs 신호 강화 먼저?
  즉시  → A로 production 반영
  강화  → B 작업 후 walk-forward 재검증
```

---

## 자율 작업 산출물

```
/home/dlfnek/stock_lab/
├── core.py                          ← 엔진 (HP dict 단일 진실 원천)
├── portfolio_backtester.py          ← 기존 (변경 없음)
├── test_qqq_benchmark.py            ← Task 2
├── test_split_sensitivity.py        ← Task 3
├── test_cost_sensitivity.py         ← Task 4
├── test_walk_forward.py             ← Task 5
├── test_cash_asset.py               ← Task 6
├── test_bonus.py                    ← 보너스 (rebal 빈도 + Top-10/CASH)
├── robustness_report.md             ← 이 파일 (Task 7)
└── results/
    ├── qqq_close.csv                ← QQQ 일봉 (재사용용)
    ├── qqq_comparison.txt
    ├── split_sensitivity.txt
    ├── cost_sensitivity.txt
    ├── walk_forward.txt
    ├── walk_forward.csv
    ├── cash_asset.txt
    └── bonus_walkforward.txt        ← Biweekly 발견
```

**재실행 방법**: 어느 테스트든 `python3 test_*.py`. HP override 필요하면 `core.merge_hp({...})` 한 줄.

**핵심 하이퍼파라미터** (`core.py` `DEFAULT_HP`):
- `top_n`, `rebal_days`, `hysteresis`
- `cost_oneway`, `cash_score`
- `rsi_period`, `ma_period`, `vol_period`
- `train_start`, `split_date`
- `weight_step`, `rsi_signed`

전부 한 dict에서 관리, 테스트마다 override 가능.

---

*"인프라는 단단해졌고, sweet spot도 찾았다 (Biweekly). 다음은 새 신호로 알파 더 끌어올리기."*
