# Module(상준) — Elastic Net / ANN 견해 모듈

FF3 Security Selection으로 고른 20종목에 대해 **매달 BL 견해(P, Q, Ω)를 만드는 모듈**입니다.
Elastic Net 버전과 ANN(신경망) 버전 두 가지가 있고, 예측 모델만 다르며 나머지는 모두 같습니다.
사후 기대수익률과 비중 계산은 BL 모듈에서 하며, 출력 형식은 근석님 `view_module.py`와 맞췄습니다.

> **Elastic Net은 비교 기준(baseline)용으로 만든 모델입니다.**
> Gu, Kelly & Xiu (2020)도 Elastic Net 같은 정규화 선형 모델을 기준으로 두고 트리·신경망 모델과 성과를 비교합니다.
> ANN이나 XGBoost같은 비선형 모델이 이 선형 기준보다 나은 견해를 만들어야
> "복잡한 ML 모델을 쓴 의미"가 있다고 볼 수 있습니다. 그래서 데이터·종목·특성·P·Ω·평가 구간을 모두 같게 두고
> 예측 모델만 바꿔서 비교할 수 있도록 만들었습니다.

---

## 1. 폴더 구성

| 파일 / 폴더 | 내용 |
|---|---|
| `view_module_ElasticNEt.py` | Elastic Net 견해 모듈 (ANN·XGBoost 등과 비교하기 위한 기준 모델) |
| `view_module_ANN.py` | ANN 견해 모듈 |
| `outputs_ElasticNet/P1`, `P2` | Elastic Net 결과 |
| `outputs_ANN/P1`, `P2` | ANN 결과 |
| `*.pdf` | 참고 논문 4편 (아래 2장) |

`P1` = 2000~2018 계속 상장 종목, `P2` = 2010~2018 계속 상장 종목입니다.

---

## 2. 참고 논문

| 역할 | 논문 | 이 모듈에서 가져온 것 |
|---|---|---|
| **Q** | Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies 33(5) | Elastic Net·신경망 모델 설정, 특성 정의, 횡단면 순위 스케일링, 시간 순서 검증, 1년마다 재학습 |
| **P** | Krauss, Do & Huck (2017), *Deep neural networks, gradient-boosted trees, random forests*, EJOR 259(2) | 예측 순위 상위 k 롱 / 하위 k 숏 구조 |
| 참고 | Fischer & Krauss (2018), *Deep Learning with LSTM Networks for Financial Market Predictions*, EJOR 270(2) | (현재 미사용. 시계열 신경망 비교용) |
| 참고 | Gu, Kelly & Xiu (2021), *Autoencoder Asset Pricing Models*, Journal of Econometrics 222(1) | (현재 미사용. 팩터 모델 확장용) |

Ω는 두 논문 모두 정의하지 않아서 직접 정했습니다 (3장).

---

## 3. 견해를 만드는 방법

### Q — 견해의 크기 (Gu, Kelly & Xiu)

1. **예측 모델**이 종목별 다음 달 수익률을 예측합니다.
   - **Elastic Net**: L1 비율 ρ = 0.5, 규제 강도 λ는 1e-4 ~ 1e-1 중에서 선택 (논문 Table A.5)
   - **ANN**: NN3 구조 (은닉층 32-16-8, 층마다 Batch Normalization + ReLU), 가중치 L1 벌점, Adam,
     조기 종료(5 epoch), 서로 다른 seed로 10개 학습 후 평균 (논문 Sec. 1.7)
2. **Q = P × 예측 수익률** (= 롱 5종목 평균 예측 − 숏 5종목 평균 예측, 월 수익률 단위)

**입력 특성 8개** (논문 특성 중 가격·거래량만으로 만들 수 있는 것)

| 특성 | 정의 |
|---|---|
| `mom1m` | t월 수익률 (단기 반전) |
| `mom6m` | t-5 ~ t-1월 누적수익률 |
| `mom12m` | t-11 ~ t-1월 누적수익률 |
| `chmom` | mom6m − (t-11 ~ t-6월 누적수익률) |
| `retvol` | 최근 12개월 월수익률 표준편차 |
| `dolvol` | log(수정주가 × 거래량) |
| `beta` | 최근 12개월 시장 베타 |
| `idiovol` | 위 회귀의 잔차 표준편차 |

- 매달 종목 간 순위로 바꿔 [-1, 1]에 맞추고, 결측은 0(횡단면 중앙값)으로 채웁니다 (논문과 동일).
- **학습 대상**: 해당 구간 계속 상장 종목 전체 (P1 1,577개, P2 2,422개). 예측은 선정된 20종목에 대해서만 씁니다.

### P — 무엇에 대한 견해인가 (Krauss et al.)

매달 20종목을 예측 수익률 순으로 정렬해 **상위 5종목 +0.2, 하위 5종목 −0.2, 나머지 0**을 줍니다.
견해는 1개(K = 1, 이름 `Top5_minus_Bottom5`)이고, 행 합은 0입니다.
**근석님 모듈과 달리 P가 매달 바뀝니다.**

### Ω — 견해를 얼마나 믿을지

**Ω = 최근 24개월 견해 예측 오차의 제곱 평균**, 즉 (실제 롱숏 스프레드 − Q)²의 평균입니다.
결정일 이전에 실현된 오차만 씁니다. 모델이 최근에 많이 틀렸을수록 Ω가 커져 BL이 견해를 덜 반영합니다.

---

## 4. 시간 구조 (근석님과 동일)

```
P1: 2013-01 ~ 2018-12  12개월마다 재학습하며 견해 생성 → 실제 결과와의 오차를 쌓음 (Ω 계산용)
    2019-01 ~ 2020-12  2018-12에 학습한 모델을 고정하고 매달 P, Q, Ω 생성 (평가 구간, 24개월)
P2: 오차를 쌓는 구간만 2015-01부터이고 나머지는 같음
```

- 결정월 t의 모델은 **t월까지 실현된 수익률만으로 학습**합니다 (특성은 t-1월까지, 타깃은 t월까지).
- 매 학습 때 학습 구간의 **마지막 12개월을 검증용**으로 떼어 하이퍼파라미터를 고릅니다 (시간 순서 유지).
  Elastic Net은 고른 뒤 전체 구간으로 다시 학습하고, ANN은 조기 종료에 검증 구간이 필요해서 다시 학습하지 않습니다 (논문과 동일).
- 2018-12-31에 만든 견해는 2019년 1월에 쓰이고, 마지막 2020-11-30 견해는 2020년 12월에 쓰입니다.

---

## 5. 실행 방법

### 설정 (각 파일 맨 위)

```python
USE_MAX_WEIGHT = True         # 종목 비중 상한 ON/OFF
MAX_WEIGHT = 0.20             # 종목당 최대 비중
LISTING_PERIOD = "2000~2018"  # "2000~2018" (P1) 또는 "2010~2018" (P2)
```

- `LISTING_PERIOD`를 바꾸면 종목 선정 파일과 ML 학습 대상 종목이 함께 바뀝니다.
- **비중 상한은 이 모듈이 직접 적용하지 않습니다.** 견해 모듈은 비중을 계산하지 않으므로,
  설정값을 `bl_input/constraints.json`에 저장해 두고 **BL 모듈이 읽어서 적용**합니다.

### 실행

```bash
pip install numpy pandas scikit-learn torch    # 개발 환경: numpy 1.26.4, pandas 2.3.3, scikit-learn 1.8.0, torch 2.14.0 (CPU)
python view_module_ElasticNEt.py    # 약 20초
python view_module_ANN.py           # 약 5~8분 (재학습마다 신경망 16개 학습)
```

- pandas는 2.1 이상이 필요합니다.
- 한 번 실행하면 `LISTING_PERIOD` 하나만 만듭니다. P1, P2를 모두 만들려면 설정을 바꿔 두 번 실행하세요.
- **데이터 경로**: 프로젝트 폴더 구조를 그대로 가정합니다. 이 폴더에서 두 단계 위(`IFE_term_project/`)에
  `Project_Data/IFE_term_stock_data.csv`, `Project_Data/ff3_factors.csv`,
  `Fama French 3 factor model security selection/ff3 2000~2018, 2010~2018/`이 있어야 합니다.
  Colab 등에서 구조가 다르면 파일 위쪽의 `STOCK_DATA`, `FF3_FACTORS`, `SELECTION_DIR`을 고치세요.

### 종목 선정

`ff3_selected_{기간}.csv`의 rank 상위 20종목을 씁니다. 같은 회사(COMNAM)가 여러 종목이면 거래량이 가장 큰 1종목만 남깁니다
(예: P2의 HEICO 2종목 → 1종목, 대신 21위 AVGO 포함). 근석님 모듈과 같은 20종목입니다.

---

## 6. 결과 파일

`outputs_{모델}/{P1|P2}/`

```
bl_input/                 ← BL 모듈 입력
  assets.csv              order, PERMNO, TICKER, COMNAM (20행). P 의 열 순서
  P.csv                   date, hold_until, view, <PERMNO 20개>   (24행, 날짜별 1행)
  Q.csv                   date, hold_until, Top5_minus_Bottom5     (24행, 월 수익률)
  Omega.csv               date, hold_until, Top5_minus_Bottom5|Top5_minus_Bottom5  (24행)
  constraints.json        {"use_max_weight": true, "max_weight": 0.2}
views_readable.csv        날짜별 롱·숏 종목, 예측 스프레드, 실제 스프레드, Ω (사람이 보는 용도)
diagnostics/
  views_timeseries.csv    오차 이력 구간 + 평가 구간 전체 (Q, 실제값, 오차, Ω, IC 등)
  model_fits.csv          재학습마다 고른 하이퍼파라미터 (ElasticNet은 계수도 포함)
  predicted_returns.csv   날짜 × 20종목 예측 수익률
  summary.json            구간별 요약 지표
```

> **근석님 형식과 다른 점:** `P.csv`가 1행 고정이 아니라 **날짜별 1행**입니다. BL 모듈에서 읽을 때 이 부분을 맞춰야 합니다.

### BL 모듈과 연결 (예시)

```python
from view_module_ANN import load_views        # ElasticNet도 같은 함수가 있음
views = load_views("outputs_ANN/P1")

for date, v in views.items():                 # "2018-12-31", ..., "2020-11-30"
    # v.assets (20,), v.P (1, 20), v.Q (1,), v.Omega (1, 1)
    # v.metadata["hold_until"], v.metadata["use_max_weight"], v.metadata["max_weight"]
    tS = tau * Sigma
    mu_bl = Pi + tS @ v.P.T @ np.linalg.solve(v.P @ tS @ v.P.T + v.Omega, v.Q - v.P @ Pi)
```

---

## 7. 현재 결과 (평가 구간 2019-01 ~ 2020-12, 24개월)

| | ElasticNet P1 | ANN P1 | ElasticNet P2 | ANN P2 |
|---|---|---|---|---|
| 견해 방향 적중률 | 42% | 62.5% | 46% | 54% |
| 20종목 순위 상관 (평균 IC) | −0.03 | +0.02 | −0.03 | −0.01 |
| 평균 예측 스프레드 Q (%/월) | 0.08 | 1.48 | 0.53 | 2.04 |
| 평균 실제 스프레드 (%/월) | −0.34 | 0.48 | −0.20 | 0.10 |
| 전체 학습 종목 대상 OOS R² | 1.2% | 0.7% | 0.9% | 0.1% |

- ANN의 적중률이 더 높지만 24개월 표본에서는 통계적으로 의미 있는 차이가 아닙니다 (P1 15/24 기준 p ≈ 0.15).
- 20종목 안에서 순위를 맞히는 능력(IC)은 두 모델 모두 0 근처입니다.
- **ANN은 Q를 크게 잡습니다** (예측 월 1.5~2% vs 실제 0.1~0.5%). BL 결과를 볼 때 견해가 과하게 반영되는지 확인이 필요합니다.
- **Elastic Net은 계수 대부분을 0으로 줄입니다.** P1의 2018-12 모델은 사실상 `dolvol`(거래대금) 하나만 써서,
  P가 "거래대금 작은 종목 롱 / 큰 종목 숏"과 비슷해집니다.

어느 모델이 나은지는 BL에 넣어 포트폴리오 성과로 비교해야 합니다.
XGBoost 등 다른 모델의 결과도 같은 표에 넣어 **Elastic Net 열을 기준으로 비교**하면 됩니다.

---

## 8. 검증한 것

- **룩어헤드 없음**: 특정 달 이후의 주가·거래량을 무작위로 망가뜨려 다시 실행했을 때,
  그 이전 견해가 완전히 같게 나오는지 확인했습니다 (2016-06 기준 43개월, 2019-06 기준 79개월 모두 일치).
  Elastic Net으로 확인했고, ANN은 같은 데이터·시간 처리 코드를 씁니다.
- 평가 24개월 모두 P의 행 합 0, 롱 5·숏 5이고, `load_views()`로 다시 읽힙니다.

---

## 9. 논문과 다른 점 · 한계

- **특성**: 논문은 94개 특성을 쓰지만, 원자료에 장부가치·발행주식수가 없어 가격·거래량 기반 8개만 씁니다.
  P2(2010년 시작)에서도 학습 기간을 확보하려고 창 길이를 최대 12개월로 줄였습니다 (36개월 모멘텀 제외, 베타 기간 단축).
  `retvol`은 논문의 일별 대신 월별 수익률로 계산합니다.
- **Elastic Net 손실 함수**: 논문은 Huber 손실이지만 일반 제곱 손실을 씁니다.
- **ANN 하이퍼파라미터**: 계산 시간을 줄이려고 신경망 1개로 (L1 벌점, 학습률)을 먼저 고른 뒤 그 설정으로 10개를 학습합니다.
- **Ω**: 논문에 없는 방식이며 직접 정했습니다. 근석님처럼 Idzorek 신뢰도 방식으로 바꿀 수 있습니다.
- **오차 이력 구간(2013~2018)은 완전한 표본 외가 아닙니다.** 20종목 자체를 2018년까지 데이터로 골랐고,
  학습 대상도 2018년까지 살아남은 종목뿐이라 이 구간으로 계산한 Ω는 낙관적일 수 있습니다.
  평가 구간(2019~2020)은 문제없습니다. 근석님 모듈도 같은 구조입니다.
- **재현성**: ANN은 같은 컴퓨터에서는 seed로 재현되지만, PyTorch 버전이나 CPU가 다르면 숫자가 약간 달라질 수 있습니다.
- **ANN과 ElasticNet 파일에 같은 코드가 들어 있습니다** (ANN 파일 단독 실행을 위해). 특성이나 Ω 계산을 바꾸면 두 파일을 모두 고쳐야 합니다.
- BL 모듈과의 연결은 아직 테스트하지 않았습니다.
