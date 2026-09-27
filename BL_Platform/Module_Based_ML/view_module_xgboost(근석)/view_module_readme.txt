View 모듈 (view_module.py)
==========================

이 파일은 ML-Enhanced Black-Litterman 프로젝트에서 BL에 들어갈 견해(view),
즉 P, Q, Ω를 만드는 코드입니다. FF3 Security Selection에서 고른 20종목을 받아
매달 "어떤 종목 묶음이 다른 묶음보다 얼마나 더 오를지"를 예측하고, 그 예측을
얼마나 믿을지까지 계산해서 BL 모듈이 바로 쓸 수 있는 형태로 저장합니다.
사후 기대수익률과 비중 계산은 BL 모듈에서 합니다.


1. 참고한 논문
--------------

P  Ko, Son & Lee (2024), "A novel integration of the Fama-French and
   Black-Litterman models to enhance portfolio management", JIFMIM 91.
Q  Barua & Sharma (2023), "Using fear, greed and machine learning for optimizing
   global portfolios: A Black-Litterman approach", Finance Research Letters 58.
Ω  Spears, Zohren & Roberts (2023), "View fusion vis-a-vis a Bayesian
   interpretation of Black-Litterman for portfolio allocation", JFDS 5(3).


2. 견해를 만드는 방법
--------------------

P - 무엇에 대한 견해인가 (Ko et al.)

  Ko et al.은 종목을 size로 먼저 나누고, 그 안에서 B/M으로 다시 나눈 뒤
  "소형·가치주 묶음이 대형·성장주 묶음보다 더 오른다"는 견해 하나를 BL에 넣습니다.
  저희도 같은 구조를 쓰되, 원자료에 시가총액과 장부가치가 없어서 FF3 회귀 계수로
  대신했습니다. s_SMB가 클수록 소형주처럼, h_HML이 클수록 가치주처럼 움직이는
  종목으로 봅니다.

  20종목을 s_SMB 기준으로 10개씩 나누고, 각 그룹을 h_HML 기준으로 다시 5개씩
  나눕니다. 소형·가치 칸 5종목에 +1/5, 대형·성장 칸 5종목에 -1/5를 주고 나머지
  10종목은 0입니다. 견해는 하나(K = 1)이고, P는 평가 기간 동안 바뀌지 않습니다.

    P1 예시  롱: MKC EXPO SHW AZO CNI   숏: MNST UGI CHD NEE ROST
    P2 예시  롱: MKTX EXPO WDFC TTC POOL  숏: DPZ NI TDG CMS NEE

  원 논문은 5x5로 나누지만 20종목으로는 칸이 비어서 2x2를 썼습니다.
  "소형·가치"는 20종목 안에서의 상대적인 구분입니다.

Q - 견해의 크기 (Barua & Sharma)

  XGBoost로 종목별 다음 달 수익률을 예측한 뒤, 롱 5종목 평균에서 숏 5종목 평균을
  뺀 값을 Q로 씁니다 (Q = P x 예측수익률). 모델 설정과 입력 지표는 논문을 따랐습니다.

    - 트리 300개, 나머지 하이퍼파라미터는 기본값
    - 입력: 이동평균 비율 9개(1·2·3개월 / 6·9·12개월) + 모멘텀 5개(1·3·6·9·12개월)
    - 각 지표는 매달 종목 간 순위로 바꿔 [-1, 1] 범위에 맞춤
    - 학습 대상: 해당 구간에 계속 상장된 종목 전체 (P1 1,577개, P2 2,422개)

  월 수익률 단위라서 Q = 0.004면 롱 묶음이 한 달에 0.4%p 더 오른다는 뜻입니다.

Ω - 견해를 얼마나 믿을지 (Barua & Sharma의 Idzorek 방식, Spears et al.의 개념)

  Spears et al.은 Ω에 "모델이 얼마나 틀릴 수 있는지"만 넣어야 한다고 봅니다.
  저희는 Barua & Sharma처럼 과거 예측 성적(OOS R²)을 신뢰도 C로 쓰는 Idzorek
  방식을 택했습니다.

    C = 최근 24개월 견해 예측의 R² (0.01 ~ 0.9로 제한)
    Ω = τ · pΣpᵀ · (1 - C) / C,   τ = 0.1,  Σ = 최근 60개월 표본공분산

  이렇게 두면 BL 사후 기대수익률이 견해 쪽으로 정확히 C만큼 움직입니다.
  과거에 잘 맞힌 만큼 반영하는 방식입니다.
  비교용으로 Spears et al.의 앙상블 분산 방식도 코드에 옵션으로 들어 있습니다.


3. 시간 구조
------------

  선정·학습은 2018년까지, 평가는 2019~2020년입니다 (회의 내용 바탕).

    P1: 2013-01 ~ 2018-11  12개월마다 재학습하면서 예측 → 실제 결과와의 오차를 쌓음
        2018-12 ~ 2020-11  2018-12에 학습한 모델을 고정하고 매달 P, Q, Ω 생성
    P2: 오차를 쌓는 구간만 2015-01부터이고 나머지는 같습니다.

  P2가 2015-01부터인 이유: P2는 2010년 이후 데이터만 쓰고, 12개월 지표 때문에 학습은
  2011년부터 가능합니다. 2013년에 시작하면 첫 모델이 2년치로만 학습되므로, 최소 4년
  (2011~2014)을 학습한 뒤 시작하도록 정했습니다. 이렇게 해도 평가 시작(2018-12) 전에
  오차가 47개월 쌓여 신뢰도 C(최근 24개월)를 계산하는 데 충분합니다.
  이 시작 시점은 논문에 정해진 값이 아니라 구현상의 선택입니다.

  평가 구간에서는 매달 말에 다음 달 견해를 하나씩 만듭니다. 2018-12-31에 만든 견해는
  2019년 1월에 쓰이고, 마지막인 2020-11-30 견해는 2020년 12월에 쓰입니다.
  그래서 P1, P2 각각 24개월치(2019-01 ~ 2020-12) 견해가 나옵니다.
  모든 견해는 만든 날까지의 데이터만 사용합니다.


4. 파일 구성
------------

  view_module.py 한 파일 안에 아래 순서로 들어 있습니다.

    1~3    버전 확인, raw data 읽기, FF3 선정 파일 읽기 (같은 회사는 거래량 큰 1종목만)
    4      ViewResult: 한 시점의 P, Q, Ω를 담는 구조
    5      입력 지표 14개 계산
    6      P 만들기 (이중정렬)
    7~8    XGBoost 예측, Q 계산
    9      Ω 계산 (R² 신뢰도 / 앙상블 분산 / 오차분산)
    10~11  검증 후 한 달치 결과 조립
    12     load_views: 저장된 결과를 BL에서 읽는 함수
    13     전체 기간 반복과 파일 저장
    14~15  P1·P2 설정값, 실행 부분


5. 실행 방법
------------

  view_module.py와 같은 폴더에 IFE_term_stock_data.csv와 FF3 결과 zip(풀지 않아도 됨)을
  두고 실행하면, P1·P2 결과가 outputs/P1/, outputs/P2/에 저장됩니다.

    pip install numpy==2.2.6 pandas==2.2.3 xgboost==3.0.5 scikit-learn==1.6.1 pyyaml==6.0.2
    python view_module.py

  하나만 다시 만들 때는 python view_module.py --period 1 (P1) 또는 --period 2 (P2)로
  실행합니다. Colab에서는 명령 앞에 !를 붙여 셀에서 실행하면 됩니다.
  
6. 결과 파일
------------

  outputs/P1/bl_input/  (P2도 같음, CSV 5개)
    assets.csv       order, PERMNO, TICKER, COMNAM (20행). 이 순서가 P의 열 순서이고,
                     BL의 Σ와 Π도 이 순서로 맞춰야 합니다.
    P.csv            견해 행렬. 행 1개(SmallValue_minus_BigGrowth) x 열 20개(PERMNO).
                     롱 5종목 0.2, 숏 5종목 -0.2, 나머지 0 (행 합 0).
    Q.csv            date, hold_until, SmallValue_minus_BigGrowth   (24행, 월 수익률)
    Omega.csv        date, hold_until, SmallValue_minus_BigGrowth|SmallValue_minus_BigGrowth
                     (24행, 견해가 1개라 Ω도 값 1개)
    confidence.csv   date, hold_until, SmallValue_minus_BigGrowth   (24행, 신뢰도 C)

    예) Q.csv 첫 줄     2018-12-31, 2019-01-31, -0.00139
        Omega.csv 첫 줄 2018-12-31, 2019-01-31,  0.00137
        confidence 첫 줄 2018-12-31, 2019-01-31,  0.0704

  date는 리밸런싱 날(원자료 날짜 그대로, 예: 2018-12-31)이고, hold_until은 다음
  리밸런싱 날입니다. views_readable.csv에서 날짜별 롱·숏 종목, 예측값, 실제값,
  신뢰도를 한 번에 볼 수 있습니다.


7. BL 모듈과 연결(예시)
-----------------

  import view_module as vm
  views = vm.load_views("outputs/P1", bl_tau=0.1)

  for date, v in views.items():         # "2018-12-31", ..., "2020-11-30"
      # v.assets (20,), v.P (1, 20), v.Q (1,), v.Omega (1, 1)
      # v.metadata["hold_until"] 까지 보유
      # BL 쪽에서 Sigma, Pi, tau 를 정한 뒤:
      tS = tau * Sigma
      mu_bl = Pi + tS @ v.P.T @ np.linalg.solve(v.P @ tS @ v.P.T + v.Omega, v.Q - v.P @ Pi)


8. 참고할 점
------------

  - 신뢰도 C는 매달 과거 24개월 성적으로 자동 조정됩니다. 평가 구간 평균은 P1 약 2%,
    P2 약 7%로, BL이 견해를 그만큼 반영하면서 사전분포(동일가중)를 기준으로 삼습니다.
 - Ω는 τ = 0.1, Σ = 최근 60개월 표본공분산 기준이라 BL에서도 같게 쓰면 신뢰도 C만큼 반영되고, 다르게 쓰면 load_views의 bl_tau, bl_sigma 옵션을 쓰시면 됩니다.

