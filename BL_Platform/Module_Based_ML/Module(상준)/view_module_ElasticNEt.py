"""Elastic Net 견해 모듈 (P, Q, Omega)

FF3 Security Selection 으로 고른 20종목에 대해 매달 BL 견해를 만듭니다.
사후 기대수익률과 비중 계산은 BL 모듈에서 합니다. (출력 형식은 근석님 view_module.py 와 맞춤)

  Q  Gu, Kelly & Xiu (2020, RFS) — Elastic Net 으로 종목별 다음 달 수익률 예측
  P  Krauss, Do & Huck (2017, EJOR) — 예측 순위 상위 k 롱 / 하위 k 숏
  Omega  최근 24개월 견해 예측 오차의 평균제곱 (모델이 얼마나 틀렸는지)

실행:  python view_module_ElasticNEt.py      (결과: outputs_ElasticNet/P1 또는 outputs_ElasticNet/P2)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet


# =============================================================================
# 0. 사용자 설정
# =============================================================================
USE_MAX_WEIGHT = True        # 종목 비중 상한 ON/OFF
MAX_WEIGHT = 0.20            # 종목당 최대 비중 (BL 모듈이 bl_input/constraints.json 으로 읽음)

LISTING_PERIOD = "2000~2018"  # "2000~2018": 2000~2018 계속 상장 종목 (P1)
                              # "2010~2018": 2010~2018 계속 상장 종목 (P2)
                              # 종목 선정 파일과 ML 학습 대상 종목 모두 이 구간을 따릅니다.

# 경로 (이 파일 기준: Module_Based_ML/Module(상준)/ -> 프로젝트 루트는 두 단계 위)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STOCK_DATA = os.path.join(PROJECT_ROOT, "Project_Data", "IFE_term_stock_data.csv")
FF3_FACTORS = os.path.join(PROJECT_ROOT, "Project_Data", "ff3_factors.csv")
SELECTION_DIR = os.path.join(PROJECT_ROOT, "Fama French 3 factor model security selection",
                             "ff3 2000~2018, 2010~2018")
OUTPUT_ROOT = os.path.join(HERE, "outputs_ElasticNet")

PERIODS = {
    #              결과 폴더,   데이터 시작,           walk-forward 오차 이력 시작 (근석 님과 동일)
    "2000~2018": dict(name="P1", data_start="2000-01", oos_history_start="2013-01"),
    "2010~2018": dict(name="P2", data_start="2010-01", oos_history_start="2015-01"),
}
N_ASSETS = 20
TRAIN_END = "2018-12"        # 선정·학습은 여기까지 (마지막 학습 타깃 월)
EVAL_END = "2020-12"         # 평가: 2019-01 ~ 2020-12 보유월

N_LONG_SHORT = 5             # P: 롱 5 / 숏 5 (Krauss et al. 의 top-k/flop-k 를 20종목에 맞춤)
VIEW_NAME = f"Top{N_LONG_SHORT}_minus_Bottom{N_LONG_SHORT}"

ENET_L1_RATIO = 0.5                       # Gu et al. Table A.5: rho = 0.5
ENET_ALPHAS = np.logspace(-4, -1, 10)     # Gu et al. Table A.5: lambda in (1e-4, 1e-1)
VAL_MONTHS = 12                           # 학습 구간 마지막 12개월 = 검증 (시간 순서 유지)
REFIT_EVERY = 12                          # Gu et al.: 1년마다 재학습

OMEGA_WINDOW = 24
OMEGA_MIN_OBS = 12
OMEGA_FLOOR = 1e-6

VALID_EXCHCD = (1, 2, 3)
RET_CAP = 5.0                # 월 500% 초과 수익률은 자료 오류로 보고 제외 (근석님과 동일)


# =============================================================================
# 1. 데이터 로딩
# =============================================================================
def load_selection(period: str) -> pd.DataFrame:
    """rank 상위 N_ASSETS 종목. 같은 회사(COMNAM)는 거래량(avg_vol)이 가장 큰 1종목만."""
    sel = pd.read_csv(os.path.join(SELECTION_DIR, f"ff3_selected_{period}.csv"))
    sel = sel.sort_values(["avg_vol", "rank"], ascending=[False, True]).drop_duplicates("COMNAM")
    return sel.sort_values("rank").head(N_ASSETS).reset_index(drop=True)


def load_universe(period: str) -> list:
    """해당 구간 계속 상장 종목 (ML 학습용 횡단면)."""
    return sorted(pd.read_csv(os.path.join(SELECTION_DIR, f"ff3_regression_{period}.csv"),
                              usecols=["PERMNO"])["PERMNO"].unique())


def load_market(permnos, start: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """wide 행렬 (index = 월 Period, columns = PERMNO): 수익률, 수정주가, 거래량."""
    raw = pd.read_csv(STOCK_DATA, parse_dates=["date"])
    raw = raw[raw["PERMNO"].isin(permnos) & raw["EXCHCD"].isin(VALID_EXCHCD)
              & (raw["ADJ_PRC"] > 0)].drop_duplicates(["PERMNO", "date"])
    raw["ym"] = raw["date"].dt.to_period("M")
    raw = raw[raw["ym"] >= pd.Period(start, "M")]
    price = raw.pivot(index="ym", columns="PERMNO", values="ADJ_PRC").sort_index()
    volume = raw.pivot(index="ym", columns="PERMNO", values="VOL").sort_index()
    full = pd.period_range(price.index.min(), price.index.max(), freq="M")
    price, volume = price.reindex(full), volume.reindex(full)   # 빠진 달은 NaN -> 건너뛴 수익률을 만들지 않음
    ret = price / price.shift(1) - 1.0
    ret = ret.mask(ret > RET_CAP)
    return ret, price, volume


def load_factors() -> pd.DataFrame:
    ff = pd.read_csv(FF3_FACTORS, parse_dates=["date"])
    ff.index = ff["date"].dt.to_period("M")
    return ff[["Mkt_RF", "RF"]]


# =============================================================================
# 2. 특성 (Gu, Kelly & Xiu 2020, Table A.6 중 가격·거래량으로 만들 수 있는 것)
# =============================================================================
# 원자료에 장부가치·발행주식수가 없어서 가격·거래량 기반 특성만 씁니다 (논문에서도 이 계열이 가장 중요).
# 모든 특성은 결정월 t 말까지의 정보만 씁니다. 창 길이는 P2(2010년 시작)에서도 학습 기간이
# 확보되도록 최대 12개월로 두었습니다 (논문의 mom36m, 3년 beta 는 제외·단축).
#   mom1m    t 월 수익률 (단기 반전)
#   mom6m    t-5 ~ t-1 누적수익률
#   mom12m   t-11 ~ t-1 누적수익률
#   chmom    mom6m - (t-11 ~ t-6 누적수익률)
#   retvol   최근 12개월 월수익률 표준편차 (논문은 일별; 월별로 대체)
#   dolvol   log(수정주가 x 거래량)
#   beta     최근 12개월 시장 베타 (초과수익률 기준)
#   idiovol  위 회귀의 잔차 표준편차
# 스케일: 논문과 같이 매달 횡단면 순위를 [-1, 1] 로 바꾸고, 결측은 횡단면 중앙값(0)으로 채웁니다.
FEATURES = ["mom1m", "mom6m", "mom12m", "chmom", "retvol", "dolvol", "beta", "idiovol"]


def _cum(ret: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """t-first ~ t-last 누적수익률 (first >= last). 한 달이라도 없으면 NaN."""
    g = 1.0 + ret
    out = g.shift(last)
    for j in range(last + 1, first + 1):
        out = out * g.shift(j)
    return out - 1.0


def build_panel(ret, price, volume, factors) -> pd.DataFrame:
    """(ym, PERMNO) 인덱스의 long 패널: 순위 특성 8개 + target(t+1 수익률) + has_history."""
    f = factors.reindex(ret.index)
    mkt, ex = f["Mkt_RF"], ret.sub(f["RF"], axis=0)
    roll = lambda x: x.rolling(12, min_periods=12).mean()
    var_m = roll(mkt ** 2) - roll(mkt) ** 2
    beta = (roll(ex.mul(mkt, axis=0)) - roll(ex).mul(roll(mkt), axis=0)).div(var_m, axis=0)
    var_ex = roll(ex ** 2) - roll(ex) ** 2
    raw = {
        "mom1m": ret,
        "mom6m": _cum(ret, 5, 1),
        "mom12m": _cum(ret, 11, 1),
        "chmom": _cum(ret, 5, 1) - _cum(ret, 11, 6),
        "retvol": ret.rolling(12, min_periods=12).std(),
        "dolvol": np.log((price * volume).where(volume > 0)),
        "beta": beta,
        "idiovol": np.sqrt((var_ex - beta.pow(2).mul(var_m, axis=0)).clip(lower=0)),
    }
    cols = {}
    for name, wide in raw.items():
        ranked = (2.0 * wide.round(10).rank(axis=1, pct=True) - 1.0).where(wide.notna())
        cols[name] = ranked.stack(future_stack=True)
    panel = pd.DataFrame(cols)
    panel["has_history"] = raw["mom12m"].notna().stack(future_stack=True)   # 12개월 이력이 있어야 학습·예측
    panel["target"] = ret.shift(-1).stack(future_stack=True)
    panel.index.names = ["ym", "PERMNO"]
    panel[FEATURES] = panel[FEATURES].fillna(0.0)
    return panel


# =============================================================================
# 3. Elastic Net (Q 의 재료: 종목별 다음 달 수익률 예측)
# =============================================================================
def split_validation(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """학습 구간의 마지막 VAL_MONTHS 개월을 검증용으로 분리 (시간 순서 유지)."""
    months = train.index.get_level_values("ym")
    val_start = months.max() - VAL_MONTHS + 1
    return train[months < val_start], train[months >= val_start]


def fit_enet(train: pd.DataFrame) -> Tuple[ElasticNet, Dict[str, float]]:
    """마지막 VAL_MONTHS 개월로 lambda 를 고른 뒤, 학습 구간 전체로 다시 학습. 반환: (모델, 기록용 정보)."""
    fit_part, val_part = split_validation(train)
    best_alpha, best_mse = None, np.inf
    for a in ENET_ALPHAS:
        m = ElasticNet(alpha=a, l1_ratio=ENET_L1_RATIO).fit(fit_part[FEATURES], fit_part["target"])
        mse = np.mean((val_part["target"] - m.predict(val_part[FEATURES])) ** 2)
        if mse < best_mse:
            best_alpha, best_mse = a, mse
    model = ElasticNet(alpha=best_alpha, l1_ratio=ENET_L1_RATIO).fit(train[FEATURES], train["target"])
    return model, {"alpha": best_alpha, **dict(zip(FEATURES, model.coef_)), "intercept": model.intercept_}


# =============================================================================
# 4. 견해 자료구조 (근석님 ViewResult 와 같은 필드)
# =============================================================================
@dataclass(frozen=True)
class ViewResult:
    date: str                      # 리밸런싱 날 (원자료 날짜 'YYYY-MM-DD')
    assets: Tuple[int, ...]        # PERMNO, P 의 열 순서
    P: np.ndarray                  # (1, N)
    Q: np.ndarray                  # (1,)
    Omega: np.ndarray              # (1, 1)
    view_names: Tuple[str, ...]
    predicted_returns: np.ndarray  # (N,) Elastic Net 의 종목별 다음 달 수익률 예측
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_P(mu: np.ndarray, assets: np.ndarray) -> np.ndarray:
    """예측 상위 k 에 +1/k, 하위 k 에 -1/k (행 합 0). 동률은 PERMNO 오름차순."""
    order = np.lexsort((assets, -mu))
    row = np.zeros(len(mu))
    row[order[:N_LONG_SHORT]] = 1.0 / N_LONG_SHORT
    row[order[-N_LONG_SHORT:]] = -1.0 / N_LONG_SHORT
    return row


# =============================================================================
# 5. 전체 기간 실행 (walk-forward)
# =============================================================================
# 결정월 t 의 모델은 타깃월 <= t (특성월 <= t-1) 인 표본으로만 학습합니다.
#   오차 이력 구간: 12개월마다 재학습하며 견해를 만들고, 다음 달 실현값과의 오차를 쌓음
#   평가 구간: 2018-12 에 학습한 모델을 고정하고 매달 P, Q, Omega 생성 (2018-12 ~ 2020-11 결정)
# fit_model: train(DataFrame) -> (predict(X) 를 가진 모델, 기록용 dict). 다른 ML 모듈(ANN 등)은 이것만 바꿔 끼웁니다.
def run(period: str = LISTING_PERIOD, fit_model=fit_enet, output_root: str = OUTPUT_ROOT,
        use_max_weight: bool = USE_MAX_WEIGHT, max_weight: float = MAX_WEIGHT) -> str:
    cfg = PERIODS[period]
    sel = load_selection(period)
    assets = sel["PERMNO"].to_numpy()
    universe = sorted(set(load_universe(period)) | set(assets))
    ret, price, volume = load_market(universe, cfg["data_start"])
    panel = build_panel(ret, price, volume, load_factors())
    raw_dates = (pd.read_csv(STOCK_DATA, usecols=["date"], parse_dates=["date"])["date"]
                 .drop_duplicates().pipe(lambda s: s.groupby(s.dt.to_period("M")).max())
                 .dt.strftime("%Y-%m-%d"))

    train_end = pd.Period(TRAIN_END, "M")
    first = pd.Period(cfg["oos_history_start"], "M") - 1
    decisions = pd.period_range(first, pd.Period(EVAL_END, "M") - 1, freq="M")
    ym = panel.index.get_level_values("ym")
    in_universe = panel.index.get_level_values("PERMNO").isin(load_universe(period))

    model, rows, results, fits = None, [], {}, []
    for t in decisions:
        if t <= train_end and ((t - first).n % REFIT_EVERY == 0 or t == train_end):
            train = panel[in_universe & (ym <= t - 1) & panel["has_history"] & panel["target"].notna()]
            model, info = fit_model(train)
            fits.append({"fit_date": str(t), "n_obs": len(train), **info})

        X = panel.xs(t, level="ym").reindex(assets)
        mu = model.predict(X[FEATURES].fillna(0.0))
        P = make_P(mu, assets)
        Q = float(P @ mu)
        realized = ret.loc[t + 1, assets].to_numpy() if t + 1 in ret.index else np.full(len(assets), np.nan)
        Q_real = float(P @ realized)

        # Omega: 결정월 t 이전에 실현된 견해 오차 (보유월 <= t) 만 사용
        past = [r["error"] for r in rows if not np.isnan(r["error"])][-OMEGA_WINDOW:]
        omega = max(np.mean(np.square(past)), OMEGA_FLOOR) if len(past) >= OMEGA_MIN_OBS else np.nan

        # 진단: 학습 유니버스 전체의 OOS 예측 (Gu et al. 의 R²_oos 계산용)
        U = panel[in_universe & (ym == t) & panel["has_history"] & panel["target"].notna()]
        u_pred = model.predict(U[FEATURES]) if len(U) else np.array([])
        phase = "eval" if t >= train_end else "history"
        rows.append({"date": raw_dates[t], "hold_until": raw_dates.get(t + 1, str(t + 1)), "phase": phase,
                     "Q": Q, "Q_real": Q_real, "error": Q_real - Q, "Omega": omega,
                     "long": " ".join(sel.set_index("PERMNO").loc[assets[P > 0], "TICKER"]),
                     "short": " ".join(sel.set_index("PERMNO").loc[assets[P < 0], "TICKER"]),
                     "ic_20": pd.Series(mu).corr(pd.Series(realized), method="spearman") if np.ptp(mu) > 0 else np.nan,
                     "sse_universe": float(np.sum((U["target"] - u_pred) ** 2)),
                     "sst_universe": float(np.sum(U["target"] ** 2))})
        if phase == "eval":
            if np.isnan(omega):
                raise RuntimeError(f"{t}: Omega 를 계산할 오차 이력이 {len(past)}개월뿐입니다.")
            results[raw_dates[t]] = ViewResult(raw_dates[t], tuple(int(a) for a in assets), P[None, :],
                                               np.array([Q]), np.array([[omega]]), (VIEW_NAME,), mu,
                                               {"hold_until": rows[-1]["hold_until"]})

    constraints = {"use_max_weight": use_max_weight, "max_weight": max_weight if use_max_weight else None}
    return export(period, sel, results, pd.DataFrame(rows), pd.DataFrame(fits), output_root, constraints)


# =============================================================================
# 6. 저장
# =============================================================================
def export(period, sel, results, rec, fits, output_root, constraints) -> str:
    out = os.path.join(output_root, PERIODS[period]["name"])
    bl_dir, dg_dir = os.path.join(out, "bl_input"), os.path.join(out, "diagnostics")
    os.makedirs(bl_dir, exist_ok=True)
    os.makedirs(dg_dir, exist_ok=True)

    a = sel[["PERMNO", "TICKER", "COMNAM"]].copy()
    a.insert(0, "order", range(len(a)))
    a.to_csv(os.path.join(bl_dir, "assets.csv"), index=False)

    # P 가 매달 바뀌므로 근석 님 형식(행 1개)과 달리 날짜별 1행입니다.
    ev = list(results.values())
    base = [{"date": r.date, "hold_until": r.metadata["hold_until"]} for r in ev]
    pd.DataFrame([{**b, "view": VIEW_NAME, **dict(zip(r.assets, r.P[0]))} for b, r in zip(base, ev)]) \
        .to_csv(os.path.join(bl_dir, "P.csv"), index=False)
    pd.DataFrame([{**b, VIEW_NAME: r.Q[0]} for b, r in zip(base, ev)]) \
        .to_csv(os.path.join(bl_dir, "Q.csv"), index=False)
    pd.DataFrame([{**b, f"{VIEW_NAME}|{VIEW_NAME}": r.Omega[0, 0]} for b, r in zip(base, ev)]) \
        .to_csv(os.path.join(bl_dir, "Omega.csv"), index=False)
    with open(os.path.join(bl_dir, "constraints.json"), "w", encoding="utf-8") as f:
        json.dump(constraints, f, ensure_ascii=False, indent=2)

    evr = rec[rec["phase"] == "eval"]
    pd.DataFrame({
        "날짜(리밸런싱)": evr["date"], "보유 종료일": evr["hold_until"], "견해": VIEW_NAME,
        "롱 종목": evr["long"], "숏 종목": evr["short"],
        "예측 스프레드 Q(%/월)": (evr["Q"] * 100).round(3),
        "실현 스프레드(%/월)": (evr["Q_real"] * 100).round(3),
        "Omega": evr["Omega"],
    }).to_csv(os.path.join(out, "views_readable.csv"), index=False, encoding="utf-8-sig")

    rec.to_csv(os.path.join(dg_dir, "views_timeseries.csv"), index=False)
    fits.to_csv(os.path.join(dg_dir, "model_fits.csv"), index=False)
    pd.DataFrame({r.date: r.predicted_returns for r in ev}, index=list(sel["PERMNO"])).T \
        .rename_axis("date").to_csv(os.path.join(dg_dir, "predicted_returns.csv"))

    summary = {"period": period, "view": VIEW_NAME, "n_eval_months": len(ev),
               "constraints": constraints}
    for ph, g in rec.groupby("phase"):
        g = g.dropna(subset=["Q_real"])
        summary[ph] = {"months": len(g),
                       "r2_oos_universe": 1 - g["sse_universe"].sum() / g["sst_universe"].sum(),
                       "mean_ic_20": g["ic_20"].mean(),
                       "view_hit_rate": float((np.sign(g["Q"]) == np.sign(g["Q_real"])).mean()),
                       "mean_Q": g["Q"].mean(), "mean_Q_real": g["Q_real"].mean()}
    with open(os.path.join(dg_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return out


# =============================================================================
# 7. BL 모듈에서 읽기
# =============================================================================
def load_views(output_dir: str) -> Dict[str, ViewResult]:
    """{날짜: ViewResult}. 사용 예:
        views = load_views("outputs_ElasticNet/P1")
        for date, v in views.items():   # v.P (1, 20), v.Q (1,), v.Omega (1, 1), v.metadata["max_weight"]
    """
    bl = os.path.join(output_dir, "bl_input")
    P = pd.read_csv(os.path.join(bl, "P.csv"))
    Q = pd.read_csv(os.path.join(bl, "Q.csv")).set_index("date")
    Om = pd.read_csv(os.path.join(bl, "Omega.csv")).set_index("date")
    with open(os.path.join(bl, "constraints.json"), encoding="utf-8") as f:
        cons = json.load(f)
    assets = tuple(int(c) for c in P.columns[3:])
    return {row["date"]: ViewResult(
        row["date"], assets, row[P.columns[3:]].to_numpy(dtype=float)[None, :],
        Q.loc[[row["date"]], [VIEW_NAME]].to_numpy(dtype=float)[0],
        Om.loc[[row["date"]]].iloc[:, 1:].to_numpy(dtype=float).reshape(1, 1),
        (VIEW_NAME,), np.array([]), {"hold_until": row["hold_until"], **cons})
        for _, row in P.iterrows()}


if __name__ == "__main__":
    print(f"[완료] 결과 폴더: {run()}")
