from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# 1. 버전 확인   
# =============================================================================
# 결과 재현에 필요한 고정 버전 
#
PINNED = {"xgboost": "3.0.5", "numpy": "2.2.6", "pandas": "2.2.3"}


def version_warnings() -> List[str]:
    import numpy
    import pandas
    import xgboost

    installed = {"xgboost": xgboost.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__}
    return [f"{name} {installed[name]} 설치되었습니다 (고정 버전 {want}).  결과가 달라질 수 있습니다 — "
            f"pip install numpy==2.2.6 pandas==2.2.3 xgboost==3.0.5 scikit-learn==1.6.1 pyyaml==6.0.2" for name, want in PINNED.items() if installed[name] != want]


# =============================================================================
# 2. raw data 로딩  
# =============================================================================

VALID_EXCHCD = (1, 2, 3)
RET_CAP = 5.0


@dataclass
class MarketData:
    """wide 행렬 모음. index = pd.PeriodIndex(M), columns = PERMNO."""

    ret: pd.DataFrame      # 월수익률
    price: pd.DataFrame    # ADJ_PRC
    volume: pd.DataFrame   # VOL
    info: pd.DataFrame     # index=PERMNO, columns=[TICKER, COMNAM, EXCHCD] (마지막 관측 기준)
    dates: Optional[pd.Series] = None   # index=ym(Period), 값='YYYY-MM-DD' 원자료의 그 달 날짜(월말 거래일)

    def restrict(self, permnos=None, start=None, end=None) -> "MarketData":
        cols = self.ret.columns if permnos is None else pd.Index(permnos).intersection(self.ret.columns)
        sl = slice(pd.Period(start, "M") if start else None, pd.Period(end, "M") if end else None)
        return MarketData(
            ret=self.ret.loc[sl, cols],
            price=self.price.loc[sl, cols],
            volume=self.volume.loc[sl, cols],
            info=self.info.reindex(cols),
            dates=self.dates,
        )


def load_stock_panel(path: str, start: Optional[str] = None, end: Optional[str] = None) -> MarketData:
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.drop_duplicates(subset=["PERMNO", "date"], keep="first")
    raw = raw[raw["ADJ_PRC"].notna() & (raw["ADJ_PRC"] > 0)]
    raw = raw[raw["EXCHCD"].isin(VALID_EXCHCD)].copy()
    raw["ym"] = raw["date"].dt.to_period("M")
    if start:
        raw = raw[raw["ym"] >= pd.Period(start, "M")]
    if end:
        raw = raw[raw["ym"] <= pd.Period(end, "M")]
    raw = raw.sort_values(["PERMNO", "ym"])

    price = raw.pivot(index="ym", columns="PERMNO", values="ADJ_PRC").sort_index()
    volume = raw.pivot(index="ym", columns="PERMNO", values="VOL").sort_index()
    # 전체 월 인덱스를 연속으로 맞춰야 shift(1) 이 '정확히 1개월 전' 을 의미합니다.
    full = pd.period_range(price.index.min(), price.index.max(), freq="M")
    price = price.reindex(full)
    volume = volume.reindex(full)
    ret = price / price.shift(1) - 1.0
    ret = ret.mask(ret > RET_CAP)
    price.index.name = volume.index.name = ret.index.name = "ym"

    info = raw.groupby("PERMNO")[["TICKER", "COMNAM", "EXCHCD"]].last()
    # 원자료 날짜 그대로 (한 달에 날짜가 하나: 그 달 마지막 거래일)
    dates = raw.groupby("ym")["date"].agg(lambda s: s.mode().iloc[0]).dt.strftime("%Y-%m-%d")
    return MarketData(ret=ret, price=price, volume=volume, info=info, dates=dates)


# =============================================================================
# 3. FF3 선정 결과 로딩  
# =============================================================================

REQUIRED = ["rank", "PERMNO", "alpha", "b_mkt", "s_smb", "h_hml", "t_alpha", "TICKER", "COMNAM"]
DEDUPE_OPTIONS = ("most_liquid", "best_rank", "none")


def dedupe_same_company(sel: pd.DataFrame, method: str = "most_liquid") -> tuple:
    """회사(COMNAM)당 한 종목만 남김. 반환: (남은 DataFrame, 제외된 DataFrame)."""
    if method not in DEDUPE_OPTIONS:
        raise ValueError(f"dedupe 방법은 {DEDUPE_OPTIONS} 중 하나여야 합니다.")
    if method == "none":
        return sel, sel.iloc[0:0]
    if method == "most_liquid" and "avg_vol" not in sel.columns:
        raise ValueError("most_liquid 는 selection 파일에 avg_vol 열이 필요합니다.")
    key = ["avg_vol", "rank"] if method == "most_liquid" else ["rank"]
    asc = [False, True] if method == "most_liquid" else [True]
    keep_idx = sel.sort_values(key, ascending=asc).drop_duplicates("COMNAM", keep="first").index
    kept = sel.loc[sel.index.isin(keep_idx)]
    dropped = sel.loc[~sel.index.isin(keep_idx)]
    return kept, dropped


def load_selection(path: str, n_assets: int, dedupe: str = "most_liquid") -> pd.DataFrame:
    """rank 오름차순 상위 n_assets (같은 회사 중복 제거 후). 자산 순서 = rank 순서.

    제외된 종목은 반환 DataFrame 의 attrs['dropped'] 에 기록됩니다.
    """
    sel = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in sel.columns]
    if missing:
        raise ValueError(f"selection 파일에 필요한 열이 없습니다: {missing}")
    if sel["PERMNO"].duplicated().any():
        raise ValueError("selection 에 중복 PERMNO 가 있습니다.")
    sel = sel.sort_values("rank").reset_index(drop=True)
    kept, dropped = dedupe_same_company(sel, dedupe)
    out = kept.sort_values("rank").head(n_assets).reset_index(drop=True)
    # 상위 n 안에 들어갈 수 있었던 종목 중 제외된 것만 기록
    cutoff = out["rank"].max()
    out.attrs["dropped"] = dropped[dropped["rank"] <= cutoff].reset_index(drop=True)
    return out


def load_universe(path: str) -> list:
    """해당 구간 balanced 유니버스 PERMNO 목록 (ML 학습용 횡단면)."""
    return sorted(pd.read_csv(path, usecols=["PERMNO"])["PERMNO"].unique().tolist())


# =============================================================================
# 4. 공통 자료구조 (ViewResult)   
# =============================================================================
# View 모듈의 공통 자료구조와 인터페이스.
#
#     P      : (K, N)  무엇에 대한 견해인가
#     Q      : (K,)    견해의 크기 
#     Omega  : (K, K)  견해의 불확실성 (월 수익률 분산 단위)
#     assets : 길이 N, PERMNO. P 의 열 순서 = assets 순서

@dataclass(frozen=True)
class ViewResult:
    date: str                      # 리밸런싱 날 (파이프라인 내부 'YYYY-MM', load_views 로 읽으면 원자료 날짜 'YYYY-MM-DD')
    assets: Tuple[int, ...]        # PERMNO, P 의 열 순서
    P: np.ndarray                  # (K, N)
    Q: np.ndarray                  # (K,)
    Omega: np.ndarray              # (K, K)
    view_names: Tuple[str, ...]    # 길이 K
    predicted_returns: np.ndarray  # (N,) XGBoost 의 종목별 다음 달 수익률 예측 (mu_hat)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.P.shape[0]

    @property
    def N(self) -> int:
        return self.P.shape[1]

    def P_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.P, index=list(self.view_names), columns=list(self.assets))

    def aligned_to(self, assets) -> "ViewResult":
        """BL 쪽 자산 순서에 맞춰 P 열과 predicted_returns 를 재배열."""
        assets = tuple(int(a) for a in assets)
        if set(assets) != set(self.assets) or len(assets) != len(self.assets):
            raise ValueError("자산 집합이 다릅니다. 순서만 바꿀 수 있습니다.")
        pos = [self.assets.index(a) for a in assets]
        return ViewResult(self.date, assets, self.P[:, pos], self.Q, self.Omega,
                          self.view_names, self.predicted_returns[pos], dict(self.metadata))


class ReturnPredictor(Protocol):
    """Q 생성용 수익률 예측 모델 인터페이스입니다.

    X: (n_samples, n_features) DataFrame, y: (n_samples,) Series.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReturnPredictor": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


# =============================================================================
# 5. Feature (Barua & Sharma Table 2)   
# =============================================================================
# Point-in-time feature 생성 — Barua & Sharma (2023) Table 2 의 기술적 지표 세트.
#
# 논문은 ETF 별 일간 데이터에 아래 14개 지표를 예측변수로 썼습니다. 또한 같은 세트를 월 단위로 계산합니다.
#     Moving Average (s, l), s ∈ {1,2,3}, l ∈ {6,9,12}   -> ma_{s}_{l} = MA_s / MA_l - 1   (9개)
#     Momentum (1), (3), (6), (9), (12)                   -> mom_{k} = 최근 k개월 누적수익률  (5개)
# MA 는 가격 수준과 무관하도록 비율(단기 이동평균 / 장기 이동평균 - 1)로 둡니다.
#
# 스케일: 논문은 지표를 [-1, 1] 로 스케일링 하였습니다. 여기서는 월별 횡단면 백분위 순위를 [-1, 1] 로
# 옮깁니다. (시점 t 의 횡단면만 쓰므로 미래 정보가 섞이지 않습니다).
#
# 모든 feature 는 결정월 t 말까지의 정보만 사용합니다. (rolling window 는 t 에서 끝납니다).
# 타깃은 t+1 월 수익률이며, feature 와 같은 (t, PERMNO) 키에 붙습니다.

MA_SHORT = (1, 2, 3)
MA_LONG = (6, 9, 12)
MOM_WINDOWS = (1, 3, 6, 9, 12)
FEATURE_NAMES = [f"ma_{s}_{l}" for s in MA_SHORT for l in MA_LONG] + [f"mom_{k}" for k in MOM_WINDOWS]


def _compound(ret: pd.DataFrame, w: int) -> pd.DataFrame:
    #w개월 누적수익률 = (1+r_t)(1+r_{t-1})...(1+r_{t-w+1}) - 1. 한 달이라도 없으면 NaN.
    
    g = 1.0 + ret
    out = g.copy()
    for j in range(1, w):
        out = out * g.shift(j)
    return out - 1.0


def _stack(wide: pd.DataFrame) -> pd.Series:
    import inspect
    if "future_stack" in inspect.signature(pd.DataFrame.stack).parameters:
        return wide.stack(future_stack=True)
    return wide.stack(dropna=False)


class ViewFeatureBuilder:
    """wide MarketData -> long feature table indexed by (ym, PERMNO)."""

    def __init__(self, transform: str = "rank"):
        if transform not in ("rank", "none"):
            raise ValueError("transform must be 'rank' or 'none'")
        self.transform = transform

    def raw_features(self, md: MarketData) -> dict:
        r, p = md.ret, md.price
        feats = {}
        for s in MA_SHORT:
            ma_s = p.rolling(s, min_periods=s).mean()
            for l in MA_LONG:
                feats[f"ma_{s}_{l}"] = ma_s / p.rolling(l, min_periods=l).mean() - 1.0
        for k in MOM_WINDOWS:
            feats[f"mom_{k}"] = _compound(r, k)
        return feats

    def build(self, md: MarketData) -> pd.DataFrame:
        cols = {}
        for name, wide in self.raw_features(md).items():
            if self.transform == "rank":
                # 월별 횡단면 백분위 순위 (0, 1] -> [-1, 1]
                # 순위 전에 소수 10자리로 반올림: 계산 경로에 따른 1e-16 수준 차이가 순위를 뒤집지 않게 합니다.
                w = (2.0 * wide.round(10).rank(axis=1, pct=True) - 1.0).where(wide.notna())
            else:
                w = wide
            cols[name] = _stack(w)
        X = pd.DataFrame(cols)[FEATURE_NAMES]
        X.index.names = ["ym", "PERMNO"]
        # 12개월 지표(mom_12, ma_*_12)가 계산되는 시점부터 학습·예측 대상입니다.
        return X[X["mom_12"].notna() & X["ma_1_12"].notna()]


def build_target(ret: pd.DataFrame, kind: str = "raw") -> pd.Series:
    """결정월 t 행에 t+1 수익률을 붙인 타깃 (ym=t, PERMNO).

    raw          : 다음 달 수익률 그대로 (기본값. Barua & Sharma 처럼 수익률 자체를 예측)
    cs_demeaned  : 다음 달 수익률 - 그 달 횡단면 평균 (선택)
    논문은 로그수익률을 예측했지만, BL 의 기대수익률은 산술수익률이므로 산술 월수익률을 씁니다.
    """
    nxt = ret.shift(-1)
    if kind == "cs_demeaned":
        nxt = nxt.sub(nxt.mean(axis=1), axis=0)
    elif kind != "raw":
        raise ValueError("target must be 'raw' or 'cs_demeaned'")
    y = _stack(nxt)
    y.index.names = ["ym", "PERMNO"]
    return y.rename("target")


# =============================================================================
# 6. P (Ko, Son & Lee 2024)   
# =============================================================================
# P 생성: FF3 factor exposure 기반 relative view.
#
# Ko, Son & Lee (2024) 의 'Fama-French 정보로 BL view structure 를 만든다' 는 아이디어를
# 응용합니다 (replication 아님: 실제 Size, B/M 대신 FF3 회귀 계수 s_SMB, h_HML 사용.
# 원자료에 발행주식수·장부가치가 없어 시가총액·B/M 계산이 불가능하기 때문입니다).
#   - s_SMB 가 클수록 '소형주처럼' 움직임  -> size 대리변수 (클수록 small)
#   - h_HML 이 클수록 '가치주처럼' 움직임  -> B/M 대리변수 (클수록 value)
#
#
# double_sort   — Ko et al. (2024) Sec. 3, Eq. (11) 과 같은 구조, K = 1
#    size 대리변수로 grid 개 그룹 → 각 그룹 안에서 value 대리변수로 grid 개 그룹 (순차 이중정렬).
#    첫 칸(가장 small & 가장 value)  +1/|D_1|,  마지막 칸(가장 big & 가장 growth)  -1/|D_last|.
#    원 논문은 5x5(=25칸, N=90)이지만 N=20 에서는 칸이 비므로 grid=2 (2x2, 칸당 5종목)를 기본으로 합니다.
#
# 각 행의 합은 0 (relative view) 이고 P_k @ mu = 롱 그룹 평균 - 숏 그룹 평균.

VIEW_LABELS = {"s_smb": "SMB", "h_hml": "HML", "b_mkt": "MKT", "alpha": "ALPHA"}
P_METHODS = ("double_sort", "independent")


def _order(ex: pd.DataFrame, col: str, ids) -> list:
    """col 내림차순, 동률은 PERMNO 오름차순 (재현성)."""
    sub = ex.loc[list(ids), [col]].reset_index()
    return sub.sort_values([col, "PERMNO"], ascending=[False, True])["PERMNO"].tolist()


class FF3ExposurePBuilder:
    def __init__(self, exposures: Sequence[str] = ("s_smb", "h_hml"), group_frac: float = 0.25,
                 method: str = "double_sort", grid: int = 2):
        if method not in P_METHODS:
            raise ValueError(f"p.method 는 {P_METHODS} 중 하나여야 합니다.")
        if method == "independent" and not 0 < group_frac <= 0.5:
            raise ValueError("group_frac 는 (0, 0.5] 범위여야 합니다.")
        if method == "double_sort":
            if len(exposures) != 2:
                raise ValueError("double_sort 는 exposure 2개 (size 대리변수, value 대리변수)가 필요합니다.")
            if grid < 2:
                raise ValueError("grid 는 2 이상이어야 합니다.")
        self.exposures = tuple(exposures)
        self.group_frac = group_frac
        self.method = method
        self.grid = grid

    def n_group(self, n_assets: int) -> int:
        g = int(round(n_assets * self.group_frac))
        if g < 1 or 2 * g > n_assets:
            raise ValueError(f"N={n_assets}, group_frac={self.group_frac} 로는 그룹을 만들 수 없습니다.")
        return g

    def build(self, exposures: pd.DataFrame, assets: Sequence[int]) -> Tuple[np.ndarray, Tuple[str, ...], pd.DataFrame]:
        """exposures: index=PERMNO, columns ⊇ self.exposures.

        반환: P (K, N), view_names, membership (PERMNO x view, 값 +1/0/-1)
        """
        assets = list(assets)
        ex = exposures.reindex(assets)
        ex.index.name = "PERMNO"
        if ex[list(self.exposures)].isna().any().any():
            raise ValueError("선정 종목 중 exposure 가 없는 종목이 있습니다.")
        if self.method == "double_sort":
            return self._double_sort(ex, assets)
        return self._independent(ex, assets)

    def _to_row(self, assets, long, short):
        m = pd.Series(0, index=assets)
        m[long] = 1
        m[short] = -1
        row = np.zeros(len(assets))
        row[m.to_numpy() > 0] = 1.0 / len(long)
        row[m.to_numpy() < 0] = -1.0 / len(short)
        return row, m

    def _independent(self, ex, assets):
        g = self.n_group(len(assets))
        P = np.zeros((len(self.exposures), len(assets)))
        member = pd.DataFrame(index=assets)
        names = []
        for k, col in enumerate(self.exposures):
            order = _order(ex, col, assets)
            name = f"{VIEW_LABELS.get(col, col)}_high_minus_low"
            P[k], member[name] = self._to_row(assets, order[:g], order[-g:])
            names.append(name)
        return P, tuple(names), member

    def _double_sort(self, ex, assets):
        size_col, value_col = self.exposures
        if len(assets) < self.grid * self.grid:
            raise ValueError(f"N={len(assets)} 로는 {self.grid}x{self.grid} 이중정렬 칸을 채울 수 없습니다.")
        # 1단계: size 대리변수 (s_SMB 큰 = small) 로 grid 등분, 2단계: 각 그룹 안에서 value 대리변수로 grid 등분
        size_groups = np.array_split(np.array(_order(ex, size_col, assets)), self.grid)
        small, big = size_groups[0].tolist(), size_groups[-1].tolist()
        long = np.array_split(np.array(_order(ex, value_col, small)), self.grid)[0].tolist()    # small & value
        short = np.array_split(np.array(_order(ex, value_col, big)), self.grid)[-1].tolist()    # big & growth
        name = "SmallValue_minus_BigGrowth"
        row, m = self._to_row(assets, long, short)
        member = pd.DataFrame({name: m}, index=assets)
        return row[None, :], (name,), member


# =============================================================================
# 7. 예측 모델 (XGBoost)   
# =============================================================================
# Q 생성용 수익률 예측 모델.
#
# XGBoost — Barua & Sharma (2023) Sec. 3.4: 트리 300개, 나머지 하이퍼파라미터는 기본값.
# BootstrapEnsemble — Spears et al. (2023) 부록 A.2: 앙상블 예측 분산 = epistemic 불확실성 (비교 결과용).
#
# 새 모델 추가 방법 (ML 담당): fit(X, y) / predict(X) 를 가진 클래스를 만들고 PREDICTORS에 등록합니다.

class XGBoostPredictor:
    def __init__(self, seed: int = 42, **params):
        from xgboost import XGBRegressor

        defaults = dict(n_estimators=300, n_jobs=-1)            # 논문: 300 trees, 나머지 기본값
        defaults.update(params)
        self.model = XGBRegressor(random_state=seed, **defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.to_numpy(dtype=float))


PREDICTORS: Dict[str, Callable[..., object]] = {
    "xgboost": XGBoostPredictor,
}


def make_predictor(name: str, seed: int = 42, **params):
    if name not in PREDICTORS:
        raise KeyError(f"알 수 없는 predictor '{name}'. 사용 가능: {list(PREDICTORS)}")
    return PREDICTORS[name](seed=seed, **params)


class BootstrapEnsemble:
    """같은 모델을 월 단위 block bootstrap 표본으로 n_members 번 학습한 앙상블입니다.

    Spears, Zohren & Roberts (2023, 부록 A.2): 부스팅 모델의 epistemic 불확실성을
    '앙상블 예측값의 분산'으로 근사합니다. (Ustimenko et al., 2020). 여기서는 학습 월(ym)을
    복원추출해서 멤버마다 다른 표본을 쓰므로, 멤버 간 예측 차이 = 추정(모델) 불확실성 입니다.

    predict(X)         -> 멤버 평균 (Q 에 사용)
    predict_members(X) -> (n_members, n) (Omega 에 사용)
    """

    def __init__(self, name: str, n_members: int = 10, seed: int = 42, **params):
        if n_members < 2:
            raise ValueError("앙상블은 멤버가 2개 이상이어야 분산을 계산할 수 있습니다.")
        self.name, self.n_members, self.seed, self.params = name, n_members, seed, params
        self.members = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        months = X.index.get_level_values("ym")
        uniq = np.unique(months)
        pos = pd.Series(np.arange(len(X))).groupby(np.asarray(months)).apply(np.asarray).to_dict()
        self.members = []
        for m in range(self.n_members):
            rng = np.random.default_rng(self.seed + 1000 * (m + 1))
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([pos[d] for d in draw])
            model = make_predictor(self.name, seed=self.seed + m, **self.params)
            model.fit(X.iloc[idx], y.iloc[idx])
            self.members.append(model)
        return self

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        return np.vstack([m.predict(X) for m in self.members])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X).mean(axis=0)


# =============================================================================
# 8. Q = P @ mu_hat  
# =============================================================================
# Q 생성: Q_t = P_t @ mu_hat_t.
#
# 모델이 Q 를 직접 예측하지 않고 종목별 다음 달 수익률 mu_hat 을 예측한 뒤
# P 로 사영하므로, P 와 Q 가 수학적으로 항상 일관됩니다.
# P 의 각 행 합이 0 이므로 mu_hat 에 공통 상수가 더해져도 Q 는 변하지 않습니다
# (-> 횡단면 평균을 뺀 타깃으로 학습해도 Q 의 의미는 동일).

class ModelQBuilder:
    def __init__(self, predictor):
        self.predictor = predictor

    def build(self, P: np.ndarray, X_assets: pd.DataFrame, assets: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """X_assets: index=PERMNO (결정월 t 의 feature). 반환: (mu_hat (N,), Q (K,))"""
        X = X_assets.reindex(list(assets))
        if X.index.isna().any() or len(X) != P.shape[1]:
            raise ValueError("feature 행과 자산 수가 맞지 않습니다.")
        if X.isna().all(axis=1).any():
            missing = X.index[X.isna().all(axis=1)].tolist()
            raise ValueError(f"결정월 feature 가 없는 자산: {missing}")
        mu = np.asarray(self.predictor.predict(X), dtype=float)
        return mu, P @ mu

    def build_members(self, P: np.ndarray, X_assets: pd.DataFrame, assets: Sequence[int]):
        """앙상블 predictor 일 때 멤버별 Q (n_members, K). 아니면 None."""
        if not hasattr(self.predictor, "predict_members"):
            return None
        X = X_assets.reindex(list(assets))
        mu_m = np.asarray(self.predictor.predict_members(X), dtype=float)   # (M, N)
        return mu_m @ P.T


# =============================================================================
# 9. Omega   
# =============================================================================
# Omega 생성: view 의 불확실성.
#
# BL 우도는 q | mu ~ N(P mu, Omega) 이다. 즉 Omega 는 '기대수익률(mu)에 대한 견해가 얼마나
# 불확실한가' 이지, '다음 달 실현수익률이 얼마나 흔들리는가' 가 아닙니다.
#
# Spears, Zohren & Roberts (2023) 는 이 구분을 명시합니다 (Section 5, 부록 A.2):
#     예측분산 = epistemic (모델/추정 불확실성)  +  aleatoric (데이터 자체의 노이즈)
#     -> Omega 에는 epistemic 만,  aleatoric 은 수익률 분산 쪽(F)에 둡니다.
# 논문의 epistemic 추정법
#     - Boost 모델 : 앙상블 멤버 예측값의 분산                 -> EnsembleEpistemicOmegaBuilder
#     - ARIMA 모델 : 최근 OOS 예측오차 분산 - 현재 aleatoric (하한 적용) -> RollingErrorOmegaBuilder
#
# 이 모듈의 세 가지 방법 (config view.omega.method)
#     oos_r2_confidence  : tau pSp' (1-C)/C, C = view 의 OOS R^2   [본 결과, Barua & Sharma 의 Idzorek 변형]
#     ensemble_epistemic : Var_m(P mu_hat^(m))                     [비교 결과, Spears Boost 방식]
#     rolling_mse_net    : max(MSE(e) - diag(P Sigma P^T), 하한)   [옵션, Spears ARIMA 방식]
# He-Litterman Omega = tau P Sigma P^T (tau=0.1) 는 Ko et al. (2024), Barua & Sharma (2023) 의 Omega 이며
# 여기서는 Idzorek 식의 기준 크기와 이력 부족 시 대체값으로 씁니다.

OMEGA_METHODS = ("oos_r2_confidence", "ensemble_epistemic", "rolling_mse_net")


def _floor_psd(M: np.ndarray, min_variance: float) -> np.ndarray:
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, min_variance)
    return (V * w) @ V.T


class _OmegaBase:
    def __init__(self, structure: str = "diag", tau: float = 0.1, min_variance: float = 1e-6, scale: float = 1.0):
        if structure not in ("diag", "full"):
            raise ValueError("structure must be 'diag' or 'full'")
        self.structure = structure
        self.tau = tau
        self.min_variance = min_variance
        self.scale = scale

    def prior(self, P: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """He-Litterman 기본 Omega 의 전체 행렬 tau * P Sigma P^T (Ko et al. 2024, Barua & Sharma 2023 의 Omega)."""
        return self.tau * P @ Sigma @ P.T

    def _finish(self, Omega: np.ndarray) -> np.ndarray:
        if self.structure == "diag":
            Omega = np.diag(np.diag(Omega))
        return _floor_psd(self.scale * Omega, self.min_variance)

    def _prior_fallback(self, P, Sigma, info):
        if Sigma is None:
            raise ValueError("Omega 를 계산할 정보가 부족하고 prior 용 Sigma 도 없습니다.")
        info["omega_source"] = "he_litterman_prior"
        return self.prior(P, Sigma)


class EnsembleEpistemicOmegaBuilder(_OmegaBase):
    """Omega = 앙상블 멤버별 Q 의 (공)분산 = epistemic 불확실성 (Spears et al. 2023, Boost 모델 방식)."""

    def build(self, past_errors: Optional[pd.DataFrame], P: np.ndarray, Sigma: Optional[np.ndarray],
              Q_members: Optional[np.ndarray] = None, past_realized=None) -> Tuple[np.ndarray, Dict]:
        info: Dict = {"n_errors": 0 if past_errors is None else int(past_errors.dropna().shape[0])}
        if Q_members is None or Q_members.shape[0] < 2:
            raise ValueError("ensemble_epistemic 은 앙상블 predictor (n_members >= 2) 가 필요합니다.")
        Omega = np.atleast_2d(np.cov(Q_members, rowvar=False, ddof=1))
        info["omega_source"] = f"ensemble_epistemic_{self.structure}"
        info["n_members"] = int(Q_members.shape[0])
        return self._finish(Omega), info


class RollingErrorOmegaBuilder(_OmegaBase):
    """Spears et al. (2023) 부록 A.2 의 ARIMA 방식 epistemic 근사.

        Omega = MSE(e) - P Sigma P^T  (하한 min_variance),   e = P r_{t+1} - Q  (최근 window 개월 OOS 오차)

    논문은 '최근 OOS 예측오차 분산 - 현재 aleatoric' 이고 여기서는 분산 대신 MSE(편향 포함)를 씁니다.
    오차가 min_obs 개 미만이면 He-Litterman prior 를 씁니다.
    """

    def __init__(self, structure: str = "diag", window: int = 24, min_obs: int = 12, tau: float = 0.1,
                 min_variance: float = 1e-6, scale: float = 1.0):
        super().__init__(structure, tau, min_variance, scale)
        if min_obs > window:
            raise ValueError("min_obs 는 window 이하여야 합니다.")
        self.window = window
        self.min_obs = min_obs

    def build(self, past_errors: pd.DataFrame, P: np.ndarray, Sigma: Optional[np.ndarray],
              Q_members: Optional[np.ndarray] = None, past_realized=None) -> Tuple[np.ndarray, Dict]:
        """past_errors: index=결정월, columns=view (K). 결정 시점에 이미 실현된 오차만 넣을 것."""
        E = past_errors.dropna().tail(self.window).to_numpy(dtype=float)
        n = E.shape[0]
        info: Dict = {"n_errors": int(n)}
        if n >= self.min_obs and Sigma is not None:
            Omega = E.T @ E / n - P @ Sigma @ P.T               # MSE - aleatoric -> epistemic 근사
            info["omega_source"] = f"rolling_mse_net_{self.structure}"
        else:
            Omega = self._prior_fallback(P, Sigma, info)
        return self._finish(Omega), info


class OOSR2ConfidenceOmegaBuilder(_OmegaBase):
    """Idzorek (2007) 식 confidence -> Omega. confidence 는 view 의 과거 OOS R^2.

    Barua & Sharma (2023, Sec. 4.3) 의 'BL-FG (Idzorek)' 변형: 예측모형 view 의 confidence 로
    XGBoost 예측의 out-of-sample R^2 를 사용. 여기서는 view 단위로 계산합니다.

        R2_k   = 1 - sum(e_k^2) / sum(Q_real_k^2)        (최근 window 개월, 벤치마크 예측 = 0 스프레드)
        C_k    = clip(R2_k, c_min, c_max)
        omega_k = tau * p_k Sigma p_k^T * (1 - C_k) / C_k

    단일 view 에서 BL posterior 가 view 쪽으로 이동하는 비율 = C_k (BL 의 tau 가 이 tau 와 같을 경우).
    즉 '과거에 설명력이 있었던 만큼만 믿습니다'. 예측력이 없으면 C = c_min -> view 거의 무시합니다.
    """

    def __init__(self, window: int = 24, min_obs: int = 12, c_min: float = 0.01, c_max: float = 0.9, **kw):
        super().__init__(**kw)
        if not 0 < c_min < c_max < 1:
            raise ValueError("0 < c_min < c_max < 1 이어야 합니다.")
        self.window, self.min_obs, self.c_min, self.c_max = window, min_obs, c_min, c_max

    def build(self, past_errors: pd.DataFrame, P: np.ndarray, Sigma: Optional[np.ndarray],
              Q_members: Optional[np.ndarray] = None, past_realized: Optional[pd.DataFrame] = None):
        if Sigma is None:
            raise ValueError("oos_r2_confidence 는 Sigma 가 필요합니다.")
        K = P.shape[0]
        info: Dict = {"n_errors": 0}
        conf = np.full(K, self.c_min)
        if past_realized is not None and len(past_errors):
            both = pd.concat([past_errors, past_realized], axis=1, keys=["e", "r"]).dropna().tail(self.window)
            info["n_errors"] = int(len(both))
            if len(both) >= self.min_obs:
                e2 = (both["e"].to_numpy() ** 2).sum(axis=0)
                r2d = (both["r"].to_numpy() ** 2).sum(axis=0)
                r2 = 1.0 - e2 / np.where(r2d > 0, r2d, np.nan)
                conf = np.clip(np.nan_to_num(r2, nan=self.c_min), self.c_min, self.c_max)
        prior = np.diag(self.prior(P, Sigma))
        Omega = np.diag(prior * (1.0 - conf) / conf)
        info["omega_source"] = "oos_r2_confidence" if info["n_errors"] >= self.min_obs else "oos_r2_confidence_cmin"
        info["confidence"] = conf.tolist()
        return self._finish(Omega), info


def make_omega_builder(ocfg: dict) -> _OmegaBase:
    method = ocfg.get("method", "oos_r2_confidence")
    common = dict(structure=ocfg.get("structure", "diag"), tau=ocfg.get("tau", 0.1),
                  min_variance=float(ocfg.get("min_variance", 1e-6)), scale=ocfg.get("scale", 1.0))
    if method == "ensemble_epistemic":
        return EnsembleEpistemicOmegaBuilder(**common)
    if method == "rolling_mse_net":
        return RollingErrorOmegaBuilder(window=ocfg.get("window", 24), min_obs=ocfg.get("min_obs", 12), **common)
    if method == "oos_r2_confidence":
        return OOSR2ConfidenceOmegaBuilder(window=ocfg.get("window", 24), min_obs=ocfg.get("min_obs", 12),
                                           c_min=ocfg.get("c_min", 0.01), c_max=ocfg.get("c_max", 0.9), **common)
    raise ValueError(f"omega.method 는 {OMEGA_METHODS} 중 하나여야 합니다.")


# =============================================================================
# 10. 검증  
# =============================================================================
# P, Q, Omega 검증. '실행은 되지만 수학적으로 틀린' 결과를 BL 로 넘기지 않기 위함입니다.

class ViewValidationError(ValueError):
    pass


def validate_view(v: ViewResult, bl_assets: Optional[Sequence[int]] = None, relative: bool = True,
                  atol: float = 1e-10) -> None:
    K, N = v.P.shape
    checks = [
        (v.P.ndim == 2, "P 는 2차원이어야 합니다."),
        (v.Q.shape == (K,), f"Q.shape={v.Q.shape}, 기대값 ({K},)"),
        (v.Omega.shape == (K, K), f"Omega.shape={v.Omega.shape}, 기대값 ({K},{K})"),
        (len(v.assets) == N, f"assets 길이 {len(v.assets)} != N={N}"),
        (len(set(v.assets)) == N, "assets 에 중복이 있습니다."),
        (len(v.view_names) == K, "view_names 길이가 K 와 다릅니다."),
        (v.predicted_returns.shape == (N,), "predicted_returns 길이가 N 과 다릅니다."),
        (np.isfinite(v.P).all() and np.isfinite(v.Q).all() and np.isfinite(v.Omega).all(), "NaN/inf 가 있습니다."),
        (np.allclose(v.Omega, v.Omega.T, atol=atol), "Omega 가 대칭이 아닙니다."),
        ((np.diag(v.Omega) > 0).all(), "Omega 대각원소가 0 이하입니다."),
        (np.linalg.eigvalsh(v.Omega).min() > 0, "Omega 가 positive definite 가 아닙니다."),
        (np.linalg.matrix_rank(v.P) == K, "P 의 행이 선형종속입니다 (view 중복)."),
    ]
    if relative:
        checks.append((np.allclose(v.P.sum(axis=1), 0.0, atol=1e-9), "relative view 인데 P 행 합이 0 이 아닙니다."))
    if bl_assets is not None:
        checks.append((tuple(int(a) for a in bl_assets) == v.assets,
                       "BL 자산 순서와 view 자산 순서가 다릅니다. ViewResult.aligned_to() 를 사용하세요."))
    errs = [msg for ok, msg in checks if not ok]
    if errs:
        raise ViewValidationError(f"[{v.date}] " + " / ".join(errs))


def same_company_warnings(selection: pd.DataFrame) -> List[str]:
    """같은 회사의 여러 주식 클래스(예: HEI, HEI.A)가 함께 선정된 경우 경고."""
    out, seen = [], set()
    for col in ("COMNAM", "TICKER"):
        dup = selection[selection[col].duplicated(keep=False)]
        for name, g in dup.groupby(col):
            key = tuple(sorted(g["PERMNO"].tolist()))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"같은 회사로 보이는 종목이 함께 선정되었습니다. ({col}='{name}'): PERMNO {list(key)} "
                       f"— 수익률 상관이 높아 Sigma 조건수가 나빠지고 한 회사에 비중이 몰릴 수 있습니다.")
    return out


# =============================================================================
# 11. 한 달치 P -> Q -> Omega 조립   
# =============================================================================
# ViewGenerator: P, Q, Omega 생성을 조율합니다.
#
# 금융 로직은 각 Builder 에 있고, 여기서는
#     1) 자산 순서 고정  2) FF3 exposure 정렬  3) P  4) Q  5) Omega  6) 검증  7) ViewResult
# 순서로 호출합니다.

OmegaBuilder = _OmegaBase   # 원래: from ... import _OmegaBase as OmegaBuilder


def sample_cov(returns: pd.DataFrame, end: pd.Period, window: int, min_obs: int = 24) -> Optional[np.ndarray]:
    #end 월까지(포함) 최근 window 개월 표본 공분산. 관측이 부족하면 None.
    hist = returns.loc[:end].tail(window).dropna()
    if len(hist) < min_obs:
        return None
    return hist.cov().to_numpy()


class ViewGenerator:
    def __init__(self, p_builder: FF3ExposurePBuilder, predictor, omega_builder: OmegaBuilder,
                 cov_window: int = 60):
        self.p_builder = p_builder
        self.q_builder = ModelQBuilder(predictor)
        self.omega_builder = omega_builder
        self.cov_window = cov_window

    @property
    def predictor(self):
        return self.q_builder.predictor

    @predictor.setter
    def predictor(self, model):
        self.q_builder.predictor = model

    def generate(
        self,
        date,
        selected_assets: Sequence[int],
        ff3_estimates: pd.DataFrame,
        features_t: pd.DataFrame,
        error_history: pd.DataFrame,
        asset_returns: pd.DataFrame,
        bl_assets: Optional[Sequence[int]] = None,
        realized_history: Optional[pd.DataFrame] = None,
    ) -> ViewResult:
        """
        date            : 결정월 t (이 달 말 정보로 t+1 견해 생성)
        selected_assets : 선정 종목 PERMNO (이 순서가 곧 P 의 열 순서)
        ff3_estimates   : index=PERMNO, s_smb/h_hml 등 exposure 열
        features_t      : 결정월 t 의 feature, index=PERMNO
        error_history   : index=결정월, columns=view_names, 값=실현 오차. t 이후 행은 무시됨
        asset_returns   : 선정 종목 월수익률 (Omega prior 의 Sigma 용). t 이후 행은 무시됨
        realized_history: index=결정월, 실현 view 스프레드 P r_{t+1}. (oos_r2_confidence 용) t 이후 행은 무시됨
        """
        t = pd.Period(date, "M")
        assets = tuple(int(a) for a in selected_assets)                  # 1)
        exposures = ff3_estimates.reindex(list(assets))                  # 2)
        P, names, _ = self.p_builder.build(exposures, assets)            # 3)
        mu, Q = self.q_builder.build(P, features_t, assets)              # 4)
        Q_members = self.q_builder.build_members(P, features_t, assets)  #    앙상블이면 (M, K)

        # 5) look-ahead 방지: 결정월 t 에서 실현이 끝난 오차는 결정월 <= t-1 인 것뿐
        past = error_history.loc[error_history.index < t] if len(error_history) else error_history
        past = past.reindex(columns=list(names))
        Sigma = sample_cov(asset_returns.loc[:t, list(assets)], t, self.cov_window)
        past_real = None
        if realized_history is not None and len(realized_history):
            past_real = realized_history.loc[realized_history.index < t].reindex(columns=list(names))
        Omega, info = self.omega_builder.build(past, P, Sigma, Q_members=Q_members, past_realized=past_real)
        if Q_members is not None:
            info["q_member_std"] = Q_members.std(axis=0, ddof=1).tolist()

        prior_var = np.diag(self.omega_builder.prior(P, Sigma)) if Sigma is not None else np.full(len(names), np.nan)
        info.update({
            # 단일 view 기준 posterior 가 view 쪽으로 이동하는 비율 = prior / (prior + omega)
            "prior_var": prior_var.tolist(),
            "view_weight": (prior_var / (prior_var + np.diag(Omega))).tolist(),
        })
        result = ViewResult(str(t), assets, P, Q, Omega, names, mu, info)
        validate_view(result, bl_assets=bl_assets)                       # 6)
        return result                                                    # 7)


# =============================================================================
# 12. BL 쪽 로더   
# =============================================================================
# 저장된 view 결과(bl_input/)를 BL 모듈에서 읽기 위한 로더입니다.
#
#     from ml_bl.views import load_views
#     views = load_views("outputs/P1")            # {'2018-12-31': ViewResult, ..., '2020-11-30': ViewResult}
#     v = views["2018-12-31"]                       # 2018-12-31 리밸런싱, 2019-01-31 까지 보유
#     mu_bl = bl_model.posterior(assets=v.assets, P=v.P, Q=v.Q, Omega=v.Omega)
#
# 날짜 키는 raw data(IFE_term_stock_data.csv)의 날짜 문자열 그대로입니다.
#
# BL 의 tau 가 View 모듈 config 의 tau(기본 0.1)와 다르면 bl_tau 를 넘기세요:
#     views = load_views("outputs/P1", bl_tau=0.05)
# 본 결과 Omega(oos_r2_confidence) 는 Omega = tau * pSp' * (1-C)/C 로 tau 에 비례하므로,
# Omega 를 bl_tau / tau 배로 재조정해 '과거 OOS R^2 만큼 믿는다' (view 반영 비중 = C) 가 유지됩니다.
# tau 와 무관한 방법(ensemble_epistemic, rolling_mse_net) 은 재조정하지 않습니다.
#
# 'view 반영 비중 = C' 가 정확히 성립하려면 Omega 를 만들 때의 Sigma 도 BL 의 Sigma 와 같아야 합니다.
# BL 이 Sigma 를 다른 방식(다른 윈도우 등)으로 추정한다면 bl_sigma 를 넘기세요:
#     views = load_views("outputs/P1", bl_tau=0.1, bl_sigma=lambda date: Sigma_BL_at(date))
#     # bl_sigma(date) -> (N, N) 공분산, 행·열 순서 = views[date].assets
# 그러면 Omega = bl_tau * p Sigma_BL p' * (1-C)/C 로 다시 계산됩니다 (idzorek_omega).

TAU_DEPENDENT_SOURCES = ("oos_r2_confidence", "oos_r2_confidence_cmin", "he_litterman_prior")


def idzorek_omega(P: np.ndarray, Sigma: np.ndarray, tau: float, confidence) -> np.ndarray:
    """Idzorek confidence C -> Omega = diag(tau * p_k Sigma p_k' * (1 - C_k) / C_k).

    단일 견해에서 BL 사후 기대수익률이 견해 쪽으로 C 만큼 이동합니다 (같은 tau, Sigma 를 BL 에서 쓸 경우).
    """
    C = np.asarray(confidence, dtype=float)
    if np.any((C <= 0) | (C >= 1)):
        raise ValueError("confidence 는 (0, 1) 범위여야 합니다.")
    base = tau * np.einsum("kn,nm,km->k", P, Sigma, P)
    return np.diag(base * (1.0 - C) / C)


def load_views(output_dir: str, bl_tau: Optional[float] = None,
               bl_sigma: Optional[Callable[[str], np.ndarray]] = None) -> Dict[str, ViewResult]:
    bl = os.path.join(output_dir, "bl_input")
    dg = os.path.join(output_dir, "diagnostics")
    P = pd.read_csv(os.path.join(bl, "P.csv"), index_col="view")
    names = tuple(P.index)
    assets = tuple(int(c) for c in P.columns)
    order = pd.read_csv(os.path.join(bl, "assets.csv"))["PERMNO"].astype(int).tolist()
    if tuple(order) != assets:
        raise ValueError("assets.csv 와 P.csv 의 자산 순서가 다릅니다.")
    Qdf = pd.read_csv(os.path.join(bl, "Q.csv"), dtype={"date": str, "hold_until": str}).set_index("date")
    Odf = pd.read_csv(os.path.join(bl, "Omega.csv"), dtype={"date": str, "hold_until": str}).set_index("date")

    view_tau, method = 0.1, None
    summ_path = os.path.join(dg, "summary.json")
    if os.path.exists(summ_path):
        with open(summ_path, encoding="utf-8") as f:
            s = json.load(f)
        view_tau, method = float(s.get("omega_tau", view_tau)), s.get("omega_method")
    ts = pd.read_csv(os.path.join(dg, "views_timeseries.csv"), dtype={"date": str})
    ts = ts[ts["phase"] == "eval"].set_index(["date", "view"])
    mu = pd.read_csv(os.path.join(dg, "predicted_returns.csv"), index_col="date", dtype={"date": str})
    mu.columns = [int(c) for c in mu.columns]

    out = {}
    for d in Qdf.index:
        Om = np.array([[Odf.loc[d, f"{a}|{b}"] for b in names] for a in names], dtype=float)
        source = ts.loc[(d, names[0]), "omega_source"]
        if bl_tau is not None and source in TAU_DEPENDENT_SOURCES:
            Om = Om * (bl_tau / view_tau)
        conf = [ts.loc[(d, n), "confidence"] for n in names] if "confidence" in ts.columns else None
        if bl_sigma is not None and source in ("oos_r2_confidence", "oos_r2_confidence_cmin"):
            Sig = np.asarray(bl_sigma(d), dtype=float)
            if Sig.shape != (len(assets), len(assets)):
                raise ValueError(f"bl_sigma({d}) 의 shape {Sig.shape} != ({len(assets)}, {len(assets)})")
            Om = idzorek_omega(P.to_numpy(dtype=float), Sig, bl_tau if bl_tau is not None else view_tau, conf)
        out[d] = ViewResult(
            date=d, assets=assets, P=P.to_numpy(dtype=float),
            Q=Qdf.loc[d, list(names)].to_numpy(dtype=float), Omega=Om, view_names=names,
            predicted_returns=mu.loc[d, list(assets)].to_numpy(dtype=float),
            metadata={"hold_until": Qdf.loc[d, "hold_until"], "omega_source": source, "omega_method": method,
                      "confidence": conf,
                      "omega_tau": bl_tau if (bl_tau is not None and source in TAU_DEPENDENT_SOURCES) else view_tau},
        )
    return out


# =============================================================================
# 13. 파이프라인 (walk-forward, 저장)  
# =============================================================================
# View 생성 전체 파이프라인 (walk-forward).
#
# 시간축 (P1 예시)
#     ~2012-12            : 초기 학습 구간
#     2013-01 ~ 2018-11   : [oos_history] 결정월. 12개월마다 재학습하며 OOS 예측 ->
#                           실현 오차를 쌓아 Omega 이력을 만듭니다. (선정·학습 구간 안).
#     2018-12 ~ 2020-11   : [eval] 결정월. 2018-12 까지의 데이터로 학습한 모델을 고정하고
#                           매월 Q 를 예측, 실현 오차가 쌓이는 대로 Omega 갱신합니다.
#                           (보유월 = 2019-01 ~ 2020-12)
#
# 결정월 t 의 ViewResult 는 t 말까지의 정보만 사용합니다.

log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class PipelineOutput:
    config: dict
    selection: pd.DataFrame
    membership: pd.DataFrame
    results: Dict[str, ViewResult]
    records: pd.DataFrame              # 결정월 x view 단위 시계열 (Q, Q_real, error, omega ...)
    ic: pd.DataFrame                   # 결정월별 예측력 (Spearman IC)
    warnings: List[str] = field(default_factory=list)
    features_used: List[str] = field(default_factory=list)
    dates: Optional[pd.Series] = None      # ym(Period) -> 원자료 날짜 'YYYY-MM-DD'


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()))


def run_pipeline(cfg: dict, root: str = ".") -> PipelineOutput:
    P_ = lambda p: p if (p is None or os.path.isabs(p)) else os.path.join(root, p)  # noqa: E731
    paths, smp, vcfg = cfg["paths"], cfg["sample"], cfg["view"]
    seed = cfg.get("seed", 42)
    warnings: List[str] = list(version_warnings())
    for w in warnings:
        log.warning(w)

    # ---- 입력 --------------------------------------------------------------
    selection = load_selection(P_(paths["selection"]), cfg["selection"]["n_assets"],
                               dedupe=cfg["selection"].get("dedupe_same_company", "most_liquid"))
    for _, d in selection.attrs.get("dropped", pd.DataFrame()).iterrows():
        warnings.append(f"같은 회사 중복 제외: PERMNO {int(d['PERMNO'])} ({d['TICKER']}, rank {int(d['rank'])}) "
                        f"— 회사당 한 종목 규칙 ({cfg['selection'].get('dedupe_same_company', 'most_liquid')})")
    assets = selection["PERMNO"].astype(int).tolist()
    universe = sorted(set(load_universe(P_(paths["universe"]))) | set(assets))
    warnings += same_company_warnings(selection)

    md = load_stock_panel(P_(paths["stock_data"]), start=smp["data_start"], end=smp["eval_end"])
    md = md.restrict(permnos=universe)

    qcfg = vcfg["q"]
    fb = ViewFeatureBuilder(transform=qcfg.get("feature_transform", "rank"))
    X_all = fb.build(md)
    y_all = build_target(md.ret, qcfg.get("target", "raw")).reindex(X_all.index)
    feature_names = X_all.columns.tolist()

    # ---- 시간축 ------------------------------------------------------------
    train_end = pd.Period(smp["train_end"], "M")
    hist_start = pd.Period(smp["oos_history_start"], "M")
    eval_start = pd.Period(smp["eval_start"], "M")
    eval_end = pd.Period(smp["eval_end"], "M")
    hist_dec = pd.period_range(hist_start, train_end - 1, freq="M")          # 타깃월 <= train_end
    eval_dec = pd.period_range(eval_start - 1, eval_end - 1, freq="M")        # 보유월 eval_start..eval_end
    if eval_start - 1 != train_end:
        warnings.append("eval_start 가 train_end 바로 다음 달이 아닙니다.")

    # ---- 모듈 --------------------------------------------------------------
    pcfg, ocfg = vcfg["p"], vcfg["omega"]
    p_builder = FF3ExposurePBuilder(pcfg["exposures"], pcfg.get("group_frac", 0.25),
                                    method=pcfg.get("method", "double_sort"), grid=pcfg.get("grid", 2))
    omega_builder = make_omega_builder(ocfg)
    pred_name = qcfg["predictor"]
    pred_params = cfg.get("predictors", {}).get(pred_name, {})
    n_members = int(qcfg.get("n_members", 1))
    if ocfg.get("method", "oos_r2_confidence") == "ensemble_epistemic" and n_members < 2:
        raise ValueError("omega.method=ensemble_epistemic 은 view.q.n_members >= 2 가 필요합니다.")
    gen = ViewGenerator(p_builder, predictor=None, omega_builder=omega_builder, cov_window=ocfg["cov_window"])

    exposures = selection.set_index("PERMNO")
    _, view_names, membership = p_builder.build(exposures, assets)
    asset_ret = md.ret[assets]
    ym_level = X_all.index.get_level_values("ym")

    def fit_model(t: pd.Period):
        """결정월 t 에서 학습: 타깃까지 실현된 표본 = 결정월 <= t-1."""
        m = (ym_level <= t - 1) & y_all.notna().to_numpy()
        if n_members > 1:
            model = BootstrapEnsemble(pred_name, n_members=n_members, seed=seed, **pred_params)
        else:
            model = make_predictor(pred_name, seed=seed, **pred_params)
        model.fit(X_all[m], y_all[m])
        log.info("fit @ %s: %d samples (through %s), members=%d", t, int(m.sum()), t - 1, n_members)
        return model, t - 1

    error_hist = pd.DataFrame(columns=list(view_names), dtype=float)
    real_hist = pd.DataFrame(columns=list(view_names), dtype=float)
    results: Dict[str, ViewResult] = {}
    rows, ic_rows = [], []
    model, trained_through, last_fit = None, None, None
    refit_every = int(qcfg.get("refit_every", 12))

    for phase, decisions in (("oos_history", hist_dec), ("eval", eval_dec)):
        for t in decisions:
            need_fit = model is None or (
                (phase == "oos_history" or qcfg.get("refit_in_eval", False)) and (t - last_fit).n >= refit_every)
            if phase == "eval" and t == eval_dec[0] and not qcfg.get("refit_in_eval", False):
                need_fit = True        # 평가 직전 train_end 까지 전부로 한 번 학습 후 고정
            if need_fit:
                model, trained_through = fit_model(t)
                last_fit = t
                gen.predictor = model

            X_t = X_all.xs(t, level="ym") if t in ym_level else pd.DataFrame(columns=feature_names)
            res = gen.generate(t, assets, exposures, X_t, error_hist, asset_ret, realized_history=real_hist)
            res.metadata.update({"phase": phase, "model": pred_name, "trained_through": str(trained_through)})
            results[str(t)] = res

            # t+1 실현 -> 오차 기록 (다음 결정월부터만 Omega 에 쓰임: generator 가 index < t 로 필터)
            r_next = asset_ret.loc[t + 1].to_numpy(dtype=float) if (t + 1) in asset_ret.index else np.full(len(assets), np.nan)
            Q_real = res.P @ r_next
            err = Q_real - res.Q
            error_hist.loc[t] = err
            real_hist.loc[t] = Q_real

            for k, name in enumerate(view_names):
                rows.append({
                    "date": str(t), "holding_month": str(t + 1), "phase": phase, "view": name,
                    "Q": res.Q[k], "Q_real": Q_real[k], "error": err[k],
                    "omega": res.Omega[k, k], "omega_source": res.metadata["omega_source"],
                    "n_errors": res.metadata["n_errors"], "prior_var": res.metadata["prior_var"][k],
                    "view_weight": res.metadata["view_weight"][k],
                    "q_member_std": res.metadata.get("q_member_std", [np.nan] * len(view_names))[k],
                    "confidence": res.metadata.get("confidence", [np.nan] * len(view_names))[k],
                })
            # 예측력: 선정 종목 / 유니버스 전체 Spearman IC (raw 다음 달 수익률 기준)
            ic_u = np.nan
            if len(X_t):
                mu_u = model.predict(X_t)
                r_u = md.ret.loc[t + 1].reindex(X_t.index).to_numpy(dtype=float) if (t + 1) in md.ret.index else np.full(len(X_t), np.nan)
                ic_u = _spearman(mu_u, r_u)
            ic_rows.append({"date": str(t), "phase": phase, "ic_selected": _spearman(res.predicted_returns, r_next),
                            "ic_universe": ic_u})

    records = pd.DataFrame(rows)
    ic = pd.DataFrame(ic_rows)
    return PipelineOutput(cfg, selection, membership, results, records, ic, warnings, feature_names, md.dates)


# ---- 요약 / 저장 -------------------------------------------------------------

def summarize(out: PipelineOutput) -> dict:
    v = out.config["view"]
    summ = {"period": out.config.get("period"), "predictor": v["q"]["predictor"],
            "p_method": v["p"].get("method", "double_sort"), "omega_method": v["omega"].get("method", "oos_r2_confidence"),
            "n_members": int(v["q"].get("n_members", 1)),
            "omega_tau": float(v["omega"].get("tau", 0.1)),
            "n_assets": len(out.selection), "features": out.features_used, "warnings": out.warnings, "phases": {}}
    for phase, g in out.records.groupby("phase"):
        ph = {"n_months": int(g["date"].nunique()), "views": {}}
        for view, gv in g.groupby("view"):
            gv = gv.dropna(subset=["Q_real"])
            ph["views"][view] = {
                "hit_rate_sign": float((np.sign(gv["Q"]) == np.sign(gv["Q_real"])).mean()),
                "corr_Q_Qreal": float(gv["Q"].corr(gv["Q_real"])),
                "rmse": float(np.sqrt((gv["error"] ** 2).mean())),
                "mean_abs_Q": float(gv["Q"].abs().mean()),
                "std_Q_real": float(gv["Q_real"].std()),
                "mean_view_weight": float(gv["view_weight"].mean()),
                "mean_omega": float(gv["omega"].mean()),
                "mean_prior_var": float(gv["prior_var"].mean()),
                "omega_sources": gv["omega_source"].value_counts().to_dict(),
            }
        gi = out.ic[out.ic["phase"] == phase]
        ph["ic_selected_mean"] = float(gi["ic_selected"].mean())
        ph["ic_universe_mean"] = float(gi["ic_universe"].mean())
        ph["ic_universe_tstat"] = float(gi["ic_universe"].mean() / gi["ic_universe"].std() * np.sqrt(gi["ic_universe"].notna().sum()))
        summ["phases"][phase] = ph
    return summ


def _raw_date(out: PipelineOutput, ym) -> str:
    """Period -> 원자료 날짜 문자열 (없으면 월말)."""
    p = pd.Period(ym, "M")
    if out.dates is not None and p in out.dates.index:
        return str(out.dates.loc[p])
    return p.to_timestamp(how="end").strftime("%Y-%m-%d")


def export(out: PipelineOutput, output_dir: str) -> dict:
    """저장 구조

    <output_dir>/
      views_readable.csv      사람이 읽는 표: 날짜별 견해 (롱/숏 종목, Q %, 신뢰도 ...)
      bl_input/               ★ BL 에 넣는 파일 (평가 구간만)
        assets.csv            자산 순서 = P 의 열 순서 (PERMNO, TICKER, 회사명)
        P.csv                 K x N 견해 행렬 (행 = 견해, 열 = PERMNO)
        Q.csv                 날짜별 Q (행 = 날짜, 열 = 견해)
        Omega.csv             날짜별 Omega (행 = 날짜, 열 = '견해i|견해j')
        confidence.csv        날짜별 Idzorek confidence C (oos_r2_confidence 일 때)
      diagnostics/            검증·분석용 (전체 구간)

    날짜 규약: date = 견해를 만든 날 = 리밸런싱 날 (원자료의 그 달 마지막 거래일, 이 날까지의 정보만 사용)
              hold_until = 다음 리밸런싱 날 (이 사이의 수익률 = 원자료에서 hold_until 행의 수익률)
    """
    os.makedirs(output_dir, exist_ok=True)
    bl_dir = os.path.join(output_dir, "bl_input")
    dg_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(bl_dir, exist_ok=True)
    os.makedirs(dg_dir, exist_ok=True)

    any_res = next(iter(out.results.values()))
    names = list(any_res.view_names)
    assets = list(any_res.assets)
    tick = out.selection.set_index("PERMNO")["TICKER"]

    # ---- bl_input ---------------------------------------------------------
    sel = out.selection[["PERMNO", "TICKER", "COMNAM"]].copy()
    sel = sel.set_index("PERMNO").loc[assets].reset_index()
    sel.insert(0, "order", range(len(sel)))
    sel.to_csv(os.path.join(bl_dir, "assets.csv"), index=False)
    any_res.P_frame().to_csv(os.path.join(bl_dir, "P.csv"), index_label="view")

    ev = [(d, r) for d, r in out.results.items() if r.metadata.get("phase") == "eval"]
    q_rows, om_rows, c_rows = [], [], []
    for d, r in ev:
        base = {"date": _raw_date(out, d), "hold_until": _raw_date(out, pd.Period(d, "M") + 1)}
        q_rows.append({**base, **dict(zip(names, r.Q))})
        om = dict(base)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                om[f"{a}|{b}"] = r.Omega[i, j]
        om_rows.append(om)
        conf = r.metadata.get("confidence")
        if conf is not None:
            c_rows.append({**base, **dict(zip(names, conf))})
    pd.DataFrame(q_rows).to_csv(os.path.join(bl_dir, "Q.csv"), index=False)
    pd.DataFrame(om_rows).to_csv(os.path.join(bl_dir, "Omega.csv"), index=False)
    if c_rows:   # Idzorek confidence C: BL 이 자기 Sigma·tau 로 Omega 를 다시 만들 때 사용
        pd.DataFrame(c_rows).to_csv(os.path.join(bl_dir, "confidence.csv"), index=False)

    # ---- views_readable.csv ------------------------------------------------
    members = {n: ([tick[a] for a, x in zip(assets, any_res.P[k]) if x > 0],
                   [tick[a] for a, x in zip(assets, any_res.P[k]) if x < 0]) for k, n in enumerate(names)}
    rec = out.records.copy()
    rec["date_raw"] = rec["date"].map(lambda d: _raw_date(out, d))
    rec["hold_raw"] = rec["holding_month"].map(lambda d: _raw_date(out, d))
    evr = rec[rec["phase"] == "eval"]
    readable = pd.DataFrame({
        "날짜(리밸런싱)": evr["date_raw"],
        "보유 종료일": evr["hold_raw"],
        "견해": evr["view"],
        "롱 종목": evr["view"].map(lambda n: " ".join(members[n][0])),
        "숏 종목": evr["view"].map(lambda n: " ".join(members[n][1])),
        "예측 스프레드 Q(%/월)": (evr["Q"] * 100).round(3),
        "실현 스프레드(%/월)": (evr["Q_real"] * 100).round(3),
        "신뢰도 C(OOS R²)": evr["confidence"].round(3),
        "Omega": evr["omega"],
        "BL 반영비중(τ=config)": evr["view_weight"].round(4),
    })
    readable.to_csv(os.path.join(output_dir, "views_readable.csv"), index=False, encoding="utf-8-sig")

    # ---- diagnostics -------------------------------------------------------
    diag = rec.drop(columns=["date", "holding_month"]).rename(columns={"date_raw": "date", "hold_raw": "hold_until"})
    diag = diag[["date", "hold_until"] + [c for c in diag.columns if c not in ("date", "hold_until")]]
    diag.to_csv(os.path.join(dg_dir, "views_timeseries.csv"), index=False)
    mu = pd.DataFrame({_raw_date(out, d): r.predicted_returns for d, r in out.results.items()}, index=assets).T
    mu.index.name = "date"
    mu.to_csv(os.path.join(dg_dir, "predicted_returns.csv"))
    ic = out.ic.copy()
    ic["date"] = ic["date"].map(lambda d: _raw_date(out, d))
    ic.to_csv(os.path.join(dg_dir, "prediction_ic.csv"), index=False)

    summ = summarize(out)
    with open(os.path.join(dg_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    return summ


# =============================================================================
# 14. 설정 
# =============================================================================
CONFIGS: Dict[str, str] = {
    "P1_2000_2018": """\
# 실험 P1: 선정 구간 2000-01 ~ 2018-12, 평가 2019-01 ~ 2020-12
period: P1
seed: 42

paths:
  stock_data: data/raw/IFE_term_stock_data.csv
  selection: data/selection/ff3_selected_2000~2018.csv     # FF3 selection 결과 (rank 순)
  universe: data/selection/ff3_regression_2000~2018.csv    # 해당 구간 balanced 유니버스 (ML 학습용)
  output_dir: outputs/P1

sample:
  data_start: "2000-01"
  train_end: "2018-12"          # 선정·학습은 여기까지 (마지막 학습 타깃 월)
  oos_history_start: "2013-01"  # walk-forward OOS 예측 시작 (Omega용 오차 이력)
  eval_start: "2019-01"         # 평가 첫 보유월
  eval_end: "2020-12"

selection:
  n_assets: 20
  dedupe_same_company: most_liquid   # 회사당 거래량 최대 클래스만 (most_liquid | best_rank | none)

view:
  p:                            # Ko, Son & Lee (2024)
    method: double_sort         # double_sort: 순차 이중정렬 단일 견해 (K=1) 
    exposures: [s_smb, h_hml]   # [size 대리변수, value 대리변수]  (실제 size·B/M 이 원자료에 없어서 FF3 계수 사용)
    grid: 2                     # 2x2 (칸당 5종목). 원 논문 5x5 는 N=20 에서 칸이 빔
    group_frac: 0.25            # independent 에서만 사용
  q:                            # Barua & Sharma (2023)
    predictor: xgboost
    n_members: 1                # 1 = 단일 XGBoost (논문). ensemble_epistemic 비교 결과에서만 10
    target: raw                 # 다음 달 산술수익률
    feature_transform: rank     # 월별 횡단면 순위 -> [-1, 1]  (논문: 지표를 [-1, 1] 로 스케일)
    refit_every: 12             # walk-forward 재학습 주기(개월)
    refit_in_eval: false        # false = 2018-12까지 학습한 모델을 평가기간에 고정
  omega:
    method: oos_r2_confidence   # oos_r2_confidence: Barua & Sharma 의 Idzorek 변형 (본 결과)
                                #   | ensemble_epistemic: Spears 의 Boost 방식 (비교 결과, n_members >= 2 필요)
                                #   | rolling_mse_net: Spears 의 ARIMA 방식 (옵션)
    structure: diag
    window: 24                  # 최근 24개월 OOS 오차로 R^2 계산
    min_obs: 12                 # 이보다 적으면 C = c_min
    c_min: 0.01                 # confidence 하한 (예측력 없음 -> view 반영 1%)
    c_max: 0.9                  # confidence 상한
    tau: 0.1                    # Ko·Barua 논문값. Omega 는 tau 에 비례 -> BL tau 가 다르면 load_views(..., bl_tau=) 로 재조정
    cov_window: 60              # Sigma 추정 윈도우 (개월)
    min_variance: 1.0e-6
    scale: 1.0

predictors:
  xgboost:                      # Barua & Sharma (2023) Sec. 3.4: 트리 300개, 나머지 기본값
    n_estimators: 300
""",
    "P2_2010_2018": """\
# 실험 P2: 선정 구간 2010-01 ~ 2018-12, 평가 2019-01 ~ 2020-12
period: P2
seed: 42

paths:
  stock_data: data/raw/IFE_term_stock_data.csv
  selection: data/selection/ff3_selected_2010~2018.csv     # FF3 selection 결과 (rank 순)
  universe: data/selection/ff3_regression_2010~2018.csv    # 해당 구간 balanced 유니버스 (ML 학습용)
  output_dir: outputs/P2

sample:
  data_start: "2010-01"
  train_end: "2018-12"          # 선정·학습은 여기까지 (마지막 학습 타깃 월)
  oos_history_start: "2015-01"  # walk-forward OOS 예측 시작 (Omega용 오차 이력)
  eval_start: "2019-01"         # 평가 첫 보유월
  eval_end: "2020-12"

selection:
  n_assets: 20
  dedupe_same_company: most_liquid   # 회사당 거래량 최대 클래스만 (most_liquid | best_rank | none)

view:
  p:                            # Ko, Son & Lee (2024)
    method: double_sort         # double_sort: 순차 이중정렬 단일 견해 (K=1)  
    exposures: [s_smb, h_hml]   # [size 대리변수, value 대리변수]  (실제 size·B/M 이 원자료에 없어서 FF3 계수 사용)
    grid: 2                     # 2x2 (칸당 5종목). 원 논문 5x5 는 N=20 에서 칸이 빔
    group_frac: 0.25            # independent 에서만 사용
  q:                            # Barua & Sharma (2023)
    predictor: xgboost
    n_members: 1                # 1 = 단일 XGBoost (논문). ensemble_epistemic 비교 결과에서만 10
    target: raw                 # 다음 달 산술수익률
    feature_transform: rank     # 월별 횡단면 순위 -> [-1, 1]  (논문: 지표를 [-1, 1] 로 스케일)
    refit_every: 12             # walk-forward 재학습 주기(개월)
    refit_in_eval: false        # false = 2018-12까지 학습한 모델을 평가기간에 고정
  omega:
    method: oos_r2_confidence   # oos_r2_confidence: Barua & Sharma 의 Idzorek 변형 (본 결과)
                                #   | ensemble_epistemic: Spears 의 Boost 방식 (비교 결과, n_members >= 2 필요)
                                #   | rolling_mse_net: Spears 의 ARIMA 방식 (옵션)
    structure: diag
    window: 24                  # 최근 24개월 OOS 오차로 R^2 계산
    min_obs: 12                 # 이보다 적으면 C = c_min
    c_min: 0.01                 # confidence 하한 (예측력 없음 -> view 반영 1%)
    c_max: 0.9                  # confidence 상한
    tau: 0.1                    # Ko·Barua 논문값. Omega 는 tau 에 비례 -> BL tau 가 다르면 load_views(..., bl_tau=) 로 재조정
    cov_window: 60              # Sigma 추정 윈도우 (개월)
    min_variance: 1.0e-6
    scale: 1.0

predictors:
  xgboost:                      # Barua & Sharma (2023) Sec. 3.4: 트리 300개, 나머지 기본값
    n_estimators: 300
""",
}


# =============================================================================
# 15. 실행 
# =============================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))     # 데이터 경로(data/...)의 기준 폴더 기본값

# 번호: (선정 구간, 설정 이름)
PERIODS = {
    "1": ("2000~2018", "P1_2000_2018"),
    "2": ("2010~2018", "P2_2010_2018"),
}
_ALIASES = {**{k: k for k in PERIODS}, **{label: k for k, (label, _) in PERIODS.items()},
            **{label.replace("~", "_"): k for k, (label, _) in PERIODS.items()}}


def resolve_period(choice) -> str:
    """'1' / '2000~2018' / '2000_2018' -> 설정 이름 ('P1_2000_2018'). 잘못된 값이면 ValueError."""
    key = _ALIASES.get(str(choice).strip())
    if key is None:
        raise ValueError(f"잘못된 선택: {choice!r} (가능: {', '.join(PERIODS)})")
    return PERIODS[key][1]


def get_config(name_or_path: str) -> dict:
    """내장 설정 이름('P1_2000_2018') 또는 yaml 파일 경로 -> dict (매번 새 복사본)."""
    if name_or_path in CONFIGS:
        return yaml.safe_load(CONFIGS[name_or_path])
    return load_config(name_or_path)


_SKIP_DIRS = {"outputs", ".git", ".venv", "venv", "__pycache__", "__MACOSX", "site-packages",
              "sample_data", "view_env", ".config", "node_modules"}


def _norm_name(name: str) -> str:
    return os.path.basename(name).replace(" ", "").replace("&", "").lower()


def _find_data(cfg: dict, root: str, verbose: bool = True) -> dict:
    """설정의 데이터 파일(data/raw/..., data/selection/...)이 정해진 위치에 없으면 찾아서 맞춘다.

    root 아래(3단계 깊이까지)의 같은 이름 파일, 또는 zip 안의 같은 이름 파일을 찾는다.
    - 풀린 파일: 그 경로를 그대로 사용
    - zip 안의 파일: 정해진 위치(data/...)로 꺼냄
    그래서 view_module.py 옆에 CSV 와 FF3 zip 을 그냥 올려 두기만 해도 실행된다.
    """
    cfg = dict(cfg, paths=dict(cfg["paths"]))
    keys = ("stock_data", "selection", "universe")
    todo = {k: cfg["paths"][k] for k in keys
            if not os.path.exists(cfg["paths"][k] if os.path.isabs(cfg["paths"][k]) else os.path.join(root, cfg["paths"][k]))}
    if not todo:
        return cfg
    files = []
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS and not d.startswith(".")]
        if dp.count(os.sep) - base_depth >= 3:
            dns[:] = []
        files += [os.path.join(dp, f) for f in fns]
    want = {_norm_name(rel): k for k, rel in todo.items()}
    for f in sorted(files):                                   # ① 풀린 파일
        k = want.get(_norm_name(f))
        if k and k in todo:
            cfg["paths"][k] = f
            todo.pop(k)
            if verbose:
                print(f"[데이터] {os.path.relpath(f, root)} 사용")
    for f in sorted(files):                                   # ② zip 안의 파일
        if not todo or not f.lower().endswith(".zip"):
            continue
        try:
            with zipfile.ZipFile(f) as z:
                for m in z.namelist():
                    k = want.get(_norm_name(m))
                    if k and k in todo and not m.endswith("/") and "__MACOSX" not in m:
                        dst = os.path.join(root, todo[k])
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with z.open(m) as src, open(dst, "wb") as out:
                            shutil.copyfileobj(src, out)
                        todo.pop(k)
                        if verbose:
                            print(f"[데이터] {os.path.basename(f)} 에서 {os.path.basename(m)} 꺼냄 -> {os.path.relpath(dst, root)}")
        except (zipfile.BadZipFile, OSError):
            continue
    if todo:
        raise FileNotFoundError(
            "데이터 파일을 찾지 못했습니다: " + ", ".join(os.path.basename(v) for v in todo.values())
            + f"\n  {root} 아래에 해당 파일(또는 그 파일이 든 zip)을 두거나, data/raw, data/selection 폴더에 넣어 주세요.")
    return cfg


def run(period: Optional[str] = None, config: Optional[str] = None, output_dir: Optional[str] = None,
        root: Optional[str] = None, verbose: bool = True) -> dict:
    """View 모듈 실행: 설정 선택 -> run_pipeline -> export. 요약(summary.json 내용) 반환."""
    root = os.path.abspath(root or ROOT)
    name = config if config else resolve_period(period)
    cfg = get_config(name)
    if output_dir:
        cfg["paths"]["output_dir"] = output_dir
    if verbose:
        print(f"[실행] {name}  (선정 파일: {os.path.basename(cfg['paths']['selection'])})")
    cfg = _find_data(cfg, root, verbose)
    out = run_pipeline(cfg, root=root)
    out_dir = cfg["paths"]["output_dir"]
    summ = export(out, out_dir if os.path.isabs(out_dir) else os.path.join(root, out_dir))
    if verbose:
        for w in out.warnings:
            print("[경고]", w)
        print(json.dumps({k: v for k, v in summ.items() if k != "features"}, ensure_ascii=False, indent=2))
        print(f"[완료] 결과 폴더: {out_dir}")
    return summ


def run_all(root: Optional[str] = None, verbose: bool = True) -> dict:
    """P1(2000~2018), P2(2010~2018)을 차례로 모두 실행. {설정 이름: 요약} 반환."""
    return {name: run(config=name, root=root, verbose=verbose) for _, name in PERIODS.values()}


def main():
    ap = argparse.ArgumentParser(description="View 모듈 (P, Q, Omega) 생성. 옵션 없이 실행하면 P1, P2 를 모두 만듭니다.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--period", help="하나만 실행할 때: 1 = 2000~2018, 2 = 2010~2018")
    g.add_argument("--config", help="yaml 설정 파일 직접 지정")
    ap.add_argument("--output-dir", help="설정의 paths.output_dir 덮어쓰기 (--period / --config 와 함께)")
    ap.add_argument("--root", help="data/, outputs/ 의 기준 폴더 (기본: 이 파일이 있는 폴더)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.config or args.period:                       # 하나만 실행
        if args.config:
            name = args.config
        else:
            try:
                name = resolve_period(args.period)
            except ValueError as e:
                ap.error(str(e))
        run(config=name, output_dir=args.output_dir, root=args.root)
        return
    if args.output_dir:
        ap.error("--output-dir 는 --period 나 --config 와 함께 쓰세요 (P1, P2 를 모두 만들 때는 outputs/P1, outputs/P2 에 저장).")
    run_all(root=args.root)                               # 기본: P1, P2 모두
    root = os.path.abspath(args.root or ROOT)
    print("\n[전체 완료] " + ", ".join(
        f"{label} -> {get_config(name)['paths']['output_dir']}" for label, name in PERIODS.values())
          + f"   (기준 폴더: {root})")


if __name__ == "__main__":
    main()
