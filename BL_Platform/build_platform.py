#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BL 플랫폼 빌더 — Module_Based_ML 의 견해 모듈 결과를 읽어 HTML 한 파일로 만든다.
=========================================================================
실행:   python build_platform.py
결과:   같은 폴더의 index.html  (더블클릭으로 열림. 인터넷·서버 불필요)
필요:   numpy, pandas, plotly (차트 라이브러리를 HTML 에 내장하기 위해서만 사용)

읽는 것
  Module_Based_ML/<모듈 폴더>/     아래의 모든 bl_input/ 폴더 (zip 안에 있어도 됨)
      bl_input/assets.csv, P.csv, Q.csv, Omega.csv  (필수)
      bl_input/confidence.csv, constraints.json, ../diagnostics/predicted_returns.csv  (있으면 사용)
    bl_input 이 하나도 없는 모듈 폴더는 드롭다운에 '결과 없음'으로만 표시한다.
    새 모듈 결과가 생기면 이 스크립트를 다시 실행하면 자동으로 추가된다.
  Project_Data/IFE_term_stock_data.csv, S&P500_Data.csv, ff3_factors.csv
  Fama French 3 factor model security selection/.../ff3_selected_<기간>.csv  (있으면 종목 카드에 FF3 지표 표시)
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE  # 모듈·데이터 폴더를 모두 BL_Platform 안에 둔다
MODULE_ROOT = PROJECT / "Module_Based_ML"
DATA_DIR = PROJECT / "Project_Data"
FF3_DIR = PROJECT / "Fama French 3 factor model security selection"

START_YEAR, END_YEAR = 2000, 2020
N_MONTHS = (END_YEAR - START_YEAR + 1) * 12
COST_ONE_WAY = 0.001
PERIOD_INFO = {"P1": ("P1 · 2000~2018 계속 상장", "2000~2018"), "P2": ("P2 · 2010~2018 계속 상장", "2010~2018")}
MODEL_NAMES = {"ann": "ANN", "elasticnet": "ElasticNet", "xgboost": "XGBoost"}


def month_label(i):
    return "%d-%02d" % (START_YEAR + i // 12, i % 12 + 1)


def month_idx(s):
    d = pd.to_datetime(s)
    return (d.year - START_YEAR) * 12 + d.month - 1


def num(v, d=8):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# 모듈 폴더 탐색: 일반 폴더와 zip을 같은 방식으로 읽는다
# ---------------------------------------------------------------------------
class DirSource:
    def __init__(self, base: Path):
        self.base = base

    def read(self, rel):
        p = self.base / rel
        return p.read_text(encoding="utf-8-sig") if p.is_file() else None

    def where(self):
        return str(self.base.relative_to(PROJECT))


class ZipSource:
    def __init__(self, zpath: Path, prefix: str):
        self.z, self.prefix, self.zpath = zipfile.ZipFile(zpath), prefix, zpath

    def read(self, rel):
        name = self.prefix + rel
        try:
            return self.z.read(name).decode("utf-8-sig")
        except KeyError:
            return None

    def where(self):
        return str(self.zpath.relative_to(PROJECT)) + " > " + self.prefix.rstrip("/")


def find_period_sources(folder: Path):
    """폴더 안에서 <모델>/<기간>/bl_input/ 구조를 모두 찾는다. -> [(모델 폴더명, 기간명, source)]"""
    out = []
    for p in sorted(folder.rglob("bl_input")):
        if ".venv" in p.parts or not p.is_dir():
            continue
        period_dir = p.parent
        out.append((period_dir.parent.name, period_dir.name, DirSource(period_dir)))
    for zp in sorted(folder.rglob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        for nm in names:
            m = re.match(r"^(.*?)([^/]+)/([^/]+)/bl_input/assets\.csv$", nm)
            if m:
                out.append((m.group(2), m.group(3), ZipSource(zp, m.group(1) + m.group(2) + "/" + m.group(3) + "/")))
    return out


def model_name(model_dir: str, folder: str):
    """outputs_ANN -> ANN,  views/outputs 처럼 모델명이 없으면 모듈 폴더명(view_module_xgboost(근석) -> XGBoost)."""
    key = model_dir[len("outputs_"):] if model_dir.lower().startswith("outputs_") else ""
    if not key:
        key = re.sub(r"\(.*?\)", "", folder)
        key = re.sub(r"(?i)view_module_?|module", "", key).strip(" _-")
    return MODEL_NAMES.get(key.lower(), key or folder)


def read_csv_text(txt):
    return pd.read_csv(io.StringIO(txt))


def load_period(src):
    assets = read_csv_text(src.read("bl_input/assets.csv")).sort_values("order")
    permnos = [str(int(x)) for x in assets["PERMNO"]]
    Pdf = read_csv_text(src.read("bl_input/P.csv"))
    Qdf = read_csv_text(src.read("bl_input/Q.csv"))
    Odf = read_csv_text(src.read("bl_input/Omega.csv"))
    ctxt = src.read("bl_input/confidence.csv")
    Cdf = read_csv_text(ctxt) if ctxt else None
    cons_txt = src.read("bl_input/constraints.json")
    cons = json.loads(cons_txt) if cons_txt else None

    view_names = [c for c in Qdf.columns if c not in ("date", "hold_until")]
    views = []
    for _, q in Qdf.iterrows():
        d = str(q["date"])[:10]
        for vn in view_names:
            if "date" in Pdf.columns:                               # 상준: 날짜별 P
                row = Pdf[(Pdf["date"].astype(str).str[:10] == d) & (Pdf["view"] == vn)]
            else:                                                   # 근석: 고정 P
                row = Pdf[Pdf["view"] == vn]
            if row.empty:
                raise ValueError("P.csv 에 %s / %s 행이 없습니다." % (d, vn))
            P = [float(row.iloc[0][p]) for p in permnos]
            orow = Odf[Odf["date"].astype(str).str[:10] == d]
            ocol = vn + "|" + vn
            omega = float(orow.iloc[0][ocol]) if not orow.empty and ocol in orow.columns else None
            conf = None
            if Cdf is not None:
                crow = Cdf[Cdf["date"].astype(str).str[:10] == d]
                if not crow.empty and vn in crow.columns:
                    conf = float(crow.iloc[0][vn])
            views.append({"date": d, "name": vn, "P": P, "Q": num(q[vn], 10), "omega": num(omega, 10), "conf": num(conf, 6)})

    pred = None
    ptxt = src.read("diagnostics/predicted_returns.csv")
    if ptxt:
        pdf = read_csv_text(ptxt)
        dates = {v["date"] for v in views}
        pdf = pdf[pdf["date"].astype(str).str[:10].isin(dates)]
        pred = {str(r["date"])[:10]: [num(r[p], 6) for p in permnos] for _, r in pdf.iterrows()}

    return {
        "permnos": [int(p) for p in permnos],
        "tickers": [str(t) for t in assets["TICKER"]],
        "names": [str(n).title() for n in assets["COMNAM"]],
        "views": views, "pred": pred, "constraints": cons, "source": src.where(),
        "hasConf": Cdf is not None,
    }


def discover_modules():
    if not MODULE_ROOT.is_dir():
        sys.exit("[오류] Module_Based_ML 폴더를 찾지 못했습니다: %s" % MODULE_ROOT)
    modules = []
    for folder in sorted(d for d in MODULE_ROOT.iterdir() if d.is_dir()):
        owner_m = re.search(r"\(([^)]+)\)", folder.name)
        owner = owner_m.group(1) if owner_m else folder.name
        found = find_period_sources(folder)
        by_model = {}
        for mdir, period, src in found:
            by_model.setdefault(model_name(mdir, folder.name), {})[period] = src
        if not by_model:
            modules.append({"id": "m%d" % len(modules), "owner": owner, "model": model_name("", folder.name),
                            "folder": folder.name, "available": False, "periods": {}})
            print("[모듈] %-28s → bl_input 결과 없음 (드롭다운에 비활성으로 표시)" % folder.name)
            continue
        for model in sorted(by_model):
            periods = {}
            for period in sorted(by_model[model]):
                periods[period] = load_period(by_model[model][period])
                print("[모듈] %-28s → %s %s: 견해 %d개, 종목 %d개  (%s)" % (
                    folder.name, model, period, len(periods[period]["views"]), len(periods[period]["tickers"]),
                    periods[period]["source"]))
            modules.append({"id": "m%d" % len(modules), "owner": owner, "model": model, "folder": folder.name,
                            "available": True, "periods": periods})
    return modules


# ---------------------------------------------------------------------------
# 원자료
# ---------------------------------------------------------------------------
def load_market(permnos):
    stock = DATA_DIR / "IFE_term_stock_data.csv"
    df = pd.read_csv(stock, usecols=["PERMNO", "date", "TICKER", "COMNAM", "ADJ_PRC"], parse_dates=["date"])
    df = df[df["PERMNO"].isin(permnos)].dropna(subset=["ADJ_PRC"])
    df = df[df["ADJ_PRC"] > 0].copy()
    df["mi"] = (df["date"].dt.year - START_YEAR) * 12 + df["date"].dt.month - 1
    df = df[(df["mi"] >= 0) & (df["mi"] < N_MONTHS)].sort_values(["PERMNO", "date"]).drop_duplicates(["PERMNO", "mi"])
    prc = df.pivot(index="mi", columns="PERMNO", values="ADJ_PRC").reindex(range(N_MONTHS))
    ret = prc / prc.shift(1) - 1.0
    stocks = {}
    for p in permnos:
        if p not in prc.columns:
            sys.exit("[오류] 주가 데이터에 PERMNO %s 가 없습니다." % p)
        stocks[str(p)] = {"p": [num(v, 4) for v in prc[p]], "r": [num(v, 8) for v in ret[p]]}

    sp = pd.read_csv(DATA_DIR / "S&P500_Data.csv", parse_dates=["date"])
    sp["mi"] = (sp["date"].dt.year - START_YEAR) * 12 + sp["date"].dt.month - 1
    sp["r"] = sp["SP500_RET"].astype(str).str.rstrip("%").astype(float) / 100.0
    SP = sp.drop_duplicates("mi").set_index("mi")["r"].reindex(range(N_MONTHS))

    ff = pd.read_csv(DATA_DIR / "ff3_factors.csv", parse_dates=["date"])
    ff["mi"] = (ff["date"].dt.year - START_YEAR) * 12 + ff["date"].dt.month - 1
    RF = ff.drop_duplicates("mi").set_index("mi")["RF"].reindex(range(N_MONTHS))
    return stocks, [num(v, 8) for v in SP], [num(v, 8) for v in RF]


def load_ff3(period_label):
    f = next(iter(FF3_DIR.rglob("ff3_selected_%s.csv" % period_label)), None) if FF3_DIR.is_dir() else None
    if f is None:
        return None
    d = pd.read_csv(f)
    return {str(int(r["PERMNO"])): {"rank": int(r["rank"]), "alpha": num(r["alpha_ann"], 5), "t": num(r["t_alpha"], 3),
                                    "b": num(r["b_mkt"], 3), "s": num(r["s_smb"], 3), "h": num(r["h_hml"], 3)}
            for _, r in d.iterrows()}


def main():
    modules = discover_modules()
    avail = [m for m in modules if m["available"]]
    if not avail:
        sys.exit("[오류] 읽을 수 있는 모듈 결과(bl_input)가 하나도 없습니다.")

    permnos = sorted({p for m in avail for per in m["periods"].values() for p in per["permnos"]})
    stocks, SP, RF = load_market(permnos)
    names = {}
    for m in avail:
        for per in m["periods"].values():
            for p, t, n in zip(per["permnos"], per["tickers"], per["names"]):
                names[str(p)] = (t, n)
    for p, (t, n) in names.items():
        stocks[p].update({"t": t, "n": n})
    print("[원자료] 종목 %d개, %s ~ %s" % (len(stocks), month_label(0), month_label(N_MONTHS - 1)))

    periods = sorted({k for m in avail for k in m["periods"]})
    ff3 = {k: load_ff3(PERIOD_INFO[k][1]) for k in periods if k in PERIOD_INFO}
    payload = {
        "meta": {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "startYear": START_YEAR,
                 "costOneWay": COST_ONE_WAY, "moduleRoot": MODULE_ROOT.name,
                 "periods": [{"key": k, "label": PERIOD_INFO.get(k, (k,))[0]} for k in periods]},
        "months": [month_label(i) for i in range(N_MONTHS)],
        "sp": SP, "rf": RF, "stocks": stocks, "ff3": ff3, "modules": modules,
    }
    for m in modules:
        for per in m["periods"].values():
            per["permnos"] = [str(p) for p in per["permnos"]]

    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    import plotly
    plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    engine_js = (HERE / "engine.js").read_text(encoding="utf-8")
    parts = {"DATA": blob, "PLOTLY": plotly_js.replace("</script", "<\\/script"), "ENGINE": engine_js}
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    html = re.sub(r"__(DATA|PLOTLY|ENGINE)__", lambda m: parts[m.group(1)], tpl)
    out = HERE / "index.html"
    out.write_text(html, encoding="utf-8")
    print("[완료] %s  (%.1f MB)" % (out, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
