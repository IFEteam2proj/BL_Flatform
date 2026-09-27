/*
 * Black-Litterman 계산 엔진 (브라우저·Node 공용)
 * framework-main/bl_engine.py 의 식을 그대로 옮기고, 월별 Σ(최근 60개월)·모듈 Ω·비중 상한·거래비용을 더했다.
 */
(function (root) {
"use strict";

/* ---------- 행렬 도구 ---------- */
function matVec(A, x) { return A.map(r => r.reduce((s, v, j) => s + v * x[j], 0)); }
function matMul(A, B) {
  const n = A.length, m = B[0].length, k = B.length;
  const C = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(m).fill(0);
    for (let t = 0; t < k; t++) { const a = A[i][t]; if (a === 0) continue; const b = B[t]; for (let j = 0; j < m; j++) row[j] += a * b[j]; }
    C.push(row);
  }
  return C;
}
function T(A) { return A[0].map((_, j) => A.map(r => r[j])); }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

// 가우스-조르단 역행렬. 특이행렬이면 아주 작은 릿지를 더해 다시 푼다 (np.linalg.pinv 대용).
function inv(A) {
  const n = A.length;
  for (let attempt = 0; attempt < 4; attempt++) {
    const ridge = attempt === 0 ? 0 : Math.pow(10, -14 + 3 * attempt) * (1 + Math.max(...A.map((r, i) => Math.abs(r[i]))));
    const M = A.map((r, i) => r.map((v, j) => v + (i === j ? ridge : 0)).concat(A.map((_, j) => (i === j ? 1 : 0))));
    let ok = true;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-300) { ok = false; break; }
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c]; if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
    }
    if (ok) return M.map(r => r.slice(n));
  }
  throw new Error("역행렬을 계산할 수 없습니다.");
}

/* ---------- 추정 ---------- */
// 표본공분산 (ddof=1). 결측이 있는 행은 뺀다.
function sampleCov(rows) {
  const ok = rows.filter(r => r.every(v => v != null && isFinite(v)));
  const n = rows[0].length, t = ok.length;
  if (t < 2) throw new Error("공분산을 추정할 관측치가 부족합니다 (" + t + "개월).");
  const mu = new Array(n).fill(0);
  ok.forEach(r => r.forEach((v, j) => (mu[j] += v / t)));
  const S = Array.from({ length: n }, () => new Array(n).fill(0));
  ok.forEach(r => {
    const d = r.map((v, j) => v - mu[j]);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) S[i][j] += d[i] * d[j];
  });
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { S[i][j] /= t - 1; S[j][i] = S[i][j]; }
  return { S, nObs: t };
}

/* ---------- BL 식 (bl_engine.py 와 동일) ---------- */
function impliedReturns(lam, S, w) { return matVec(S, w).map(v => lam * v); }

function baseViewVar(P, tau, S) { return P.map(p => tau * dot(p, matVec(S, p))); }

function omegaHL(P, tau, S) { return baseViewVar(P, tau, S).map(v => (v <= 1e-12 ? 1e-6 : v)); }

function omegaConf(P, tau, S, conf) {
  return baseViewVar(P, tau, S).map((v, k) => {
    const c = Math.min(0.999, Math.max(0.001, conf[k]));
    const o = ((1 - c) / c) * v;
    return o <= 1e-12 ? 1e-6 : o;
  });
}

// mu_BL = Pi + tS P' (P tS P' + Omega)^-1 (Q - P Pi),  Sigma_BL = Sigma + tS - tS P' (..)^-1 P tS
function posterior(tau, S, pi, P, Q, omegaDiag) {
  const n = S.length;
  const tS = S.map(r => r.map(v => v * tau));
  if (!P.length) return { mu: pi.slice(), sigma: S.map((r, i) => r.map((v, j) => v + tS[i][j])) };
  const tSPt = matMul(tS, T(P));                       // n x K
  const mid = matMul(P, tSPt);                         // K x K
  omegaDiag.forEach((o, k) => (mid[k][k] += o));
  const midInv = inv(mid);
  const gain = matMul(tSPt, midInv);                   // n x K
  const diff = Q.map((q, k) => q - dot(P[k], pi));
  const mu = pi.map((p, i) => p + dot(gain[i], diff));
  const PtS = T(tSPt);                                 // K x n  (= P tS, tS 대칭)
  const adj = matMul(gain, PtS);
  const sigma = [];
  for (let i = 0; i < n; i++) { const r = []; for (let j = 0; j < n; j++) r.push(S[i][j] + tS[i][j] - adj[i][j]); sigma.push(r); }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const a = (sigma[i][j] + sigma[j][i]) / 2; sigma[i][j] = a; sigma[j][i] = a; }
  return { mu, sigma };
}

/* ---------- 최적화: min 0.5 lam w'Sw - mu'w,  sum w = 1,  lo <= w <= hi ---------- */
function projectBox(v, lo, hi) {
  let a = Math.min(...v) - hi - 1, b = Math.max(...v) - lo + 1;
  const f = th => v.reduce((s, x) => s + Math.min(hi, Math.max(lo, x - th)), 0) - 1;
  for (let it = 0; it < 200; it++) { const m = (a + b) / 2; if (f(m) > 0) a = m; else b = m; if (b - a < 1e-15) break; }
  const th = (a + b) / 2;
  return v.map(x => Math.min(hi, Math.max(lo, x - th)));
}
function maxEig(S) {
  let x = S.map(() => 1), l = 0;
  for (let it = 0; it < 200; it++) {
    const y = matVec(S, x); const nrm = Math.sqrt(dot(y, y)) || 1;
    const nl = nrm / (Math.sqrt(dot(x, x)) || 1); x = y.map(v => v / nrm);
    if (Math.abs(nl - l) < 1e-12 * Math.max(1, nl)) { l = nl; break; }
    l = nl;
  }
  return l;
}
function optimize(mu, S, lam, lo, hi) {
  const n = mu.length;
  const L = lam * maxEig(S) * 1.02 + 1e-12;
  // 가속 투영경사법(FISTA) + 적응형 재시작(O'Donoghue & Candès, 2015)
  let w = projectBox(new Array(n).fill(1 / n), lo, hi), y = w.slice(), t = 1;
  for (let it = 0; it < 50000; it++) {
    const g = matVec(S, y).map((v, i) => lam * v - mu[i]);
    const wn = projectBox(y.map((v, i) => v - g[i] / L), lo, hi);
    let delta = 0, prog = 0;
    for (let i = 0; i < n; i++) { delta = Math.max(delta, Math.abs(wn[i] - w[i])); prog += g[i] * (wn[i] - w[i]); }
    if (prog > 0) { t = 1; y = wn.slice(); }
    else { const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2; y = wn.map((v, i) => v + ((t - 1) / tn) * (v - w[i])); t = tn; }
    w = wn;
    if (delta < 1e-14) break;
  }
  // bl_engine.py 와 같은 정리: 아주 작은 비중은 0, 합 1로 재정규화
  w = w.map(v => (Math.abs(v) < 1e-6 ? 0 : v));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map(v => v / s);
}

/* ---------- 성과 지표 (bl_engine.run_black_litterman_backtest 와 같은 정의) ---------- */
function perfStats(r, rfAnnual) {
  const n = r.length;
  let eq = 1, peak = 1, mdd = 0;
  const cum = [], dd = [];
  r.forEach(x => { eq *= 1 + x; peak = Math.max(peak, eq); const d = (eq - peak) / peak; mdd = Math.min(mdd, d); cum.push(eq - 1); dd.push(d); });
  const total = eq - 1, years = Math.max(n / 12, 1 / 12);
  const cagr = total > -1 ? Math.pow(1 + total, 1 / years) - 1 : -1;
  const mean = r.reduce((a, b) => a + b, 0) / n;
  const vol = n > 1 ? Math.sqrt(r.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (n - 1)) * Math.sqrt(12) : 0;
  const sharpe = vol > 1e-6 ? (cagr - rfAnnual) / vol : 0;
  return { total, cagr, vol, sharpe, mdd, cum, dd };
}

// 드리프트 반영 회전율·거래비용 (BL_Dashboard 와 같은 방식). prev = 직전 달 말 보유비중(드리프트 후)
function tradeMonth(prev, wNew, ret, costRate) {
  const n = wNew.length;
  let traded = 0; for (let i = 0; i < n; i++) traded += Math.abs(wNew[i] - (prev ? prev[i] : 0));
  let wsum = 0, gross = 0;
  for (let i = 0; i < n; i++) if (ret[i] != null && isFinite(ret[i])) { wsum += wNew[i]; gross += wNew[i] * ret[i]; }
  gross = Math.abs(wsum) > 1e-12 ? gross / wsum : 0;
  const cost = costRate * traded;
  const net = (1 + gross) * (1 - cost) - 1;
  const drift = wNew.map((w, i) => (ret[i] != null && isFinite(ret[i]) && Math.abs(wsum) > 1e-12 ? (w / wsum) * (1 + ret[i]) / (1 + gross) : 0));
  return { gross, net, turnover: 0.5 * traded, cost, drift };
}

/* ---------- 월별 백테스트 ---------- */
// cfg: { months, R (월 x 자산), sp, rf, views[{date,name,P,Q,omega,conf}], lam, tau, omegaMethod('module'|'conf'|'hl'),
//        allowShort, cap(null|number), costRate, prior(null|[n]), fixedCov(null|n x n), retOverride(null|{ym:[n]}), covWindow, rfAnnual,
//        covMode('rolling'|'fixed') }  fixed = framework 방식: fixedCov 가 없으면 첫 견해 시점의 최근 covWindow 개월로 한 번 추정해 모든 달에 사용
function monthIndexOf(months, date) { return months.indexOf(String(date).slice(0, 7)); }

function runBacktest(cfg) {
  const n = cfg.R[0].length, months = cfg.months, win = cfg.covWindow || 60;
  const warnings = [];
  let prior = cfg.prior ? cfg.prior.slice() : new Array(n).fill(1 / n);
  const ps = prior.reduce((a, b) => a + b, 0); prior = prior.map(v => v / ps);

  let hi = cfg.cap != null ? cfg.cap : 1;
  if (hi * n < 1 - 1e-12) { warnings.push("비중 상한 " + (hi * 100).toFixed(1) + "% × " + n + "종목 < 100% 라서 상한을 " + (100 / n).toFixed(2) + "%로 올려 계산했습니다."); hi = 1 / n; }
  const lo = cfg.allowShort ? -hi : 0;

  const byDate = new Map();
  cfg.views.forEach(v => {
    if (!v.date) return;
    const k = String(v.date).slice(0, 10);
    if (!byDate.has(k)) byDate.set(k, []);
    byDate.get(k).push(v);
  });
  const dates = Array.from(byDate.keys()).sort();

  const recs = [];
  let prevBL = null, prevEW = null, fixedEst = null, covDate = null;
  dates.forEach(d => {
    const t = monthIndexOf(months, d);
    if (t < 0) { warnings.push(d + " 견해: 주가 데이터 기간 밖이라 건너뜀"); return; }
    const h = t + 1;
    if (h >= months.length) { warnings.push(d + " 견해: 다음 달 실현수익률이 없어 건너뜀"); return; }
    const vs = byDate.get(d).filter(v => v.Q != null && isFinite(v.Q) && v.P.some(x => x !== 0));

    let S, nObs = null;
    if (cfg.fixedCov) S = cfg.fixedCov;
    else if (cfg.covMode === "fixed" && fixedEst) { S = fixedEst.S; nObs = fixedEst.nObs; }
    else {
      const c = sampleCov(cfg.R.slice(Math.max(0, t - win + 1), t + 1)); S = c.S; nObs = c.nObs;
      if (cfg.covMode === "fixed") { fixedEst = c; covDate = d; }
    }
    const pi = impliedReturns(cfg.lam, S, prior);
    const P = vs.map(v => v.P), Q = vs.map(v => v.Q);
    let om;
    if (!P.length) om = [];
    else if (cfg.omegaMethod === "hl") om = omegaHL(P, cfg.tau, S);
    else if (cfg.omegaMethod === "conf") om = omegaConf(P, cfg.tau, S, vs.map(v => (v.conf != null && isFinite(v.conf) ? v.conf : 0.5)));
    else { const hl = omegaHL(P, cfg.tau, S); om = vs.map((v, k) => (v.omega != null && isFinite(v.omega) && v.omega > 0 ? v.omega : hl[k])); }
    const post = posterior(cfg.tau, S, pi, P, Q, om);
    const w = optimize(post.mu, post.sigma, cfg.lam, lo, hi);

    const ym = months[h];
    const ret = cfg.retOverride && cfg.retOverride[ym] ? cfg.retOverride[ym] : cfg.R[h];
    if (ret.some(v => v == null || !isFinite(v))) warnings.push(ym + ": 일부 종목 수익률이 없어 해당 종목을 빼고 계산");
    const ew = prior.slice();
    const bl = tradeMonth(prevBL, w, ret, cfg.costRate);
    const eq = tradeMonth(prevEW, ew, ret, cfg.costRate);
    prevBL = bl.drift; prevEW = eq.drift;
    recs.push({
      viewDate: d, ym, t, h, w, pi, mu: post.mu, vol: S.map((r, i) => Math.sqrt(r[i] * 12)), S, nObs,
      views: vs.map((v, k) => ({ name: v.name, P: v.P, Q: v.Q, omega: om[k] })),
      ret, sp: cfg.sp[h], rf: cfg.rf[h],
      gross: bl.gross, net: bl.net, turnover: bl.turnover, cost: bl.cost,
      ewGross: eq.gross, ewNet: eq.net, ewTurnover: eq.turnover, ewCost: eq.cost,
      k: w.filter(x => Math.abs(x) > 1e-9).length
    });
  });
  if (!recs.length) throw new Error("백테스트할 수 있는 견해 월이 없습니다.");
  return { recs, prior, warnings, cap: hi, lo, covDate };
}

function summarize(recs, useNet, rfAnnual) {
  const bl = recs.map(r => (useNet ? r.net : r.gross));
  const ew = recs.map(r => (useNet ? r.ewNet : r.ewGross));
  const sp = recs.map(r => r.sp);
  const exc = bl.map((x, i) => x - ew[i]);
  return {
    bl: perfStats(bl, rfAnnual), ew: perfStats(ew, rfAnnual), sp: perfStats(sp, rfAnnual),
    blR: bl, ewR: ew, spR: sp, exc, winRate: exc.filter(x => x > 0).length / exc.length
  };
}

const api = { matVec, matMul, inv, sampleCov, impliedReturns, omegaHL, omegaConf, posterior, projectBox, optimize,
  perfStats, tradeMonth, runBacktest, summarize };
if (typeof module !== "undefined" && module.exports) module.exports = api;
else root.BLEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
