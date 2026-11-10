"""timing v2：買進狀態、中線出場、W1 警示。定義依 research/timing/protocol_v2.json（事前登錄）。

python research/timing/timing2.py universe OUT                       # 回測（上市前 150 檔）
python research/timing/timing2.py daily OUT [--positions 檔案.csv]    # 每日狀態（selected pool；yfinance）
positions.csv 欄位：code,entry_date,entry_price（私人資料，不要放進公開 repo）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import timing as tm  # noqa: E402

HERE = Path(__file__).parent
COST = 0.005
RULES = ["E0_抱滿", "E1_破120線", "E2_破120線且線下彎", "E3_高點回落25%", "E4_高點回落30%", "E5_Chandelier5"]
STATES = ["上升趨勢", "剛站回", "線下", "深跌"]


def state_of(I, t):
    C, m60, m120, h = I["C"][t], I["ma60"][t], I["ma120"][t], I["h250"][t]
    if np.isnan(m120) or np.isnan(h):
        return None
    if C > m120:
        return "上升趨勢" if m60 > m120 else "剛站回"
    return "深跌" if C <= 0.70 * h else "線下"


def exit_line(I, k, e, hc, hh, armed, rule):
    """回傳 rule 在第 k 日是否成立（不含 −30% 停損與 250 日上限）。"""
    C, m = I["C"][k], I["ma120"][k]
    if rule == "E1_破120線":
        return armed and C < m
    if rule == "E2_破120線且線下彎":
        return armed and C < m and k >= 20 and m < I["ma120"][k - 20]
    if rule == "E3_高點回落25%":
        return C < hc * 0.75
    if rule == "E4_高點回落30%":
        return C < hc * 0.70
    if rule == "E5_Chandelier5":
        return C < hh
    return False


def sim(I, t, rule):
    O, H, L, C, n = I["O"], I["H"], I["L"], I["C"], I["n"]
    e = O[t + 1]
    if not e > 0 or (H[t + 1] == L[t + 1] and e / C[t] - 1 >= 0.095):
        return None
    hc, ch, armed, mx = e, -np.inf, False, 0.0
    for k in range(t + 1, n):
        hc, mx = max(hc, C[k]), max(mx, H[k] / e - 1)
        if C[k] > I["ma120"][k]:
            armed_now = True
        else:
            armed_now = armed
        if not np.isnan(I["atr22"][k]):
            ch = max(ch, max(H[t + 1:k + 1]) - 5 * I["atr22"][k]) if rule == "E5_Chandelier5" else ch
        stop = C[k] < 0.70 * e
        hit = stop or exit_line(I, k, e, hc, ch, armed, rule) or (k - t >= 250)
        armed = armed_now
        if hit:
            x = k + 1
            while x < n and H[x] == L[x] and O[x] / C[x - 1] - 1 <= -0.095:
                x += 1
            if x >= n:
                return None
            return O[x] / e - 1 - COST, x - t - 1, stop, mx
    return None


def universe(out):
    sys.path.insert(0, str(HERE.parent / "tw20"))
    import tw20
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cal, P, F, _, tx = tw20.panel(out)
    L, _, _ = tw20.label_frame(cal, P, F)
    c = P["close"].where(P["close"] > 0)
    ret1 = c / (c.ffill().shift() * F) - 1
    jump = ret1.abs() > tw20.JUMP
    px = (1 + ret1.fillna(0)).cumprod().where(c.notna())
    adj = px / c
    gp = pd.Series(np.arange(len(cal)), index=cal)
    members = L.groupby("code").date.apply(set)
    rows, eps, W = [], [], []
    for code in sorted(members.index):
        d = pd.DataFrame({"open": P["open"][code] * adj[code], "high": P["high"][code] * adj[code], "low": P["low"][code] * adj[code],
                          "close": px[code], "amount": P["amount"][code]}).dropna(subset=["open", "high", "low", "close"])
        if len(d) < 300:
            continue
        clean = ~(jump[code].reindex(d.index).fillna(False).astype(float).rolling(251, min_periods=1).max().astype(bool)).to_numpy()
        fut = ~(jump[code].reindex(d.index).fillna(False).astype(float)[::-1].rolling(252, min_periods=1).max()[::-1].astype(bool)).to_numpy()
        univ = d.index.isin(members[code]) & clean
        I = tm.indicators(d, tx.reindex(d.index).ffill().to_numpy())
        n, O, C = I["n"], I["O"], I["C"]
        g = gp.reindex(d.index).to_numpy()
        for t in np.flatnonzero(univ & fut & (g % 20 == 0)):
            st = state_of(I, t)
            if st is None or t + 252 >= n:
                continue
            r = {"code": code, "date": d.index[t], "state": st}
            for rule in RULES:
                s = sim(I, t, rule)
                if s is None:
                    r = None
                    break
                r[rule], r[rule + "_d"], r[rule + "_stop"], r[rule + "_mx"] = s
            if r:
                rows.append(r)
        deep = I["deep"]
        prior = pd.Series(deep.astype(float)).shift(1).rolling(60, min_periods=60).max().to_numpy()
        for t in np.flatnonzero(deep & (prior == 0) & univ & fut):
            if t + 251 >= n:
                continue
            end = C[t + 250]
            Lr = end / O[t + 1] - 1 - COST
            Tr = np.mean([end / O[t + j] - 1 for j in (1, 21, 41)]) - COST
            k = np.flatnonzero(C[t + 1:t + 250] > I["ma120"][t + 1:t + 250])
            Wr = end / O[t + 1 + k[0] + 1] - 1 - COST if len(k) and t + 1 + k[0] + 1 <= t + 250 else 0.0
            eps.append({"code": code, "date": d.index[t], "L": Lr, "T": Tr, "W": Wr, "W_filled": bool(len(k))})
        W.append(tm.warn_frame(code, d, I, univ & fut))
    R, E, W = pd.DataFrame(rows), pd.DataFrame(eps), pd.concat(W)
    R.to_csv(out / "v2_samples.csv.gz", index=False)
    E.to_csv(out / "v2_episodes.csv", index=False)
    per = lambda dt: np.where((dt >= tm.DEV[0]) & (dt <= tm.DEV[1]), "開發", np.where(dt >= tm.VAL0, "驗證", ""))
    R["期間"], E["期間"] = per(R.date), per(E.date)
    R, E = R[R["期間"] != ""].copy(), E[E["期間"] != ""].copy()
    days = {"開發": cal[(cal >= tm.DEV[0]) & (cal <= tm.DEV[1])], "驗證": cal[cal >= tm.VAL0]}
    ci = lambda x, p: tm.diff_ci(x.date, x.v.to_numpy(), None, None, days[p], 120)
    f = lambda lo, hi, k=100: f"[{k * lo:+.1f}, {k * hi:+.1f}]"

    # Part 1
    R["own"] = R.E0_抱滿 - R.groupby(["code", "期間"]).E0_抱滿.transform("mean")
    R["xsec"] = R.E0_抱滿 - R.groupby("date").E0_抱滿.transform("mean")
    p1 = []
    for st in STATES:
        est = {}
        for p in ("開發", "驗證"):
            x = R[(R.state == st) & (R["期間"] == p)]
            lo, hi = ci(x.assign(v=x.own), p)
            lo2, hi2 = ci(x.assign(v=x.xsec), p)
            est[p] = (x.own.mean(), lo, hi)
            p1.append({"狀態": st, "期間": p, "樣本數": len(x), "平均報酬%": 100 * x.E0_抱滿.mean(), "中位數%": 100 * x.E0_抱滿.median(),
                       "觸發−30%停損%": 100 * x.E0_抱滿_stop.mean(), "曾達+50%": 100 * (x.E0_抱滿_mx >= .5).mean(),
                       "同檔超額pp": 100 * x.own.mean(), "95%區間": f(lo, hi), "同日超額pp": 100 * x.xsec.mean(), "同日95%區間": f(lo2, hi2)})
        d, v = est["開發"], est["驗證"]
        p1[-1]["判定"] = "較好 ✓" if v[1] > 0 and d[0] > 0 else "較差 ✓" if v[2] < 0 and d[0] < 0 else "不確定"
    pe = []
    for p in ("開發", "驗證"):
        x = E[E["期間"] == p]
        if not len(x):
            continue
        row = {"期間": p, "起點數": len(x), "W 成交比例%": 100 * x.W_filled.mean()}
        for k, nm in (("L", "一次買"), ("T", "分三批"), ("W", "等站回")):
            row[f"{nm} 平均%"], row[f"{nm} 中位%"], row[f"{nm} ≤−30%"] = 100 * x[k].mean(), 100 * x[k].median(), 100 * (x[k] <= -.3).mean()
        for k, nm in (("T", "分三批−一次買"), ("W", "等站回−一次買")):
            lo, hi = ci(x.assign(v=x[k] - x.L), p)
            row[nm + " pp"] = f"{100 * (x[k] - x.L).mean():+.1f} {f(lo, hi)}"
        pe.append(row)

    # Part 2
    p2, cand = [], {}
    for scope in ["全部"] + STATES:
        for p in ("開發", "驗證"):
            x = R[R["期間"] == p] if scope == "全部" else R[(R["期間"] == p) & (R.state == scope)]
            q0 = x.E0_抱滿.quantile(.05)
            for rule in RULES:
                v, dd = x[rule], x[rule + "_d"]
                lo, hi = ci(x.assign(v=v - x.E0_抱滿), p) if rule != RULES[0] else (0, 0)
                tail = v.quantile(.05)
                okc = rule != RULES[0] and 100 * (tail - q0) >= 5 and 100 * lo > -3
                if scope == "全部":
                    cand.setdefault(rule, {})[p] = (okc, v.mean())
                p2.append({"進場": scope, "期間": p, "出場": rule, "樣本數": len(x), "平均報酬%": 100 * v.mean(), "中位數%": 100 * v.median(),
                           "最差5%": 100 * tail, "≤−25%比例": 100 * (v <= -.25).mean(),
                           "曾達+50%且最終≥+30%": 100 * ((x[rule + "_mx"] >= .5) & (v >= .3)).mean(), "持有天數": dd.mean(),
                           "每250日報酬%": 100 * v.mean() / dd.mean() * 250,
                           "對E0差pp": "" if rule == RULES[0] else f"{100 * (v - x.E0_抱滿).mean():+.1f} {f(lo, hi)}"})
    good = [r for r, v in cand.items() if v.get("開發", (0,))[0] and v.get("驗證", (0,))[0]]
    pref = max(good, key=lambda r: cand[r]["驗證"][1]) if good else RULES[0]

    # Part 3
    Wv = W[W.univ & W.dd30_250.notna()].copy()
    Wv["期間"] = per(Wv.date)
    k = "W1_弱於大盤"
    xs = Wv[~Wv[k]].groupby("date")[["dd30_250", "end_250"]].mean()
    Ev = Wv[Wv[k + "_onset"] & (Wv["期間"] != "")].join(xs, on="date", rsuffix="_x").dropna(subset=["dd30_250_x"])
    p3, est = [], {}
    for p in ("開發", "驗證"):
        e = Ev[Ev["期間"] == p]
        lo, hi = ci(e.assign(v=e.dd30_250 - e.dd30_250_x), p)
        est[p] = ((e.dd30_250 - e.dd30_250_x).mean(), lo)
        p3.append({"期間": p, "事件數": len(e), "其後250日曾跌30%": 100 * e.dd30_250.mean(), "同日未警示股票": 100 * e.dd30_250_x.mean(),
                   "差pp": f"{100 * est[p][0]:+.1f} {f(lo, hi)}", "250日期末報酬%": 100 * e.end_250.mean(),
                   "同日未警示期末報酬%": 100 * e.end_250_x.mean(), "其後曾漲30%（誤報）": 100 * e.up30_250.mean()})
    w1ok = est["驗證"][1] > 0 and est["開發"][0] > 0
    res = {"preferred_exit": pref, "exit_candidates": good, "w1_pass": bool(w1ok)}
    (out / "v2_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    P2 = pd.DataFrame(p2)
    txt = "\n".join(["# timing v2：買進狀態、中線出場、W1 警示（上市前 150 檔）", "",
                     "所有報酬已扣來回成本 0.5%；出場以收盤確認、隔日開盤成交。區間為進場日 moving-block bootstrap（120 日）。"
                     "樣本彼此重疊，不是獨立交易；結果是歷史統計，不是交易績效保證。", "",
                     "## 1. 在哪種狀態買進（出場：抱滿 250 日，收盤跌破 −30% 停損）", "", tm.md(pd.DataFrame(p1)), "",
                     "### 深跌後：一次買、分三批、等站回 120 日線（起點後 250 日結算，不設停損）", "", tm.md(pd.DataFrame(pe)), "",
                     "## 2. 中線出場規則（全部樣本日隨機進場）", "", tm.md(P2[P2.進場 == "全部"].drop(columns="進場")), "",
                     f"**事前規則選出的出場：{pref}**（候選：{good or '無'}）", "",
                     "### 依進場狀態（驗證期）", "", tm.md(P2[(P2.進場 != "全部") & (P2.期間 == "驗證")].drop(columns="期間")), "",
                     "## 3. W1 弱於大盤警示（同日橫斷面對照）", "", tm.md(pd.DataFrame(p3)), "",
                     f"**W1 {'通過' if w1ok else '未通過'}事前標準**（此比較在登錄前已看過，確認價值有限）。", ""])
    (out / "v2_universe.md").write_text(txt, encoding="utf-8")
    print(txt)


def position_status(I, dates, entry_date, entry_price):
    """持倉狀態：各出場規則今天（最後一根 K）是否成立。entry_price 為實際成本（還原尺度以最新價為準，除權息後需自行調整）。"""
    t0 = int(np.searchsorted(dates, pd.Timestamp(entry_date)))
    i, C, H = I["n"] - 1, I["C"], I["H"]
    hc, hh = max(entry_price, np.nanmax(C[t0:i + 1])), np.nanmax(H[t0:i + 1])
    armed = bool((C[t0:i] > I["ma120"][t0:i]).any())
    ch = np.nanmax([np.nanmax(H[t0:k + 1]) - 5 * I["atr22"][k] for k in range(t0, i + 1)])
    return {"持有天數": i - t0, "報酬%": round(100 * (C[i] / entry_price - 1), 1), "自高點回落%": round(100 * (C[i] / hc - 1), 1),
            "−30%停損價": round(0.70 * entry_price, 2), "停損成立": bool(C[i] < 0.70 * entry_price),
            "E1_破120線": bool(armed and C[i] < I["ma120"][i]),
            "E2_破120線且線下彎": bool(armed and C[i] < I["ma120"][i] and I["ma120"][i] < I["ma120"][i - 20]),
            "E3_高點回落25%": bool(C[i] < hc * 0.75), "E4_高點回落30%": bool(C[i] < hc * 0.70), "E5_Chandelier5": bool(C[i] < ch),
            "已滿250日": bool(i - t0 >= 250)}


def daily(out, positions=None):
    import yfinance as yf
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def get(sym):
        d = yf.download(sym, start="2014-01-01", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.columns = [x.lower() for x in d.columns]
        return d
    bench = {"TW": get("^TWII").close, "US": get("^GSPC").close, "NQ": get("^IXIC").close}
    names = [x.split()[0] for x in (HERE / "pool.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]

    def bkey(t):  # 台股比加權；美股比 S&P 500，Nasdaq 掛牌者比 Nasdaq 綜合
        if t[0].isdigit():
            return "TW"
        try:
            return "NQ" if yf.Ticker(t).fast_info["exchange"] in ("NMS", "NGM", "NCM") else "US"
        except Exception:  # ponytail: 查不到交易所就退回 S&P 500
            return "US"
    pos = pd.read_csv(positions, dtype={"code": str}) if positions else pd.DataFrame(columns=["code", "entry_date", "entry_price"])
    rows, prow = [], []
    for t in names:
        for sym in ([f"{t}.TW", f"{t}.TWO"] if t[0].isdigit() else [t]):
            d = get(sym)
            if len(d) > 260:
                break
        else:
            rows.append({"代號": t, "狀態": "資料不足"})
            continue
        d = d[d.volume > 0].dropna(subset=["open", "high", "low", "close"])
        d["amount"] = d.volume * d.close
        bk = bkey(t)
        I = tm.indicators(d, bench[bk].reindex(d.index).ffill().to_numpy())
        i = I["n"] - 1
        w1 = bool(np.nan_to_num(I["W1_弱於大盤"][i], nan=0))
        onset = w1 and not bool(np.nan_to_num(I["W1_弱於大盤"][i - 1], nan=0))
        rows.append({"代號": t, "資料日": f"{d.index[i]:%Y-%m-%d}", "收盤": round(I["C"][i], 2), "狀態": state_of(I, i) or "資料不足",
                     "距一年高點%": round(100 * (I["C"][i] / I["h250"][i] - 1), 1), "對120日線%": round(100 * (I["C"][i] / I["ma120"][i] - 1), 1),
                     "W1弱於大盤": "今日新增" if onset else "是" if w1 else "—",
                     "W1基準": {"TW": "加權", "US": "S&P500", "NQ": "Nasdaq"}[bk]})
        for p in pos[pos.code == t].itertuples():
            prow.append({"代號": t, "進場日": p.entry_date, "成本": p.entry_price, **position_status(I, d.index, p.entry_date, p.entry_price)})
    S = pd.DataFrame(rows)
    S.to_csv(out / "daily_status.csv", index=False)
    txt = ["# 每日狀態（selected pool）", "", S.astype(object).where(S.notna(), "").astype(str).to_markdown(index=False, disable_numparse=True), ""]
    if prow:
        Pp = pd.DataFrame(prow)
        Pp.to_csv(out / "daily_positions.csv", index=False)
        txt += ["## 持倉", "", Pp.astype(str).to_markdown(index=False, disable_numparse=True), ""]
    (out / "daily.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["universe", "daily"])
    a.add_argument("out")
    a.add_argument("--positions")
    x = a.parse_args()
    universe(x.out) if x.cmd == "universe" else daily(x.out, x.positions)
