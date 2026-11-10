"""timing v1：進場 A／B／C、出場、警示層。定義一律依 research/timing/protocol.json（事前登錄）。

python research/timing/timing.py universe OUT   # 上市前 150 檔：開發期與驗證期
python research/timing/timing.py pool OUT       # selected pool（yfinance）：只報告＋今日狀態
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PROTO = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
COST = PROTO["execution"]["cost_round_trip"]
DEV = (pd.Timestamp("2015-07-01"), pd.Timestamp("2021-12-31"))
VAL0 = pd.Timestamp("2023-01-01")
SCN = ["A_回檔轉強", "B60_突破回測", "B120_突破回測", "C_低檔止跌"]
STATE = {"A_回檔轉強": "trend", "B60_突破回測": "trend", "B120_突破回測": "trend", "C_低檔止跌": "deep"}
WS = ["W1_弱於大盤", "W2_結構破壞", "W3_一年新低", "W4_爆量長黑"]


def rsi(c, n=14):
    d = np.diff(c, prepend=np.nan)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = pd.Series(dn).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + au / ad)).to_numpy()


def indicators(d, bench):
    """d：還原 open/high/low/close/amount（index 為該檔有成交的交易日）；bench：對齊後的大盤收盤。"""
    O, H, L, C = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    s, n = pd.Series(C), len(C)
    pc = np.r_[np.nan, C[:-1]]
    tr = pd.Series(np.fmax(H - L, np.fmax(abs(H - pc), abs(L - pc))))
    I = dict(O=O, H=H, L=L, C=C, n=n, ma20=s.rolling(20).mean().to_numpy(), ma60=s.rolling(60).mean().to_numpy(),
             ma120=s.rolling(120).mean().to_numpy(), atr14=tr.rolling(14).mean().to_numpy(), atr22=tr.rolling(22).mean().to_numpy(),
             rsi=rsi(C), h250=s.rolling(250, min_periods=200).max().to_numpy(), ret1=C / pc - 1)
    I["trend"] = (C > I["ma120"]) & (I["ma60"] > I["ma120"])
    I["deep"] = (C < I["ma120"]) & (C <= 0.70 * I["h250"])
    # 擺盪低點：嚴格低於前 5 日與後 5 日最低；第 j+5 日收盤後才確認
    lo = pd.Series(L)
    is_sw = (L < lo.shift(1).rolling(5).min().to_numpy()) & (L < lo[::-1].shift(1).rolling(5).min()[::-1].to_numpy())
    p1, i1, p2 = np.full(n, np.nan), np.full(n, -1), np.full(n, np.nan)
    a, ai, b = np.nan, -1, np.nan
    for t in range(n):
        j = t - 5
        if j >= 0 and is_sw[j]:
            b, a, ai = a, L[j], j
        p1[t], i1[t], p2[t] = a, ai, b
    I.update(sl1=p1, sl1i=i1, sl2=p2)
    # 警示
    r60 = C / np.r_[np.full(60, np.nan), C[:-60]] - 1
    b60 = bench / np.r_[np.full(60, np.nan), bench[:-60]] - 1
    amt = pd.Series(d["amount"].to_numpy(float))
    I["W1_弱於大盤"] = (r60 - b60) <= -0.20
    I["W2_結構破壞"] = (C < p1) & (C < I["ma120"]) & (I["ma120"] < np.r_[np.full(20, np.nan), I["ma120"][:-20]])
    I["W3_一年新低"] = C < s.shift(1).rolling(250, min_periods=200).min().to_numpy()
    I["W4_爆量長黑"] = (I["ret1"] <= -0.05) & (amt >= 2 * amt.shift(1).rolling(20).mean()).to_numpy() & (C < I["ma60"])
    return I


def entry_signals(I):
    O, H, L, C, n = I["O"], I["H"], I["L"], I["C"], I["n"]
    s = pd.Series(C)
    hc60 = s.shift(1).rolling(60).max().to_numpy()
    sig = {k: np.zeros(n, bool) for k in SCN}
    usedA, usedC = set(), set()
    for t in range(1, n):
        j = I["sl1i"][t]
        if j >= 0 and j + 5 <= t <= j + 25:
            if (j not in usedA and I["trend"][t] and C[t] > I["ma20"][t] and I["rsi"][t] >= 50 and I["rsi"][t - 1] < 50
                    and L[j] <= 0.90 * hc60[j]):
                sig["A_回檔轉強"][t] = True
                usedA.add(j)
            if j not in usedC and I["deep"][t] and C[t] > I["ma60"][t] and I["sl1"][t] > I["sl2"][t]:
                sig["C_低檔止跌"][t] = True
                usedC.add(j)
    for N in (60, 120):
        pm = s.shift(1).rolling(N).max().to_numpy()
        b, bp, ret = -1, np.nan, False
        for t in range(1, n):
            if C[t] > pm[t]:
                b, bp, ret = t, pm[t], False
            elif b >= 0 and t <= b + 20:
                if not ret:
                    ret = bool(L[t] <= bp * 1.03 and C[t] >= bp * 0.97)
                elif C[t] > H[t - 1] and I["trend"][t]:
                    sig[f"B{N}_突破回測"][t] = True
                    b = -1
    return sig


def simulate(I, t, alt=False):
    """t 日收盤後訊號 → t+1 開盤買；出場以收盤確認、下一個可成交日開盤賣。未能在資料內完成則回傳 None。"""
    O, H, L, C, n = I["O"], I["H"], I["L"], I["C"], I["n"]
    if t + 1 >= n or np.isnan(I["sl1"][t]) or np.isnan(I["atr14"][t]):
        return None
    e = O[t + 1]
    if not e > 0 or (H[t + 1] == L[t + 1] and e / C[t] - 1 >= 0.095):
        return None
    S0 = min(max(I["sl1"][t] - I["atr14"][t], 0.70 * e), 0.95 * e)
    line, hh, armed, mae, mx = S0, e, False, 0.0, 0.0
    for k in range(t + 1, n):
        hh, mae, mx = max(hh, H[k]), min(mae, L[k] / e - 1), max(mx, H[k] / e - 1)
        d, reason = k - t, None
        if not alt:
            if not np.isnan(I["atr22"][k]):
                line = max(line, hh - 3 * I["atr22"][k])
            if C[k] < line:
                reason = "初始停損" if line == S0 else "移動停利"
        elif C[k] < S0:
            reason = "初始停損"
        elif C[k] > I["ma120"][k]:
            armed = True
        elif armed:
            reason = "移動停利"
        if reason is None and d >= 250:
            reason = "到期"
        if reason is None and d >= 120 and C[k] / e - 1 < 0.10:
            reason = "時間"
        if reason:
            x = k + 1
            while x < n and H[x] == L[x] and O[x] / C[x - 1] - 1 <= -0.095:
                x += 1
            if x >= n:
                return None
            ret = O[x] / e - 1 - COST
            return dict(t=t, x=x, ret=ret, R=ret / ((e - S0) / e), risk=(e - S0) / e, reason=reason, mae=mae, mx=mx, days=x - t - 1)
    return None


def run_stock(code, d, bench, univ=None, gpos=None, controls=True):
    I = indicators(d, bench)
    sig = entry_signals(I)
    n, dates = I["n"], d.index
    univ = np.ones(n, bool) if univ is None else univ
    T, K = [], []
    for alt in (False, True):
        for sc in SCN:
            free = -1
            for t in np.flatnonzero(sig[sc] & univ):
                if t < free:
                    continue
                r = simulate(I, t, alt)
                if r is None:
                    continue
                free = r["x"]
                T.append(dict(code=code, scn=sc, alt=alt, date=dates[t], full=t + 252 < n, **r))
        if controls:
            for st in ("trend", "deep"):
                for t in np.flatnonzero(I[st] & univ & (gpos % 20 == 0)):
                    r = simulate(I, t, alt)
                    if r is not None:
                        K.append(dict(code=code, state=st, alt=alt, date=dates[t], full=t + 252 < n, **r))
    return I, sig, T, K


def warn_frame(code, d, I, univ):
    n, O, H, L, C = I["n"], I["O"], I["H"], I["L"], I["C"]
    ref = np.r_[O[1:], np.nan]
    W = pd.DataFrame({k: np.nan_to_num(I[k], nan=0).astype(bool) for k in WS})
    cnt = W.sum(1).to_numpy()
    W["黃燈"], W["紅燈"] = cnt == 1, cnt >= 2
    out = pd.DataFrame({"code": code, "date": d.index, "univ": univ})
    for h in (120, 250):
        fmin = pd.Series(L)[::-1].rolling(h).min()[::-1].shift(-1).to_numpy()
        out[f"dd30_{h}"] = np.where(np.isnan(fmin) | np.isnan(ref), np.nan, (fmin / ref - 1 <= -0.30).astype(float))
        out[f"end_{h}"] = np.r_[C[h:], np.full(h, np.nan)] / ref - 1
    fmax = pd.Series(H)[::-1].rolling(250).max()[::-1].shift(-1).to_numpy()
    out["up30_250"] = np.where(np.isnan(fmax) | np.isnan(ref), np.nan, (fmax / ref - 1 >= 0.30).astype(float))
    for k in W:
        out[k] = W[k].to_numpy()
        out[k + "_onset"] = W[k].to_numpy() & ~np.r_[False, W[k].to_numpy()[:-1]]
    return out


# ---------- 統計 ----------
def block_index(n, block, reps, seed=42):
    rng = np.random.default_rng(seed)
    block = min(block, n)
    k = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, size=(reps, k))
    return (st[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]


def diff_ci(a_dates, a_val, b_dates, b_val, days, block):
    """mean(a) − mean(b)（b 為 None 時只算 mean(a)），以 days 為單位做 moving-block bootstrap。"""
    pos = pd.Series(np.arange(len(days)), index=days)
    agg = lambda dt, v: (np.bincount(pos[dt].to_numpy(), weights=v, minlength=len(days)), np.bincount(pos[dt].to_numpy(), minlength=len(days)))
    sa, na = agg(a_dates, a_val)
    idx = block_index(len(days), block, 2000)
    A, NA = sa[idx].sum(1), na[idx].sum(1)
    ok = NA > 0
    v = A[ok] / NA[ok]
    if b_dates is not None:
        sb, nb = agg(b_dates, b_val)
        B, NB = sb[idx].sum(1), nb[idx].sum(1)
        ok = (NA > 0) & (NB > 0)
        v = A[ok] / NA[ok] - B[ok] / NB[ok]
    return tuple(np.percentile(v, [2.5, 97.5])) if len(v) else (np.nan, np.nan)


def md(df):
    df = df.copy()
    if "通過" in df:
        df["通過"] = df["通過"].fillna("")
    f = lambda c: c.map(lambda v: "" if pd.isna(v) else (f"{v:.2f}" if "R" in c.name else f"{v:.1f}")) if c.dtype.kind == "f" else c.astype(str)
    return df.apply(f).to_markdown(index=False, disable_numparse=True)


def period_of(dt, full):
    return np.where((dt >= DEV[0]) & (dt <= DEV[1]), "開發", np.where((dt >= VAL0) & full, "驗證", ""))


def entry_tables(T, K, cal):
    T["期間"], K["期間"] = period_of(T.date, T.full), period_of(K.date, K.full)
    rows = []
    for alt in (False, True):
        for sc in SCN:
            est = {}
            for per in ("開發", "驗證"):
                s = T[(T.scn == sc) & (T.alt == alt) & (T["期間"] == per)]
                c = K[(K.state == STATE[sc]) & (K.alt == alt) & (K["期間"] == per)]
                days = cal[(cal >= (DEV[0] if per == "開發" else VAL0)) & (cal <= (DEV[1] if per == "開發" else T.date[T["期間"] == "驗證"].max()))]
                if not len(s):
                    continue
                d = s.R.mean() - c.R.mean()
                ci = {b: diff_ci(s.date, s.R.to_numpy(), c.date, c.R.to_numpy(), days, b) for b in (120, 60, 250)}
                est[per] = (d, ci, len(s), s.date.dt.to_period("M").nunique())
                rows.append({"情境": sc, "出場": "MA120" if alt else "Chandelier", "期間": per, "交易數": len(s), "月數": est[per][3],
                             "平均R": s.R.mean(), "對照R": c.R.mean(), "R差": d, "R差95%[120日]": f"[{ci[120][0]:+.2f}, {ci[120][1]:+.2f}]",
                             "R差下限 60／250日": f"{ci[60][0]:+.2f}／{ci[250][0]:+.2f}",
                             "平均報酬%": 100 * s.ret.mean(), "對照報酬%": 100 * c.ret.mean(), "最終獲利%": 100 * (s.ret > 0).mean(),
                             "對照獲利%": 100 * (c.ret > 0).mean(), "初始停損%": 100 * (s.reason == "初始停損").mean(),
                             "曾達+50%": 100 * (s.mx >= .5).mean(), "曾達+100%": 100 * (s.mx >= 1).mean(), "MAE中位%": 100 * s.mae.median(),
                             "持有天數": s.days.mean(), "對照筆數": len(c)})
            if "驗證" in est:
                d, ci, n, m = est["驗證"]
                ok = n >= 100 and m >= 24 and ci[120][0] > 0 and "開發" in est and est["開發"][0] > 0
                why = [w for w, c in (("交易數<100", n < 100), ("月數<24", m < 24), ("區間下限≤0", not ci[120][0] > 0),
                                      ("開發期不同號", not ("開發" in est and est["開發"][0] > 0))) if c]
                rows[-1]["通過"] = "✓" if ok else "✗（" + "、".join(why) + "）"
    return pd.DataFrame(rows)


def warn_tables(W, cal):
    W = W[W.univ & W.dd30_250.notna()].copy()
    W["期間"] = np.where((W.date >= DEV[0]) & (W.date <= DEV[1]), "開發", np.where(W.date >= VAL0, "驗證", ""))
    W["year"] = W.date.dt.year
    rows = []
    for k in WS + ["黃燈", "紅燈"]:
        base = W[~W[k]].groupby(["code", "year"])[["dd30_250", "dd30_120", "end_250", "up30_250"]].mean()
        xs = W[~W[k]].groupby("date")[["dd30_250", "end_250"]].mean()        # 事後補充：同一天、未警示的其他股票
        E = W[W[k + "_onset"] & (W["期間"] != "")].join(base, on=["code", "year"], rsuffix="_b").join(xs, on="date", rsuffix="_x").dropna(subset=["dd30_250_b"])
        est = {}
        for per in ("開發", "驗證"):
            e = E[E["期間"] == per]
            if not len(e):
                continue
            days = cal[(cal >= e.date.min()) & (cal <= e.date.max())]
            x = (e.dd30_250 - e.dd30_250_b).to_numpy()
            lo, hi = diff_ci(e.date, x, None, None, days, 120)
            est[per] = (x.mean(), lo)
            y = (e.dd30_250 - e.dd30_250_x).to_numpy()
            ylo, yhi = diff_ci(e.date, y, None, None, days, 120)
            post = {"［事後補充］同日未警示股票": 100 * e.dd30_250_x.mean(), "［事後］差pp": f"{100 * y.mean():+.1f} [{100 * ylo:+.1f}, {100 * yhi:+.1f}]",
                    "［事後］期末報酬差pp": 100 * (e.end_250 - e.end_250_x).mean()}
            rows.append({"警示": k, "期間": per, "事件數": len(e), "其後250日曾跌破−30%": 100 * e.dd30_250.mean(), "同檔同年未警示": 100 * e.dd30_250_b.mean(),
                         "差pp": 100 * x.mean(), "95%區間": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
                         "120日 差pp": 100 * (e.dd30_120 - e.dd30_120_b).mean(), "250日期末報酬%": 100 * e.end_250.mean(),
                         "未警示期末報酬%": 100 * e.end_250_b.mean(), "誤報率（其後曾達+30%）": 100 * e.up30_250.mean(), **post})
        if "驗證" in est:
            ok = est["驗證"][1] > 0 and "開發" in est and est["開發"][0] > 0
            rows[-1]["通過"] = "✓" if ok else "✗"
    return pd.DataFrame(rows)


def universe(out):
    sys.path.insert(0, str(HERE.parent / "tw20"))
    import tw20
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cal, P, F, _, tx = tw20.panel(out)
    L, _, _ = tw20.label_frame(cal, P, F)
    c = P["close"].where(P["close"] > 0)
    ret1 = c / (c.ffill().shift() * F) - 1
    jump = (ret1.abs() > tw20.JUMP)
    px = (1 + ret1.fillna(0)).cumprod().where(c.notna())
    adj = px / c
    gp = pd.Series(np.arange(len(cal)), index=cal)
    T, K, W = [], [], []
    members = L.groupby("code").date.apply(set)
    for code in sorted(members.index):
        d = pd.DataFrame({"open": P["open"][code] * adj[code], "high": P["high"][code] * adj[code], "low": P["low"][code] * adj[code],
                          "close": px[code], "amount": P["amount"][code]}).dropna(subset=["open", "high", "low", "close"])
        if len(d) < 300:
            continue
        clean = ~(jump[code].reindex(d.index).fillna(False).astype(float).rolling(251, min_periods=1).max().astype(bool)).to_numpy()
        univ = d.index.isin(members[code]) & clean      # 入池，且過去 250 日內沒有無法還原的跳空
        I, sig, t, k = run_stock(code, d, tx.reindex(d.index).ffill().to_numpy(), univ, gp.reindex(d.index).to_numpy())
        T += t
        K += k
        W.append(warn_frame(code, d, I, univ))
    T, K, W = pd.DataFrame(T), pd.DataFrame(K), pd.concat(W)
    T.to_csv(out / "trades.csv.gz", index=False)
    K.to_csv(out / "controls.csv.gz", index=False)
    E, Wt = entry_tables(T, K, cal), warn_tables(W, cal)
    E.to_csv(out / "entries.csv", index=False)
    Wt.to_csv(out / "warnings.csv", index=False)
    txt = "\n".join(["# timing v1：上市前 150 檔（開發期與驗證期）", "",
                     "R = 扣成本報酬 ÷ 初始風險；對照 = 同一狀態下隨機進場、相同出場規則。驗證期只含最長持有期（250 日）完整落在資料內的進場日，"
                     "避免只留下已出場的交易造成偏差。區間為進場日 moving-block bootstrap（120 日；另列 60／250 日下限）。", "",
                     "## 進場情境", "", md(E), "", "## 警示", "", md(Wt), ""])
    (out / "universe.md").write_text(txt, encoding="utf-8")
    print(txt)


def pool(out):
    import yfinance as yf
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def get(sym):
        d = yf.download(sym, start="2014-01-01", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.columns = [x.lower() for x in d.columns]
        return d
    bench = {"TW": get("^TWII").close, "US": get("^GSPC").close}
    names = [x.split()[0] for x in (HERE / "pool.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
    T, rows = [], []
    for t in names:
        for sym in ([f"{t}.TW", f"{t}.TWO"] if t[0].isdigit() else [t]):
            d = get(sym)
            if len(d) > 60:
                break
        else:
            rows.append({"代號": t, "狀態": "抓不到資料"})
            continue
        d = d[d.volume > 0].dropna(subset=["open", "high", "low", "close"])
        d["amount"] = d.volume * d.close
        b = bench["TW" if t[0].isdigit() else "US"].reindex(d.index).ffill().to_numpy()
        I, sig, tr, _ = run_stock(t, d, b, controls=False)
        T += tr
        i, C = I["n"] - 1, I["C"]
        recent = [f"{k.split('_')[0]}（{d.index[j]:%m-%d}）" for k in SCN for j in np.flatnonzero(sig[k][-5:]) + I["n"] - 5]
        ws = [k.split("_")[0] for k in WS if np.nan_to_num(I[k][i], nan=0)]
        stop = I["sl1"][i] - I["atr14"][i] if not np.isnan(I["sl1"][i]) else np.nan
        rows.append({"代號": t, "資料日": f"{d.index[i]:%m-%d}", "收盤": round(C[i], 2),
                     "狀態": "趨勢向上" if I["trend"][i] else "低檔深跌" if I["deep"][i] else "120日線下" if C[i] < I["ma120"][i] else "120日線上（60<120）",
                     "距一年高點%": round(100 * (C[i] / I["h250"][i] - 1), 1) if not np.isnan(I["h250"][i]) else None,
                     "對60日線%": round(100 * (C[i] / I["ma60"][i] - 1), 1), "對120日線%": round(100 * (C[i] / I["ma120"][i] - 1), 1),
                     "RSI14": round(I["rsi"][i], 0), "近5日進場訊號": "、".join(recent) or "—", "警示": "、".join(ws) or "—",
                     "燈號": "紅" if len(ws) >= 2 else "黃" if len(ws) == 1 else "—",
                     "擺盪低點−1ATR（停損參考）": round(stop, 2) if stop == stop else None,
                     "停損距離%": round(100 * (stop / C[i] - 1), 1) if stop == stop else None})
    S = pd.DataFrame(rows)
    T = pd.DataFrame(T)
    T.to_csv(out / "pool_trades.csv", index=False)
    g = T[~T.alt].groupby("scn").agg(交易數=("R", "size"), 平均R=("R", "mean"), **{"平均報酬%": ("ret", lambda s: 100 * s.mean()),
                                    "最終獲利%": ("ret", lambda s: 100 * (s > 0).mean()), "初始停損%": ("reason", lambda s: 100 * (s == "初始停損").mean()),
                                    "曾達+50%": ("mx", lambda s: 100 * (s >= .5).mean()), "持有天數": ("days", "mean")}).reset_index()
    txt = "\n".join(["# timing v1：selected pool（只報告，不用來調規則）", "", "## 今日狀態", "",
                     S.astype(object).where(S.notna(), "").astype(str).to_markdown(index=False, disable_numparse=True), "",
                     "## 歷史上各情境的已完成交易（Chandelier 出場；樣本少、且為事後選出的標的，僅供參考）", "", md(g), ""])
    (out / "pool.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    {"universe": universe, "pool": pool}[sys.argv[1]](sys.argv[2])
