"""US v1：美股個股的買進狀態、出場、相對強弱、警示。定義依 research/timing/protocol_us_v1.json（事前登錄）。

python research/timing/us1.py run OUT     # 需要網路（Wikipedia、yfinance）
python research/timing/us1.py demo        # 離線自檢（合成資料）
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import timing as tm  # noqa: E402
import timing2 as t2  # noqa: E402

COST = t2.COST
RULES = ["E0_抱滿", "E1_破120線", "E4_高點回落30%"]
NAMES = {"N0": "N0 不停損", "E0_抱滿": "E0 −30%停損", "E1_破120線": "E1 跌破120日線", "E4_高點回落30%": "E4 高點回落30%"}


def fetch():
    import requests
    import yfinance as yf
    tick, member = [], {}
    for page in ("List_of_S%26P_500_companies", "List_of_S%26P_400_companies"):
        html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers={"User-Agent": "kronos-tw-backtest research (github.com/mues990)"}, timeout=60).text
        tab = next(x for x in pd.read_html(io.StringIO(html)) if "Symbol" in x.columns)
        sym = [str(s).strip().replace(".", "-") for s in tab["Symbol"]]
        tick += sym
        member.update({k: "S&P500" if "500" in page else "S&P400" for k in sym})
    tick = sorted(set(tick))
    assert len(tick) > 800, f"成分股清單異常：{len(tick)}"
    bench = yf.download("^GSPC", start="2014-01-01", auto_adjust=True, progress=False)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)
    data = {}
    for i in range(0, len(tick), 100):
        raw = yf.download(tick[i:i + 100], start="2014-01-01", auto_adjust=True, progress=False, group_by="ticker", threads=True)
        for k in tick[i:i + 100]:
            if k in raw.columns.get_level_values(0):
                d = raw[k].rename(columns=str.lower).dropna(subset=["open", "high", "low", "close"])
                if len(d):
                    data[k] = d
        time.sleep(2)
    return tick, data, bench["Close"].dropna(), member


def samples(data, bench):
    """回傳 (進場樣本, 深跌起點樣本, 排除清單)。"""
    cal = bench.index
    gp = pd.Series(np.arange(len(cal)), index=cal)
    rows, eps, bad = [], [], []
    for code, d in sorted(data.items()):
        d = d[(d.volume > 0) & d.index.isin(cal)]
        if len(d) < 300:
            bad.append((code, "日數不足"))
            continue
        r1 = d.close.pct_change()
        if (r1 > 2.0).any() or (r1 < -0.75).any():
            bad.append((code, "單日變動異常"))
            continue
        d = d.assign(amount=d.volume * d.close)
        b = bench.reindex(d.index).ffill().to_numpy()
        I = tm.indicators(d, b)
        n, O, C = I["n"], I["O"], I["C"]
        g = gp.reindex(d.index).to_numpy()
        vol60 = d.close.pct_change().rolling(60).std().to_numpy()
        dv60 = d.amount.rolling(60).mean().to_numpy()
        for t in np.flatnonzero(g % 20 == 0):
            st = t2.state_of(I, t)
            if st is None or t + 252 >= n or t < 250:
                continue
            e = O[t + 1]
            r = {"code": code, "date": d.index[t], "state": st, "N0": O[t + 251] / e - 1 - COST, "N0_d": 250,
                 "RS": C[t - 20] / C[t - 250] - 1 - (b[t - 20] / b[t - 250] - 1), "W1": bool(np.nan_to_num(I["W1_弱於大盤"][t])),
                 "跌30": bool(C[t + 1:t + 251].min() < 0.70 * e), "VOL60": vol60[t], "DV60": dv60[t]}
            for rule in RULES:
                s = t2.sim(I, t, rule)
                if s is None:
                    r = None
                    break
                r[rule], r[rule + "_d"] = s[0], s[1]
            if r:
                rows.append(r)
        deep = I["deep"]
        prior = pd.Series(deep.astype(float)).shift(1).rolling(60, min_periods=60).max().to_numpy()
        for t in np.flatnonzero(deep & (prior == 0)):
            if t + 251 >= n:
                continue
            end = C[t + 250]
            k = np.flatnonzero(C[t + 1:t + 250] > I["ma120"][t + 1:t + 250])
            ok = len(k) and t + 1 + k[0] + 1 <= t + 250
            eps.append({"code": code, "date": d.index[t], "L": end / O[t + 1] - 1 - COST,
                        "W": end / O[t + 1 + k[0] + 1] - 1 - COST if ok else 0.0, "W_filled": bool(ok)})
    return pd.DataFrame(rows), pd.DataFrame(eps), bad


def three(dev, val, pos="較好", neg="較差"):
    """dev、val：[(點, lo, hi), …]；全部同號才下結論。"""
    if all(v[1] > 0 for v in val) and all(d[0] > 0 for d in dev):
        return pos
    if all(v[2] < 0 for v in val) and all(d[0] < 0 for d in dev):
        return neg
    return "不確定"


def report(R, E, bad, cal, n_univ, out):
    per = lambda dt: np.where((dt >= tm.DEV[0]) & (dt <= tm.DEV[1]), "開發", np.where(dt >= tm.VAL0, "驗證", ""))
    R, E = R.assign(期間=per(R.date)), E.assign(期間=per(E.date))
    R, E = R[R["期間"] != ""].copy(), E[E["期間"] != ""].copy()
    days = {"開發": cal[(cal >= tm.DEV[0]) & (cal <= tm.DEV[1])], "驗證": cal[cal >= tm.VAL0]}
    P = ("開發", "驗證")

    def ci(x, col, p):
        if not len(x):
            return np.nan, np.nan, np.nan
        lo, hi = tm.diff_ci(x.date, x[col].to_numpy(float), None, None, days[p], 120)
        return x[col].mean(), lo, hi
    f = lambda m, lo, hi: f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    tbl = lambda rows: pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
    R["win"] = (R.N0 > 0).astype(float)
    for c in ("N0", "win"):
        R[c + "_x"] = R[c] - R.groupby("date")[c].transform("mean")
    txt = ["# US v1：美股個股（S&P 500＋S&P 400 現存成分股）", "",
           f"規格：`research/timing/protocol_us_v1.json`。成分股 {n_univ} 檔，排除 {len(bad)} 檔，樣本 {len(R)} 筆（{R.code.nunique()} 檔）。",
           "**限制：只含現存成分股（存活者偏差），會高估抱著不動與深跌買進、低估停損。**", ""]
    res = {}

    # Q1 出場
    rows, est = [], {}
    for rule in ["N0"] + RULES:
        for p in P:
            x = R[R["期間"] == p]
            row = {"規則": NAMES[rule], "期間": p, "樣本": len(x), "平均%": f"{100 * x[rule].mean():.1f}", "中位%": f"{100 * x[rule].median():.1f}",
                   "勝率%": f"{100 * (x[rule] > 0).mean():.1f}", "最差5%": f"{100 * x[rule].quantile(.05):.1f}",
                   "≤−50%比例": f"{100 * (x[rule] <= -.5).mean():.1f}", "平均持有日": f"{x[rule + '_d'].mean():.0f}"}
            if rule != "N0":
                e = ci(x.assign(v=x[rule] - x.N0), "v", p)
                est[(rule, p)] = e
                row["減N0 pp"] = f(*e)
            rows.append(row)
        if rule != "N0":
            res[rule] = three([est[(rule, "開發")]], [est[(rule, "驗證")]])
            rows[-1]["判定"] = res[rule]
    txt += ["## Q1 出場（基準 N0：不停損抱滿 250 日）", "", tbl(rows).replace("nan", ""), ""]

    # Q2 狀態
    rows = []
    for st in t2.STATES:
        e = {p: (ci(R[(R.state == st) & (R["期間"] == p)], "N0_x", p), ci(R[(R.state == st) & (R["期間"] == p)], "win_x", p)) for p in P}
        for p in P:
            x = R[(R.state == st) & (R["期間"] == p)]
            rows.append({"狀態": st, "期間": p, "樣本": len(x), "平均%": f"{100 * x.N0.mean():.1f}", "勝率%": f"{100 * x.win.mean():.1f}",
                         "其後跌30%比例": f"{100 * x['跌30'].mean():.1f}", "同日報酬超額pp": f(*e[p][0]), "同日勝率超額pp": f(*e[p][1])})
        res["state_" + st] = rows[-1]["判定"] = three(list(e["開發"]), list(e["驗證"]))
    txt += ["## Q2 買進時的狀態（N0）", "", tbl(rows).replace("nan", ""), ""]

    # Q3 相對強弱
    H = R[R.RS.notna()].copy()
    H["rk"] = H.groupby("date").RS.rank(pct=True)
    e = {p: (ci(H[(H.rk >= .5) & (H["期間"] == p)], "N0_x", p), ci(H[(H.rk >= .5) & (H["期間"] == p)], "win_x", p)) for p in P}
    res["RS"] = three(list(e["開發"]), list(e["驗證"]), "強勢較好", "弱勢較好")
    rows = [{"期間": p, "強勢樣本": int(((H.rk >= .5) & (H["期間"] == p)).sum()), "同日報酬超額pp": f(*e[p][0]), "同日勝率超額pp": f(*e[p][1])} for p in P]
    H["五分位"] = pd.cut(H.rk, [0, .2, .4, .6, .8, 1], labels=["Q1弱", "Q2", "Q3", "Q4", "Q5強"])
    q = H.groupby(["期間", "五分位"], observed=True).agg(樣本=("N0", "size"), 勝率=("win", "mean"), 平均=("N0", "mean"), 中位=("N0", "median"), 其後跌30=("跌30", "mean")).reset_index()
    for k in ("勝率", "平均", "中位", "其後跌30"):
        q[k] = (100 * q[k]).round(1)
    txt += ["## Q3 相對強弱（12-1 月，對 S&P 500）", "", f"**判定：{res['RS']}**", "", tbl(rows), "", q.to_markdown(index=False, disable_numparse=True), ""]

    # Q4 W1
    rows, e = [], {}
    for p in P:
        x = R[R["期間"] == p]
        a, b = x[x.W1], x[~x.W1]
        d30 = tm.diff_ci(a.date, a["跌30"].to_numpy(float), b.date, b["跌30"].to_numpy(float), days[p], 120) if len(a) and len(b) else (np.nan, np.nan)
        dn0 = tm.diff_ci(a.date, a.N0.to_numpy(float), b.date, b.N0.to_numpy(float), days[p], 120) if len(a) and len(b) else (np.nan, np.nan)
        e[p] = (a["跌30"].mean() - b["跌30"].mean(), *d30)
        rows.append({"期間": p, "W1樣本": len(a), "W1 其後跌30%": f"{100 * a['跌30'].mean():.1f}", "其他 其後跌30%": f"{100 * b['跌30'].mean():.1f}",
                     "比例差pp": f(*e[p]), "W1 勝率%": f"{100 * a.win.mean():.1f}", "其他 勝率%": f"{100 * b.win.mean():.1f}",
                     "N0 平均差pp": f(a.N0.mean() - b.N0.mean(), *dn0)})
    res["W1"] = "警示成立" if e["驗證"][1] > 0 and e["開發"][0] > 0 else "不成立"
    txt += ["## Q4 W1 弱於大盤（60 日落後 S&P 500 ≥ 20 pp）", "", f"**判定：{res['W1']}**", "", tbl(rows).replace("nan", ""), ""]

    # Q5 深跌起點
    rows, e = [], {}
    for p in P:
        x = E[E["期間"] == p] if len(E) else E
        if not len(x):
            e[p] = (np.nan, np.nan, np.nan)
            continue
        e[p] = ci(x.assign(v=x.W - x.L), "v", p)
        rows.append({"期間": p, "起點數": len(x), "一次買 平均%": f"{100 * x.L.mean():.1f}", "一次買 中位%": f"{100 * x.L.median():.1f}",
                     "一次買 勝率%": f"{100 * (x.L > 0).mean():.1f}", "一次買 ≤−30%": f"{100 * (x.L <= -.3).mean():.1f}",
                     "等站回 平均%": f"{100 * x.W.mean():.1f}", "等站回 成交比例%": f"{100 * x.W_filled.mean():.1f}", "等站回−一次買 pp": f(*e[p])})
    res["deep"] = three([e["開發"]], [e["驗證"]], "等站回較好", "一次買較好")
    txt += ["## Q5 深跌起點：現在買還是等站回 120 日線", "", f"**判定：{res['deep']}**", "", tbl(rows) if rows else "無樣本", ""]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "us1_universe.md").write_text("\n".join(txt), encoding="utf-8")
    R.to_csv(out / "us1_samples.csv.gz", index=False)
    print("\n".join(txt))
    return res


def report2(R, cal, member, out):
    """US v2：出場規則依波動與規模分組（protocol_us_v2.json）。"""
    per = lambda dt: np.where((dt >= tm.DEV[0]) & (dt <= tm.DEV[1]), "開發", np.where(dt >= tm.VAL0, "驗證", ""))
    R = R.assign(期間=per(R.date))
    R = R[(R["期間"] != "") & R.VOL60.notna() & R.DV60.notna()].copy()
    days = {"開發": cal[(cal >= tm.DEV[0]) & (cal <= tm.DEV[1])], "驗證": cal[cal >= tm.VAL0]}
    R["波動"] = pd.cut(R.groupby("date").VOL60.rank(pct=True), [0, .2, .4, .6, .8, 1], labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"]).astype(str)
    R["規模"] = pd.cut(R.groupby("date").DV60.rank(pct=True), [0, 1 / 3, 2 / 3, 1], labels=["小", "中", "大"]).astype(str)
    R["指數"] = R.code.map(member).fillna("?")
    R["交叉"] = np.where((R["波動"] == "Q5高") & (R["規模"] == "小"), "Q5高×小", "其他")
    f = lambda m, lo, hi: f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"

    def table(col, order):
        rows, est = [], {}
        for gname in order:
            for p in ("開發", "驗證"):
                x = R[(R[col] == gname) & (R["期間"] == p)]
                if not len(x):
                    continue
                row = {col: gname, "期間": p, "樣本": len(x), "N0 平均%": f"{100 * x.N0.mean():.1f}", "N0 中位%": f"{100 * x.N0.median():.1f}",
                       "N0 勝率%": f"{100 * (x.N0 > 0).mean():.1f}", "N0 最差5%": f"{100 * x.N0.quantile(.05):.1f}",
                       "N0 ≤−50%": f"{100 * (x.N0 <= -.5).mean():.1f}", "其後跌30%": f"{100 * x['跌30'].mean():.1f}"}
                for rule, nm in (("E0_抱滿", "E0−N0 pp"), ("E1_破120線", "E1−N0 pp"), ("E4_高點回落30%", "E4−N0 pp")):
                    v = (x[rule] - x.N0).to_numpy(float)
                    lo, hi = tm.diff_ci(x.date, v, None, None, days[p], 120)
                    row[nm] = f(v.mean(), lo, hi)
                    est[(gname, p, rule)] = (v.mean(), lo, hi)
                rows.append(row)
        return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True), est
    t1, e1 = table("波動", ["Q1低", "Q2", "Q3", "Q4", "Q5高"])
    t2_, _ = table("規模", ["小", "中", "大"])
    t3, _ = table("指數", ["S&P500", "S&P400"])
    t4, _ = table("交叉", ["Q5高×小", "其他"])
    d, v = e1.get(("Q5高", "開發", "E0_抱滿"), (np.nan,) * 3), e1.get(("Q5高", "驗證", "E0_抱滿"), (np.nan,) * 3)
    verdict = "高波動股該停損" if v[1] > 0 and d[0] > 0 else "高波動股停損仍較差" if v[2] < 0 and d[0] < 0 else "不確定"
    txt = ["# US v2：出場規則依波動與規模分組", "", "規格：`research/timing/protocol_us_v2.json`。N0 = 不停損抱滿 250 日；E0 = −30% 停損；E1 = 跌破 120 日線；E4 = 高點回落 30%。",
           "**限制：存活者偏差在高波動、小規模組最嚴重，這些組的不停損被高估最多；指數外小型股不在樣本內。**", "",
           f"## 依波動（VOL60 同日五分位）　**判定：{verdict}**", "", t1, "", "## 依規模代理（DV60 同日三分位，描述）", "", t2_, "",
           "## 依指數（現在的成分，描述）", "", t3, "", "## 最高波動 × 小規模（描述）", "", t4, ""]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "us2_groups.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    return verdict


def run2(out):
    tick, data, bench, member = fetch()
    R, _, _ = samples(data, bench)
    R.to_csv(Path(out).mkdir(parents=True, exist_ok=True) or Path(out) / "us2_samples.csv.gz", index=False)
    report2(R, bench.index, member, out)


def run(out):
    tick, data, bench, _ = fetch()
    R, E, bad = samples(data, bench)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "us1_tickers.txt").write_text(f"# {pd.Timestamp.now(tz='UTC'):%Y-%m-%d} S&P 500＋S&P 400，{len(tick)} 檔；排除：{bad}\n" + "\n".join(tick) + "\n")
    report(R, E, bad, bench.index, len(tick), out)


def demo():
    """離線自檢：合成資料跑完整流程，並核對 N0 與 E0 的關係。"""
    import tempfile
    rng = np.random.default_rng(0)
    cal = pd.bdate_range("2014-01-01", "2026-10-30")
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, .01, len(cal)))), index=cal)
    data = {}
    for i in range(25):
        c = 50 * np.exp(np.cumsum(rng.normal(3e-4, .03, len(cal))))
        data[f"T{i:02d}"] = pd.DataFrame({"open": c * (1 + rng.normal(0, .003, len(cal))), "high": c * 1.01, "low": c * .99, "close": c, "volume": 1e6}, index=cal)
    data["BAD"] = data["T00"].assign(close=lambda d: d.close.where(d.index != cal[500], d.close * 5))
    R, E, bad = samples(data, bench)
    assert ("BAD", "單日變動異常") in bad and "BAD" not in set(R.code)
    r = R.iloc[100]
    d = data[r.code]
    t = d.index.get_loc(r.date)
    assert abs(r.N0 - (d.open.iloc[t + 251] / d.open.iloc[t + 1] - 1 - COST)) < 1e-12
    same = R[~R["跌30"]]
    assert len(same) and np.allclose(same.N0, same["E0_抱滿"]), "沒跌破 −30% 時 E0 應等於 N0"
    assert (R[R["跌30"]]["E0_抱滿_d"] <= 250).all() and (R[R["跌30"]]["E0_抱滿_d"] < 250).mean() > 0.9 and len(E)  # 最後一天才跌破者仍為 250
    with tempfile.TemporaryDirectory() as tmp:
        res = report(R, E, bad, cal, len(data), tmp)
        v = report2(R, cal, {k: "S&P500" if int(k[1:]) % 2 else "S&P400" for k in data if k != "BAD"}, tmp)
    assert set(res) >= {"E0_抱滿", "RS", "W1", "deep"} and v in ("高波動股該停損", "高波動股停損仍較差", "不確定")
    print("us1 demo ok", len(R), len(E))


if __name__ == "__main__":
    demo() if sys.argv[1] == "demo" else {"run": run, "run2": run2}[sys.argv[1]](sys.argv[2])
