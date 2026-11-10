"""timing v3：技術面濾網（H2 相對強弱、H3 市場寬度）。定義依 research/timing/protocol_v3.json（事前登錄）。

python research/timing/timing3.py OUT      # 讀 OUT/v2_samples.csv.gz；沒有就先跑 timing2.py universe OUT
python research/timing/timing3.py rs_universe OUT   # 產生 R1 每日報告的比較股票池 rs_universe.txt（protocol_v4_forward.json）
H1（模型 A）已登錄但需要 tw20 artifact，本程式不執行。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tw20"))
import timing as tm  # noqa: E402
import timing2 as t2  # noqa: E402
import tw20  # noqa: E402

COST = t2.COST
LOW_BREADTH = 0.30


def features(out):
    """回傳 (cal, RS, breadth, R60)：RS、R60 為 date × code，breadth 為 date 序列。還原方式同 timing2.universe。"""
    cal, P, F, _, tx = tw20.panel(out)
    c = P["close"].where(P["close"] > 0)
    ret1 = c / (c.ffill().shift() * F) - 1
    px = (1 + ret1.fillna(0)).cumprod().where(c.notna())
    pxf = px.ffill()
    txf = tx.ffill()
    rs = (pxf.shift(20) / pxf.shift(250) - 1).where(px.shift(250).notna()).sub(txf.shift(20) / txf.shift(250) - 1, axis=0)
    common = [k for k in c.columns if len(k) == 4 and k.isdigit() and 1000 <= int(k) <= 8999]
    hist = c[common].notna().cumsum()
    ma = pxf[common].rolling(120).mean()
    elig = (hist >= 120) & c[common].notna() & ma.notna()
    breadth = (pxf[common] > ma).astype(float).where(elig).mean(axis=1)
    entry = (P["open"] * px / c).shift(-1)
    r60 = px.shift(-60) / entry - 1 - COST
    return cal, rs, breadth, r60


def attach(S, rs, breadth, r60):
    look = lambda M: M.stack().rename("v")
    S = S.join(look(rs).rename("RS"), on=["date", "code"]).join(look(r60).rename("R60"), on=["date", "code"])
    S["breadth"] = breadth.reindex(S.date).to_numpy()
    return S


def ci(x, col, days):
    lo, hi = tm.diff_ci(x.date, x[col].to_numpy(float), None, None, days, 120)
    return x[col].mean(), lo, hi


def ci2(a, b, col, days):
    lo, hi = tm.diff_ci(a.date, a[col].to_numpy(float), b.date, b[col].to_numpy(float), days, 120)
    return a[col].mean() - b[col].mean(), lo, hi


def fmt(m, lo, hi, k=100):
    return f"{k * m:+.1f} [{k * lo:+.1f}, {k * hi:+.1f}]"


def judge_pos(rows):
    """rows：{期間: (報酬點, 報酬lo, 報酬hi, 勝率點, 勝率lo, 勝率hi)}"""
    d, v = rows["開發"], rows["驗證"]
    if v[1] > 0 and v[4] > 0 and d[0] > 0 and d[3] > 0:
        return "通過"
    if v[2] < 0 and v[5] < 0:
        return "弱勢較好（不採用）"
    return "不通過"


def run(out):
    out = Path(out)
    f = out / "v2_samples.csv.gz"
    if not f.exists():
        t2.universe(out)
    S = pd.read_csv(f, dtype={"code": str}, parse_dates=["date"])
    cal, rs, breadth, r60 = features(out)
    S = attach(S, rs, breadth, r60)
    S["E0"] = S["E0_抱滿"]
    S["win"] = (S.E0 > 0).astype(float)
    S["win60"] = (S.R60 > 0).astype(float).where(S.R60.notna())
    S["期間"] = np.where((S.date >= tm.DEV[0]) & (S.date <= tm.DEV[1]), "開發", np.where(S.date >= tm.VAL0, "驗證", ""))
    S = S[S["期間"] != ""].copy()
    days = {"開發": cal[(cal >= tm.DEV[0]) & (cal <= tm.DEV[1])], "驗證": cal[cal >= tm.VAL0]}
    res, txt = {}, ["# timing v3：技術面濾網（上市前 150 檔）", "", "規格：`research/timing/protocol_v3.json`。勝 = E0（抱滿 250 日、−30% 停損）扣成本報酬 > 0。", ""]

    # H2 相對強弱
    H = S[S.RS.notna()].copy()
    H["rank"] = H.groupby("date").RS.rank(pct=True)
    H["強"] = H["rank"] >= 0.5
    for col in ("E0", "win", "R60", "win60"):
        H[col + "_x"] = H[col] - H.groupby("date")[col].transform("mean")
        H[col + "_own"] = H[col] - H.groupby(["code", "期間"])[col].transform("mean")
    rows, tab = {}, []
    for p in ("開發", "驗證"):
        x = H[(H["期間"] == p) & H["強"]]
        r, w = ci(x, "E0_x", days[p]), ci(x, "win_x", days[p])
        rows[p] = r + w
        o = ci(x, "E0_own", days[p])
        r3 = ci(x.dropna(subset=["R60_x"]), "R60_x", days[p])
        tab.append({"期間": p, "強勢樣本": len(x), "RS缺值排除": int(S[S["期間"] == p].RS.isna().sum()),
                    "強勢勝率%": f"{100 * x.win.mean():.1f}", "全部勝率%": f"{100 * H[H['期間'] == p].win.mean():.1f}",
                    "同日勝率超額pp": fmt(*w), "同日報酬超額pp": fmt(*r), "同檔報酬超額pp": fmt(*o), "三個月報酬超額pp（參考）": fmt(*r3)})
    j2 = judge_pos(rows)
    res["H2"] = {"判定": j2, "rows": rows}
    q = H.assign(五分位=pd.qcut(H.groupby("date").RS.rank(pct=True), 5, labels=["Q1弱", "Q2", "Q3", "Q4", "Q5強"]))
    qt = q.groupby(["期間", "五分位"], observed=True).agg(樣本=("E0", "size"), 勝率=("win", "mean"), 平均報酬=("E0", "mean"),
                                                     同日超額=("E0_x", "mean"), 停損比例=("E0_抱滿_stop", "mean")).reset_index()
    for k in ("勝率", "平均報酬", "同日超額", "停損比例"):
        qt[k] = (100 * qt[k]).round(1)
    st = H[H["強"]].groupby(["期間", "state"]).agg(樣本=("E0", "size"), 勝率超額=("win_x", "mean"), 報酬超額=("E0_x", "mean")).reset_index()
    for k in ("勝率超額", "報酬超額"):
        st[k] = (100 * st[k]).round(1)
    txt += ["## H2 相對強弱（12-1 月，強勢 = 同日排名前一半）", "", f"**判定：{j2}**", "",
            pd.DataFrame(tab).to_markdown(index=False, disable_numparse=True), "", "### RS 五分位（描述）", "",
            qt.to_markdown(index=False, disable_numparse=True), "", "### 強勢樣本依 v2 狀態分列（描述，pp）", "",
            st.to_markdown(index=False, disable_numparse=True), ""]

    # H3 市場寬度
    S["低寬度"] = S.breadth <= LOW_BREADTH
    rows, tab = {}, []
    for p in ("開發", "驗證"):
        x = S[S["期間"] == p]
        a, b = x[x["低寬度"]], x[~x["低寬度"]]
        r, w = ci2(a, b, "E0", days[p]), ci2(a, b, "win", days[p])
        rows[p] = r + w
        r3 = ci2(a.dropna(subset=["R60"]), b.dropna(subset=["R60"]), "R60", days[p])
        tab.append({"期間": p, "低寬度進場日": a.date.nunique(), "其他進場日": b.date.nunique(), "低寬度樣本": len(a),
                    "低寬度勝率%": f"{100 * a.win.mean():.1f}", "其他勝率%": f"{100 * b.win.mean():.1f}",
                    "勝率差pp": fmt(*w), "報酬差pp": fmt(*r), "三個月報酬差pp（參考）": fmt(*r3)})
    d, v = rows["開發"], rows["驗證"]
    if S[(S["期間"] == "驗證") & S["低寬度"]].date.nunique() < 10:
        j3 = "樣本不足"
    elif v[1] > 0 and v[4] > 0 and d[0] > 0 and d[3] > 0:
        j3 = "通過：低寬度時進場較好"
    elif v[2] < 0 and v[5] < 0 and d[0] < 0 and d[3] < 0:
        j3 = "通過：低寬度時暫停進場"
    else:
        j3 = "不通過"
    res["H3"] = {"判定": j3, "rows": rows}
    S["寬度組"] = pd.cut(S.breadth, [-0.01, 0.30, 0.60, 1.01], labels=["≤0.30", "0.30–0.60", ">0.60"])
    bt = S.groupby(["期間", "寬度組"], observed=True).agg(進場日=("date", "nunique"), 樣本=("E0", "size"), 勝率=("win", "mean"),
                                                     平均報酬=("E0", "mean"), 停損比例=("E0_抱滿_stop", "mean")).reset_index()
    for k in ("勝率", "平均報酬", "停損比例"):
        bt[k] = (100 * bt[k]).round(1)
    txt += ["## H3 市場寬度（站上 120 日線比例 ≤ 0.30 為低寬度）", "", f"**判定：{j3}**", "",
            pd.DataFrame(tab).to_markdown(index=False, disable_numparse=True), "", "### 寬度三分位（描述）", "",
            bt.to_markdown(index=False, disable_numparse=True), "", "## H1 模型 A", "", "已登錄、未執行（需要 tw20 第 2 段 artifact）。", ""]
    (out / "v3_universe.md").write_text("\n".join(txt), encoding="utf-8")
    (out / "v3_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    H.to_csv(out / "v3_samples.csv.gz", index=False)
    breadth.rename("breadth").to_csv(out / "v3_breadth.csv")
    print("\n".join(txt))


def rs_universe(out):
    """repo 內證交所資料最新日的上市流動性前 150 檔（tw20 股票池規則）→ research/timing/rs_universe.txt。"""
    cal, P, F, _, _ = tw20.panel(Path(out))
    L, _, _ = tw20.label_frame(cal, P, F)
    last = L.date.max()
    codes = sorted(L[L.date == last].code)
    (HERE / "rs_universe.txt").write_text(f"# R1 比較股票池：上市流動性前 {len(codes)} 檔，資料日 {last:%Y-%m-%d}\n" + "\n".join(codes) + "\n")
    print(last.date(), len(codes))


if __name__ == "__main__":
    rs_universe(sys.argv[2]) if sys.argv[1] == "rs_universe" else run(sys.argv[1])
