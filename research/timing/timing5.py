"""timing v5（台股）：量能與波動濾網。定義依 research/timing/protocol_v5.json（事前登錄）。

python research/timing/timing5.py OUT      # 讀 OUT/v2_samples.csv.gz（timing2.py universe OUT）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tw20"))
import timing as tm  # noqa: E402
import timing3 as t3  # noqa: E402
import tw20  # noqa: E402

Q5 = ["Q1低", "Q2", "Q3", "Q4", "Q5高"]


def features(out):
    cal, P, F, _, _ = tw20.panel(Path(out))
    c = P["close"].where(P["close"] > 0)
    ret = c / (c.ffill().shift() * F) - 1
    amt = P["amount"].where(c.notna()).fillna(0.0)
    roll = lambda M, n: M.rolling(n, min_periods=int(n * .8))
    return {"VOL60": roll(ret, 60).std(), "UP60": roll(amt.where(ret > 0, 0.0), 60).sum() / roll(amt, 60).sum(),
            "DRY": roll(amt, 20).mean() / roll(amt, 250).mean(), "HEAT": roll(amt, 60).mean() / roll(amt, 250).mean()}


def verdict(win, ret):
    """win、ret：{期間: (點, lo, hi)}。"""
    if win["驗證"][1] > 0 and win["開發"][0] > 0 and ret["驗證"][0] >= 0 and ret["開發"][0] >= 0:
        return "通過"
    if win["驗證"][2] < 0 and win["開發"][2] < 0:
        return "相反"
    return "不通過"


def run(out):
    out = Path(out)
    S = pd.read_csv(out / "v2_samples.csv.gz", dtype={"code": str}, parse_dates=["date"])
    cal, rs, _, _ = t3.features(out)
    feats = features(out)
    feats["RS"] = rs
    for k, M in feats.items():
        S = S.join(M.stack().rename(k), on=["date", "code"])
    S["E0"], S["stop"] = S["E0_抱滿"], S["E0_抱滿_stop"].astype(float)
    S["win"] = (S.E0 > 0).astype(float)
    S["期間"] = np.where((S.date >= tm.DEV[0]) & (S.date <= tm.DEV[1]), "開發", np.where(S.date >= tm.VAL0, "驗證", ""))
    S = S[S["期間"] != ""].copy()
    n0 = len(S)
    S = S.dropna(subset=["VOL60", "UP60", "DRY", "HEAT", "RS"]).copy()
    days = {"開發": cal[(cal >= tm.DEV[0]) & (cal <= tm.DEV[1])], "驗證": cal[cal >= tm.VAL0]}
    P = ("開發", "驗證")
    for k in ("VOL60", "UP60", "HEAT", "RS"):
        S[k + "_r"] = S.groupby("date")[k].rank(pct=True)
    S["vq"] = pd.cut(S.VOL60_r, [0, .2, .4, .6, .8, 1], labels=Q5).astype(str)
    S["rq"] = pd.cut(S.RS_r, [0, .2, .4, .6, .8, 1], labels=Q5).astype(str)
    S["hq"] = pd.cut(S.HEAT_r, [0, .2, .4, .6, .8, 1], labels=Q5).astype(str)
    f = lambda m, lo, hi: f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    tbl = lambda rows: pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)

    def excess(mask, base=("date",), within=None):
        """買進組相對 base 分組平均的勝率／報酬超額。within：只在該子樣本內計算。"""
        X = S if within is None else S[within]
        g = X.groupby(list(base))
        X = X.assign(wx=X.win - g.win.transform("mean"), rx=X.E0 - g.E0.transform("mean"))
        o = {}
        for p in P:
            x = X[mask.reindex(X.index) & (X["期間"] == p)]
            o[p] = {"n": len(x), "wr": x.win.mean(), "all": X[X["期間"] == p].win.mean(), "stop": x.stop.mean(),
                    "win": (x.wx.mean(), *tm.diff_ci(x.date, x.wx.to_numpy(), None, None, days[p], 120)),
                    "ret": (x.rx.mean(), *tm.diff_ci(x.date, x.rx.to_numpy(), None, None, days[p], 120))}
        return o

    def block(title, o, extra=""):
        v = verdict({p: o[p]["win"] for p in P}, {p: o[p]["ret"] for p in P})
        rows = [{"期間": p, "買進組樣本": o[p]["n"], "買進組勝率%": f"{100 * o[p]['wr']:.1f}", "全部勝率%": f"{100 * o[p]['all']:.1f}",
                 "買進組停損%": f"{100 * o[p]['stop']:.1f}", "同日勝率超額pp": f(*o[p]["win"]), "同日報酬超額pp": f(*o[p]["ret"])} for p in P]
        return v, [f"## {title}", "", f"**判定：{v}**{extra}", "", tbl(rows), ""]

    def quint(col):
        q = S.groupby(["期間", col]).agg(樣本=("E0", "size"), 勝率=("win", "mean"), 平均報酬=("E0", "mean"), 中位數=("E0", "median"), 停損比例=("stop", "mean")).reset_index()
        for k in ("勝率", "平均報酬", "中位數", "停損比例"):
            q[k] = (100 * q[k]).round(1)
        return q.to_markdown(index=False, disable_numparse=True)
    txt = ["# timing v5（台股）：量能與波動濾網（上市前 150 檔）", "",
           f"規格：`research/timing/protocol_v5.json`。勝 = E0（抱滿 250 日、−30% 停損）扣成本報酬 > 0。樣本 {len(S)} 筆（特徵缺值排除 {n0 - len(S)} 筆）。", ""]
    res = {}
    m_h4, m_v1, m_v3, m_r1 = S.VOL60_r < .8, S.UP60_r >= .5, S.HEAT_r < .8, S.RS_r >= .8
    res["H4"], t = block("H4 波動濾網：避開 VOL60 最高 20%", excess(m_h4))
    txt += t + ["### VOL60 五分位（描述）", "", quint("vq"), ""]
    res["V1"], t = block("V1 承接力道：UP60 同日排名前一半", excess(m_v1))
    deep = S.state == "深跌"
    o = excess(S.UP60 >= S[deep].groupby("date").UP60.transform("median").reindex(S.index), within=deep)
    txt += t + ["深跌樣本內（UP60 ≥ 同日深跌樣本中位數，描述）：" + "；".join(f"{p} 勝率超額 {f(*o[p]['win'])}、報酬超額 {f(*o[p]['ret'])}（n={o[p]['n']}）" for p in P), ""]

    # V2 量縮止跌
    rows, e = [], {}
    for p in P:
        x = S[deep & (S["期間"] == p)]
        a, b = x[x.DRY < .5], x[x.DRY >= .5]
        if a.date.nunique() < 10:
            e[p] = None
            rows.append({"期間": p, "量縮樣本": len(a), "量縮進場日": a.date.nunique()})
            continue
        dw = (a.win.mean() - b.win.mean(), *tm.diff_ci(a.date, a.win.to_numpy(), b.date, b.win.to_numpy(), days[p], 120))
        dr = (a.E0.mean() - b.E0.mean(), *tm.diff_ci(a.date, a.E0.to_numpy(), b.date, b.E0.to_numpy(), days[p], 120))
        e[p] = (dw, dr)
        rows.append({"期間": p, "量縮樣本": len(a), "量縮進場日": a.date.nunique(), "未量縮樣本": len(b), "量縮勝率%": f"{100 * a.win.mean():.1f}",
                     "未量縮勝率%": f"{100 * b.win.mean():.1f}", "勝率差pp": f(*dw), "報酬差pp": f(*dr)})
    res["V2"] = "樣本不足" if None in e.values() else verdict({p: e[p][0] for p in P}, {p: e[p][1] for p in P})
    txt += ["## V2 量縮止跌：深跌且 DRY < 0.5", "", f"**判定：{res['V2']}**", "", tbl(rows), ""]
    res["V3"], t = block("V3 異常放量：避開 HEAT 最高 20%", excess(m_v3))
    txt += t + ["### HEAT 五分位（描述）", "", quint("hq"), ""]

    # 控制
    rows = []
    for name, mask, base in (("V1 控制波動", m_v1, ("date", "vq")), ("V3 控制波動", m_v3, ("date", "vq")), ("R1（RS 前 20%）未控制", m_r1, ("date",)),
                             ("R1 控制波動", m_r1, ("date", "vq")), ("H4 控制 RS", m_h4, ("date", "rq")), ("高波動（VOL60 前 20%）未控制", ~m_h4, ("date",)),
                             ("高波動 控制 RS", ~m_h4, ("date", "rq"))):
        o = excess(mask, base)
        res[name] = {p: o[p]["win"] for p in P}
        rows.append({"項目": name, **{f"{p} 勝率超額pp": f(*o[p]["win"]) for p in P}, **{f"{p} 報酬超額pp": f(*o[p]["ret"]) for p in P}})
    ct = S.assign(R1=np.where(m_r1, "R1", "非R1"), 高波動=np.where(~m_h4, "高波動", "其他")).groupby(["期間", "R1", "高波動"]).agg(
        樣本=("E0", "size"), 勝率=("win", "mean"), 平均報酬=("E0", "mean"), 停損比例=("stop", "mean")).reset_index()
    for k in ("勝率", "平均報酬", "停損比例"):
        ct[k] = (100 * ct[k]).round(1)
    txt += ["## 控制檢查：量能、R1 與波動誰是主因", "", tbl(rows), "", "### R1 × 高波動 交叉（描述）", "", ct.to_markdown(index=False, disable_numparse=True), ""]
    (out / "v5_universe.md").write_text("\n".join(txt), encoding="utf-8")
    S.to_csv(out / "v5_samples.csv.gz", index=False)
    print("\n".join(txt))
    return res


if __name__ == "__main__":
    assert verdict({"開發": (.02, 0, 0), "驗證": (.03, .01, .05)}, {"開發": (0.0, 0, 0), "驗證": (.01, 0, 0)}) == "通過"
    assert verdict({"開發": (.02, 0, 0), "驗證": (.03, .01, .05)}, {"開發": (-.01, 0, 0), "驗證": (.01, 0, 0)}) == "不通過"
    assert verdict({"開發": (-.02, -.03, -.01), "驗證": (-.03, -.05, -.01)}, {"開發": (0, 0, 0), "驗證": (0, 0, 0)}) == "相反"
    run(sys.argv[1])
