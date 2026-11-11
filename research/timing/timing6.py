"""timing v6（台股）：哪些技術條件提高大漲（E0 ≥ +50%）機率。定義依 research/timing/protocol_v6.json（事前登錄）。

python research/timing/timing6.py OUT      # 讀 OUT/v2_samples.csv.gz
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
import timing5 as t5  # noqa: E402
import tw20  # noqa: E402

FEATS = ["VOL60", "RS", "DH", "GAP", "R60", "UP60", "DRY", "HEAT"]


def price_features(out):
    cal, P, F, _, _ = tw20.panel(Path(out))
    c = P["close"].where(P["close"] > 0)
    px = (1 + (c / (c.ffill().shift() * F) - 1).fillna(0)).cumprod().where(c.notna())
    pf = px.ffill()
    return {"DH": px / pf.rolling(250, min_periods=200).max(), "GAP": px / pf.rolling(120).mean() - 1, "R60": px / pf.shift(60) - 1}


def run(out):
    import lightgbm as lgb
    out = Path(out)
    S = pd.read_csv(out / "v2_samples.csv.gz", dtype={"code": str}, parse_dates=["date"])
    cal, rs, _, _ = t3.features(out)
    feats = {**t5.features(out), **price_features(out), "RS": rs}
    for k in FEATS:
        S = S.join(feats[k].stack().rename(k), on=["date", "code"])
    S["E0"], S["stop"] = S["E0_抱滿"], S["E0_抱滿_stop"].astype(float)
    S["big"], S["win"], S["touch"] = (S.E0 >= .5).astype(float), (S.E0 > 0).astype(float), (S["E0_抱滿_mx"] >= .5).astype(float)
    S["期間"] = np.where((S.date >= tm.DEV[0]) & (S.date <= tm.DEV[1]), "開發", np.where(S.date >= tm.VAL0, "驗證", ""))
    S = S[S["期間"] != ""].copy()
    n0 = len(S)
    S = S.dropna(subset=FEATS).reset_index(drop=True)
    for k in FEATS:
        S[k + "_r"] = S.groupby("date")[k].rank(pct=True)
    days = {"開發": cal[(cal >= tm.DEV[0]) & (cal <= tm.DEV[1])], "驗證": cal[cal >= tm.VAL0]}
    g = S.groupby("date")
    for c in ("big", "E0", "stop", "win"):
        S[c + "_x"] = S[c] - g[c].transform("mean")
    f = lambda m, lo, hi: f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"

    def ci(x, col, p):
        return (x[col].mean(), *tm.diff_ci(x.date, x[col].to_numpy(float), None, None, days[p], 120)) if len(x) else (np.nan,) * 3

    def rows_for(name, mask, periods=("開發", "驗證")):
        rows, est = [], {}
        for p in periods:
            x, a = S[mask & (S["期間"] == p)], S[S["期間"] == p]
            est[p] = (ci(x, "big_x", p), ci(x, "E0_x", p))
            rows.append({"組別": name, "期間": p, "樣本": len(x), "大漲%": f"{100 * x.big.mean():.1f}", "全部大漲%": f"{100 * a.big.mean():.1f}",
                         "曾觸及+50%": f"{100 * x.touch.mean():.1f}", "勝率%": f"{100 * x.win.mean():.1f}", "平均%": f"{100 * x.E0.mean():.1f}",
                         "中位%": f"{100 * x.E0.median():.1f}", "停損%": f"{100 * x.stop.mean():.1f}", "全部停損%": f"{100 * a.stop.mean():.1f}",
                         "大漲機率超額pp": f(*est[p][0]), "平均報酬超額pp": f(*est[p][1]), "停損比例超額pp": f(*ci(x, "stop_x", p))})
        return rows, est

    def judge(est):
        d, v = est["開發"], est["驗證"]
        return "通過" if v[0][1] > 0 and d[0][0] > 0 and v[1][0] >= 0 and d[1][0] >= 0 else "不通過"
    tbl = lambda rows: pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
    hv, r1 = S.VOL60_r >= .8, S.RS_r >= .8
    res, txt = {}, ["# timing v6（台股）：大漲機率（上市前 150 檔）", "",
                    f"規格：`research/timing/protocol_v6.json`。大漲 = E0（抱滿 250 日、−30% 停損）扣成本報酬 ≥ +50%。樣本 {len(S)} 筆（特徵缺值排除 {n0 - len(S)} 筆）。", ""]
    a, e = rows_for("G1 高波動且非R1", hv & ~r1)
    b, _ = rows_for("對照：高波動且R1", hv & r1)
    c, _ = rows_for("對照：非高波動", ~hv)
    res["G1"] = judge(e)
    txt += ["## G1 高波動、未追高（VOL60 前 20% 且 RS 非前 20%）", "", f"**判定：{res['G1']}**", "", tbl(a + b + c), ""]
    a, e = rows_for("G2 離高點最遠20%", S.DH_r < .2)
    res["G2"] = judge(e)
    q = S.assign(DH五分位=pd.cut(S.DH_r, [0, .2, .4, .6, .8, 1], labels=["Q1最遠", "Q2", "Q3", "Q4", "Q5最近"])).groupby(["期間", "DH五分位"], observed=True).agg(
        樣本=("E0", "size"), 大漲=("big", "mean"), 勝率=("win", "mean"), 平均=("E0", "mean"), 中位=("E0", "median"), 停損=("stop", "mean")).reset_index()
    for k in ("大漲", "勝率", "平均", "中位", "停損"):
        q[k] = (100 * q[k]).round(1)
    txt += ["## G2 離一年高點最遠的 20%", "", f"**判定：{res['G2']}**", "", tbl(a), "", q.to_markdown(index=False, disable_numparse=True), ""]

    # G3 模型：開發期訓練、凍結、驗證期評估
    X = [k + "_r" for k in FEATS]
    tr = S[S["期間"] == "開發"]
    m = lgb.LGBMClassifier(num_leaves=8, learning_rate=.03, n_estimators=300, min_child_samples=100, subsample=.8, subsample_freq=1,
                           colsample_bytree=.8, random_state=42, verbose=-1).fit(tr[X], tr.big)
    S["score"] = m.predict_proba(S[X])[:, 1]
    S["score_r"] = S.groupby("date").score.rank(pct=True)
    g3 = S.score_r >= .8
    a, e = rows_for("G3 模型前20%", g3, ("驗證",))
    b, e2 = rows_for("G3 中不屬於 G1 者", g3 & ~(hv & ~r1), ("驗證",))
    c, _ = rows_for("G1 中不屬於 G3 者", (hv & ~r1) & ~g3, ("驗證",))
    v = e["驗證"]
    res["G3"] = "通過" if v[0][1] > 0 and v[1][0] >= 0 else "不通過"
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATS)
    imp = (100 * imp / imp.sum()).round(1).sort_values(ascending=False)
    txt += ["## G3 模型（8 個特徵，開發期訓練後凍結，只看驗證期）", "", f"**判定：{res['G3']}**", "", tbl(a + b + c), "",
            "特徵重要性（gain %）：" + "、".join(f"{k} {v_}" for k, v_ in imp.items()), ""]
    (out / "v6_universe.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    return res


if __name__ == "__main__":
    run(sys.argv[1])
