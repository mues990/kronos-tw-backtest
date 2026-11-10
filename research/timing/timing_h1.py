"""timing v3 H1：凍結模型 A 當進場濾網（規格 protocol_v3.json 的 H1_model_A）。只讀 tw20 第 2 段 artifact，不重訓。

python research/timing/timing_h1.py OUT TW20_DIR
OUT：含 v2_samples.csv.gz（timing2.py universe）；TW20_DIR：tw20-stage2 artifact（stage2/model_A.txt、features.parquet）。
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import timing as tm  # noqa: E402
import timing3 as t3  # noqa: E402

MODEL_SHA = "d1ca62aa23f13952b986cc68e883bd52e4e0983563364d54eeae358c1f2c0cf3"   # tw20 frozen.json models.A.sha256
THR_UP = 0.23873360094451                                                       # tw20 frozen.json thresholds.A.UP.90
UP, DOWN = 2, 0                                                                 # tw20 class_map
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")


def signal(p):
    """同 tw20 stage2.signals：UP 分數 ≥ 門檻且 > DOWN 分數。"""
    return (p[:, UP] >= THR_UP) & (p[:, UP] > p[:, DOWN])


def score(S, X, booster):
    feats = booster.feature_name()
    D = S.merge(X[["date", "code"] + feats], on=["date", "code"], how="left", indicator="_x")
    miss = int((D._x != "both").sum())
    D = D[D._x == "both"].drop(columns="_x").copy()
    D[feats] = D[feats].replace([np.inf, -np.inf], np.nan)
    D["入選"] = signal(booster.predict(D[feats]))
    return D.drop(columns=feats), miss


def run(out, tw):
    import lightgbm as lgb
    out, tw = Path(out), Path(tw)
    mf = tw / "stage2" / "model_A.txt"
    assert hashlib.sha256(mf.read_bytes()).hexdigest() == MODEL_SHA, "模型 A 雜湊不符"
    S = pd.read_csv(out / "v2_samples.csv.gz", dtype={"code": str}, parse_dates=["date"])
    S = S[(S.date >= START) & (S.date <= END)].copy()
    assert S.date.max() <= END, "不得評分 2025 年以後"
    X = pd.read_parquet(tw / "features.parquet")
    X["code"] = X["code"].astype(str)
    X = X[(X.date >= START) & (X.date <= END)]
    D, miss = score(S, X, lgb.Booster(model_file=str(mf)))
    cal, _, _, r60 = t3.features(out)
    D = D.join(r60.stack().rename("R60"), on=["date", "code"])
    D["E0"] = D["E0_抱滿"]
    D["win"] = (D.E0 > 0).astype(float)
    for c in ("E0", "win", "R60"):
        D[c + "_x"] = D[c] - D.groupby("date")[c].transform("mean")
    D["E0_own"] = D.E0 - D.groupby("code").E0.transform("mean")
    days = cal[(cal >= START) & (cal <= END)]
    a = D[D["入選"]]
    r, w = t3.ci(a, "E0_x", days), t3.ci(a, "win_x", days)
    o, q = t3.ci(a, "E0_own", days), t3.ci(a.dropna(subset=["R60_x"]), "R60_x", days)
    verdict = "通過" if r[1] > 0 and w[1] > 0 else ("較差" if r[2] < 0 and w[2] < 0 else "不通過")
    tab = pd.DataFrame([{"樣本": len(D), "缺特徵": miss, "入選": len(a), "入選日數": a.date.nunique(),
                         "入選勝率%": f"{100 * a.win.mean():.1f}", "全部勝率%": f"{100 * D.win.mean():.1f}",
                         "入選平均%": f"{100 * a.E0.mean():.1f}", "全部平均%": f"{100 * D.E0.mean():.1f}",
                         "同日勝率超額pp": t3.fmt(*w), "同日報酬超額pp": t3.fmt(*r), "同檔報酬超額pp": t3.fmt(*o),
                         "三個月報酬超額pp（參考）": t3.fmt(*q)}])
    st = a.groupby("state").agg(樣本=("E0", "size"), 勝率超額=("win_x", "mean"), 報酬超額=("E0_x", "mean"),
                                停損比例=("E0_抱滿_stop", "mean")).reset_index()
    for k in ("勝率超額", "報酬超額", "停損比例"):
        st[k] = (100 * st[k]).round(1)
    yr = a.groupby(a.date.dt.year).agg(樣本=("E0", "size"), 勝率超額=("win_x", "mean"), 報酬超額=("E0_x", "mean")).reset_index()
    for k in ("勝率超額", "報酬超額"):
        yr[k] = (100 * yr[k]).round(1)
    txt = ["# timing v3 H1：模型 A 做多 P90 當進場濾網（進場日 2023–2024）", "",
           f"模型 A SHA-256 已核對（{MODEL_SHA[:12]}…）；門檻 {THR_UP:.4f}；不重訓、不評分 2025 年以後。", "",
           f"**判定：{verdict}**", "", tab.to_markdown(index=False, disable_numparse=True), "",
           "### 入選樣本依 v2 狀態分列（描述，pp）", "", st.to_markdown(index=False, disable_numparse=True), "",
           "### 入選樣本分年（描述，pp）", "", yr.to_markdown(index=False, disable_numparse=True), ""]
    (out / "h1_universe.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


def demo():
    """離線自檢：signal 規則與 score 的合併。"""
    p = np.array([[0.1, 0.5, 0.4], [0.5, 0.2, 0.3], [0.1, 0.8, 0.1], [0.3, 0.4, 0.3]])
    assert signal(p).tolist() == [True, False, False, False]

    class B:
        feature_name = staticmethod(lambda: ["f"])
        predict = staticmethod(lambda X: np.c_[np.zeros(len(X)), 1 - X.f.to_numpy(), X.f.to_numpy()])
    S = pd.DataFrame({"date": pd.to_datetime(["2023-01-03"] * 3), "code": ["1101", "2330", "9999"]})
    X = pd.DataFrame({"date": pd.to_datetime(["2023-01-03"] * 2), "code": ["1101", "2330"], "f": [0.9, 0.1]})
    D, miss = score(S, X, B)
    assert miss == 1 and D.set_index("code")["入選"].to_dict() == {"1101": True, "2330": False}
    print("timing_h1 demo ok")


if __name__ == "__main__":
    demo() if sys.argv[1] == "demo" else run(sys.argv[1], sys.argv[2])
