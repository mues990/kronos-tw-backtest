"""tw20 第 2 段補充分析：只讀既有結果（凍結模型、凍結門檻、逐筆驗證預測），不重訓、不搜尋新門檻、不碰 2025 年以後。

python research/tw20/stage2_extra.py OUT    # OUT = 第 2 段 artifact（tw20-stage2）的內容 → OUT/stage2_extra/

1. 保守成交：排除該方向 t+1 一字鎖死的樣本（第 2 段已算，這裡完整列出）
2. 未知標籤敏感度：驗證期內因停牌、缺價、無法還原而不能評分的樣本，用凍結模型與門檻判斷是否會發訊號，再給達標率的上下界
3. 主要候選與 A 做多各門檻的 2023／2024 分年
4. 分數分箱與機率校準（描述用，不用來選門檻）
5. R20 分布（不是交易績效：未計成本、滑價、部位）
6. 舊二分類（backtest/market/qlib）與本輪三分類的差異
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import stage2 as s2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UNKNOWN = ["label_event_unreliable", "no_entry_price", "no_exit_price"]


def f1(v, d=1):
    return "" if pd.isna(v) else f"{v:.{d}f}"


def md(df):
    return df.astype(object).where(df.notna(), "").astype(str).to_markdown(index=False, disable_numparse=True)


def cooldown(S, cal, days=20):
    pos = pd.Series(np.arange(len(cal)), index=cal)
    S = S.assign(ix=S.date.map(pos).to_numpy()).sort_values(["code", "ix"])
    keep, last = [], {}
    for r in S.itertuples():
        if r.ix - last.get(r.code, -10**9) >= days:
            keep.append(r.Index)
            last[r.code] = r.ix
    return S.loc[keep].drop(columns="ix")


def main(out):
    import lightgbm as lgb
    out = Path(out)
    res, dst = out / "stage2", out / "stage2_extra"
    dst.mkdir(exist_ok=True)
    fz = json.loads((res / "frozen.json").read_text(encoding="utf-8"))
    C = pd.read_csv(res / "candidates.csv")
    V = pd.read_parquet(res / "validation_predictions.parquet")
    assert V.date.max() <= s2.VAL_END and V.label_end_date.max() <= s2.VAL_END
    b = s2.PROTO["bootstrap"]
    dates = pd.DatetimeIndex(sorted(V.date.unique()))
    IDX = {x: s2.block_index(len(dates), x, b["n"], b["seed"]) for x in [b["block_days"], *b["sensitivity_block_days"]]}
    prim = {k: v for k, v in fz["primary"].items() if v}
    lines = ["# tw20 第 2 段補充分析", "",
             "只讀第 2 段的凍結模型、凍結門檻與逐筆驗證預測；不重訓、不搜尋新門檻、不載入 2025 年以後的樣本。"
             "第 4 節的分箱只是描述，不作為選門檻依據。", ""]

    # 1. 保守成交
    T = C[["模型", "方向", "百分位", "訊號數", "達標率%", "提升pp", "排除鎖死_訊號數", "排除鎖死_達標率%", "排除鎖死_提升pp", "通過"]].copy()
    T["鎖死訊號數"] = T.訊號數 - T.排除鎖死_訊號數
    for c in ("達標率%", "提升pp", "排除鎖死_達標率%", "排除鎖死_提升pp"):
        T[c] = T[c].map(f1)
    lines += ["## 1. 保守成交版本（排除該方向 t+1 一字鎖死的樣本）", "",
              "UP 排除 t+1 一字漲停（買不到）；DOWN 排除 t+1 一字跌停（賣不掉）。基準在排除後的樣本上重算。", "", md(T), ""]

    # 2. 未知標籤敏感度
    L = pd.read_parquet(out / "labels.parquet")
    U = L[(L.date >= s2.VAL_START) & (L.label_end_date <= s2.VAL_END) & L.status.isin(UNKNOWN)].copy()
    fe = L[(L.date >= s2.VAL_START) & (L.label_end_date <= s2.VAL_END) & (L.status == "feature_event_unreliable")]
    X = pd.read_parquet(out / "features.parquet")
    feats_a = [c for c in X.columns if c not in ("code", "date")]
    U = U.merge(X, on=["date", "code"], how="left").merge(pd.read_parquet(out / "bfeat.parquet"), on=["date", "code"], how="left")
    U[feats_a + s2.BF] = U[feats_a + s2.BF].replace([np.inf, -np.inf], np.nan)
    U = U.copy()
    e = fz["vol_group_edges"]
    U["volgrp"] = pd.cut(U.vol20, [-np.inf, *e, np.inf], labels=["低", "中", "高"]).astype(object).fillna("缺")
    gb = {side: V.groupby(["date", "volgrp"])[f"base_{side}"].first() for side in s2.DIRS}
    rows = []
    for name, feats in (("A", feats_a), ("B", feats_a + s2.BF)):
        p = lgb.Booster(model_file=str(res / f"model_{name}.txt")).predict(U[feats])
        U = U.assign(**{f"p{name}_{k}": p[:, i] for k, i in s2.CLS.items()})
        for side, other in s2.DIRS.items():
            for pct in s2.PROTO["signals"]["candidate_percentiles"]:
                t = fz["thresholds"][name][side][str(pct)]
                su = s2.signals(U[f"p{name}_{side}"].to_numpy(), U[f"p{name}_{other}"].to_numpy(), t)
                sv = s2.signals(V[f"p{name}_{side}"].to_numpy(), V[f"p{name}_{other}"].to_numpy(), t)
                S, Su = V[sv], U[su]
                day_mean = V.groupby("date")[f"base_{side}"].mean()
                bu = pd.Series([gb[side].get((d, g), day_mean.get(d, np.nan)) for d, g in zip(Su.date, Su.volgrp)], dtype=float)
                n, u, h = len(S), len(Su), int((S.cls == side).sum())
                ex = (S.cls == side).astype(float).sum() - S[f"base_{side}"].sum()
                rows.append({"模型": name, "方向": side, "百分位": pct, "可評分訊號": n, "未知標籤訊號": u,
                             "未知_狀態": "、".join(f"{k}{v}" for k, v in Su.status.value_counts().items()),
                             "達標率%": f1(100 * h / n), "達標率下界%": f1(100 * h / (n + u)), "達標率上界%": f1(100 * (h + u) / (n + u)),
                             "提升pp": f1(100 * ex / n), "提升下界pp": f1(100 * (ex - bu.sum()) / (n + u)),
                             "提升上界pp": f1(100 * (ex + u - bu.sum()) / (n + u))})
    lines += ["## 2. 未知標籤敏感度", "",
              f"驗證期內不能評分的樣本：{U.status.value_counts().to_dict()}（共 {len(U)} 筆）。"
              f"另有 feature_event_unreliable {len(fe)} 筆依事前規則排除（訊號時已知），不列入未知。",
              "用凍結模型與門檻判斷它們是否會發訊號；下界把會發訊號的未知樣本全算未達標，上界全算達標。", "", md(pd.DataFrame(rows)), ""]

    # 3. 分年
    keyp = {(v["模型"], side, v["百分位"]) for side, v in prim.items()}
    Y = C[C.模型.eq("A") | pd.Series([(r.模型, r.方向, r.百分位) in keyp for r in C.itertuples()], index=C.index)]
    Y = pd.DataFrame([{"模型": r.模型, "方向": r.方向, "百分位": r.百分位, "全期 達標/提升": f"{f1(r['達標率%'])} / {f1(r['提升pp'])}",
                       "2023 訊號數": r["2023訊號數"], "2023 達標/提升": f"{f1(r['2023達標率%'])} / {f1(r['2023提升pp'])}",
                       "2024 訊號數": r["2024訊號數"], "2024 達標/提升": f"{f1(r['2024達標率%'])} / {f1(r['2024提升pp'])}",
                       "通過": "✓" if r.通過 else "✗"} for _, r in Y.iterrows()])
    lines += ["## 3. 分年結果（模型 A 全部候選＋主要候選）", "", md(Y), ""]

    # 4. 分數分箱與校準
    edges = [0, 50, 80, 90, 95, 99, 100]
    rows = []
    for name in ("A", "B"):
        for side in s2.DIRS:
            sc = V[f"p{name}_{side}"].to_numpy()
            q = np.quantile(sc, np.array(edges) / 100)
            q[-1] += 1e-12
            for lo, hi, a, z in zip(edges[:-1], edges[1:], q[:-1], q[1:]):
                m = (sc >= a) & (sc < z)
                r = s2.evaluate(V, m, side, dates, IDX)
                rows.append({"模型": name, "方向": side, "分數百分位": f"{lo}–{hi}", "樣本數": r["訊號數"], "日期數": r["不同日期數"],
                             "平均預測機率%": f1(100 * sc[m].mean()), "實際達標率%": f1(r["達標率%"]), "基準%": f1(r["基準%"]),
                             "提升pp [40日]": f"{f1(r['提升pp'])} [{f1(r['提升lo40'])}, {f1(r['提升hi40'])}]"})
    lines += ["## 4. 分數分箱與機率校準（描述用）", "",
              "依模型對該方向的分數，在全部驗證樣本上切成百分位區間。這裡不套用「大於另一方向」的規則，所以和第 2 段的候選不完全相同。"
              "「平均預測機率」對照「實際達標率」可看出模型機率是否偏高或偏低。", "", md(pd.DataFrame(rows)), ""]

    # 5. R20 分布
    grp = V.groupby(["date", "volgrp"]).R20.transform("mean")
    rows = []
    for side, v in prim.items():
        n = v["模型"]
        t = fz["thresholds"][n][side][str(v["百分位"])]
        sig = s2.signals(V[f"p{n}_{side}"].to_numpy(), V[f"p{n}_{s2.DIRS[side]}"].to_numpy(), t)
        for lab, m in ((f"{side} 主要候選（{n} P{v['百分位']}）", sig), ("全部驗證樣本", np.ones(len(V), bool))):
            r = V.R20[m]
            rows.append({"樣本": lab, "筆數": int(m.sum()), "平均R20%": f1(100 * r.mean()), "同日同波動組平均R20%": f1(100 * grp[m].mean()),
                         "中位數%": f1(100 * r.median()), "≥+10%": f1(100 * (r >= .1).mean()), "0～+10%": f1(100 * ((r >= 0) & (r < .1)).mean()),
                         "−10%～0": f1(100 * ((r > -.1) & (r < 0)).mean()), "≤−10%": f1(100 * (r <= -.1).mean()),
                         "P10%": f1(100 * r.quantile(.1)), "P90%": f1(100 * r.quantile(.9))})
    lines += ["## 5. R20 分布（不是交易績效）", "",
              "R20 是 t+1 開盤到 t+20 收盤的還原報酬，未計交易成本、滑價與部位；DOWN 訊號的 R20 越負代表看空越準，不代表可實現的放空收益。", "",
              md(pd.DataFrame(rows)), ""]

    # 6. 舊二分類 vs 本輪三分類
    old_cfg = json.loads((ROOT / "backtest/market/qlib/config.json").read_text(encoding="utf-8"))
    O = pd.read_csv(ROOT / "backtest/market/qlib/selections.csv.gz", dtype={"code": str}, parse_dates=["date"])
    O = O[(O.方向 == "long") & (O.期間 == "驗證")]
    diff = pd.DataFrame([
        ["標籤", "R20 ≥ +10%（二分類，做多、看空各一個模型）", "UP／DOWN／OTHER（一個三分類模型）"],
        ["樣本排除", "t−60..t 有無法還原跳空；另排除 t−20..t+20（含未來資訊）", "status == ok（t−60..t 為事前規則；t+1..t+20 為資料品質排除）"],
        ["訓練", "2015–2020 擬合、2021–2022 early stop，不重訓", "2015–2020 擬合、2021–2022 early stop，再以固定迭代數重訓 2015–2022"],
        ["LightGBM", f"lr {old_cfg['lightgbm']['learning_rate']}、leaves {old_cfg['lightgbm']['num_leaves']}、min_data {old_cfg['lightgbm']['min_data_in_leaf']}、子抽樣 0.8",
         "lr 0.03、leaves 15、max_depth 4、min_child 500、reg_lambda 10、無子抽樣"],
        ["門檻來源", "early-stop 期間（2021–2022）預測機率的前 X% 分位", "驗證期（2023–2024）分數的第 90／95／99 百分位"],
        ["訊號規則", "機率 ≥ 門檻", "該方向分數 ≥ 門檻，且大於另一方向分數"],
        ["重複訊號", "同檔觸發後 20 個交易日不重複計入", "每日重疊訊號全部計入"],
        ["區間", "月份群集 bootstrap", "40 日 moving-block bootstrap"]], columns=["項目", "舊二分類（backtest/market/qlib）", "本輪三分類（tw20）"])
    k = O.merge(L[["date", "code", "cls", "status"]], on=["date", "code"], how="left")
    rows = []
    for sel, g in k.groupby("選股", sort=False):
        rows.append({"樣本": f"舊二分類 做多{sel}（驗證期、已冷卻）", "筆數": len(g), "舊標籤達標率%": f1(100 * g.long.mean()),
                     "新標籤達標率%": f1(100 * (g.cls == "UP").mean()), "新標籤狀態非 ok": int((g.status != "ok").sum())})
    a_up = s2.signals(V.pA_UP.to_numpy(), V.pA_DOWN.to_numpy(), fz["thresholds"]["A"]["UP"]["90"])
    for pct in (90, 95, 99):
        sig = s2.signals(V.pA_UP.to_numpy(), V.pA_DOWN.to_numpy(), fz["thresholds"]["A"]["UP"][str(pct)])
        S = V[sig]
        Sc = cooldown(S, dates)
        mc = V.index.isin(Sc.index)
        r = s2.evaluate(V, mc, "UP", dates, IDX)
        rows.append({"樣本": f"本輪 A 做多 P{pct}", "筆數": len(S), "新標籤達標率%": f1(100 * (S.cls == "UP").mean())})
        rows.append({"樣本": f"本輪 A 做多 P{pct}＋20 日冷卻（同舊規則）", "筆數": len(Sc), "新標籤達標率%": f1(r["達標率%"]),
                     "基準%": f1(r["基準%"]), "提升pp [40日]": f"{f1(r['提升pp'])} [{f1(r['提升lo40'])}, {f1(r['提升hi40'])}]"})
    vk = V[a_up][["date", "code"]].assign(新A_P90=True)
    ov = O.merge(vk, on=["date", "code"], how="left")
    ovl = {sel: f"{100 * g.新A_P90.fillna(False).astype(bool).mean():.0f}%" for sel, g in ov.groupby("選股", sort=False)}
    lines += ["## 6. 舊二分類與本輪三分類的差異", "", "### 設計差異", "", md(diff), "", "### 數字對照（驗證期 2023–2024）", "",
              md(pd.DataFrame(rows)), "",
              f"舊二分類的做多選股中，同一天同一檔也是本輪 A 做多 P90 訊號的比例：{ovl}。", "",
              "注意：舊研究在驗證期用 early-stop 期間的分位當門檻，入選比例不等於本輪的 10%／5%／1%；"
              "加上冷卻規則，舊研究每檔每 20 天最多算 1 筆，筆數遠少於本輪，兩邊的數字不能直接比較。", ""]
    (dst / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
