"""tw20 第 2 段：A／B 模型（Alpha158＋LightGBM 三分類；B 另加 8 個市場環境特徵）與驗證期評估。
規格一律讀 research/tw20/protocol.json（事前登錄）。本段不載入、不預測、不評分 2025 年以後的樣本。

python research/tw20/stage2.py bfeat OUT              # 8 個 B 特徵 → OUT/bfeat.parquet（需先有 OUT/labels.parquet）
python research/tw20/stage2.py run   OUT [--placebo]  # 訓練 A／B、驗證期評估 → OUT/stage2[_placebo]/
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tw20  # noqa: E402

PROTO_FILE = Path(__file__).parent / "protocol.json"
PROTO = json.loads(PROTO_FILE.read_text(encoding="utf-8"))
CLS = PROTO["target"]["class_map"]
BF = list(PROTO["models"]["B_features"])
FIT_END, ES_START, DEV_END = pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")
VAL_START, VAL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
DIRS = {"UP": "DOWN", "DOWN": "UP"}


# ---------- B 特徵 ----------
def bfeat(out):
    out = Path(out)
    cal, P, F, _, tx = tw20.panel(out)
    L = pd.read_parquet(out / "labels.parquet", columns=["date", "code"])
    c = P["close"].where(P["close"] > 0)
    ret1 = c / (c.ffill().shift() * F) - 1
    px = (1 + ret1.fillna(0)).cumprod().where(c.ffill().notna())      # 還原收盤；停牌日沿用前值，上市前為缺值
    univ = pd.DataFrame(False, index=cal, columns=c.columns)
    univ.values[cal.get_indexer(L.date), c.columns.get_indexer(L.code)] = True
    ma60 = px.rolling(60, min_periods=60).mean()
    day = pd.DataFrame({
        "MKT_RET20": tx / tx.shift(20) - 1, "MKT_RET60": tx / tx.shift(60) - 1,
        "MKT_MA60_GAP": tx / tx.rolling(60, min_periods=60).mean() - 1,
        "MKT_VOL20": (tx / tx.shift() - 1).rolling(20, min_periods=20).std(ddof=1),
        "BREADTH_MA60": ((px > ma60) & univ).sum(1) / (ma60.notna() & univ).sum(1),
        "BREADTH_UP": ((ret1 > 0) & univ).sum(1) / (ret1.notna() & univ).sum(1)}, index=cal)
    ii, jj = cal.get_indexer(L.date), c.columns.get_indexer(L.code)
    B = L.copy()
    for k in day:
        B[k] = day[k].to_numpy()[ii]
    B["REL_RET20"] = (px / px.shift(20) - 1).to_numpy()[ii, jj] - B.MKT_RET20
    B["REL_RET60"] = (px / px.shift(60) - 1).to_numpy()[ii, jj] - B.MKT_RET60
    B = B[["date", "code"] + BF]
    B.to_parquet(out / "bfeat.parquet", index=False)
    print(B[BF].describe().T[["count", "mean", "min", "max"]].to_string())


# ---------- 評估 ----------
def signals(p_dir, p_other, thr):
    """該方向分數 ≥ 門檻 且 > 另一方向分數；相等則棄權。"""
    return (p_dir >= thr) & (p_dir > p_other)


def block_index(n, block, reps, seed):
    """moving-block bootstrap 的日期索引（reps × n）；所有候選共用。"""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, k))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]


def ci(num, den, idx):
    """Σnum/Σden 在各 bootstrap 樣本的 95% 區間（沒有訊號的樣本略過）。"""
    d = den[idx].sum(1)
    v = num[idx].sum(1)[d > 0] / d[d > 0]
    return (np.nan, np.nan) if not len(v) else tuple(np.percentile(v, [2.5, 97.5]))


def evaluate(V, sig, side, dates, IDX):
    other = DIRS[side]
    S = V[sig]
    pos = pd.Series(np.arange(len(dates)), index=dates)
    hit, base = (S.cls == side).to_numpy(float), S[f"base_{side}"].to_numpy()
    agg = lambda x: np.bincount(pos[S.date].to_numpy(), weights=x, minlength=len(dates))
    n_d, h_d, e_d = agg(np.ones(len(S))), agg(hit), agg(hit - base)
    r = {"訊號數": len(S), "不同日期數": int((n_d > 0).sum()), "覆蓋率%": 100 * len(S) / len(V),
         "達標率%": 100 * hit.mean() if len(S) else np.nan, "反向達標率%": 100 * (S.cls == other).mean() if len(S) else np.nan,
         "基準%": 100 * base.mean() if len(S) else np.nan, "提升pp": 100 * (hit - base).mean() if len(S) else np.nan}
    for b, idx in IDX.items():
        lo, hi = ci(e_d, n_d, idx)
        r[f"提升lo{b}"], r[f"提升hi{b}"] = 100 * lo, 100 * hi
    lo, hi = ci(h_d, n_d, IDX[PROTO["bootstrap"]["block_days"]])
    r["達標率lo"], r["達標率hi"] = 100 * lo, 100 * hi
    for y in (2023, 2024):
        m = (S.date.dt.year == y).to_numpy()
        r[f"{y}訊號數"] = int(m.sum())
        r[f"{y}達標率%"] = 100 * hit[m].mean() if m.any() else np.nan
        r[f"{y}提升pp"] = 100 * (hit[m] - base[m]).mean() if m.any() else np.nan
    return r


def passed(r):
    R, b = PROTO["pass_rule"], PROTO["bootstrap"]
    checks = {"訊號數": r["訊號數"] >= R["min_signals"], "日期數": r["不同日期數"] >= R["min_distinct_dates"],
              "分年皆正": bool(r["2023提升pp"] > 0 and r["2024提升pp"] > 0), "主要區間": bool(r[f"提升lo{b['block_days']}"] > 0),
              "敏感性": all(bool(r[f"提升lo{x}"] > 0) for x in b["sensitivity_block_days"])}
    return all(checks.values()), "、".join(k for k, v in checks.items() if not v)


def pick(rows):
    """合格候選中達標率最高，同分取覆蓋率較高。"""
    ok = [r for r in rows if r["通過"]]
    return max(ok, key=lambda r: (r["達標率%"], r["覆蓋率%"])) if ok else None


def run(out, placebo=False):
    import lightgbm as lgb
    out = Path(out)
    res = out / ("stage2_placebo" if placebo else "stage2")
    res.mkdir(parents=True, exist_ok=True)
    L = pd.read_parquet(out / "labels.parquet")
    counts = L[L.date <= VAL_END].groupby([L.date.dt.year.rename("year"), "status"]).size().unstack(fill_value=0)
    L = L[(L.label_end_date <= VAL_END) & (L.status == "ok")]                 # 保留期與未成熟樣本在這裡就丟掉
    X = pd.read_parquet(out / "features.parquet")
    feats_a = [c for c in X.columns if c not in ("code", "date")]
    D = L.merge(X, on=["date", "code"], how="left", indicator="_x").merge(pd.read_parquet(out / "bfeat.parquet"), on=["date", "code"], how="left")
    assert D.date.max() <= VAL_END and D.label_end_date.max() <= VAL_END, "不得載入保留期樣本"
    no_feat = int((D._x != "both").sum())
    D = D[D._x == "both"].drop(columns="_x").reset_index(drop=True)
    D[feats_a + BF] = D[feats_a + BF].replace([np.inf, -np.inf], np.nan)
    D = D.copy()
    if placebo:                                                             # 負向對照：同一訊號日內打亂類別
        rng = np.random.default_rng(42)
        D["cls"] = D.groupby("date").cls.transform(lambda s: rng.permutation(s.to_numpy()))
    D["y"] = D.cls.map(CLS).astype(int)
    fit = D[D.label_end_date <= FIT_END]
    es = D[(D.date >= ES_START) & (D.label_end_date <= DEV_END)]
    dev = D[D.label_end_date <= DEV_END]
    V = D[D.date >= VAL_START].copy()
    assert fit.label_end_date.max() < es.date.min() and dev.label_end_date.max() < V.date.min(), "purge 失敗"
    edges = dev.vol20.quantile([1 / 3, 2 / 3]).to_list()
    V["volgrp"] = pd.cut(V.vol20, [-np.inf, *edges, np.inf], labels=["低", "中", "高"]).astype(object).fillna("缺")   # vol20 缺值自成一組
    for side in DIRS:
        V[f"base_{side}"] = (V.cls == side).groupby([V.date, V.volgrp], observed=True).transform("mean")

    P = PROTO["lightgbm"]
    params = {k: P[k] for k in ("objective", "num_class", "metric", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
                                "reg_lambda", "seed", "num_threads", "deterministic", "force_row_wise", "verbose")}
    models, imp = {}, {}
    for name, feats in (("A", feats_a), ("B", feats_a + BF)):
        t0 = time.time()
        m0 = lgb.train(params, lgb.Dataset(fit[feats], fit.y), P["n_estimators_max"], valid_sets=[lgb.Dataset(es[feats], es.y)],
                       callbacks=[lgb.early_stopping(P["early_stopping_rounds"], verbose=False)])
        it = int(m0.best_iteration)
        m = lgb.train(params, lgb.Dataset(dev[feats], dev.y), it)
        f = res / f"model_{name}.txt"
        m.save_model(str(f))
        p = m.predict(V[feats])
        for k, i in CLS.items():
            V[f"p{name}_{k}"] = p[:, i]
        g = pd.Series(m.feature_importance("gain"), index=feats)
        imp[name] = (100 * g / g.sum()).sort_values(ascending=False).head(15).round(1).to_dict()
        models[name] = {"best_iteration": it, "early_stop_logloss": float(m0.best_score["valid_0"]["multi_logloss"]),
                        "n_features": len(feats), "model_file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                        "train_secs": round(time.time() - t0, 1)}
        print(name, models[name], flush=True)

    b = PROTO["bootstrap"]
    dates = pd.DatetimeIndex(sorted(V.date.unique()))
    IDX = {x: block_index(len(dates), x, b["n"], b["seed"]) for x in [b["block_days"], *b["sensitivity_block_days"]]}
    rows, thr = [], {}
    for name in models:
        for side, other in DIRS.items():
            ps, po = V[f"p{name}_{side}"].to_numpy(), V[f"p{name}_{other}"].to_numpy()
            for pct in PROTO["signals"]["candidate_percentiles"]:
                t = float(np.quantile(ps, pct / 100))
                sig = signals(ps, po, t)
                r = {"模型": name, "方向": side, "百分位": pct, "門檻": t, **evaluate(V, sig, side, dates, IDX)}
                r["通過"], r["未過項目"] = passed(r)
                lock = V[f"entry_locked_{'up' if side == 'UP' else 'down'}"].to_numpy(bool)   # 參考：排除該方向 t+1 鎖死樣本
                V2 = V[~lock].copy()
                V2[f"base_{side}"] = (V2.cls == side).groupby([V2.date, V2.volgrp], observed=True).transform("mean")
                e2 = evaluate(V2, sig[~lock], side, dates, IDX)
                r.update({"排除鎖死_訊號數": e2["訊號數"], "排除鎖死_達標率%": e2["達標率%"], "排除鎖死_提升pp": e2["提升pp"]})
                rows.append(r)
                thr.setdefault(name, {}).setdefault(side, {})[str(pct)] = t
    primary = {}
    for side in DIRS:
        best = pick([x for x in (pick([r for r in rows if r["模型"] == n and r["方向"] == side]) for n in models) if x])
        primary[side] = None if best is None else {"模型": best["模型"], "百分位": best["百分位"], "門檻": best["門檻"],
                                                   "驗證達標率%": best["達標率%"], "驗證提升pp": best["提升pp"]}
    gate = any(v and v["驗證達標率%"] > 50 for v in primary.values())
    goal = 100 * PROTO["target"]["goal_hit_rate"]
    R = pd.DataFrame(rows)
    R.to_csv(res / "candidates.csv", index=False)
    cols = ["date", "code", "label_end_date", "R20", "cls", "vol20", "volgrp", "base_UP", "base_DOWN", "entry_locked_up", "entry_locked_down"]
    V[cols + [c for c in V.columns if c[:2] in ("pA", "pB")]].to_parquet(res / "validation_predictions.parquet", index=False)
    frozen = {"placebo": placebo, "protocol_sha256": hashlib.sha256(PROTO_FILE.read_bytes()).hexdigest(), "class_map": CLS,
              "vol_group_edges": edges, "models": models, "thresholds": thr, "primary": primary, "holdout_gate_open": bool(gate),
              "samples": {"fit": len(fit), "early_stop": len(es), "refit": len(dev), "validation": len(V), "validation_dates": len(dates),
                          "no_feature_rows_dropped": no_feat, "validation_vol20_missing": int(V.vol20.isna().sum()),
                          "fit_range": [f"{fit.date.min():%Y-%m-%d}", f"{fit.label_end_date.max():%Y-%m-%d}"],
                          "validation_range": [f"{V.date.min():%Y-%m-%d}", f"{V.label_end_date.max():%Y-%m-%d}"]},
              "validation_class_share": V.cls.value_counts(normalize=True).round(4).to_dict()}
    (res / "frozen.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")

    f1 = lambda v: "" if pd.isna(v) else f"{v:.1f}"
    T = pd.DataFrame([{
        "模型": r["模型"], "方向": r["方向"], "門檻": f"P{r['百分位']}（{r['門檻']:.3f}）", "訊號數": r["訊號數"], "日期數": r["不同日期數"],
        "覆蓋率%": f"{r['覆蓋率%']:.2f}", "達標率%": f"{f1(r['達標率%'])} [{f1(r['達標率lo'])}, {f1(r['達標率hi'])}]", "反向%": f1(r["反向達標率%"]),
        "基準%": f1(r["基準%"]), "提升pp [40日]": f"{f1(r['提升pp'])} [{f1(r['提升lo40'])}, {f1(r['提升hi40'])}]",
        "區間 20日": f"[{f1(r['提升lo20'])}, {f1(r['提升hi20'])}]", "區間 60日": f"[{f1(r['提升lo60'])}, {f1(r['提升hi60'])}]",
        "2023 達標/提升（n）": f"{f1(r['2023達標率%'])} / {f1(r['2023提升pp'])}（{r['2023訊號數']}）",
        "2024 達標/提升（n）": f"{f1(r['2024達標率%'])} / {f1(r['2024提升pp'])}（{r['2024訊號數']}）",
        "排除鎖死 達標/提升（n）": f"{f1(r['排除鎖死_達標率%'])} / {f1(r['排除鎖死_提升pp'])}（{r['排除鎖死_訊號數']}）",
        "通過": "✓" if r["通過"] else f"✗（{r['未過項目']}）", f"達 {goal:.0f}%": "是" if r["達標率%"] >= goal else "未達標"} for r in rows])
    md = [f"# tw20 第 2 段：A／B 驗證期結果{'（placebo：標籤已打亂，僅為負向對照）' if placebo else ''}", "",
          f"樣本：fit {len(fit):,}、early stop {len(es):,}、refit {len(dev):,}、驗證 {len(V):,}（{len(dates)} 個交易日，"
          f"{V.date.min():%Y-%m-%d}～{V.date.max():%Y-%m-%d}）。缺特徵而剔除 {no_feat} 筆。",
          f"驗證期類別比例：{frozen['validation_class_share']}。波動組切點：{edges[0]:.4f}、{edges[1]:.4f}。",
          "模型：" + "；".join(f"{k} 迭代 {v['best_iteration']}（early-stop logloss {v['early_stop_logloss']:.4f}）" for k, v in models.items()) + "。",
          "達標率與提升的區間為 moving-block bootstrap（2000 次、seed 42）；門檻是模型分數，不是校準後的勝率；每日重疊訊號不是獨立交易。", "",
          "## 候選比較", "", T.to_markdown(index=False, disable_numparse=True), "",
          "## 主要模型（事前規則選出）", ""]
    for side, v in primary.items():
        md.append(f"- **{side}**：" + ("沒有候選通過，記錄為未通過。" if v is None else
                                     f"模型 {v['模型']}、P{v['百分位']}（門檻 {v['門檻']:.4f}），驗證達標率 {v['驗證達標率%']:.1f}%、提升 {v['驗證提升pp']:+.1f} pp；"
                                     f"{'達' if v['驗證達標率%'] >= goal else '未達'} {goal:.0f}% 目標。"))
    md += ["", f"**保留期閘門（至少一個主要候選通過且驗證達標率 > 50%）：{'開啟' if gate else '不開啟'}。**", "",
           "## 重要特徵（gain 佔比 %，前 15）", ""] + [f"- **{k}**：" + "、".join(f"{a} {b}" for a, b in v.items()) for k, v in imp.items()]
    md += ["", "## 驗證期以前各年樣本狀態", "", counts.to_markdown(), ""]
    (res / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("<<<FROZEN_JSON\n" + json.dumps(frozen, ensure_ascii=False) + "\nFROZEN_JSON>>>")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["bfeat", "run"])
    a.add_argument("out")
    a.add_argument("--placebo", action="store_true")
    x = a.parse_args()
    bfeat(x.out) if x.cmd == "bfeat" else run(x.out, x.placebo)
