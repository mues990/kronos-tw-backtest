"""Qlib Alpha158（純 K 線＋量）＋ LightGBM：預測 20 日 ±10%，看「模型高分」時勝率能否提高。
設定 backtest/market/qlib/config.json（事前登錄）。需 Python ≤3.12 + pyqlib 0.9.7 + lightgbm + pyarrow。

python backtest/qlib_study.py dump      WORKDIR   # 還原價量 → Qlib 二進位格式（WORKDIR/qlib_data）
python backtest/qlib_study.py features  WORKDIR   # Alpha158 158 個因子，只留股票池樣本 → WORKDIR/features.parquet
python backtest/qlib_study.py model     WORKDIR   # 訓練、選股門檻、驗證期評估（通過才算保留期）→ backtest/market/qlib/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import market
from conditions import boot
from market import OUT

D = OUT / "qlib"
C = json.loads((D / "config.json").read_text(encoding="utf-8"))
FIELDS = ("open", "high", "low", "close", "volume", "vwap", "factor")


def frames_and_samples():
    fr = market.frames()
    s, q = market.samples(fr)
    clean = ~fr["jump"].astype(float).rolling(61, min_periods=1).max().fillna(1).astype(bool)  # 與第 1 批相同：60 日內無異常跳空
    key = clean.stack()
    s = s[s.set_index(["date", "code"]).index.isin(key[key].index)].reset_index(drop=True)
    for side in ("long", "short"):
        s[f"base_{side}"] = s.groupby(["date", "波動組"], observed=True)[side].transform("mean")
    return fr, s


def dump(work):
    fr, s = frames_and_samples()
    cal, c = fr["cal"], fr["c"]
    px = (1 + fr["ret1"].fillna(0)).cumprod().where(c.notna())
    adj = px / c
    data = {"open": fr["o"] * adj, "high": fr["h"] * adj, "low": fr["l"] * adj, "close": px,
            "volume": fr["v"] / adj, "vwap": fr["amt"] / fr["v"].where(fr["v"] > 0) * adj, "factor": adj}
    root = Path(work) / "qlib_data"
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(exist_ok=True)
    (root / "calendars" / "day.txt").write_text("\n".join(f"{d:%Y-%m-%d}" for d in cal) + "\n")
    codes = sorted(s.code.unique())
    inst = []
    for code in codes:
        col = c[code].to_numpy()
        idx = np.flatnonzero(~np.isnan(col))
        a, b = idx[0], idx[-1]
        fdir = root / "features" / code.lower()
        fdir.mkdir(parents=True, exist_ok=True)
        for k in FIELDS:
            v = data[k][code].to_numpy()[a:b + 1].astype("<f4")
            np.hstack([np.array([a], dtype="<f4"), v]).astype("<f4").tofile(fdir / f"{k}.day.bin")
        inst.append(f"{code}\t{cal[a]:%Y-%m-%d}\t{cal[b]:%Y-%m-%d}")
    (root / "instruments" / "all.txt").write_text("\n".join(inst) + "\n")
    s.to_parquet(Path(work) / "samples.parquet")
    print(f"{len(codes)} 檔、{len(cal)} 個交易日 → {root}")


def features(work):
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D as QD
    qlib.init(provider_uri=str(Path(work) / "qlib_data"), expression_cache=None, dataset_cache=None, kernels=2)
    fields, names = Alpha158DL.get_feature_config()
    s = pd.read_parquet(Path(work) / "samples.parquet")
    keep = pd.MultiIndex.from_arrays([s.code.str.lower(), s.date])
    codes = sorted(s.code.str.lower().unique())
    parts = []
    for i in range(0, len(codes), 40):
        f = QD.features(codes[i:i + 40], fields, start_time="2015-01-01", end_time="2026-12-31", freq="day")
        f.columns = names
        f = f[f.index.isin(keep)].astype("float32")
        parts.append(f)
        print(f"{min(i + 40, len(codes))}/{len(codes)} 檔", flush=True)
    F = pd.concat(parts)
    F.index = F.index.set_names(["code", "date"])
    F = F.reset_index()
    F["code"] = F.code.str.upper()
    F.to_parquet(Path(work) / "features.parquet")
    print(F.shape)


def cooldown(x, cal, days):
    pos = pd.Series(np.arange(len(cal)), index=cal)
    x = x.assign(i=x.date.map(pos)).sort_values(["code", "i"])
    keep, last = [], {}
    for r in x.itertuples():
        if r.i - last.get(r.code, -10**9) >= days:
            keep.append(r.Index)
            last[r.code] = r.i
    return x.loc[keep].drop(columns="i")


def evaluate(x, side):
    other = "short" if side == "long" else "long"
    x = x.assign(excess=x[side].astype(float) - x[f"base_{side}"])
    lo, hi = boot(x, **json.loads((OUT / "conditions" / "config.json").read_text(encoding="utf-8"))["bootstrap"])
    anyw = (x.long | x.short).mean()
    return {"筆數": len(x), "月數": x.date.dt.to_period("M").nunique(), "檔數": x.code.nunique(),
            "勝率%": 100 * x[side].mean(), "基準%": 100 * x[f"base_{side}"].mean(), "提升pp": 100 * x.excess.mean(),
            "95%區間": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]", "lo": 100 * lo,
            "反向勝率%": 100 * x[other].mean(), "反向基準%": 100 * x[f"base_{other}"].mean(),
            "任一方向±10%": 100 * anyw, "任一基準%": 100 * (x.base_long + x.base_short).mean(),
            "平均20日報酬%": 100 * x.ret.mean()}


def robust(s, S):
    s = s.copy()
    s["vdec"] = s.groupby("date").vol20.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
    for side in ("long", "short"):
        s[f"d_{side}"] = s.groupby(["date", "vdec"])[side].transform("mean")
    S = S.merge(s[["date", "code", "vdec", "d_long", "d_short"]], on=["date", "code"])
    bs = json.loads((OUT / "conditions" / "config.json").read_text(encoding="utf-8"))["bootstrap"]
    rows = []
    for (side, sel, per), x in S[S["期間"] != "early_stop"].groupby(["方向", "選股", "期間"], sort=False):
        other = "short" if side == "long" else "long"
        x = x.assign(excess=x[side].astype(float) - x[f"d_{side}"])
        lo, hi = boot(x, **bs)
        rows.append({"方向": "做多" if side == "long" else "長空", "選股": sel, "期間": per, "筆數": float(len(x)),
                     "勝率%": 100 * x[side].mean(), "十分位基準%": 100 * x[f"d_{side}"].mean(), "提升pp": 100 * x.excess.mean(),
                     "95%區間": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]", "反向勝率%": 100 * x[other].mean(),
                     "反向十分位基準%": 100 * x[f"d_{other}"].mean(), "最高波動十分位佔比%": 100 * (x.vdec == 9).mean()})
    return pd.DataFrame(rows)


def model(work):
    import lightgbm as lgb
    S = C["split"]
    s = pd.read_parquet(Path(work) / "samples.parquet")
    F = pd.read_parquet(Path(work) / "features.parquet")
    x = s.merge(F, on=["code", "date"], how="inner")
    feats = [c for c in F.columns if c not in ("code", "date")]
    cal = pd.DatetimeIndex(sorted(s.date.unique()))
    tr_end = pd.Timestamp(S["train"][1])
    purge_from = cal[cal <= tr_end][-S["purge_days"]]
    seg = lambda a, b: x[(x.date >= a) & (x.date <= b)]
    train = x[(x.date >= S["train"][0]) & (x.date < purge_from)]
    es, val = seg(*S["early_stop"]), seg(*S["validation"])
    P = dict(C["lightgbm"])
    nbr, esr = P.pop("num_boost_round"), P.pop("early_stopping_rounds")
    rows, imp, sel_out, best = [], {}, [], {}
    for side in C["targets"]:
        m = lgb.train(P, lgb.Dataset(train[feats], train[side].astype(int)), nbr,
                      valid_sets=[lgb.Dataset(es[feats], es[side].astype(int))],
                      callbacks=[lgb.early_stopping(esr, verbose=False)])
        g = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
        imp[side] = (100 * g / g.sum()).head(15).round(1)
        from sklearn.metrics import roc_auc_score  # noqa: E402 (lightgbm 的相依)
        p_es, p_val = m.predict(es[feats], num_iteration=m.best_iteration), m.predict(val[feats], num_iteration=m.best_iteration)
        auc = {"early_stop": roc_auc_score(es[side], p_es), "validation": roc_auc_score(val[side], p_val)}
        for pct in C["selection"]["top_pct"]:
            thr = np.quantile(p_es, 1 - pct / 100)
            for period, df, p in (("early_stop", es, p_es), ("驗證", val, p_val)):
                pick = cooldown(df[p >= thr][["date", "code", "ret", "long", "short", "base_long", "base_short", "波動組"]],
                                cal, C["selection"]["cooldown_days"])
                r = evaluate(pick, side)
                passed = (r["lo"] > C["pass"]["驗證"]["lift_ci_low_gt"]) if period == "驗證" else None
                rows.append({"方向": "做多" if side == "long" else "長空", "選股": f"前{pct}%", "期間": period, "門檻機率": thr,
                             "AUC": auc["validation" if period == "驗證" else "early_stop"], **r,
                             "通過": "—" if passed is None else ("✓" if passed else "✗")})
                sel_out.append(pick.assign(方向=side, 選股=f"前{pct}%", 期間=period))
                if passed:  # 只有通過驗證的門檻才看保留期（同一個模型，不重訓）
                    ho = seg(*S["holdout"])
                    p_ho = m.predict(ho[feats], num_iteration=m.best_iteration)
                    pk = cooldown(ho[p_ho >= thr][["date", "code", "ret", "long", "short", "base_long", "base_short", "波動組"]],
                                  cal, C["selection"]["cooldown_days"])
                    rows.append({"方向": rows[-1]["方向"], "選股": f"前{pct}%", "期間": "保留", "門檻機率": thr,
                                 "AUC": roc_auc_score(ho[side], p_ho), **evaluate(pk, side), "通過": "—"})
                    sel_out.append(pk.assign(方向=side, 選股=f"前{pct}%", 期間="保留"))
        best[side] = m.best_iteration
    R = pd.DataFrame(rows)
    R.drop(columns=["lo"], errors="ignore").to_csv(D / "results.csv", index=False)
    pd.concat(sel_out).to_csv(D / "selections.csv.gz", index=False, compression="gzip")
    fmt = lambda t: t.apply(lambda c: c.map(lambda v: "" if pd.isna(v) else f"{v:.3f}" if c.name in ("門檻機率", "AUC")
                                            else f"{v:.0f}" if c.name in ("筆數", "月數") else f"{v:.1f}")
                            if c.dtype.kind == "f" else c.astype(str).replace("nan", "")).to_markdown(index=False, disable_numparse=True)
    cols = ["方向", "選股", "期間", "AUC", "門檻機率", "筆數", "月數", "勝率%", "基準%", "提升pp", "95%區間", "反向勝率%", "反向基準%",
            "任一方向±10%", "任一基準%", "平均20日報酬%", "通過"]
    main = R[cols]
    lines = ["# 條件研究｜Qlib Alpha158 ＋ LightGBM（純 K 線＋量）", "",
             f"特徵：Qlib Alpha158 的 {len(feats)} 個因子。訓練 {S['train'][0]}～{S['train'][1]}（最後 {S['purge_days']} 個交易日去除，避免標籤重疊），"
             f"early stopping {S['early_stop'][0]}～{S['early_stop'][1]}，驗證 {S['validation'][0]}～{S['validation'][1]}。"
             f"樣本：訓練 {len(train):,}、early stop {len(es):,}、驗證 {len(val):,}。LightGBM 最佳迭代：做多 {best['long']}、長空 {best['short']}。",
             "選股：模型機率 ≥ early-stop 期間的前 X% 分位門檻；同檔觸發後 20 個交易日不重複。勝負、基準、月份群集與第 1 批相同。",
             "「反向勝率」= 同一批選股往另一方向動 10% 的比例；兩邊都升高代表只是挑到大波動，不是判斷方向。", "",
             "## 結果", "", fmt(main), "", "## 重要因子（gain 佔比 %，前 15）", ""]
    for side, g in imp.items():
        lines += [f"**{'做多' if side == 'long' else '長空'}**：" + "、".join(f"{k} {v}" for k, v in g.items()), ""]
    lines += ["## 穩健性檢查：基準改為同日期 × 同波動十分位（事後加做，不改變上面的事前判定）", "", fmt(robust(s, pd.concat(sel_out))), ""]
    lines += ["> 勝率是事件統計，不是交易績效（未計交易成本、滑價；長空未計借券限制）。"]
    (D / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"dump": dump, "features": features, "model": model}[sys.argv[1]](sys.argv[2])
