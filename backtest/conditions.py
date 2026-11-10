"""條件研究：事前登錄的量價條件 → 觸發後的 20 日 ±10% 勝率，與同日期 × 同波動組基準比較。

python backtest/conditions.py      # 設定 backtest/market/conditions/config.json → 結果 backtest/market/conditions/
流程：開發期全部評估 → 只有通過的進驗證期 → 只有再通過的才計算保留期。
"""
import json

import numpy as np
import pandas as pd

import market
from market import HERE, OUT

D = OUT / "conditions"
C = json.loads((D / "config.json").read_text(encoding="utf-8"))


def signals(fr):
    """各條件的布林矩陣（日期 × 檔），只用 t 日收盤前資訊。"""
    ret1, amt, tx = fr["ret1"], fr["amt"], fr["tx"]
    px = (1 + ret1.fillna(0)).cumprod().where(fr["c"].notna())  # 還原收盤指數（只用來比相對高低與均線）
    ma = {n: px.rolling(n, min_periods=n).mean() for n in (5, 20, 60)}
    vol_x2 = amt >= 2 * amt.shift().rolling(20, min_periods=15).mean()
    r20 = px / px.shift(20) - 1
    hi60, lo60 = px >= px.rolling(60, min_periods=60).max(), px <= px.rolling(60, min_periods=60).min()
    v = fr["vol20"].shift()
    v_low = v <= v.rolling(250, min_periods=200).quantile(0.2)
    above = tx > tx.rolling(60).mean()
    cross_up = (px > ma[20]) & (px.shift() <= ma[20].shift())
    cross_dn = (px < ma[20]) & (px.shift() >= ma[20].shift())
    return {
        "L1_突破量增": hi60 & vol_x2,
        "L2_急跌反彈": (r20 <= -0.15) & above.values[:, None],
        "L3_多頭首站20MA": (ma[5] > ma[20]) & (ma[20] > ma[60]) & cross_up,
        "L4_收斂後放量長紅": v_low & (ret1 >= 0.04) & vol_x2,
        "S1_破底量增": lo60 & vol_x2,
        "S2_高檔爆量長黑": (px.shift() / px.shift(61) - 1 >= 0.40) & (ret1 <= -0.05) & vol_x2,
        "S3_空頭首破20MA": (ma[5] < ma[20]) & (ma[20] < ma[60]) & cross_dn,
    }


def cooldown(sig, ok, days):
    """同一檔同一條件觸發後 days 個交易日內不再計入（等同已持有部位）。只在可評分樣本上觸發。"""
    a = (sig & ok).to_numpy()
    out = np.zeros_like(a)
    last = np.full(a.shape[1], -10**9)
    for i in range(a.shape[0]):
        hit = a[i] & (i - last >= days)
        out[i] = hit
        last[hit] = i
    return pd.DataFrame(out, index=sig.index, columns=sig.columns)


def boot(x, n, seed):
    """以月份為群集的 bootstrap：Σ(勝−基準)/筆數 的 95% 區間。"""
    g = x.groupby(x.date.dt.to_period("M"))
    e, m = g.excess.sum().to_numpy(), g.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(m), (n, len(m)))
    return np.percentile(e[idx].sum(1) / m[idx].sum(1), [2.5, 97.5])


def evaluate(x, side):
    w = x[side].astype(bool)
    x = x.assign(excess=w.astype(float) - x[f"base_{side}"])
    lo, hi = boot(x, **C["bootstrap"])
    return {"筆數": len(x), "月數": x.date.dt.to_period("M").nunique(), "檔數": x.code.nunique(),
            "勝率%": 100 * w.mean(), "基準%": 100 * x[f"base_{side}"].mean(), "提升pp": 100 * x.excess.mean(),
            "95%區間": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]", "lo": 100 * lo, "倍數": w.mean() / x[f"base_{side}"].mean(),
            "平均20日報酬%": 100 * x.ret.mean()}


def main():
    fr = market.frames()
    s, q = market.samples(fr)
    # 基準：同日期 × 同波動組的勝率
    for side in ("long", "short"):
        s[f"base_{side}"] = s.groupby(["date", "波動組"], observed=True)[side].transform("mean")
    clean = ~fr["jump"].astype(float).rolling(C["lookback_clean_days"] + 1, min_periods=1).max().fillna(1).astype(bool)
    ok = fr["ok"] & clean
    sigs = signals(fr)
    key = s.set_index(["date", "code"])
    rows, trades = [], []
    status = {}
    for name, spec in C["conditions"].items():
        side = spec["side"]
        hit = cooldown(sigs[name], ok, C["cooldown_days"]).stack()
        hit = hit[hit].index
        x = key.reindex(pd.MultiIndex.from_tuples(list(hit), names=["date", "code"])).dropna(subset=["ret"]).reset_index()
        x["條件"] = name
        trades.append(x)
        res = {}
        for period in ("開發", "驗證", "保留"):
            if period != "開發" and status.get(name) != period:
                break  # 上一期沒過，不看下一期
            y = x[x["期間"] == period]
            if not len(y):
                break
            r = evaluate(y, side)
            p = C["pass"][period] if period != "保留" else None
            if period == "開發":
                passed = r["筆數"] >= p["min_trades"] and r["月數"] >= p["min_months"] and r["lo"] > p["lift_ci_low_gt"]
            elif period == "驗證":
                passed = r["lo"] > p["lift_ci_low_gt"]
            else:
                passed = None
            status[name] = {"開發": "驗證", "驗證": "保留"}.get(period) if passed else None
            rows.append({"條件": name, "方向": "做多" if side == "long" else "長空", "期間": period, **r,
                         "通過": "—" if passed is None else ("✓" if passed else "✗")})
            # 開發期細分：波動組、大盤（描述用，不作為通過依據）
            if period == "開發":
                for col in ("波動組", "大盤"):
                    for k, z in y.groupby(col, observed=True):
                        if len(z) >= 30:
                            rows.append({"條件": name, "方向": "", "期間": f"開發｜{col}={k}", **evaluate(z, side), "通過": ""})
    R = pd.DataFrame(rows)
    T = pd.concat(trades)
    T.to_csv(D / "trades.csv.gz", index=False, compression="gzip")
    R.drop(columns=["lo"]).to_csv(D / "results.csv", index=False)
    show = R.drop(columns=["lo"])
    md = lambda t: t.apply(lambda c: c.map(lambda v: f"{v:.1f}" if c.name != "倍數" else f"{v:.2f}") if c.dtype.kind == "f"
                           else c.astype(str)).to_markdown(index=False, disable_numparse=True)
    main_rows = show[show["期間"].isin(["開發", "驗證", "保留"])]
    lines = ["# 條件研究｜第 1 批：量價條件", "",
             "勝負：訊號日 t 收盤後 → t+1 開盤買 → t+20 收盤結算；做多勝 ≥ +10%、長空勝 ≤ −10%。",
             f"基準：{C['baseline']}。提升 = 勝率 − 基準；95% 區間以月份為群集 bootstrap。"
             f"同檔同條件觸發後 {C['cooldown_days']} 個交易日內不重複計入。",
             "通過標準（事前登錄）：開發期 ≥100 筆、≥30 個月、提升區間下限 > 0 → 驗證期提升區間下限 > 0 → 才計算保留期。", "",
             "## 主結果", "", md(main_rows), "", "## 開發期細分（描述用）", "", md(show[show["期間"].str.startswith("開發｜")]), "",
             "> 勝率是事件統計，不是交易績效（未計交易成本、滑價；長空未計借券限制）。"]
    (D / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
