"""合併各標的預測 → 標記（流動性、處置、事件）→ 方向命中率／MAE 與基準比較 → 逐筆 CSV 與摘要。

python backtest/score.py            # 讀 backtest/work/，寫 backtest/results/（試跑寫 results_pilot/）
"""
import json
import sys

import numpy as np
import pandas as pd

from twse import CFG, HERE, WORK, mode, tickers

LBS = CFG["lookbacks"]
ETF = set(CFG["etf"])
FLAT = {1: "漲", 0: "平", -1: "跌"}


def tick(p, etf):
    """TWSE 升降單位。ETF：<50 為 0.01，否則 0.05。"""
    if etf:
        return 0.01 if p < 50 else 0.05
    for lim, t in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)):
        if p < lim:
            return t
    return 5.0


def direction(ret, thr):
    """漲跌幅小於半個升降單位 → 平（未還原期間就是收盤價完全相同）。"""
    return np.where(ret > thr, 1, np.where(ret < -thr, -1, 0))


def load():
    cal = pd.DatetimeIndex(pd.read_csv(WORK / "calendar.csv", parse_dates=["date"]).date)
    g = pd.read_csv(WORK / "grid.csv", parse_dates=["origin", "target"])
    g = g[(g.period != "gap") & (g.pilot if mode() == "pilot" else True)]
    ev = pd.read_csv(WORK / "events.csv", dtype={"ticker": str}, parse_dates=["ex_date"])
    pun = pd.read_csv(WORK / "punish.csv", dtype={"ticker": str}, parse_dates=["announce", "start", "end"]) \
        if (WORK / "punish.csv").stat().st_size > 1 else pd.DataFrame(columns=["ticker", "announce", "start", "end"])
    noti = pd.read_csv(WORK / "notice.csv", dtype={"ticker": str}, parse_dates=["date"]) \
        if (WORK / "notice.csv").stat().st_size > 1 else pd.DataFrame(columns=["ticker", "date"])
    meta = json.loads((WORK / "prep_meta.json").read_text(encoding="utf-8"))
    B, P = {}, []
    for t in tickers():
        f = WORK / f"bars_{t}.csv"
        if f.exists():
            B[t] = pd.read_csv(f, index_col="date", parse_dates=["date"])
        f = WORK / f"preds_{t}.csv"
        if f.exists():
            P.append(pd.read_csv(f, dtype={"ticker": str}, parse_dates=["origin"]))
    preds = pd.concat(P) if P else pd.DataFrame()
    return cal, g, ev, pun, noti, meta, B, preds


def labels(cal, g, ev, pun, noti, failed_months, B):
    pos = {d: i for i, d in enumerate(cal)}
    rows = []
    for t in tickers():
        etf = t in ETF
        b = B.get(t)
        c = b.close.dropna() if b is not None else pd.Series(dtype=float)
        e, pu = ev[ev.ticker == t], pun[pun.ticker == t]
        nt = set(noti[noti.ticker == t].date)
        for r in g.itertuples():
            i = pos[r.origin]
            oc, tc = c.get(r.origin, np.nan), c.get(r.target, np.nan)
            fw = e[(e.ex_date > r.origin) & (e.ex_date <= r.target)]
            prev = c[c.index <= cal[i - 5]]
            f_mom = e[(e.ex_date > cal[i - 5]) & (e.ex_date <= r.origin)].factor.prod()
            traded = c.index.intersection(cal[i - CFG["liquidity_days"] + 1: i + 1])
            liq = b.amount.reindex(traded).median() if b is not None else np.nan
            known = pu[pu.announce <= r.origin]
            active = ((known.start <= r.origin) & (known.end >= r.origin)).any()
            upcoming = ((known.start > r.origin) & (known.start <= r.target)).any()
            need = {f"{m:%Y-%m}" for m in pd.date_range(cal[i - CFG["freq_disp_days"] + 1], r.target, freq="D")}
            unknown = bool(need & set(failed_months))
            rows.append({
                "ticker": t, "group": "ETF" if etf else "個股", "origin": r.origin, "target": r.target, "period": r.period,
                "origin_close": oc, "target_close_raw": tc,
                "events_in_window": ";".join(f"{d:%Y-%m-%d}{k}" for d, k in zip(fw.ex_date, fw.kind)),
                "window_factor": fw.factor.prod(),
                "actual_ret": tc / (oc * fw.factor.prod()) - 1,
                "mom_ret": oc / (prev.iloc[-1] * f_mom) - 1 if len(prev) else np.nan,
                "thr": tick(oc, etf) / 2 / oc if oc == oc else np.nan,
                "liq_median20_twd": liq, "liq_pass": True if etf else bool(liq >= CFG["liquidity_min_twd"]),
                "disp_state": "未知" if unknown else "處置中" if active else "已公告將處置" if upcoming else "無",
                "post_disp_announced": bool(((pu.announce > r.origin) & (pu.announce <= r.target)).any()),
                "disp_count_60d": pu.announce[(pu.announce >= cal[i - CFG["freq_disp_days"] + 1]) & (pu.announce <= r.origin)].nunique(),
                "attention_5d": any(d in nt for d in cal[i - 4: i + 1]),
            })
    L = pd.DataFrame(rows)
    L["reason"] = np.select(
        [L.origin_close.isna(), L.target_close_raw.isna(), ~L.liq_pass, L.disp_state == "處置中", L.disp_state == "已公告將處置"],
        ["起點無成交", "第5日無有效收盤(無法評分)", "流動性不足", "處置中", "已公告將處置"], "")
    L["scorable"] = L.origin_close.notna() & L.target_close_raw.notna()
    L["main"] = L.reason == ""
    L["freq_disp"] = L.disp_count_60d >= CFG["freq_disp_min"]
    L["act_dir"] = np.where(L.actual_ret.notna(), direction(L.actual_ret, L.thr), np.nan)
    L["mom_dir"] = np.where(L.mom_ret.notna(), direction(L.mom_ret, L.thr), np.nan)
    return L


def data_checks(cal, B, ev, cutoff):
    """缺漏/停牌（交易所有開市但該檔無有效收盤）與無對應公司行動的跳空（|日報酬|>10.5%，超過漲跌幅限制）。"""
    rows, gaps = [], []
    for t, b in B.items():
        c = b.close.dropna()
        days = cal[(cal >= c.index[0]) & (cal <= cutoff)]
        miss = days.difference(c.index)
        rows.append({"ticker": t, "首日": f"{c.index[0]:%Y-%m-%d}", "末日": f"{c.index[-1]:%Y-%m-%d}", "有效K棒": len(c),
                     "無有效收盤的交易日": len(miss), "日期": ", ".join(f"{d:%Y-%m-%d}" for d in miss[:12]) + (" …" if len(miss) > 12 else "")})
        r = c / c.shift() - 1
        evd = set(ev[ev.ticker == t].ex_date)
        gaps += [{"ticker": t, "date": f"{d:%Y-%m-%d}", "ret%": 100 * v} for d, v in r[r.abs() > 0.105].items() if d not in evd]
    return pd.DataFrame(rows), pd.DataFrame(gaps, columns=["ticker", "date", "ret%"])


def merge(L, preds):
    d = preds.merge(L, on=["ticker", "origin"], suffixes=("_pred", ""))
    bad = (d.origin_close_pred - d.origin_close).abs() > 1e-6
    assert not bad.any(), f"預測輸入的最後收盤與原始收盤不符：\n{d[bad]}"
    d["pred_ret"] = d.pred_close_d5 / d.origin_close - 1
    d["pred_dir"] = direction(d.pred_ret, d.thr)
    d["hit"] = (d.pred_dir == d.act_dir).astype(int)
    return d.drop(columns="origin_close_pred")


def common(d):
    """四個窗口都有預測、且可評分的「標的 × 日期」。"""
    d = d[d.scorable]
    k = d.groupby(["ticker", "origin"]).lookback.transform("nunique")
    return d[k == len(LBS)]


def metrics(y):
    hit, up, mom = y.hit.mean(), (y.act_dir == 1).mean(), (y.mom_dir == y.act_dir).mean()
    return pd.Series({
        "樣本數": len(y), "命中率%": 100 * hit, "一律看漲%": 100 * up, "動能延續%": 100 * mom,
        "vs看漲pp": 100 * (hit - up), "vs動能pp": 100 * (hit - mom),
        "MAE%": 100 * (y.pred_ret - y.actual_ret).abs().mean(), "MAE不變%": 100 * y.actual_ret.abs().mean()})


def summ(d, keys, L=None, Lmask=None):
    """覆蓋率 = 進入比較的「標的×日期」/ 該組全部格點（含無法評分、被排除者）。"""
    out = d.groupby(keys + ["lookback"]).apply(metrics, include_groups=False).reset_index()
    out["樣本數"] = out["樣本數"].astype(int)
    if L is not None:
        den = L[Lmask].groupby(keys).size() if keys else Lmask.sum()
        out["覆蓋率%"] = 100 * out["樣本數"] / (out[keys[0]].map(den) if keys else den)
    return out


def boot(y, col, n=2000, seed=0):
    """以預測日期為群集的 bootstrap：命中率 − 基準 的 95% 區間（pp）。"""
    rng = np.random.default_rng(seed)
    by = y.groupby("origin")
    s = pd.DataFrame({"h": by.hit.sum(), "b": by[col].sum(), "n": by.size()})
    idx = rng.integers(0, len(s), (n, len(s)))
    h, b, m = s.h.values[idx].sum(1), s.b.values[idx].sum(1), s.n.values[idx].sum(1)
    return np.percentile(100 * (h - b) / m, [2.5, 97.5])


def md(df, fmt=".1f"):
    return df.to_markdown(index=False, floatfmt=fmt) if len(df) else "（無樣本）"


def main():
    pilot = mode() == "pilot"
    out = HERE / ("results_pilot" if pilot else "results")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    cal, g, ev, pun, noti, meta, B, preds = load()
    failed = meta["punish_failed_months"]
    L = labels(cal, g, ev, pun, noti, failed, B)
    L.to_csv(out / "labels.csv", index=False)
    d = merge(L, preds)
    d["up_hit"] = (d.act_dir == 1).astype(int)
    d["mom_hit"] = (d.mom_dir == d.act_dir).astype(int)
    d.to_csv(out / "predictions_scored.csv", index=False)

    lines = [f"# Kronos 台股 5 日方向回測{'（試跑：只驗證流程，不作為選窗或改參數依據）' if pilot else ''}", "",
             f"資料截止日 {meta['data_cutoff']}；模型 {CFG['model']}；pred_len={CFG['pred_len']}，T={CFG['T']}，"
             f"top_p={CFG['top_p']}，sample_count={CFG['sample_count']}，seed 基數 {CFG['seed']}。", ""]
    T = {}

    # 1. 開發期選窗（個股主組、共同樣本），同分取較短
    def grp(period, group, extra=None):
        m = (d.period == period) & (d.group == group) & d.main
        if extra is not None:
            m &= extra
        return common(d[m])

    dev_s = grp("dev", "個股")
    if len(dev_s):
        hits = dev_s.groupby("lookback").hit.sum()
        sel = int(min(hits[hits == hits.max()].index))
    else:
        sel = None
    (out / "selected_window.json").write_text(json.dumps({"selected_lookback": sel, "rule": "開發期個股主組方向命中率最高，同分取較短",
                                                         "dev_hits": {int(k): int(v) for k, v in (hits.items() if sel else [])}}, ensure_ascii=False))

    for period in ("dev", "holdout"):
        for group in ("個股", "ETF"):
            x = grp(period, group)
            Lm = (L.period == period) & (L.group == group)
            t = summ(x, [], L, Lm) if len(x) else pd.DataFrame()
            if len(t):
                t.insert(0, "選定", np.where(t.lookback == sel, "★", ""))
            T[f"{period}_{group}_main"] = t
            T[f"{period}_{group}_by_ticker"] = summ(x, ["ticker"], L, Lm) if len(x) else pd.DataFrame()

    # 2. 處置分組（個股、流動性達標、可評分；不要求四窗共同以外的條件）
    for period in ("dev", "holdout"):
        base = (d.period == period) & (d.group == "個股") & d.scorable
        parts = {
            "主組": base & d.main,
            "處置中或已公告將處置（主組外，另組評分）": base & d.liq_pass & d.disp_state.isin(["處置中", "已公告將處置"]),
            "主組中預測後才公告處置（事後標籤）": base & d.main & d.post_disp_announced,
            "主組中近5日注意股": base & d.main & d.attention_5d,
            "流動性不足（主組外）": base & ~d.liq_pass,
            "處置資料未知": base & (d.disp_state == "未知"),
        }
        rows = [summ(common(d[m]), []).assign(分組=k) for k, m in parts.items() if common(d[m]).size]
        T[f"{period}_disposition_groups"] = pd.concat(rows)[["分組"] + list(rows[0].columns[:-1])] if rows else pd.DataFrame()

        # 頻繁處置篩選：主組 vs 主組排除近60日≥2次處置公告
        before = common(d[base & d.main])
        after = common(d[base & d.main & ~d.freq_disp])
        Lm = (L.period == period) & (L.group == "個股")
        rows = []
        for name, x in (("套用前（主組）", before), ("套用後（排除近60日≥2次處置）", after)):
            if len(x):
                rows.append(summ(x, [], L, Lm).assign(樣本=name))
        T[f"{period}_freq_filter"] = pd.concat(rows)[["樣本"] + list(rows[0].columns[:-1])] if rows else pd.DataFrame()

    # 3. 集中度與信賴區間（選定窗口）
    conc = []
    for period in ("dev", "holdout"):
        for group in ("個股", "ETF"):
            x = grp(period, group)
            x = x[x.lookback == sel] if sel else x.iloc[:0]
            if not len(x):
                continue
            row = {"期間": period, "組別": group, "樣本數": len(x), "日期數": x.origin.nunique()}
            for col, nm in (("up_hit", "看漲"), ("mom_hit", "動能")):
                lo, hi = boot(x, col)
                row[f"vs{nm}pp"] = 100 * (x.hit.mean() - x[col].mean())
                row[f"vs{nm} 95%區間"] = f"[{lo:+.1f}, {hi:+.1f}]"
                loo = [100 * (x[x.ticker != t].hit.mean() - x[x.ticker != t][col].mean()) for t in x.ticker.unique()]
                row[f"vs{nm} 逐一剔除單檔後範圍"] = f"[{min(loo):+.1f}, {max(loo):+.1f}]" if len(loo) > 1 else "—"
                ex = x.groupby("ticker").apply(lambda y: (y.hit - y[col]).sum(), include_groups=False).sort_values(ascending=False)
                row[f"vs{nm} 前3檔佔淨超額%"] = f"{100 * ex.head(3).sum() / ex.sum():.0f}" if ex.sum() > 0 else "淨超額≤0"
            conc.append(row)
            T[f"{period}_{group}_by_month"] = summ(x.assign(月份=x.origin.dt.strftime("%Y-%m")), ["月份"])
    T["concentration"] = pd.DataFrame(conc)

    for k, t in T.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)

    # 摘要
    names = {"dev": "開發期 2024/07–2025/12", "holdout": f"保留期 2026/01–{meta['data_cutoff']}"}
    lines += [f"**事前選定窗口：lookback = {sel}**（{'開發期個股主組方向命中率最高，同分取較短' if sel else '無法選定'}）", ""]
    for period in ("dev", "holdout"):
        for group in ("個股", "ETF"):
            lines += [f"## {names[period]}｜{group}主組（四窗共同樣本）", "", md(T[f"{period}_{group}_main"]), ""]
    for period in ("dev", "holdout"):
        lines += [f"## {names[period]}｜頻繁處置篩選", "", md(T[f"{period}_freq_filter"]), "",
                  f"## {names[period]}｜處置與流動性分組", "", md(T[f"{period}_disposition_groups"]), ""]
    lines += ["## 集中度（選定窗口）", "", md(T["concentration"]), ""]
    for period in ("dev", "holdout"):
        for group in ("個股", "ETF"):
            x = T[f"{period}_{group}_by_ticker"]
            if len(x):
                lines += [f"## {names[period]}｜{group}各檔（選定窗口）", "", md(x[x.lookback == sel]), ""]
    miss, gaps = data_checks(cal, B, ev, pd.Timestamp(meta["data_cutoff"]))
    miss.to_csv(out / "tables" / "data_missing.csv", index=False)
    gaps.to_csv(out / "tables" / "data_unexplained_gaps.csv", index=False)
    lines += ["## 資料檢查", "", "缺漏／停牌（不補成交易日）：", "", md(miss, ".0f"), "",
              f"無對應公司行動的跳空（|日報酬|>10.5%）：{len(gaps)} 筆", "", md(gaps) if len(gaps) else "", ""]
    excl = L.groupby(["period", "group", "reason"]).size().rename("筆數").reset_index()
    excl["reason"] = excl.reason.replace("", "進入主組")
    lines += ["## 樣本去向（標的×日期）", "", md(excl, ".0f"), "",
              f"處置公告查詢失敗月份：{failed or '無'}；注意股查詢失敗月份：{meta['notice_failed_months'] or '無'}", "",
              "> 方向命中率是統計預測準確度，不代表交易績效（未計成本、滑價、部位）。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
