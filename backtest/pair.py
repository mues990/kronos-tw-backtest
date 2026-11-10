"""原版 Base vs 台股微調版：5 日方向配對比較（探索性）。設定在 backtest/compare/config.json。

python backtest/pair.py prep  --mode pilot|full   # 解析 HF revision、存模型卡與 config、算快取鍵 → runs/<模型>-<鍵>/
python backtest/pair.py infer 2421 --mode pilot   # 兩個模型依序推論（需 PYTHONPATH 指到 Kronos；TWSE_OFFLINE=1 用 repo 內原始資料）
python backtest/pair.py score --mode pilot|full   # 共同樣本配對評分 → compare/results[_pilot]/
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from score import direction
from twse import CFG, HERE, RAW, bars, now_tw

C = json.loads((HERE / "compare" / "config.json").read_text(encoding="utf-8"))
SNAP = HERE.parent / C["snapshot"]
RUNS = HERE / "compare" / "runs"
LB = C["lookback"]
P = {k: C[k] for k in ("pred_len", "T", "top_p", "sample_count")}
TICKERS = CFG["etf"] + CFG["stocks"]
assert C["seed"] == CFG["seed"], "種子基數必須與原版回測相同"


def sha(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(HERE.parent)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def labels():
    L = pd.read_csv(SNAP / "labels.csv", dtype={"ticker": str}, parse_dates=["origin", "target"])
    return L[L.period == C["period"]]


def origins(ticker, mode):
    if mode == "pilot":
        return pd.to_datetime(C["pilot"].get(ticker, []))
    return pd.DatetimeIndex(sorted(labels().origin.unique()))


def current(mode):
    return json.loads((RUNS / f"current_{mode}.json").read_text(encoding="utf-8"))


# ---------- prep ----------
def prep(mode):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    data_files = [SNAP / "labels.csv", SNAP / "run/events.csv", SNAP / "run/prep_meta.json", SNAP / "run/grid.csv"]
    data_files += sorted(p for t in TICKERS for p in (RAW / "stock" / t).glob("*.json"))
    data_version = sha(*data_files)
    code_version = sha(HERE / "infer.py", HERE / "twse.py", HERE / "pair.py")
    cur = {}
    for name, m in C["models"].items():
        rev = {k: api.model_info(m[k]).sha for k in ("model", "tokenizer")}
        key = {"model": m["model"], "model_rev": rev["model"], "tokenizer": m["tokenizer"], "tok_rev": rev["tokenizer"],
               "lookback": LB, "max_context": C["max_context"], **P, "seed": C["seed"], "seed_rule": C["seed_rule"],
               "data_version": data_version, "data_cutoff": json.loads((SNAP / "run/prep_meta.json").read_text())["data_cutoff"],
               "kronos_commit": CFG["kronos_commit"], "code_version": code_version}
        kid = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
        d = RUNS / f"{name}-{kid}"
        (d / "card").mkdir(parents=True, exist_ok=True)
        (d / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
        for k in ("model", "tokenizer"):
            for fn in ("README.md", "config.json"):
                try:
                    f = hf_hub_download(m[k], fn, revision=rev[k])
                    (d / "card" / f"{k}_{fn}").write_bytes(Path(f).read_bytes())
                except Exception as e:  # 沒有 README 也記下來
                    (d / "card" / f"{k}_{fn}.missing").write_text(str(e))
            files = [s.rfilename for s in api.model_info(m[k], revision=rev[k], files_metadata=True).siblings]
            (d / "card" / f"{k}_files.json").write_text(json.dumps(files, indent=1))
        cur[name] = {"dir": d.name, **key}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"current_{mode}.json").write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    ticks = sorted(C["pilot"]) if mode == "pilot" else TICKERS
    print(json.dumps(cur, ensure_ascii=False, indent=1), file=sys.stderr)
    print(f"tickers={json.dumps(ticks)}")


# ---------- infer ----------
def infer(ticker, mode):
    import torch
    from infer import load_predictor, predict_rows
    torch.set_num_threads(os.cpu_count())
    cutoff = pd.Timestamp(json.loads((SNAP / "run/prep_meta.json").read_text())["data_cutoff"])
    b = bars(ticker, cutoff)
    b = b[b.index <= cutoff]
    ev = pd.read_csv(SNAP / "run/events.csv", dtype={"ticker": str}, parse_dates=["ex_date"])
    ev = ev[ev.ticker == ticker]
    o = origins(ticker, mode)

    def extra(x, p):
        v = p[["open", "high", "low", "close", "volume", "amount"]].to_numpy()
        return {"input_hash": hashlib.sha1(np.ascontiguousarray(x[["open", "high", "low", "close", "volume", "amount"]]
                                                                .to_numpy(np.float64)).tobytes()).hexdigest()[:16],
                "input_last_date": f"{x.index[-1]:%Y-%m-%d}", "pred_d5_date": f"{p.index[-1]:%Y-%m-%d}",
                "finite": bool(np.isfinite(v).all())}

    for name, k in current(mode).items():
        d = RUNS / k["dir"]
        t0 = time.time()
        pr = load_predictor(k["model"], k["tokenizer"], k["model_rev"], k["tok_rev"], C["max_context"])
        env = {"ticker": ticker, "model": k["model"], "model_rev": k["model_rev"], "tokenizer": k["tokenizer"],
               "tok_rev": k["tok_rev"], "load_secs": round(time.time() - t0, 1),
               "model_training": pr.model.training, "tokenizer_training": pr.tokenizer.training,
               "device": str(next(pr.model.parameters()).device),
               "dtypes": sorted({str(p.dtype) for p in list(pr.model.parameters()) + list(pr.tokenizer.parameters())}),
               "n_params_model": sum(p.numel() for p in pr.model.parameters()),
               "n_params_tokenizer": sum(p.numel() for p in pr.tokenizer.parameters()),
               "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(),
               "threads": torch.get_num_threads(), "cpu_count": os.cpu_count(), "cpu": platform.processor() or platform.machine(),
               "kronos_commit": CFG["kronos_commit"], "code_rev": os.environ.get("GITHUB_SHA"), "started": now_tw().isoformat()}
        assert not env["model_training"] and not env["tokenizer_training"], "模型或 tokenizer 不在推論模式"
        t0 = time.time()
        n = predict_rows(ticker, pr, b, ev, o, [LB], d / f"preds_{ticker}.csv", P, extra)
        env.update(new_rows=n, infer_secs=round(time.time() - t0, 1), finished=now_tw().isoformat())
        (d / "env").mkdir(exist_ok=True)
        (d / "env" / f"{mode}_{ticker}.json").write_text(json.dumps(env, indent=1))
        print(f"{name} {ticker}: {n} 筆新預測，{env['infer_secs']}s", flush=True)


# ---------- score ----------
def boot(y, a, b, n=4000, seed=0):
    """以預測日期為群集的配對 bootstrap：mean(a) − mean(b) 的 95% 區間（a、b 為逐筆欄位）。"""
    rng = np.random.default_rng(seed)
    g = y.groupby("origin")
    A, B, N = g[a].sum().to_numpy(), g[b].sum().to_numpy(), g.size().to_numpy()
    i = rng.integers(0, len(N), (n, len(N)))
    return np.percentile((A[i].sum(1) - B[i].sum(1)) / N[i].sum(1), [2.5, 97.5])


def paired_table(mode):
    L = labels()
    if mode == "pilot":
        L = L[[o in set(origins(t, mode)) for t, o in zip(L.ticker, L.origin)]]
    out = L.copy()
    for name, k in current(mode).items():
        parts = [pd.read_csv(p, dtype={"ticker": str}, parse_dates=["origin"]) for p in (RUNS / k["dir"]).glob("preds_*.csv")]
        p = pd.concat(parts) if parts else pd.DataFrame(columns=["ticker", "origin", "lookback"])
        p = p[p.lookback == LB].drop_duplicates(["ticker", "origin"])
        keep = ["ticker", "origin", "origin_close", "pred_close_d1", "pred_close_d5", "input_hash", "input_start", "secs"]
        out = out.merge(p[keep].rename(columns={c: f"{c}_{name}" for c in keep[2:]}), on=["ticker", "origin"], how="left")
    st = pd.read_csv(SNAP / "predictions_scored.csv", dtype={"ticker": str}, parse_dates=["origin"])
    st = st[st.lookback == LB][["ticker", "origin", "pred_close_d1", "pred_close_d5", "input_start"]]
    out = out.merge(st.rename(columns=lambda c: c if c in ("ticker", "origin") else f"{c}_stored"), on=["ticker", "origin"], how="left")
    for name in current(mode):
        has = out[f"pred_close_d5_{name}"].notna()
        bad = has & ((out[f"origin_close_{name}"] - out.origin_close).abs() > 1e-6)
        assert not bad.any(), f"{name} 預測輸入的最後收盤與原始收盤不符"
        out[f"ret_{name}"] = out[f"pred_close_d5_{name}"] / out.origin_close - 1
        out[f"dir_{name}"] = np.where(has, direction(out[f"ret_{name}"], out.thr), np.nan)
        out[f"hit_{name}"] = (out[f"dir_{name}"] == out.act_dir).astype(float).where(has)
        out[f"ae_{name}"] = (out[f"ret_{name}"] - out.actual_ret).abs()
        out[f"down_{name}"] = (out[f"dir_{name}"] == -1).astype(float).where(has)
    out["ret_stored"] = out.pred_close_d5_stored / out.origin_close - 1
    out["base_vs_stored_d5_rel"] = out.pred_close_d5_base / out.pred_close_d5_stored - 1
    out["same_input"] = out.input_hash_base == out.input_hash_tw
    out["up_hit"] = (out.act_dir == 1).astype(float)
    out["mom_hit"] = (out.mom_dir == out.act_dir).astype(float)
    out["flat_hit"] = (out.act_dir == 0).astype(float)
    out["ae_flat"] = out.actual_ret.abs()
    miss = np.select([out.pred_close_d5_base.isna() & out.pred_close_d5_tw.isna(), out.pred_close_d5_base.isna(),
                      out.pred_close_d5_tw.isna()], ["兩模型皆無預測", "Base 無預測", "TW 無預測"], "")
    out["pair_status"] = np.where(out.reason.fillna("") != "", out.reason.fillna(""),
                                  np.where(miss != "", miss, "配對"))
    out["paired"] = out.pair_status == "配對"
    return out


def metrics(y):
    r = {"樣本數": len(y), "日期數": y.origin.nunique()}
    for nm, k in (("Base", "base"), ("TW", "tw")):
        r[f"{nm}命中%"] = 100 * y[f"hit_{k}"].mean()
        r[f"{nm} MAE%"] = 100 * y[f"ae_{k}"].mean()
        r[f"{nm}平均預測報酬%"] = 100 * y[f"ret_{k}"].mean()
        r[f"{nm}預測跌%"] = 100 * y[f"down_{k}"].mean()
    r.update({"一律看漲%": 100 * y.up_hit.mean(), "近5日延續%": 100 * y.mom_hit.mean(), "收盤不變(命中平)%": 100 * y.flat_hit.mean(),
              "收盤不變 MAE%": 100 * y.ae_flat.mean(), "實際平均報酬%": 100 * y.actual_ret.mean()})
    return pd.Series(r)


def gaps(y):
    rows = []
    for a, b, nm in (("hit_tw", "hit_base", "命中：TW − Base"), ("hit_tw", "up_hit", "命中：TW − 一律看漲"),
                     ("hit_tw", "mom_hit", "命中：TW − 近5日延續"), ("hit_base", "up_hit", "命中：Base − 一律看漲"),
                     ("hit_base", "mom_hit", "命中：Base − 近5日延續"),
                     ("ae_tw", "ae_base", "MAE：TW − Base（負=較好）"), ("ae_tw", "ae_flat", "MAE：TW − 收盤不變（負=較好）"),
                     ("ae_base", "ae_flat", "MAE：Base − 收盤不變（負=較好）")):
        lo, hi = boot(y, a, b)
        rows.append({"比較": nm, "差距(pp)": 100 * (y[a].mean() - y[b].mean()), "95%區間": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
                     "lo": 100 * lo, "hi": 100 * hi})
    return pd.DataFrame(rows)


def concentration(y):
    loo = [100 * (y[y.ticker != t].hit_tw.mean() - y[y.ticker != t].hit_base.mean()) for t in y.ticker.unique()]
    d = y.assign(ex=y.hit_tw - y.hit_base).groupby("origin").ex.sum()
    tk = y.assign(ex=y.hit_tw - y.hit_base).groupby("ticker").ex.sum()
    share = lambda s: f"{100 * s.nlargest(3).sum() / s.sum():.0f}%" if s.sum() > 0 else "淨差≤0"
    return {"逐一剔除單檔後 TW−Base 範圍(pp)": f"[{min(loo):+.1f}, {max(loo):+.1f}]",
            "TW 比 Base 多命中的淨筆數": int(d.sum()), "前3檔佔淨差": share(tk), "前3日期佔淨差": share(d),
            "TW 較好日期數 / 較差 / 相同": f"{(d > 0).sum()} / {(d < 0).sum()} / {(d == 0).sum()}"}


def verdict(g):
    r = g.set_index("比較")
    lo, hi = r.loc["命中：TW − Base", "lo"], r.loc["命中：TW − Base", "hi"]
    beat = all(r.loc[k, "lo"] > 0 for k in ("命中：TW − Base", "命中：TW − 一律看漲", "命中：TW − 近5日延續"))
    if beat:
        return "有可靠改善：TW 對 Base 與兩個方向基準的 95% 區間下限都大於 0。"
    if hi < 0:
        return "有明顯惡化：TW 對 Base 的命中率差距 95% 區間上限小於 0。"
    return f"目前結果不確定：TW − Base 命中率差距 95% 區間 [{lo:+.1f}, {hi:+.1f}] pp 包含 0，尚無足夠證據支持改善。"


def md(df, fmt=".1f"):
    if not len(df):
        return "（無樣本）"
    df = df.copy()
    for c in ("樣本數", "日期數"):
        if c in df:
            df[c] = df[c].astype(int)
    df = df.apply(lambda s: s.map(lambda v: f"{v:{fmt}}") if s.dtype.kind == "f" else s.astype(str))
    return df.to_markdown(index=False, disable_numparse=True)


def score(mode):
    out_dir = HERE / "compare" / ("results_pilot" if mode == "pilot" else "results")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    d = paired_table(mode)
    d.to_csv(out_dir / "paired.csv", index=False)
    cur = current(mode)
    lines = [f"# 原版 Base vs 台股微調版｜5 日方向配對比較{'（試跑）' if mode == 'pilot' else ''}", "",
             "**探索性比較**：2026 年 Base 結果先前已看過；本輪只回答 5 日用途，微調版原本的 10 日用途另案評估。", "",
             f"固定條件：lookback={LB}、pred_len={P['pred_len']}、max_context={C['max_context']}、T={P['T']}、top_p={P['top_p']}、"
             f"sample_count={P['sample_count']}；種子規則 {C['seed_rule']}；價格口徑、流動性、處置、方向分類沿用 `{C['snapshot']}`。", ""]
    lines += ["## 模型版本", "", md(pd.DataFrame([{"組": n, "model": k["model"], "model_rev": k["model_rev"][:12],
                                                  "tokenizer": k["tokenizer"], "tok_rev": k["tok_rev"][:12], "快取目錄": k["dir"]}
                                                 for n, k in cur.items()])), "",
              "台股版訓練期間 2010-01-01–2024-12-31、613 檔 TWSE 上市股、訓練預測窗 10 日，均為**作者聲明**（見 `card/`）；"
              "還原方式與量額單位模型卡未載明，列為未知。", ""]

    # 樣本去向
    flow = d.groupby(["group", "pair_status"]).size().rename("筆數").reset_index()
    lines += ["## 樣本去向（標的 × 日期）", "", md(flow, ".0f"), ""]
    unp = d[~d.paired][["ticker", "origin", "group", "pair_status"]]
    unp.to_csv(out_dir / "unpaired.csv", index=False)
    lines += [f"未配對 {len(unp)} 筆，逐筆見 `unpaired.csv`。", ""]

    # 重現性與輸入一致
    h = d[d.pred_close_d5_base.notna() & d.pred_close_d5_stored.notna()]
    rep = {"比對筆數": len(h), "pred_d5 完全相同": int((h.pred_close_d5_base == h.pred_close_d5_stored).sum()),
           "最大相對差": f"{h.base_vs_stored_d5_rel.abs().max():.2e}" if len(h) else "—",
           "方向不同筆數": int((direction(h.ret_base, h.thr) != direction(h.ret_stored, h.thr)).sum()),
           "input_start 不同": int((h.input_start_base != h.input_start_stored).sum()),
           "Base/TW 輸入雜湊不同": int((d.pred_close_d5_tw.notna() & d.pred_close_d5_base.notna() & ~d.same_input).sum())}
    lines += ["## Base 重跑 vs 已存結果、兩模型輸入一致性", "", md(pd.DataFrame([rep]), ".0f"), ""]

    T, verdicts = {}, {}
    for group in ("個股", "ETF"):
        y = d[d.paired & (d.group == group)]
        if not len(y):
            continue
        m = metrics(y).to_frame().T
        g = gaps(y)
        T[f"{group}_main"], T[f"{group}_gaps"] = m, g
        T[f"{group}_by_ticker"] = y.groupby("ticker").apply(metrics, include_groups=False).reset_index()
        T[f"{group}_by_month"] = y.assign(月份=y.origin.dt.strftime("%Y-%m")).groupby("月份").apply(metrics, include_groups=False).reset_index()
        verdicts[group] = verdict(g)
        lines += [f"## {group}{'（主要評估）' if group == '個股' else '（另報）'}", "", md(m), "", md(g.drop(columns=["lo", "hi"])), "",
                  f"**判讀：{verdicts[group]}**", ""]
        if y.ticker.nunique() > 1:
            lines += ["集中度：", "", md(pd.DataFrame([concentration(y)])), ""]
        lines += ["各檔：", "", md(T[f"{group}_by_ticker"][["ticker", "樣本數", "Base命中%", "TW命中%", "一律看漲%", "近5日延續%",
                                                           "Base MAE%", "TW MAE%", "Base預測跌%", "TW預測跌%"]]), "",
                  "各月：", "", md(T[f"{group}_by_month"][["月份", "樣本數", "Base命中%", "TW命中%", "一律看漲%", "近5日延續%",
                                                         "Base MAE%", "TW MAE%"]]), ""]
    for k, t in T.items():
        t.to_csv(out_dir / "tables" / f"{k}.csv", index=False)
    if mode == "pilot":
        cols = ["ticker", "origin", "events_in_window", "origin_close", "pred_close_d5_stored", "pred_close_d5_base",
                "base_vs_stored_d5_rel", "pred_close_d5_tw", "same_input", "pair_status"]
        lines += ["## 試跑逐筆", "", md(d[cols], ".4f"), ""]
    envs = [json.loads(p.read_text()) for k in cur.values() for p in sorted((RUNS / k["dir"] / "env").glob(f"{mode}_*.json"))]
    if envs:
        e = pd.DataFrame(envs)
        lines += ["## 執行環境", "", md(e[["model", "ticker", "torch", "numpy", "pandas", "python", "device", "dtypes", "threads",
                                         "model_training", "tokenizer_training", "load_secs", "infer_secs", "new_rows"]]), ""]
    lines += ["> 方向命中率是預測準確度，不是交易績效（未計交易成本、滑價與部位）。"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "verdict.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["prep", "infer", "score"])
    a.add_argument("ticker", nargs="?")
    a.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    x = a.parse_args()
    {"prep": lambda: prep(x.mode), "infer": lambda: infer(x.ticker, x.mode), "score": lambda: score(x.mode)}[x.cmd]()
