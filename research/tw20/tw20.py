"""台股 20 日達標分類（tw20）第 1 段：資料接入、標籤與樣本狀態、資料檢查、Qlib Alpha158 特徵。

python research/tw20/tw20.py events   OUT                 # 證交所「變更股票面額」事件（TWTB7U）原始 JSON → OUT/raw/
python research/tw20/tw20.py labels   OUT                 # 標籤與樣本狀態 → OUT/labels.parquet、OUT/jumps.csv
python research/tw20/tw20.py features OUT [--codes A,B]   # 完整歷史 → Qlib 二進位 → Alpha158，再套每日股票池 → OUT/features.parquet
python research/tw20/tw20.py checks   OUT                 # 資料檢查與抽查 → OUT/checks.json、OUT/checks.md
python research/tw20/tw20.py hashes   OUT                 # 輸入與輸出的 SHA-256 → OUT/hashes.json

任務定義（固定）：t 日收盤後產生訊號；t+1 交易日開盤為基準（第 1 日），t+20 交易日收盤評分。
R20 = close[t+20] / (open[t+1] × Π factor(t+1, t+20]) − 1。R20 ≥ +10% → UP，≤ −10% → DOWN，其餘 OTHER。
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest"))
import market  # noqa: E402
import twse  # noqa: E402
from twse import BASE, num, pick, roc, tables  # noqa: E402

H, UP, DOWN = 20, 0.10, -0.10
TOP_N, MIN_PRICE, MIN_HIST, LIQ_DAYS, LIQ_MINP = 150, 10.0, 120, 60, 40
FEAT_WIN = 61                        # Alpha158 最長視窗 60 日，部分因子另用 Ref(…, 1)
JUMP = 0.105                         # 超過 10% 漲跌幅限制（2015-06-01 前為 7%）又無對應公司行動 → 無法可靠還原
LIMIT_CHANGE = pd.Timestamp("2015-06-01")
CUTOFF = pd.Timestamp("2026-11-09")
YEARS = range(2015, 2027)
STATUS = ["ok", "immature", "no_entry_price", "no_exit_price", "label_event_unreliable", "feature_event_unreliable"]


# ---------- 變更面額事件 ----------
def fetch_par_change(out):
    """TWTB7U：上市公司變更股票面額。每年一次請求；原始 JSON 存檔、印出欄位供核對。"""
    raw = Path(out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log = []
    for y in YEARS:
        end = min(pd.Timestamp(y, 12, 31), CUTOFF)
        url = BASE + "change/TWTB7U?" + urllib.parse.urlencode({"startDate": f"{y}0101", "endDate": f"{end:%Y%m%d}", "response": "json"})
        try:
            raw_bytes, js = market.get(url)
        except RuntimeError as e:
            log.append({"year": y, "url": url, "error": str(e)})
            continue
        (raw / f"TWTB7U_{y}.json").write_bytes(raw_bytes)
        t = tables(js)
        log.append({"year": y, "url": url, "stat": js.get("stat"), "rows": sum(len(x) for x in t),
                    "fields": list(t[0].columns) if t else [], "sha256": hashlib.sha256(raw_bytes).hexdigest()})
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)
        if t:
            print(t[0].head(3).to_string(), flush=True)
    (raw / "TWTB7U_fetch_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def par_change_events(out):
    """解析 TWTB7U → (code, ex_date=恢復買賣日, factor)。factor 優先用 恢復買賣參考價 ÷ 停止買賣前收盤價（與減資相同口徑）。"""
    rows, method = [], set()
    for f in sorted((Path(out) / "raw").glob("TWTB7U_*.json")):
        if "log" in f.name:
            continue
        for t in tables(json.loads(f.read_text(encoding="utf-8"))):
            code = t[pick(t, "代號")].str.strip()
            day = [roc(x) for x in t[pick(t, "恢復買賣日期", "恢復買賣", "日期")]]
            cols = "".join(t.columns)
            if "參考價" in cols and "收盤" in cols:
                fac = num(t[pick(t, "參考價")]) / num(t[pick(t, "停止買賣前收盤", "收盤")])
                method.add("參考價/停止買賣前收盤")
            elif "換股率" in cols:
                fac = 1 / num(t[pick(t, "換股率")])
                method.add("1/換股率")
            else:
                raise KeyError(f"TWTB7U 欄位無法換算 factor：{list(t.columns)}")
            rows.append(pd.DataFrame({"code": code, "ex_date": day, "factor": fac.values}))
    if not rows:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["code", "ex_date"])), []
    ev = pd.concat(rows)
    ev = ev[np.isfinite(ev.factor) & (ev.factor > 0)]
    return ev.groupby(["code", "ex_date"]).factor.prod(), sorted(method)


# ---------- 面板與標籤 ----------
def panel(out):
    files = [market.OUT / f"daily_{y}.csv.gz" for y in YEARS]
    d = pd.concat(pd.read_csv(f, dtype={"code": str}, parse_dates=["date"]) for f in files)
    cal = pd.DatetimeIndex(sorted(d.date.unique()))
    P = {k: d.pivot(index="date", columns="code", values=k).reindex(cal) for k in ("open", "high", "low", "close", "volume", "amount")}
    ev0 = market.load_events(YEARS)
    ev1, method = par_change_events(out)
    ev = pd.concat([ev0, ev1]).groupby(level=[0, 1]).prod().rename("factor").rename_axis(["code", "ex_date"])
    cols = P["close"].columns
    f = np.ones((len(cal), len(cols)))
    e = ev.reset_index()
    e = e[e.code.isin(cols) & e.ex_date.isin(cal)]
    f[cal.get_indexer(e.ex_date), cols.get_indexer(e.code)] = e.factor.values
    F = pd.DataFrame(f, index=cal, columns=cols)
    taiex = pd.concat(pd.read_csv(market.OUT / f"manifest_{y}.csv", parse_dates=["date"]) for y in YEARS).set_index("date").taiex_close
    return cal, P, F, {"ex_right_reduction": int(len(ev0)), "par_change": int(len(ev1)), "par_change_method": method,
                       "par_change_codes": sorted(ev1.index.get_level_values(0).unique().tolist()) if len(ev1) else []}, taiex.reindex(cal)


def label_frame(cal, P, F):
    """回傳每個「t × 股票池內個股」一列：R20、類別、狀態、交易可行旗標、訊號前波動、label_end_date。"""
    o, h, l, c, amt = P["open"], P["high"], P["low"], P["close"].where(P["close"] > 0), P["amount"]
    ret1 = c / (c.ffill().shift() * F) - 1
    jump = ret1.abs() > JUMP
    vol20 = ret1.rolling(20, min_periods=18).std() * np.sqrt(20)   # 訊號前 20 日（含 t）還原日報酬標準差，ddof=1
    hist = c.notna().cumsum()
    liq = amt.where(c.notna()).rolling(LIQ_DAYS, min_periods=LIQ_MINP).median()
    elig = (hist >= MIN_HIST) & (c >= MIN_PRICE) & liq.notna()
    rank = liq.where(elig).rank(axis=1, ascending=False)           # 與舊基準相同（同值取平均名次）
    univ = rank <= TOP_N

    fwin = np.exp(np.log(F).rolling(H - 1).sum()).shift(-H)      # t+2..t+20 的 factor 乘積
    entry, exitc = o.shift(-1), c.shift(-H)
    r20 = exitc / (entry * fwin) - 1
    jf = jump.astype(float)
    lab_jump = jf[::-1].rolling(H, min_periods=1).max()[::-1].shift(-1).fillna(0).astype(bool)   # t+1..t+20
    feat_jump = jf.rolling(FEAT_WIN, min_periods=1).max().fillna(0).astype(bool)               # t−60..t
    limit = pd.Series(np.where(cal < LIMIT_CHANGE, 0.07, 0.10), index=cal)
    open_ret = (o / (c.shift() * F) - 1).shift(-1)                 # t+1 開盤相對 t 收盤（含 t+1 除權息）
    single = (h == l).shift(-1, fill_value=False)                  # t+1 全日一價
    lim1 = limit.shift(-1).to_numpy()[:, None]
    locked_up = single & (open_ret >= lim1 - 0.005)
    locked_dn = single & (open_ret <= -lim1 + 0.005)

    n = len(cal)
    pos = np.arange(n)[:, None] * np.ones((1, c.shape[1]), dtype=int)
    immature = pd.DataFrame(pos + H > n - 1, index=cal, columns=c.columns)
    status = np.select([x.to_numpy() for x in (immature, ~(entry > 0), exitc.isna(), lab_jump, feat_jump)], STATUS[1:], "ok")
    keep = univ.to_numpy()
    ii, jj = np.nonzero(keep)
    end_idx = np.minimum(ii + H, n - 1)
    L = pd.DataFrame({
        "date": cal[ii], "code": c.columns[jj], "label_end_date": pd.DatetimeIndex(cal[end_idx]).where(ii + H <= n - 1),
        "R20": r20.to_numpy()[ii, jj], "status": status[ii, jj],
        "entry_locked_up": locked_up.to_numpy()[ii, jj].astype(bool), "entry_locked_down": locked_dn.to_numpy()[ii, jj].astype(bool),
        "vol20": vol20.to_numpy()[ii, jj], "liq_rank": rank.to_numpy()[ii, jj], "close_raw": c.to_numpy()[ii, jj],
        "entry_open_raw": entry.to_numpy()[ii, jj], "exit_close_raw": exitc.to_numpy()[ii, jj], "window_factor": fwin.to_numpy()[ii, jj]})
    L["cls"] = np.where(L.R20.isna(), None, np.where(L.R20 >= UP, "UP", np.where(L.R20 <= DOWN, "DOWN", "OTHER")))
    L.loc[L.status == "immature", "cls"] = None
    jl = jump.stack()
    jl = jl[jl]
    jumps = pd.DataFrame({"date": jl.index.get_level_values(0), "code": jl.index.get_level_values(1)})
    jumps["ret"] = [ret1.at[d, k] for d, k in zip(jumps.date, jumps.code)]
    return L, jumps, ret1


def build_labels(out):
    cal, P, F, evinfo, taiex = panel(out)
    L, jumps, _ = label_frame(cal, P, F)
    L.to_parquet(Path(out) / "labels.parquet", index=False)
    jumps.to_csv(Path(out) / "jumps.csv", index=False)
    (Path(out) / "events_info.json").write_text(json.dumps(evinfo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(L.status.value_counts().to_string())
    print(f"無法還原的跳空 {len(jumps)} 筆；事件 {evinfo}")


# ---------- Qlib ----------
def features(out, codes=None):
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D as QD
    out = Path(out)
    cal, P, F, _, _ = panel(out)
    L = pd.read_parquet(out / "labels.parquet")
    c = P["close"].where(P["close"] > 0)
    ret1 = c / (c.ffill().shift() * F) - 1
    px = (1 + ret1.fillna(0)).cumprod().where(c.notna())            # 還原收盤（與標籤同一組 factor）
    adj = px / c                                                    # 調整係數：開高低、VWAP 乘上，量除以
    v = P["volume"].where(P["volume"] > 0)
    data = {"open": P["open"] * adj, "high": P["high"] * adj, "low": P["low"] * adj, "close": px,
            "volume": P["volume"] / adj, "vwap": P["amount"] / v * adj, "factor": adj}
    want = sorted(L.code.unique()) if not codes else codes
    root = out / "qlib_data"
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(exist_ok=True)
    (root / "calendars" / "day.txt").write_text("\n".join(f"{d:%Y-%m-%d}" for d in cal) + "\n")
    inst = []
    for code in want:
        idx = np.flatnonzero(c[code].notna().to_numpy())
        a, b = idx[0], idx[-1]                                      # 完整歷史（含不在股票池的日子）
        fd = root / "features" / code.lower()
        fd.mkdir(parents=True, exist_ok=True)
        for k, m in data.items():
            np.hstack([[a], m[code].to_numpy()[a:b + 1]]).astype("<f4").tofile(fd / f"{k}.day.bin")
        inst.append(f"{code}\t{cal[a]:%Y-%m-%d}\t{cal[b]:%Y-%m-%d}")
    (root / "instruments" / "all.txt").write_text("\n".join(inst) + "\n")
    qlib.init(provider_uri=str(root), expression_cache=None, dataset_cache=None, kernels=2)
    fields, names = Alpha158DL.get_feature_config()
    keep = pd.MultiIndex.from_arrays([L.code.str.lower(), L.date])
    parts, chk = [], []
    low = [x.lower() for x in want]
    for i in range(0, len(low), 40):
        g = low[i:i + 40]
        f = QD.features(g, fields, start_time="2015-01-01", end_time=f"{CUTOFF:%Y-%m-%d}", freq="day")
        f.columns = names
        parts.append(f[f.index.isin(keep)].astype("float32"))       # 先用完整歷史算，再套每日股票池
        k = QD.features(g, ["Ref($close,-20)/Ref($open,-1)-1", "$vwap", "$low", "$high", "$close/$factor"],
                        start_time="2015-01-01", end_time=f"{CUTOFF:%Y-%m-%d}", freq="day")
        k.columns = ["R20_qlib", "vwap", "low", "high", "close_raw_qlib"]
        chk.append(k[k.index.isin(keep)])
        print(f"{min(i + 40, len(low))}/{len(low)} 檔", flush=True)
    X = pd.concat(parts)
    X.index = X.index.set_names(["code", "date"])
    X = X.reset_index()
    X["code"] = X.code.str.upper()
    X.to_parquet(out / "features.parquet", index=False)
    K = pd.concat(chk)
    K.index = K.index.set_names(["code", "date"])
    K = K.reset_index()
    K["code"] = K.code.str.upper()
    K.to_parquet(out / "feature_label_check.parquet", index=False)
    meta = {"n_features": len(names), "feature_names": names, "codes": want, "rows": len(X),
            "qlib": qlib.__version__, "definition": "qlib.contrib.data.loader.Alpha158DL.get_feature_config()（官方 Alpha158 handler 同一組設定）"}
    (out / "features_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Alpha158 {len(names)} 個因子；{len(X):,} 列")


# ---------- 檢查 ----------
SPOT = [  # (代號, 訊號日, 用途)
    ("2330", "2019-06-03", "一般日期"), ("2317", "2018-06-01", "一般日期"),
    ("2454", "2024-06-19", "窗口內除息"), ("2603", "2021-06-28", "窗口內大跌（DOWN）"), ("1476", "2015-12-18", "窗口內除權（配股）"), ("3413", "2016-05-27", "窗口內除權息"),
    ("2327", "2025-07-28", "窗口內停牌＋變更面額（舊基準無法還原）"), ("2327", "2025-09-01", "變更面額之後"),
    ("1101", "2020-12-15", "訓練期末：label_end 跨入 2021"), ("2881", "2022-12-15", "開發期末：label_end 跨入 2023"),
    ("2308", "2024-12-16", "驗證期末：label_end 跨入 2025"), ("2382", "last_mature", "最後一個成熟訊號日（t+20＝截止日）"),
    ("2382", "first_immature", "第一個未成熟訊號日（t+20 超過截止日）"),
]


def checks(out):
    out = Path(out)
    cal, P, F, evinfo, taiex = panel(out)
    L = pd.read_parquet(out / "labels.parquet")
    jumps = pd.read_csv(out / "jumps.csv", dtype={"code": str}, parse_dates=["date"])
    R = {}
    man = pd.concat(pd.read_csv(market.OUT / f"manifest_{y}.csv", parse_dates=["date"]) for y in YEARS)
    R["資料涵蓋"] = {"交易日": len(cal), "起": f"{cal[0]:%Y-%m-%d}", "迄": f"{cal[-1]:%Y-%m-%d}", "截止日": f"{CUTOFF:%Y-%m-%d}",
                 "無資料日": int((man.n_stocks == 0).sum()), "每日檔數範圍": [int(man.n_stocks.min()), int(man.n_stocks.max())],
                 "加權指數缺值日": int(taiex.isna().sum())}
    # 單位：VWAP（金額÷股數）須落在當日最低～最高價之間
    v, a = P["volume"], P["amount"]
    vw = a / v.where(v > 0)
    inside = ((vw >= P["low"] * 0.999) & (vw <= P["high"] * 1.001))
    valid = vw.notna() & P["low"].notna()
    R["單位_VWAP"] = {"說明": "成交金額（元）÷ 成交股數（股）＝ 每股均價；與開高低同一原始尺度，特徵時再乘同一調整係數",
                    "可算筆數": int(valid.sum().sum()), "落在高低價區間": float(inside[valid].sum().sum() / valid.sum().sum()),
                    "超出區間筆數": int((valid & ~inside).sum().sum()), "成交股數為0但有價": int(((v == 0) & P["close"].notna()).sum().sum()),
                    "備註": "證交所成交股數／金額為全日官方總量（STOCK_DAY 與 MI_INDEX 兩來源一致），推測含鉅額、盤後等非盤中撮合成交，故少數日子均價落在盤中高低價外；照原值使用，不修改"}
    U = L[["date", "code"]]
    vin = pd.Series(inside.stack(), name="in").reset_index().set_axis(["date", "code", "in"], axis=1)
    vin = U.merge(vin, on=["date", "code"], how="left")
    R["單位_VWAP"]["股票池樣本中超出區間比例"] = float(1 - vin["in"].fillna(True).astype(bool).mean())
    # 公司行動
    R["公司行動"] = {**evinfo, "無法還原跳空": int(len(jumps)),
                 "跳空清單": [f"{r.code} {r.date:%Y-%m-%d} {r.ret:+.1%}" for r in jumps.sort_values("date").itertuples()][:60]}
    # 樣本狀態
    st = L.groupby([L.date.dt.year, "status"]).size().unstack(fill_value=0)
    R["樣本狀態_全期"] = L.status.value_counts().to_dict()
    R["交易限制旗標"] = {"t+1 一字漲停（UP 無法買進）": int(L.entry_locked_up.sum()), "t+1 一字跌停（DOWN 無法賣出）": int(L.entry_locked_down.sum())}
    R["類別分布_ok"] = L[L.status == "ok"].cls.value_counts().to_dict()
    # 未來資訊：舊基準的 t−20..t+20 排除
    R["事件排除與未來資訊"] = {
        "舊基準（backtest/market/base）": "排除 t−20..t+20 內有無法還原跳空的樣本；其中 t+1..t+20 需要事後才知道，不能當事前規則",
        "本研究": "t+1..t+20 內有無法還原跳空 → label_event_unreliable（資料品質排除，非選股規則）；t−60..t 內 → feature_event_unreliable（訊號時已知）",
        "label_event_unreliable": int((L.status == "label_event_unreliable").sum()),
        "feature_event_unreliable": int((L.status == "feature_event_unreliable").sum())}
    # 與舊基準比較
    old = pd.read_csv(market.OUT / "base" / "samples.csv.gz", dtype={"code": str}, parse_dates=["date"])
    m = L.merge(old[["date", "code", "ret"]], on=["date", "code"], how="outer", indicator=True)
    both = m[m._merge == "both"]
    diff = both[(both.R20 - both.ret).abs() > 1e-9]
    only_new = m[m._merge == "left_only"]
    R["與舊基準比較"] = {"兩邊都有": int(len(both)), "R20 不同": int(len(diff)), "R20 最大差": float((both.R20 - both.ret).abs().max()),
                    "只在本研究": int(len(only_new)), "只在本研究_狀態": only_new.status.value_counts().to_dict(),
                    "只在舊基準": int((m._merge == "right_only").sum()),
                    "R20 不同_例": [f"{r.code} {r.date:%Y-%m-%d} 新 {r.R20:+.4f} 舊 {r.ret:+.4f}" for r in diff.head(10).itertuples()]}
    # 特徵與標籤口徑（Qlib 讀回）
    fk = out / "feature_label_check.parquet"
    if fk.exists():
        K = pd.read_parquet(fk).merge(L[["date", "code", "R20", "status", "close_raw"]], on=["date", "code"])
        ok = K[K.R20.notna() & K.R20_qlib.notna()]
        R["特徵與標籤口徑"] = {"比對筆數": int(len(ok)), "R20 最大絕對差（float32）": float((ok.R20 - ok.R20_qlib).abs().max()),
                         "還原後 VWAP 在還原高低價區間": float(((K.vwap >= K.low * 0.999) & (K.vwap <= K.high * 1.001))[K.vwap.notna()].mean()),
                         "$close/$factor 對原始收盤 最大相對差": float(((K.close_raw_qlib / K.close_raw) - 1).abs().max())}
    # 抽查：獨立路徑重算（直接讀原始日 K 與事件表）
    ev0 = market.load_events(YEARS)
    ev1, _ = par_change_events(out)
    ev = pd.concat([ev0, ev1]).groupby(level=[0, 1]).prod().rename("factor").rename_axis(["code", "ex_date"])
    spots = []
    for code, day, why in SPOT:
        i = {"last_mature": len(cal) - 1 - H, "first_immature": len(cal) - H}.get(day)
        i = cal.searchsorted(pd.Timestamp(day)) if i is None else i   # 非交易日 → 往後取最近交易日
        t = cal[i]
        day = f"{t:%Y-%m-%d}"
        row = L[(L.code == code) & (L.date == t)]
        if not len(row):
            spots.append({"代號": code, "訊號日": day, "用途": why, "結果": "不在當日股票池"})
            continue
        r = row.iloc[0]
        if i + H > len(cal) - 1:
            spots.append({"代號": code, "訊號日": day, "用途": why, "狀態": r.status, "類別": r.cls, "結果": "未成熟，不評分"})
            continue
        d1, d20 = cal[i + 1], cal[i + H]
        o1, c20 = P["open"].at[d1, code], P["close"].at[d20, code]
        e = ev.loc[code] if code in ev.index.get_level_values(0) else pd.Series(dtype=float)
        fw = e[(e.index > d1) & (e.index <= d20)]
        r_ind = c20 / (o1 * fw.prod()) - 1
        o_old = old[(old.code == code) & (old.date == t)]
        spots.append({"代號": code, "訊號日": day, "用途": why, "t+1": f"{d1:%Y-%m-%d}", "t+20": f"{d20:%Y-%m-%d}", "開盤t+1": o1,
                      "收盤t+20": c20, "窗口事件": ";".join(f"{k:%m-%d}×{v:.4f}" for k, v in fw.items()), "獨立重算R20": round(r_ind, 6),
                      "管線R20": round(r.R20, 6), "一致": bool(abs(r_ind - r.R20) < 1e-9), "類別": r.cls, "狀態": r.status,
                      "label_end_date": f"{r.label_end_date:%Y-%m-%d}",
                      "舊基準R20": round(o_old.ret.iloc[0], 6) if len(o_old) else "（舊基準排除）"})
    R["抽查"] = spots
    (out / "checks.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    st.to_csv(out / "status_by_year.csv")
    md = ["# tw20 第 1 段資料檢查", ""]
    for k, v in R.items():
        if k == "抽查":
            md += ["## 抽查", "", pd.DataFrame(v).fillna("").astype(str).to_markdown(index=False, disable_numparse=True), ""]
        else:
            md += [f"## {k}", "", "```json", json.dumps(v, ensure_ascii=False, indent=1, default=str), "```", ""]
    md += ["## 各年樣本狀態", "", st.to_markdown(), ""]
    (out / "checks.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def hashes(out):
    out = Path(out)
    files = sorted([*market.OUT.glob("daily_[0-9]*.csv.gz"), *market.OUT.glob("manifest_[0-9]*.csv"),
                    *(twse.RAW / "events").glob("*.json"), ROOT / "backtest" / "splits.csv",
                    *(out / "raw").glob("TWTB7U_*.json"), *[out / x for x in ("labels.parquet", "features.parquet", "jumps.csv")]])
    h = {str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
         for f in files if f.exists()}
    (out / "hashes.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
    print(f"{len(h)} 個檔案")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["events", "labels", "features", "checks", "hashes"])
    a.add_argument("out")
    a.add_argument("--codes", default="")
    x = a.parse_args()
    Path(x.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {"events": lambda: fetch_par_change(x.out), "labels": lambda: build_labels(x.out),
     "features": lambda: features(x.out, [c for c in x.codes.split(",") if c] or None),
     "checks": lambda: checks(x.out), "hashes": lambda: hashes(x.out)}[x.cmd]()
    print(f"{x.cmd} 完成，{time.time() - t0:.0f} 秒")
