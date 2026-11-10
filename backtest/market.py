"""全市場日 K（TWSE 上市普通股）與「條件研究」的基準表。

python backtest/market.py fetch 2019                 # 抓一年：交易日曆 + 每日全市場行情 → backtest/market/daily_2019.csv.gz
python backtest/market.py fetch 2026 --days 20261030  # 只抓指定日期（試抓、驗證欄位）
python backtest/market.py base                       # 合併各年 → 逐日流動性前 N 檔 → 第 20 日 ±10% 基準表

勝負定義（固定）：訊號日 t 收盤後出現訊號，t+1 開盤進場，t+20 收盤結算（期間除權息以 factor 還原）。
做多勝 = 報酬 ≥ +10%；長空勝 = 報酬 ≤ −10%。
"""
import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import twse
from twse import BASE, HERE, SLEEP, fetch, months, now_tw, num, pick, roc, tables

OUT = HERE / "market"
H, WIN = 20, 0.10
TOP_N, MIN_PRICE, MIN_HIST, LIQ_DAYS = 150, 10.0, 120, 60
PERIODS = {"開發": (2015, 2022), "驗證": (2023, 2024), "保留": (2025, 2026)}
COMMON = re.compile(r"^[1-8]\d{3}$")  # 上市普通股：4 碼、非 00 開頭（ETF）、非 91 開頭（TDR）


def get(url):
    """不落地的 TWSE 請求（全市場行情每天約數百 KB，不存原始 JSON，只存解析結果與雜湊）。"""
    err = None
    for i in range(5):
        time.sleep(SLEEP * (1 + 3 * i))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kronos-backtest)"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            return raw, json.loads(raw.decode("utf-8"))
        except Exception as e:
            err = e
    raise RuntimeError(f"{url} 抓取失敗：{err}")


def trading_days(year, today):
    days = []
    for m in months(pd.Timestamp(year, 1, 1), min(pd.Timestamp(year, 12, 31), today)):
        js = fetch("afterTrading/FMTQIK", {"date": f"{m:%Y%m}01"}, f"calendar/FMTQIK_{m:%Y%m}",
                   twse.done(m + pd.offsets.MonthEnd(0), today))
        for t in tables(js):
            days += [roc(x) for x in t[pick(t, "日期")]]
    return sorted(set(days))


def parse_day(js, d):
    """回傳（普通股日 K, 加權指數收盤）。查無資料（休市）回傳空表。"""
    stock, idx = None, np.nan
    for t in tables(js):
        cols = "".join(t.columns)
        if "證券代號" in cols and "收盤價" in cols:
            stock = t
        elif "指數" in cols and "收盤指數" in cols:
            row = t[t[t.columns[0]].str.contains("發行量加權股價指數")]
            if len(row):
                idx = float(num(row[pick(row, "收盤指數")]).iloc[0])
    if stock is None:
        return pd.DataFrame(), idx
    code = stock[pick(stock, "證券代號")].str.strip()
    df = pd.DataFrame({"date": d, "code": code, "name": stock[pick(stock, "證券名稱")].str.strip(),
                       "open": num(stock[pick(stock, "開盤價")]), "high": num(stock[pick(stock, "最高價")]),
                       "low": num(stock[pick(stock, "最低價")]), "close": num(stock[pick(stock, "收盤價")]),
                       "volume": num(stock[pick(stock, "成交股數")]), "amount": num(stock[pick(stock, "成交金額")])})
    return df[df.code.str.match(COMMON)], idx


def fetch_year(year, only=None):
    OUT.mkdir(exist_ok=True)
    now = now_tw()
    today = pd.Timestamp(now.date())
    days = trading_days(year, today)
    if days and days[-1] == today and now.hour < 15:
        days = days[:-1]  # 盤中：今天未收盤
    if only:
        days = [pd.Timestamp(x) for x in only]
    f, mf = OUT / f"daily_{year}.csv.gz", OUT / f"manifest_{year}.csv"
    old = pd.read_csv(f, dtype={"code": str}, parse_dates=["date"]) if f.exists() and not only else pd.DataFrame()
    man = pd.read_csv(mf, parse_dates=["date"]) if mf.exists() and not only else pd.DataFrame()
    done = set(man[man.n_stocks > 0].date) if len(man) else set()
    parts, mrows = [old], []
    for d in days:
        if d in done:
            continue
        url = BASE + "afterTrading/MI_INDEX?" + urllib.parse.urlencode({"date": f"{d:%Y%m%d}", "type": "ALLBUT0999", "response": "json"})
        raw, js = get(url)
        df, idx = parse_day(js, d)
        parts.append(df)
        mrows.append({"date": d, "url": url, "fetched_at": now_tw().isoformat(timespec="seconds"), "stat": js.get("stat"),
                      "n_stocks": len(df), "taiex_close": idx, "sha256": hashlib.sha256(raw).hexdigest()})
        print(f"{d:%Y-%m-%d} {len(df)} 檔 加權 {idx}", flush=True)
    daily = pd.concat(parts).sort_values(["date", "code"]) if any(len(p) for p in parts) else pd.DataFrame()
    man = pd.concat([man, pd.DataFrame(mrows)]).drop_duplicates("date", keep="last").sort_values("date")
    tag = "" if not only else "_try"
    daily.to_csv(OUT / f"daily_{year}{tag}.csv.gz", index=False, compression="gzip")
    man.to_csv(OUT / f"manifest_{year}{tag}.csv", index=False)
    # 除權息、減資（全市場；原始 JSON 存 raw/events/，與原版回測同格式）
    if not only:
        b = min(pd.Timestamp(year, 12, 31), today)
        for path, name in (("exRight/TWT49U", "TWT49U"), ("reducation/TWTAUU", "TWTAUU")):
            fetch(path, {"startDate": f"{year}0101", "endDate": f"{b:%Y%m%d}"}, f"events/{name}_{year}",
                  twse.done(pd.Timestamp(year, 12, 31), today))
    bad = man[man.n_stocks == 0]
    print(f"{year}：{len(days)} 個交易日，失敗/無資料 {len(bad)} 天：{[f'{x:%Y-%m-%d}' for x in bad.date]}")


# ---------- 基準 ----------
def load_events(years):
    rows = []
    for y in years:
        for name, cols in (("TWT49U", ("資料日期", "除權息前收盤價", "除權息參考價")),
                           ("TWTAUU", ("恢復買賣日期", "停止買賣前收盤", "恢復買賣參考價"))):
            f = twse.RAW / "events" / f"{name}_{y}.json"
            if not f.exists():
                continue
            for t in tables(json.loads(f.read_text(encoding="utf-8"))):
                rows.append(pd.DataFrame({"code": t[pick(t, "股票代號")].str.strip(),
                                          "ex_date": [roc(x) for x in t[pick(t, cols[0])]],
                                          "factor": (num(t[pick(t, cols[2])]) / num(t[pick(t, cols[1], "收盤")])).values}))
    ev = pd.concat(rows)
    ev = ev[np.isfinite(ev.factor) & (ev.factor > 0)]
    return ev.groupby(["code", "ex_date"]).factor.prod()  # 同日多事件相乘


def panel():
    files = sorted(OUT.glob("daily_[0-9][0-9][0-9][0-9].csv.gz"))
    d = pd.concat(pd.read_csv(f, dtype={"code": str}, parse_dates=["date"]) for f in files)
    years = sorted(d.date.dt.year.unique())
    cal = pd.DatetimeIndex(sorted(d.date.unique()))
    ev = load_events(years)
    P = {c: d.pivot(index="date", columns="code", values=c).reindex(cal) for c in ("open", "close", "amount")}
    cols = P["close"].columns
    f = np.ones((len(cal), len(cols)))  # 每日每檔的還原 factor（非事件日 = 1）
    e = ev.reset_index()
    e = e[e.code.isin(cols) & e.ex_date.isin(cal)]
    f[cal.get_indexer(e.ex_date), cols.get_indexer(e.code)] = e.factor.values
    F = pd.DataFrame(f, index=cal, columns=cols)
    return cal, P, F, ev


def frames():
    """逐日 × 逐檔的矩陣：價量、還原日報酬、波動、股票池、20 日結算報酬、可評分遮罩、大盤狀態。"""
    cal, P, F, ev = panel()
    o, c, amt = P["open"], P["close"], P["amount"]
    c = c.where(c > 0)
    ret1 = c / (c.ffill().shift() * F) - 1               # 還原後日報酬（對上一個有成交的收盤，停牌空檔也比得到）
    jump = ret1.abs() > 0.105                            # 超過漲跌幅限制、又沒有對應公司行動 → 視為面額變更／資料異常
    vol20 = ret1.rolling(20, min_periods=18).std() * np.sqrt(20)
    hist = c.notna().cumsum()
    liq = amt.where(c.notna()).rolling(LIQ_DAYS, min_periods=40).median()
    elig = (hist >= MIN_HIST) & (c >= MIN_PRICE) & liq.notna()
    rank = liq.where(elig).rank(axis=1, ascending=False)
    univ = rank <= TOP_N

    # 結算：t+1 開盤進場、t+20 收盤；(t+1, t+20] 的事件 factor 相乘
    fwin = np.exp(np.log(F).rolling(H - 1).sum()).shift(-H)  # 視窗 t+2..t+20 的 factor
    entry, exitc = o.shift(-1), c.shift(-H)
    r = exitc / (entry * fwin) - 1
    bad_win = jump.astype(float).rolling(H + 21, min_periods=1).max().shift(-H).fillna(1).astype(bool)  # t−20..t+20 內有異常跳空
    ok = univ & r.notna() & entry.gt(0) & ~bad_win
    taiex = pd.concat(pd.read_csv(f, parse_dates=["date"]) for f in sorted(OUT.glob("manifest_[0-9][0-9][0-9][0-9].csv")))
    tx = taiex.set_index("date").taiex_close.reindex(cal)
    ma = tx.rolling(60).mean()
    regime = pd.Series(np.where(tx > ma, "加權>60MA", "加權≤60MA"), index=cal).where(ma.notna())
    return dict(cal=cal, o=o, c=c, amt=amt, F=F, ret1=ret1, jump=jump, vol20=vol20, univ=univ, r=r, ok=ok, tx=tx, regime=regime)


def samples(fr):
    """可評分的「日期 × 檔」長表，附勝負、期間、波動組（開發期三分位固定）、大盤狀態。"""
    ok, r, vol20 = fr["ok"], fr["r"], fr["vol20"]
    keep = ok.stack()
    keep = keep[keep].index  # 明確挑出入選的「日期 × 檔」（不依賴 stack 是否丟 NaN）
    s = pd.DataFrame({"ret": r.stack().reindex(keep), "vol20": vol20.stack().reindex(keep)}).reset_index()
    s.columns = ["date", "code", "ret", "vol20"]
    s["long"], s["short"] = s.ret >= WIN, s.ret <= -WIN
    s["year"] = s.date.dt.year
    s["期間"] = pd.cut(s.year, [0, 2022, 2024, 9999], labels=list(PERIODS))
    q = s[s["期間"] == "開發"].vol20.quantile([1 / 3, 2 / 3]).values  # 波動分組門檻用開發期固定
    s["波動組"] = pd.cut(s.vol20, [-np.inf, q[0], q[1], np.inf], labels=["低", "中", "高"])
    s["大盤"] = s.date.map(fr["regime"])
    return s, q


def base():
    fr = frames()
    cal = fr["cal"]
    s, q = samples(fr)

    def tab(by):
        g = s.groupby(by, observed=True)
        return pd.DataFrame({"樣本數": g.size(), "日期數": g.date.nunique(), "檔數": g.code.nunique(),
                             "做多勝%": 100 * g.long.mean(), "長空勝%": 100 * g.short.mean(),
                             "平均20日報酬%": 100 * g.ret.mean()}).reset_index()

    out = HERE / "market" / "base"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    T = {"期間": tab("期間"), "年": tab("year"), "期間×波動組": tab(["期間", "波動組"]), "期間×大盤": tab(["期間", "大盤"])}
    T["檔_開發期"] = s[s["期間"] == "開發"].groupby("code").agg(樣本數=("ret", "size"), 做多勝=("long", "mean"),
                                                            長空勝=("short", "mean")).reset_index()
    for k, t in T.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    s.to_csv(out / "samples.csv.gz", index=False, compression="gzip")
    md = lambda t: t.apply(lambda c: c.map(lambda v: f"{v:.1f}") if c.dtype.kind == "f" else c.astype(str)).to_markdown(
        index=False, disable_numparse=True)
    tk = T["檔_開發期"]
    lines = ["# 條件研究基準（不加任何條件）", "",
             f"上市普通股，每日流動性前 {TOP_N} 檔（{LIQ_DAYS} 日成交金額中位數，逐日決定、不看未來），收盤 ≥ {MIN_PRICE:.0f} 元、"
             f"上市滿 {MIN_HIST} 個交易日。訊號日 t 收盤後進場判斷：t+1 開盤買，t+{H} 收盤結算，除權息與減資以 factor 還原。",
             f"做多勝 = 報酬 ≥ +{WIN:.0%}；長空勝 = 報酬 ≤ −{WIN:.0%}。每個交易日都算一筆（視窗重疊，信賴區間要以日期為群集）。",
             f"資料：{cal[0]:%Y-%m-%d}～{cal[-1]:%Y-%m-%d}，{len(cal)} 個交易日；異常跳空（|還原日報酬|>10.5%）前後 20 日的樣本排除。",
             f"波動分組門檻（開發期三分位，20 日報酬標準差）：{q[0]:.1%}、{q[1]:.1%}。", "",
             "## 各期間", "", md(T["期間"]), "", "## 各年", "", md(T["年"]), "",
             "## 期間 × 波動組", "", md(T["期間×波動組"]), "", "## 期間 × 大盤（加權指數 vs 60 日均線）", "", md(T["期間×大盤"]), "",
             f"## 開發期各檔分布（{len(tk)} 檔）", "",
             md((100 * tk[["做多勝", "長空勝"]]).describe(percentiles=[.1, .25, .5, .75, .9]).T.reset_index().rename(columns={"index": "勝率%"})), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["fetch", "base"])
    a.add_argument("year", nargs="?", type=int)
    a.add_argument("--days", nargs="*")
    x = a.parse_args()
    fetch_year(x.year, x.days) if x.cmd == "fetch" else base()
