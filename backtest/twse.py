"""TWSE 官方資料：抓取（原始 JSON 存 backtest/raw/）、解析、還原，以及共用資料準備。

python backtest/twse.py prep        # 交易日曆、除權息、減資、處置、注意股 → backtest/work/，並產生預測起點格點
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW, WORK = HERE / "raw", HERE / "work"
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
BASE = "https://www.twse.com.tw/rwd/zh/"
TZ = dt.timezone(dt.timedelta(hours=8))
SLEEP = 3.0  # TWSE 對短時間大量請求會封 IP，官方建議 5 秒內不超過 3 次
PRICES = ["open", "high", "low", "close"]
DATA_START = pd.Timestamp(CFG["data_start"])


def mode():
    return os.environ.get("BT_MODE") or CFG["mode"]


def tickers():
    return CFG["pilot_tickers"] if mode() == "pilot" else CFG["etf"] + CFG["stocks"]


def now_tw():
    return dt.datetime.now(TZ)


# ---------- 抓取 ----------
def fetch(path, params, name, final):
    """有 _final 的快取直接用；否則重抓並覆寫（當月等未完結期間）。"""
    f = RAW / f"{name}.json"
    if f.exists():
        js = json.loads(f.read_text(encoding="utf-8"))
        if js.get("_final"):
            return js
    url = BASE + path + "?" + urllib.parse.urlencode({**params, "response": "json"})
    err = None
    for i in range(5):
        time.sleep(SLEEP * (1 + 3 * i))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kronos-backtest)"})
            js = json.loads(urllib.request.urlopen(req, timeout=40).read().decode("utf-8"))
            break
        except Exception as e:  # 非 JSON（被擋時會回 HTML）也算失敗
            err = e
    else:
        raise RuntimeError(f"{url} 抓取失敗：{err}")
    js.update(_url=url, _fetched_at=now_tw().isoformat(timespec="seconds"), _final=final)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    return js


def tables(js):
    """TWSE 回應有 {fields,data} 與 {tables:[{fields,data}]} 兩種格式；查無資料時 stat 不是 OK。"""
    if js.get("stat", "OK") != "OK":
        return []
    out = []
    for t in js.get("tables") or [js]:
        if t.get("fields") and t.get("data"):
            df = pd.DataFrame(t["data"]).astype(str)
            df.columns = (list(t["fields"]) + [f"_c{i}" for i in range(df.shape[1])])[: df.shape[1]]
            out.append(df)
    return out


def pick(df, *keys):
    for k in keys:
        for c in df.columns:
            if k in c:
                return c
    raise KeyError(f"找不到欄位 {keys}，實際欄位：{list(df.columns)}")


def roc(s):
    """'113/07/01'、'113年07月01日'、'1130701'、'2024-07-01' → Timestamp。"""
    n = re.findall(r"\d+", str(s))
    if len(n) == 1 and len(n[0]) in (7, 8):
        v = n[0]
        n = [v[:-4], v[-4:-2], v[-2:]]
    y, m, d = (int(x) for x in n[:3])
    return pd.Timestamp(y + 1911 if y < 1911 else y, m, d)


def roc_pair(s):
    """'113/07/02～113/07/15' → (start, end)。"""
    n = re.findall(r"\d+", str(s))
    return roc("/".join(n[0:3])), roc("/".join(n[3:6]))


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def months(a, b):
    d = pd.Timestamp(a).replace(day=1)
    while d <= b:
        yield d
        d = d + pd.offsets.MonthBegin(1)


def done(period_end, today):
    return bool(period_end < today)


# ---------- 各資料集 ----------
def calendar(today):
    """交易所交易日 = FMTQIK（每日市場成交資訊）有列出的日期，颱風假等休市日自然不在內。"""
    days = []
    for m in months(DATA_START, today):
        js = fetch("afterTrading/FMTQIK", {"date": f"{m:%Y%m}01"}, f"calendar/FMTQIK_{m:%Y%m}",
                   done(m + pd.offsets.MonthEnd(0), today))
        for t in tables(js):
            days += [roc(x) for x in t[pick(t, "日期")]]
    return pd.DatetimeIndex(sorted(set(days)))


def bars(ticker, today):
    """個股/ETF 日成交資訊（未還原）。股數、元。close 為 NaN 的列 = 當日無成交或停牌。"""
    parts = []
    for m in months(DATA_START, today):
        js = fetch("afterTrading/STOCK_DAY", {"date": f"{m:%Y%m}01", "stockNo": ticker},
                   f"stock/{ticker}/STOCK_DAY_{m:%Y%m}", done(m + pd.offsets.MonthEnd(0), today))
        for t in tables(js):
            parts.append(pd.DataFrame({
                "date": [roc(x) for x in t[pick(t, "日期")]],
                "open": num(t[pick(t, "開盤")]), "high": num(t[pick(t, "最高")]),
                "low": num(t[pick(t, "最低")]), "close": num(t[pick(t, "收盤")]),
                "volume": num(t[pick(t, "成交股數")]), "amount": num(t[pick(t, "成交金額")]),
                "trades": num(t[pick(t, "成交筆數")]),
            }))
    if not parts:
        raise RuntimeError(f"{ticker} 沒有任何日成交資料")
    b = pd.concat(parts).drop_duplicates("date").set_index("date").sort_index()
    b.loc[b.close <= 0, "close"] = np.nan
    return b


def yearly(path, name, today, parse):
    out = []
    for y in range(DATA_START.year, today.year + 1):
        a, b = pd.Timestamp(y, 1, 1), min(pd.Timestamp(y, 12, 31), today)
        js = fetch(path, {"startDate": f"{a:%Y%m%d}", "endDate": f"{b:%Y%m%d}"}, f"events/{name}_{y}",
                   done(pd.Timestamp(y, 12, 31), today))
        out += [parse(t) for t in tables(js)]
    return pd.concat(out) if out else pd.DataFrame()


def events(today, keep):
    """還原事件：除權息(TWT49U)、減資恢復買賣(TWTAUU)、分割(splits.csv，手動附來源)。
    factor = 參考價 / 事件前收盤；事件日(ex_date)之前的價格 × factor。"""
    div = yearly("exRight/TWT49U", "TWT49U", today, lambda t: pd.DataFrame({
        "ticker": t[pick(t, "股票代號")].str.strip(), "ex_date": [roc(x) for x in t[pick(t, "資料日期")]],
        "prev_close": num(t[pick(t, "除權息前收盤價", "收盤價")]), "ref": num(t[pick(t, "除權息參考價", "參考價")]),
        "kind": "除" + t[pick(t, "權/息")].str.strip(), "vol_adjust": False, "source": "TWSE TWT49U"}))
    red = yearly("reducation/TWTAUU", "TWTAUU", today, lambda t: pd.DataFrame({
        "ticker": t[pick(t, "股票代號")].str.strip(), "ex_date": [roc(x) for x in t[pick(t, "恢復買賣日期")]],
        "prev_close": num(t[pick(t, "停止買賣前收盤")]), "ref": num(t[pick(t, "恢復買賣參考價")]),
        "kind": "減資", "vol_adjust": True, "source": "TWSE TWTAUU"}))
    ev = pd.concat([div, red])
    ev["factor"] = ev.ref / ev.prev_close
    sp = pd.read_csv(HERE / "splits.csv", dtype={"ticker": str}, parse_dates=["ex_date"])
    sp = sp.assign(kind="分割", vol_adjust=True, prev_close=np.nan, ref=np.nan)
    ev = pd.concat([ev, sp])
    return ev[ev.ticker.isin(keep)].sort_values(["ticker", "ex_date"]).reset_index(drop=True)


def monthly_list(path, name, today, parse):
    """處置／注意股公告按月抓；抓不到的月份記下來，對應樣本標記為「未知」。"""
    out, failed = [], []
    for m in months(DATA_START, today):
        e = min(m + pd.offsets.MonthEnd(0), today)
        try:
            js = fetch(path, {"startDate": f"{m:%Y%m%d}", "endDate": f"{e:%Y%m%d}"}, f"{name}/{name}_{m:%Y%m}",
                       done(m + pd.offsets.MonthEnd(0), today))
        except RuntimeError as err:
            print(err, file=sys.stderr)
            failed.append(f"{m:%Y-%m}")
            continue
        out += [parse(t) for t in tables(js)]
    return (pd.concat(out) if out else pd.DataFrame()), failed


def punish(today, keep):
    def parse(t):
        per = [roc_pair(x) for x in t[pick(t, "處置起迄")]]
        return pd.DataFrame({
            "ticker": t[pick(t, "證券代號")].str.strip(), "announce": [roc(x) for x in t[pick(t, "公布日期")]],
            "start": [p[0] for p in per], "end": [p[1] for p in per],
            "cumulative": t[pick(t, "累計")], "measure": t[pick(t, "處置措施")]})
    df, failed = monthly_list("announcement/punish", "punish", today, parse)
    if len(df):
        df = df[df.ticker.isin(keep)].drop_duplicates(["ticker", "announce", "start", "end"])
    return df, failed


def notice(today, keep):
    def parse(t):
        return pd.DataFrame({"ticker": t[pick(t, "證券代號")].str.strip(),
                             "date": [roc(x) for x in t[pick(t, "日期")]]})
    df, failed = monthly_list("announcement/notice", "notice", today, parse)
    if len(df):
        df = df[df.ticker.isin(keep)].drop_duplicates()
    return df, failed


# ---------- 還原 ----------
def adjusted(b, ev, asof):
    """截至 asof 收盤的還原 K 線：只套用 ex_date <= asof 的事件（預測當時已生效）。最後一根 = 原始價。"""
    b = b[b.index <= asof].copy()
    for e in ev[ev.ex_date <= asof].itertuples():
        m = b.index < e.ex_date
        b.loc[m, PRICES] *= e.factor
        if e.vol_adjust:
            b.loc[m, "volume"] /= e.factor
    return b


# ---------- 共用資料準備 ----------
def grid(cal, cutoff):
    """每隔 step 個交易所交易日一個起點；目標 = 第 pred_len 個交易所交易日。跨期的起點標 gap 不跑。"""
    cal = cal[cal <= cutoff]
    dev0, dev1 = (pd.Timestamp(x) for x in CFG["dev"])
    hold0 = pd.Timestamp(CFG["holdout_start"])
    i0, h = int(np.searchsorted(cal, dev0)), CFG["pred_len"]
    rows = []
    for i in range(i0, len(cal) - h, CFG["step"]):
        o, tg = cal[i], cal[i + h]
        rows.append((o, tg, "dev" if tg <= dev1 else "holdout" if o >= hold0 else "gap"))
    return pd.DataFrame(rows, columns=["origin", "target", "period"])


def prep():
    WORK.mkdir(exist_ok=True)
    now = now_tw()
    today = pd.Timestamp(now.date())
    cal = calendar(today)
    cutoff = cal[-1]
    if cutoff == today and now.hour < 15:  # 盤中執行：今天的 K 棒未完成
        cutoff = cal[-2]
    keep = CFG["etf"] + CFG["stocks"]
    pd.Series(cal, name="date").to_csv(WORK / "calendar.csv", index=False)
    ev = events(today, keep)
    ev.to_csv(WORK / "events.csv", index=False)
    pun, pun_fail = punish(today, keep)
    pun.to_csv(WORK / "punish.csv", index=False)
    noti, noti_fail = notice(today, keep)
    noti.to_csv(WORK / "notice.csv", index=False)

    g = grid(cal, cutoff)
    pick_dates = [g[g.period != "gap"].origin.iloc[-1] if d == "last" else pd.Timestamp(d) for d in CFG["pilot_origins"]]
    idx = {int(np.searchsorted(g.origin, d)) for d in pick_dates}
    for t in CFG["pilot_tickers"]:  # 再各加一個「預測期間內有還原事件」的起點，驗證事後評分口徑
        for e in ev[ev.ticker == t].ex_date:
            hit = g.index[(g.origin < e) & (e <= g.target) & (g.period != "gap")]
            if len(hit):
                idx.add(int(hit[0]))
                break
    g["pilot"] = g.index.isin(idx)
    g.to_csv(WORK / "grid.csv", index=False)

    meta = {"run_at_tw": now.isoformat(timespec="seconds"), "data_cutoff": f"{cutoff:%Y-%m-%d}",
            "calendar_days": len(cal), "punish_failed_months": pun_fail, "notice_failed_months": noti_fail,
            "grid": g.period.value_counts().to_dict(), "pilot_origins": [f"{d:%Y-%m-%d}" for d in g[g.pilot].origin],
            "events": len(ev), "punish_rows": len(pun), "notice_rows": len(noti)}
    (WORK / "prep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(ev.to_string())
    print(pun.to_string())


if __name__ == "__main__":
    {"prep": prep}[sys.argv[1]]()
