"""離線檢查 market.py：解析、除息還原、勝負判定、流動性篩選、異常跳空排除。python backtest/test_market.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import market  # noqa: E402
import twse  # noqa: E402

# 解析：只留普通股，抓到加權指數
js = {"stat": "OK", "tables": [
    {"fields": ["指數", "收盤指數", "漲跌(+/-)"], "data": [["發行量加權股價指數", "23,456.78", "+"]]},
    {"fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價"],
     "data": [["2330", "台積電", "1,000", "10", "1,000,000", "1,000.00", "1,010.00", "990.00", "1,005.00"],
              ["0050", "元大台灣50", "1", "1", "1", "1", "1", "1", "1"], ["9105", "TDR", "1", "1", "1", "1", "1", "1", "1"],
              ["1101", "台泥", "5", "1", "50", "--", "--", "--", "--"]]}]}
df, idx = market.parse_day(js, pd.Timestamp("2026-10-30"))
assert idx == 23456.78 and list(df.code) == ["2330", "1101"] and df.close.iloc[0] == 1005 and np.isnan(df.close.iloc[1])

# 合成資料：A 每天 +0.6%（20 日約 +12.7%，做多勝）；B 平盤但第 150 天除息 10%（還原後報酬 0）；C 第 200 天異常跳空 +30%；
# D 成交金額最小（TOP_N=3 時不入選）
tmp = Path(tempfile.mkdtemp())
market.OUT, market.HERE, twse.RAW = tmp / "market", tmp, tmp / "raw"
market.TOP_N, market.MIN_HIST = 7, 30
cal = pd.bdate_range("2015-01-05", periods=300)
px = {"1111": 20 * 1.006 ** np.arange(300), "2222": np.where(np.arange(300) >= 150, 18.0, 20.0),
      "3333": np.where(np.arange(300) >= 200, 26.0, 20.0), "4444": np.full(300, 20.0),
      "5555": np.where(np.arange(300) >= 100, 20.0, 80.0)}  # E：第 95–99 天停牌、恢復後 1 拆 4（無事件資料）
rng = np.random.default_rng(0)
px |= {k: 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))) for k in ("6666", "7777", "8888")}  # 隨機漫步，讓波動分組有門檻
amt = {"1111": 1e9, "2222": 1e9, "3333": 1e9, "4444": 1e3, "5555": 1e10, "6666": 1e9, "7777": 1e9, "8888": 1e9}
rows = [{"date": d, "code": k, "name": k, "open": v[i], "high": v[i], "low": v[i], "close": v[i], "volume": 1, "amount": amt[k]}
        for k, v in px.items() for i, d in enumerate(cal) if not (k == "5555" and 95 <= i < 100)]
market.OUT.mkdir(parents=True)
pd.DataFrame(rows).to_csv(market.OUT / "daily_2015.csv.gz", index=False, compression="gzip")
pd.DataFrame({"date": cal, "taiex_close": np.linspace(9000, 9500, 300), "n_stocks": 4}).to_csv(market.OUT / "manifest_2015.csv", index=False)
(twse.RAW / "events").mkdir(parents=True)
ex = cal[150]
(twse.RAW / "events" / "TWT49U_2015.json").write_text(json.dumps({"stat": "OK", "fields": ["資料日期", "股票代號", "除權息前收盤價", "除權息參考價"],
                                                                  "data": [[f"{ex.year - 1911}年{ex:%m月%d日}", "2222", "20.00", "18.00"]]}))
market.base()
s = pd.read_csv(market.OUT / "base" / "samples.csv.gz", dtype={"code": str}, parse_dates=["date"])
assert set(s[s.code == "4444"].date) <= set(cal[95:100]), "流動性最低者只在 5555 停牌時遞補入選"
a, b, c = (s[s.code == k] for k in ("1111", "2222", "3333"))
assert a.long.all() and abs(a.ret.iloc[0] - (1.006 ** 19 - 1)) < 1e-9
win = b[(b.date >= cal[150 - 20]) & (b.date < cal[150])]
assert len(win) and (win.ret.abs() < 1e-9).all(), "除息應被還原"
assert not ((c.date >= cal[180]) & (c.date <= cal[200])).any(), "異常跳空前後 20 日應排除"
e = s[s.code == "5555"]
assert len(e) and not ((e.date >= cal[80]) & (e.date <= cal[100])).any() and (e.ret.abs() < 1e-9).all(), "停牌後的拆分應被偵測並排除"
shutil.rmtree(tmp)
print("OK")
