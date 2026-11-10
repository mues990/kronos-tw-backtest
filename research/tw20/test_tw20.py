"""離線檢查 tw20.label_frame：R20 還原、UP/DOWN/OTHER、各種樣本狀態、t+1 一字漲跌停旗標。python research/tw20/test_tw20.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tw20  # noqa: E402

n = 200
cal = pd.bdate_range("2016-01-04", periods=n)
codes = ["1111", "2222", "3333", "4444", "5555"]
base = {k: np.full(n, 50.0) for k in codes}
base["1111"] = 50 * 1.006 ** np.arange(n)                  # 穩定上漲：20 日約 +12% → UP
base["2222"] = np.where(np.arange(n) >= 150, 45.0, 50.0)   # 第 150 日除息 factor 0.9 → 還原後 R20 = 0
base["3333"] = np.where(np.arange(n) >= 160, 70.0, 50.0)   # 第 160 日無事件跳空 +40% → 無法還原
P = {k: pd.DataFrame({c: v.copy() for c, v in base.items()}, index=cal) for k in ("open", "high", "low", "close")}
P["volume"] = pd.DataFrame(1e6, index=cal, columns=codes)
P["amount"] = P["close"] * 1e6
for k in ("open", "high", "low", "close"):
    P[k].loc[cal[130], "4444"] = np.nan                     # 4444：第 130 日停牌
P["high"].loc[cal[100], "5555"] = P["low"].loc[cal[100], "5555"] = P["open"].loc[cal[100], "5555"] = 55.0  # 5555：第 100 日一字漲停
P["close"].loc[cal[100], "5555"] = 55.0
F = pd.DataFrame(1.0, index=cal, columns=codes)
F.loc[cal[150], "2222"] = 0.9
tw20.MIN_HIST, tw20.LIQ_MINP = 30, 20
L, jumps, _ = tw20.label_frame(cal, P, F)
g = lambda c, i: L[(L.code == c) & (L.date == cal[i])].iloc[0]

assert g("1111", 50).cls == "UP" and abs(g("1111", 50).R20 - (1.006 ** 19 - 1)) < 1e-12
assert all(abs(g("2222", i).R20) < 1e-12 for i in range(129, 150)), "除息應還原"
assert g("2222", 140).cls == "OTHER" and g("2222", 140).status == "ok"
assert {g("3333", i).status for i in range(140, 160)} == {"label_event_unreliable"}, "t+1..t+20 內無法還原 → 標籤不可靠"
assert {g("3333", i).status for i in range(160, 179)} == {"feature_event_unreliable"}, "t−60..t 內 → 特徵不可靠"
assert g("4444", 129).status == "no_entry_price" and g("4444", 110).status == "no_exit_price"
assert g("5555", 99).entry_locked_up and not g("5555", 98).entry_locked_up
assert set(L[L.date > cal[n - 1 - 20]].status) == {"immature"} and L[L.date > cal[n - 1 - 20]].cls.isna().all()
assert g("1111", n - 21).status == "ok" and g("1111", n - 21).label_end_date == cal[-1]
assert len(jumps) and set(jumps.code) == {"3333"}
print("OK")
