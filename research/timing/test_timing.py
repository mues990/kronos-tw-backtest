"""離線檢查：擺盪低點確認延遲、出場模擬。python research/timing/test_timing.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import timing as tm

n = 400
idx = pd.bdate_range("2020-01-01", periods=n)
c = 100 + np.arange(n) * 0.2
c[200:206] = [130, 125, 118, 124, 130, 136]                     # 第 202 天是低點
d = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "amount": 1e8}, index=idx)
I = tm.indicators(d, np.full(n, 100.0))
assert I["sl1i"][206] != 202 and I["sl1i"][207] == 202 and I["sl1"][207] == c[202] - 1, "擺盪低點要等 5 天後才確認"
r = tm.simulate(I, 300)
assert r and r["reason"] == "到期" and r["days"] == 250 - 0 and abs(r["ret"] - (c[300 + 251] / c[301] - 1 - tm.COST)) < 1e-9 if 300 + 251 < n else True
c2 = c.copy(); c2[320:] = 60                                    # 第 320 天起崩跌 → 收盤確認、隔天開盤出場
d2 = pd.DataFrame({"open": c2, "high": c2 + 1, "low": c2 - 1, "close": c2, "amount": 1e8}, index=idx)
I2 = tm.indicators(d2, np.full(n, 100.0))
r = tm.simulate(I2, 300)
assert r and r["x"] == 321 and r["reason"] in ("初始停損", "移動停利") and abs(r["ret"] - (60 / c2[301] - 1 - tm.COST)) < 1e-9
assert abs(r["R"] - r["ret"] / r["risk"]) < 1e-12 and 0.05 - 1e-9 <= r["risk"] <= 0.30 + 1e-9
assert tm.simulate(I, n - 10) is None                           # 資料內未完成 → 不計
print("OK")

# R1 追高（protocol_v4_forward.json）
import timing2 as t2
b = pd.Series(100.0, index=pd.bdate_range("2023-01-02", periods=300))
cl = pd.Series(np.r_[np.full(49, 50.0), np.linspace(50, 100, 251)], index=b.index)
assert abs(t2.rs_last(cl, b) - (cl.iloc[-21] / cl.iloc[-251] - 1)) < 1e-12        # 大盤持平 → RS = 個股 t−250→t−20 報酬
assert np.isnan(t2.rs_last(cl.iloc[-100:], b))                                     # 歷史不足 → 不判斷
pool = np.arange(150) / 100.0                                                        # 0.00…1.49；第 80 百分位 = 1.192
assert t2.r1(1.20, pool) == (120 / 150, True) and t2.r1(1.19, pool)[1] is False
assert t2.r1(1.0, np.r_[pool[:100], np.full(50, np.nan)])[1] is None                # 有效檔數 < 120 → 不判斷
print("R1 OK")
