"""離線檢查 stage2 的訊號規則、區塊抽樣、評估與通過判定。python research/tw20/test_stage2.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import stage2 as s  # noqa: E402

# 訊號規則：達門檻且大於另一方向；相等棄權
assert list(s.signals(np.array([.5, .5, .3, .6]), np.array([.4, .5, .1, .7]), .5)) == [True, False, False, False]

# 區塊抽樣：形狀、每列由連續 40 日區塊組成、可重現
idx = s.block_index(481, 40, 2000, 42)
assert idx.shape == (2000, 481) and idx.min() >= 0 and idx.max() <= 480
assert (np.diff(idx[:, :40], axis=1) == 1).all() and (idx == s.block_index(481, 40, 2000, 42)).all()

# 評估：200 個交易日 × 30 檔，訊號樣本達標率 60%、其餘 10%；基準為同日同波動組平均
rng = np.random.default_rng(0)
dates = pd.bdate_range("2023-01-02", periods=200)
V = pd.DataFrame({"date": np.repeat(dates, 30), "volgrp": "中"})
sig = np.tile(np.arange(30) < 3, 200)
V["cls"] = np.where(rng.random(len(V)) < np.where(sig, .6, .1), "UP", "OTHER")
V["base_UP"] = (V.cls == "UP").groupby([V.date, V.volgrp]).transform("mean")
IDX = {b: s.block_index(200, b, 2000, 42) for b in (40, 20, 60)}
r = s.evaluate(V, sig, "UP", pd.DatetimeIndex(dates), IDX)
assert r["訊號數"] == 600 and r["不同日期數"] == 200 and abs(r["覆蓋率%"] - 10) < 1e-9
assert abs(r["達標率%"] - 100 * (V.cls[sig] == "UP").mean()) < 1e-9 and 35 < r["提升pp"] < 55
assert r["提升lo40"] < r["提升pp"] < r["提升hi40"] and r["2023訊號數"] == 600 and r["2024訊號數"] == 0
ok, why = s.passed(r)
assert not ok and why == "分年皆正"          # 2024 沒有訊號 → 分年條件不成立
r.update({"2024提升pp": 1.0})
assert s.passed(r) == (True, "")
r2 = dict(r, 訊號數=150)
assert s.passed(r2) == (False, "訊號數")

# 選擇：合格者中達標率最高，同分取覆蓋率較高
c = [dict(通過=True, **{"達標率%": 40, "覆蓋率%": 1}), dict(通過=True, **{"達標率%": 40, "覆蓋率%": 5}), dict(通過=False, **{"達標率%": 90, "覆蓋率%": 1})]
assert s.pick(c) is c[1] and s.pick([c[2]]) is None
print("OK")
