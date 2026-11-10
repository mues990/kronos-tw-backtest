"""離線自我檢查（不需網路與模型）：python backtest/test_backtest.py"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
os.environ["BT_MODE"] = "full"
import score  # noqa: E402
import twse  # noqa: E402

# 日期格式
assert twse.roc("113/07/01") == twse.roc("113年07月01日") == twse.roc("1130701") == pd.Timestamp("2024-07-01")
assert twse.roc_pair("114/06/02～114/06/13") == (pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-13"))

# 分割還原：事件日前價格 ×0.25、量 ×4；事件生效前的起點不受影響
cal = pd.bdate_range("2023-03-01", "2026-02-27")
b = pd.DataFrame({c: 100.0 for c in twse.PRICES} | {"volume": 1000.0, "amount": 1e8}, index=cal)
split = pd.Timestamp("2025-06-18")
b.loc[b.index >= split, twse.PRICES] = 25.0
ev = pd.DataFrame({"ticker": ["0050"], "ex_date": [split], "factor": [0.25], "vol_adjust": [True], "kind": ["分割"]})
a = twse.adjusted(b, ev, split)
assert (a.close == 25).all() and a.volume.iloc[0] == 4000 and a.volume.iloc[-1] == 1000
assert (twse.adjusted(b, ev, split - pd.offsets.BDay(1)).close == 100).all()

# 格點：開發期目標不跨入保留期，保留期起點在 2026 之後，間隔 5 個交易日
g = twse.grid(cal, cal[-1])
assert (g[g.period == "dev"].target <= "2025-12-31").all() and (g[g.period == "holdout"].origin >= "2026-01-01").all()
assert set(np.diff(np.searchsorted(cal, g.origin))) == {5}

# 端到端：跨分割的實際報酬應為 0（平），選窗同分取較短
tmp = Path(tempfile.mkdtemp())
for m in (twse, score):
    m.WORK, m.HERE = tmp, tmp
twse.CFG.update(etf=["0050"], stocks=["2330"])
pd.Series(cal, name="date").to_csv(tmp / "calendar.csv", index=False)
g.assign(pilot=False).to_csv(tmp / "grid.csv", index=False)
ev.to_csv(tmp / "events.csv", index=False)
pd.DataFrame({"ticker": ["2330"], "announce": [pd.Timestamp("2025-03-03")], "start": [pd.Timestamp("2025-03-04")],
              "end": [pd.Timestamp("2025-03-17")]}).to_csv(tmp / "punish.csv", index=False)
pd.DataFrame(columns=["ticker", "date"]).to_csv(tmp / "notice.csv", index=False)
(tmp / "prep_meta.json").write_text(json.dumps({"data_cutoff": "2026-02-27", "punish_failed_months": [], "notice_failed_months": []}))
b.rename_axis("date").to_csv(tmp / "bars_0050.csv")
s = b.copy()
s["close"] = 600 + np.arange(len(s)) % 7  # 2330：週期性漲跌
s.rename_axis("date").to_csv(tmp / "bars_2330.csv")
rows = []
for t, bb in (("0050", b), ("2330", s)):
    for o in g[g.period != "gap"].origin:
        for lb in twse.CFG["lookbacks"]:
            rows.append({"ticker": t, "origin": o, "lookback": lb, "origin_close": bb.close[o], "pred_close_d5": bb.close[o] * 1.01})
for t, x in pd.DataFrame(rows).groupby("ticker"):
    x.to_csv(tmp / f"preds_{t}.csv", index=False)
score.main()
d = pd.read_csv(tmp / "results" / "predictions_scored.csv", dtype={"ticker": str}, parse_dates=["origin"])
x = d[(d.ticker == "0050") & d.events_in_window.fillna("").str.contains("分割")]
assert len(x) and np.allclose(x.actual_ret, 0) and (x.act_dir == 0).all()
assert json.loads((tmp / "results" / "selected_window.json").read_text())["selected_lookback"] == 40  # 預測全相同 → 同分取最短
lab = pd.read_csv(tmp / "results" / "labels.csv", dtype={"ticker": str}, parse_dates=["origin"])
inp = lab[(lab.ticker == "2330") & lab.origin.between("2025-03-04", "2025-03-17")]
assert len(inp) and (inp.reason == "處置中").all()
shutil.rmtree(tmp)
print("OK")
