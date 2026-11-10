"""離線檢查 pair.py 評分：用已存 Base 預測當兩組模型，配對評分須重現原版 lookback 90 結果。python backtest/test_pair.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pair  # noqa: E402

tmp = Path(tempfile.mkdtemp())
pair.RUNS = tmp / "runs"
pair.HERE = tmp  # 輸出寫到暫存目錄
st = pd.read_csv(pair.SNAP / "predictions_scored.csv", dtype={"ticker": str}, parse_dates=["origin"])
st = st[st.lookback == pair.LB]
cur = {}
for name in ("base", "tw"):
    d = pair.RUNS / f"{name}-test"
    d.mkdir(parents=True)
    p = st[["ticker", "origin", "lookback", "origin_close", "pred_close_d1", "pred_close_d5", "input_start", "secs"]].assign(input_hash="x", input_last_date=lambda x: x.origin, pred_d5_date=lambda x: x.origin, finite=True)
    if name == "tw":
        p = p.assign(pred_close_d5=p.origin_close * 1.01)  # TW：一律預測漲 1%
    p.drop(index=p.index[:3]).to_csv(d / "preds_all.csv", index=False) if name == "tw" else p.to_csv(d / "preds_all.csv", index=False)
    cur[name] = {"dir": d.name, "model": name, "model_rev": "r" * 12, "tokenizer": name, "tok_rev": "t" * 12}
(pair.RUNS / "current_full.json").write_text(json.dumps(cur))
d = pair.paired_table("full")
h = st[st.period == "holdout"][["ticker", "origin", "hit"]].merge(d, on=["ticker", "origin"])
h = h[h.paired]
assert (h.hit_base == h.hit).all(), "Base 配對命中與原版不符"
assert len(d[d.pair_status == "TW 無預測"]) <= 3 and d[d.paired].same_input.all()
y = d[d.paired & (d.group == "個股")]
assert abs(y.hit_tw.mean() - y.up_hit.mean()) < 0.05  # 一律預測漲（扣掉平盤）應接近一律看漲
assert (y.down_tw == 0).all()
g = pair.gaps(y).set_index("比較")
assert abs(g.loc["命中：Base − 一律看漲", "差距(pp)"] - 100 * (y.hit_base.mean() - y.up_hit.mean())) < 1e-9
pair.score("full")
shutil.rmtree(tmp)
print("OK")
