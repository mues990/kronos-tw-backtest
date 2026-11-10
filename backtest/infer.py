"""單一標的的 Kronos 滾動預測（不微調）。需 PYTHONPATH 指到 Kronos repo。

python backtest/infer.py 2330      # 環境變數 MODEL_REV / TOK_REV 指定 Hugging Face 版本
輸出 backtest/work/preds_2330.csv（逐筆附加，中斷後重跑會跳過已完成的列）
"""
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch

from twse import CFG, WORK, adjusted, bars, mode, now_tw

COLS = ["open", "high", "low", "close", "volume", "amount"]


def load_predictor():
    from model import Kronos, KronosPredictor, KronosTokenizer
    tok = KronosTokenizer.from_pretrained(CFG["tokenizer"], revision=os.environ.get("TOK_REV"))
    model = Kronos.from_pretrained(CFG["model"], revision=os.environ.get("MODEL_REV"))
    model.eval(), tok.eval()
    return KronosPredictor(model, tok, device="cpu", max_context=CFG["max_context"])


def seed_of(ticker, origin, lb):
    return (CFG["seed"] + zlib.crc32(f"{ticker}|{origin:%Y-%m-%d}|{lb}".encode())) % 2**31


def main(ticker):
    g = pd.read_csv(WORK / "grid.csv", parse_dates=["origin", "target"])
    g = g[(g.period != "gap") & (g.pilot if mode() == "pilot" else True)]
    cutoff = pd.Timestamp(json.loads((WORK / "prep_meta.json").read_text())["data_cutoff"])
    b = bars(ticker, pd.Timestamp(now_tw().date()))
    b = b[b.index <= cutoff]  # 截止日之後（未完成）的 K 棒不用
    b.to_csv(WORK / f"bars_{ticker}.csv")
    valid = b.dropna(subset=COLS)
    ev = pd.read_csv(WORK / "events.csv", dtype={"ticker": str}, parse_dates=["ex_date"])
    ev = ev[ev.ticker == ticker]

    out = WORK / f"preds_{ticker}.csv"
    done = set()
    if out.exists():
        p = pd.read_csv(out, parse_dates=["origin"])
        done = set(zip(p.origin, p.lookback))
    torch.set_num_threads(os.cpu_count())  # runner 預設只用 1 個執行緒
    predictor = load_predictor()
    meta = {"ticker": ticker, "torch": torch.__version__, "threads": torch.get_num_threads(), "cpu_count": os.cpu_count(),
            "model_rev": os.environ.get("MODEL_REV"), "tok_rev": os.environ.get("TOK_REV"),
            "kronos_commit": CFG["kronos_commit"], "code_rev": os.environ.get("GITHUB_SHA"), "started": now_tw().isoformat()}
    (WORK / f"meta_{ticker}.json").write_text(json.dumps(meta, indent=1))

    n = 0
    for o in g.origin:
        if o not in valid.index:
            continue  # 起點當天無成交：評分時標記 origin_no_trade
        x_all = adjusted(valid, ev, o)
        for lb in CFG["lookbacks"]:
            if (o, lb) in done:
                continue
            x = x_all.iloc[-lb:]
            if len(x) < lb:
                continue
            s = seed_of(ticker, o, lb)
            torch.manual_seed(s)
            np.random.seed(s)
            # ponytail: 未來時間戳只跳週末，不用事後才知道的休市日（颱風假）；只影響模型的日期特徵
            y_ts = pd.Series(pd.bdate_range(o + pd.offsets.BDay(1), periods=CFG["pred_len"]))
            t0 = time.time()
            p = predictor.predict(df=x[COLS].reset_index(drop=True), x_timestamp=pd.Series(x.index), y_timestamp=y_ts,
                                  pred_len=CFG["pred_len"], T=CFG["T"], top_p=CFG["top_p"],
                                  sample_count=CFG["sample_count"], verbose=False)
            row = {"ticker": ticker, "origin": f"{o:%Y-%m-%d}", "lookback": lb, "seed": s, "n_bars": len(x),
                   "input_start": f"{x.index[0]:%Y-%m-%d}", "origin_close": x.close.iloc[-1],
                   **{f"pred_close_d{i + 1}": round(float(v), 4) for i, v in enumerate(p.close)},
                   "secs": round(time.time() - t0, 2)}
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            n += 1
            print(f"{ticker} {o:%Y-%m-%d} lb={lb} {row['secs']}s", flush=True)
    print(f"{ticker}: {n} 筆新預測")


if __name__ == "__main__":
    main(sys.argv[1])
