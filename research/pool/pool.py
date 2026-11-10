"""selected pool：不看時機的基準勝率，以及出場規則（停損／停利）比較。探索性分析。

python research/pool/pool.py OUT      # 用 yfinance 還原日 K（auto_adjust）；結果寫到 OUT/
每個交易日都當作一次進場：訊號日 t，t+1 開盤買；基準是持有到 t+H 收盤。報酬扣來回成本 0.5%。
出場規則的成交假設：盤中碰到停損價即以停損價成交，開盤已跌破則以開盤價成交；同一天同時碰到停損與停利，算停損。
未處理漲跌停鎖死賣不掉的情況，實際結果會比這裡差一些。樣本彼此重疊，不是獨立交易。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COST = 0.005
RULES = {  # 名稱: 參數
    "持有到期（基準）": {},
    "停損 −8%": {"stop": .08},
    "停損 −10%": {"stop": .10},
    "停損 2×ATR14": {"atr_k": 2},
    "移動停損 10%（從最高收盤回落）": {"stop": .10, "trail": .10},
    "收盤跌破 20 日線，隔天開盤出": {"ma": True},
    "停損 −8%＋停利 +15%": {"stop": .08, "tp": .15},
    "停損 −8%＋停利 +10%": {"stop": .08, "tp": .10},
}


def sim(O, Hh, L, C, ATR, MA, i, n, stop=None, tp=None, trail=None, ma=False, atr_k=None):
    e = O[i + 1]
    s = e * (1 - stop) if stop else (e - atr_k * ATR[i] if atr_k else None)
    hi = e
    for k in range(i + 1, i + n + 1):
        if s is not None and L[k] <= s:
            return min(O[k], s) / e - 1, k - i
        if tp and Hh[k] >= e * (1 + tp):
            return max(O[k], e * (1 + tp)) / e - 1, k - i
        if trail:
            hi = max(hi, C[k])
            s = max(s, hi * (1 - trail))
        if ma and C[k] < MA[k] and k < i + n:
            return O[k + 1] / e - 1, k + 1 - i
    return C[i + n] / e - 1, n


def evaluate(data, horizons=(20, 60)):
    """data: {代號: DataFrame(index=日期, columns=open/high/low/close，已還原)}"""
    rows = []
    for code, d in data.items():
        d = d.dropna(subset=["open", "high", "low", "close"])
        O, Hh, L, C = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        pc = np.r_[np.nan, C[:-1]]
        ATR = pd.Series(np.fmax(Hh - L, np.fmax(abs(Hh - pc), abs(L - pc)))).rolling(14).mean().to_numpy()
        MA = pd.Series(C).rolling(20).mean().to_numpy()
        for n in horizons:
            for i in range(20, len(d) - n):
                for name, kw in RULES.items():
                    r, days = sim(O, Hh, L, C, ATR, MA, i, n, **kw)
                    rows.append((code, d.index[i], n, name, r - COST, days))
    return pd.DataFrame(rows, columns=["code", "date", "H", "rule", "ret", "days"])


def tables(R):
    R["期間"] = pd.cut(R.date.dt.year, [0, 2022, 2024, 9999], labels=["2015–2022", "2023–2024", "2025–2026"])
    out = []
    f = lambda t: t.apply(lambda c: c.map(lambda v: f"{v:.1f}") if c.dtype.kind == "f" else c.astype(str)).to_markdown(index=False, disable_numparse=True)
    b = R[(R.rule == "持有到期（基準）") & (R.H == 20)]
    g = b.groupby("code").agg(樣本日=("ret", "size"), 起=("date", lambda s: f"{s.min():%Y-%m}"),
                              **{"≥+10%": ("ret", lambda s: 100 * (s + COST >= .10).mean()), "有賺（扣成本）": ("ret", lambda s: 100 * (s > 0).mean()),
                                 "≤−10%": ("ret", lambda s: 100 * (s + COST <= -.10).mean()), "平均報酬%": ("ret", lambda s: 100 * s.mean())}).reset_index()
    out += ["## 基準：不看時機、持有 20 天（各檔）", "", f(g), ""]
    for n in sorted(R.H.unique()):
        x = R[R.H == n]
        t = x.groupby("rule", sort=False).agg(**{"平均報酬%": ("ret", lambda s: 100 * s.mean()), "中位數%": ("ret", lambda s: 100 * s.median()),
                                                 "有賺%": ("ret", lambda s: 100 * (s > 0).mean()), "≥+10%": ("ret", lambda s: 100 * (s + COST >= .10).mean()),
                                                 "≤−10%": ("ret", lambda s: 100 * (s + COST <= -.10).mean()), "最差 5%": ("ret", lambda s: 100 * s.quantile(.05)),
                                                 "平均持有天數": ("days", "mean")}).reset_index()
        p = x.pivot_table(index="rule", columns="期間", values="ret", aggfunc=lambda s: 100 * s.mean(), observed=True, sort=False)
        p.columns = [f"平均報酬% {c}" for c in p.columns]
        out += [f"## 出場規則比較：最長持有 {n} 天（全部標的合併，{x.code.nunique()} 檔、{len(x) // len(RULES):,} 個進場日）", "", f(t.merge(p.reset_index(), on="rule")), ""]
    return "\n".join(out)


def fetch(pool):
    import yfinance as yf
    data, log = {}, []
    for t in pool:
        for sym in ([f"{t}.TW", f"{t}.TWO"] if t[0].isdigit() else [t]):
            d = yf.download(sym, start="2015-01-01", end="2026-11-10", auto_adjust=True, progress=False)
            if len(d) > 60:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d.columns = [c.lower() for c in d.columns]
                data[t] = d[d.volume > 0]
                log.append(f"{t}: {sym} {d.index[0]:%Y-%m-%d}～{d.index[-1]:%Y-%m-%d} {len(d)} 列")
                break
        else:
            log.append(f"{t}: 抓不到資料")
    return data, log


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    pool = [x.split()[0] for x in (Path(__file__).parent / "pool.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
    data, log = fetch(pool)
    R = evaluate(data)
    R.to_parquet(out / "pool_trades.parquet", index=False)
    md = "# selected pool：基準勝率與出場規則（探索性）\n\n資料：yfinance 還原日 K。\n\n" + tables(R) + "\n## 資料\n\n" + "\n".join(f"- {x}" for x in log) + "\n"
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
