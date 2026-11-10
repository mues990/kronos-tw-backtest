# Kronos 台股 5 日方向滾動回測

用官方 `NeoQuasar/Kronos-base` ＋ `NeoQuasar/Kronos-Tokenizer-base`（不微調），比較 lookback = 40／90／120／250 根日 K，
預測第 5 個交易日收盤相對起點收盤的漲／跌／平。

## 執行

| 方式 | 指令 |
|---|---|
| Actions（建議） | Actions → kronos-backtest → Run workflow（選 `pilot` 或 `full`） |
| 本機 | `python backtest/twse.py prep` → `PYTHONPATH=<Kronos repo> python backtest/infer.py 2330`（每檔各跑一次）→ `python backtest/score.py` |
| 自我檢查 | `python backtest/test_backtest.py`（離線、不需模型） |

流程：`prep` 抓共用資料並產生起點格點、解析 Hugging Face 版本 → `infer` 每檔一個 job 平行推論（逐筆附加，可續跑）→ `score` 合併評分，把 `raw/` 與 `results/` 推回分支。

## 參數（`config.json`）

`pred_len=5`、`max_context=512`、`T=1.0`、`top_p=0.9`（top_k=0）、`sample_count=10`（Kronos 內部對 10 條樣本路徑取平均）。
每筆的隨機種子 = `(20261109 + crc32("代號|起點|lookback")) mod 2^31`，記錄在逐筆 CSV 的 `seed` 欄。
模型版本、Kronos 程式 commit、本 repo commit、torch 版本記錄在 `results/run/meta_<代號>.json`。

## 資料來源（全部 TWSE 官方，原始 JSON 存在 `raw/`）

| 用途 | 端點 | 說明 |
|---|---|---|
| 交易所交易日 | `afterTrading/FMTQIK` | 有列出的日期才算交易日（颱風假等休市日自然排除） |
| 日 K | `afterTrading/STOCK_DAY` | 未還原 OHLC；成交股數（股）、成交金額（元）。無有效收盤 = 停牌/無成交，不補值 |
| 除權息 | `exRight/TWT49U` | factor = 除權息參考價 ÷ 除權息前收盤價 |
| 減資 | `reducation/TWTAUU` | factor = 恢復買賣參考價 ÷ 停止買賣前收盤；量同步調整（本名單期間內無資料） |
| 分割 | `splits.csv`（手動，附公告來源） | 0050：2025-06-18 起 1 拆 4，factor 0.25，量 ×4 |
| 處置 | `announcement/punish` | 公布日期、處置起迄 |
| 注意股 | `announcement/notice` | 只做標記 |

## 口徑

- **輸入**：截至起點收盤；只套用 `ex_date ≤ 起點` 的公司行動做向後還原，最後一根是原始價。未來 5 天的時間戳只跳週末（不用事後才知道的休市日）。
- **評分**：目標 = 起點後第 5 個交易所交易日；該日無有效收盤 → 無法評分（不順延）。實際報酬 = 目標收盤 ÷（起點收盤 × 期間內事件 factor 乘積）− 1。
- **平**：漲跌幅小於半個升降單位（以起點價計）即為平；預測、實際、動能基準都用同一門檻。
- **格點**：2024-07-01 起每 5 個交易所交易日一個起點。開發期 = 目標日 ≤ 2025-12-31；保留期 = 起點 ≥ 2026-01-01 且目標日 ≤ 資料截止日；跨期的 1 個起點不跑。
- **選窗**：開發期個股主組、四窗共同樣本的方向命中率最高者，同分取較短，寫入 `selected_window.json` 後才看保留期。
- **個股主組**：近 20 個交易日成交金額中位數 ≥ 5,000 萬元，且起點時不在處置中、也沒有已公告將於 5 日內開始的處置（以公布日 ≤ 起點為「已知」）。預測後才公告的處置留在主組並加 `post_disp_announced` 標籤。
- **頻繁處置篩選**：另外排除起點往前 60 個交易日內（含起點）不同公布日的處置公告 ≥ 2 次者。
- **基準**：一律看漲；過去 5 個交易日方向延續（同一還原口徑）；MAE 基準為「收盤不變」（= |實際報酬|）。

## 輸出（`results/`）

- `predictions_scored.csv`：逐筆（標的 × 起點 × lookback），含預測 1–5 日收盤、實際報酬與方向、基準、所有標記與種子。
- `labels.csv`：標的 × 起點，含排除原因（`reason`）。
- `summary.md`、`tables/*.csv`：各組彙總、各檔、處置分組、頻繁處置篩選、集中度、資料檢查。
- `selected_window.json`、`run/`（格點、事件、處置、各檔原始預測與版本紀錄）。

方向命中率是預測準確度，不是交易績效（未計交易成本、滑價與部位）。
