# tw20 交接文件（第 1、2 段完成）

更新：2026-11-10。分支 `research/tw20`。到目前為止沒有載入、預測或評分任何 2025 年以後的樣本。

## 第 2 段結果（A／B 模型，驗證期 2023–2024）

- **規格：** `research/tw20/protocol.json`，在訓練前提交（commit `ba86ca9`）。之後有 2 項修訂，都在看到任何真實驗證績效之前，記在檔內 `amendments`：加權指數改用 FMTQIK；vol20 缺值自成一個波動組。
- **程式：** `research/tw20/stage2.py`、`test_stage2.py`。
- **執行：** <https://github.com/mues990/kronos-tw-backtest/actions/runs/42762827049>（commit `e6f6790`，約 5.7 分鐘，標準 runner）。Artifact `tw20-stage2` 保留到 2027-02-08，內含兩個模型、`frozen.json`、逐筆驗證預測、`candidates.csv`、`summary.md`、placebo 結果、雜湊。
- **repo 內的精簡結果：** `research/tw20/results/stage2/summary.md`、`frozen.json`（凍結門檻、迭代數、模型 SHA-256、波動組切點）。
- **樣本：** fit 198,995、early stop 70,270、refit 272,251、驗證 68,922（461 個交易日）。A 迭代 131，B 迭代 48。

| 方向 | 主要模型（事前規則選出） | 訊號數／日期數 | 驗證達標率 | 同日同波動基準 | 提升（40 日區塊 95% 區間） |
|---|---|---|---|---|---|
| UP | A、P90（門檻 0.2387） | 5,437／448 | 27.6% [23.5, 33.8] | 21.8% | +5.9 pp [+4.1, +8.2] |
| DOWN | A、P99（門檻 0.3216） | 685／312 | 29.6% [22.7, 36.8] | 22.2% | +7.4 pp [+1.0, +15.3] |

- 12 個候選中 9 個通過事前規則（A：UP P90／P95、DOWN P90／P95／P99；B：UP P90／P95、DOWN P90／P95）。
- **70% 目標未達成。** 所有候選的驗證達標率都在 14%–30% 之間。
- **B 沒有比 A 好。** 同門檻下 B 的達標率與提升都較低，訊號集中在較少的日期（UP P90：B 244 天，A 448 天）。
- **DOWN 主要候選不穩。** 2023 年提升只有 +0.4 pp，2024 年 +15.8 pp。
- **保留期閘門：不開啟。** 沒有任何通過的候選驗證達標率超過 50%（protocol 的 `holdout_gate`）。依規則不進行保留期評估。
- **負向對照（標籤在同一天內打亂）：** A 的提升在 −1.5 到 +0.2 pp 之間，沒有任何候選通過，符合預期。
- **重現性：** 本機重跑得到的兩個模型檔 SHA-256 與 Actions 完全相同。

## 第 3 段的狀態

依 `holdout_gate`，不開啟保留期模型績效。下一步是結案報告，以及以新日期做前瞻檢驗的規格（凍結 `frozen.json` 的模型與門檻，從 2026-11-10 之後的新訊號日開始累積，不重訓、不調門檻）。尚未建立任何排程。

## 任務定義（固定）

- 訊號在 t 日收盤後產生；以 t+1 交易日開盤為基準（第 1 日），t+20 交易日收盤評分。
- `R20 = close[t+20] / (open[t+1] × Π factor(t+1, t+20]) − 1`。除權息、減資、變更面額以 factor 還原。
- `R20 ≥ +10%` → UP；`R20 ≤ −10%` → DOWN；其餘 OTHER。期間曾達標不算。DOWN 只代表看空／避跌訊號。
- 股票池逐日決定：上市普通股（代號 1000–8999）、已有 ≥120 個有成交的交易日、當日原始收盤 ≥ 10 元、過去 60 日成交金額中位數（至少 40 個有成交日）前 150 名。離開股票池不影響已產生訊號的追蹤。
- 資料截止日 2026-11-09。最後一個成熟訊號日是 2026-10-07。

## 檔案與資料位置

| 項目 | 位置 |
|---|---|
| 程式 | `research/tw20/tw20.py`（子命令 `events`、`labels`、`features`、`checks`、`hashes`） |
| 離線檢查 | `research/tw20/test_tw20.py` |
| 依賴鎖定 | `research/tw20/requirements.lock`（Python 3.12；`pyqlib==0.9.7`、`lightgbm==4.7.0`、`pandas==2.2.3`、`numpy==2.2.6`、`pyarrow==26.0.1`） |
| Workflow | `.github/workflows/tw20-data.yml`（只有 `workflow_dispatch`；ubuntu-24.04；timeout 45 分；concurrency；輸入檢查；不寫回 repo） |
| 日 K（輸入） | `backtest/market/daily_YYYY.csv.gz`、`manifest_YYYY.csv`（含加權指數收盤）；證交所 MI_INDEX，2015-01-05～2026-11-09，2,886 個交易日 |
| 除權息、減資（輸入） | `backtest/raw/events/TWT49U_YYYY.json`、`TWTAUU_YYYY.json` |
| 變更面額（輸入） | Actions 執行時抓證交所 `rwd/zh/change/TWTB7U`，原始 JSON 存在 artifact 的 `raw/` |
| 輸出（artifact `tw20-stage1-smoke`，保留 30 天） | `labels.parquet`、`features.parquet`（smoke 只含指定代號）、`feature_label_check.parquet`、`jumps.csv`、`checks.md`、`checks.json`、`status_by_year.csv`、`hashes.json`、`pip-freeze.txt`、`run.txt` |

Smoke test 執行：<https://github.com/mues990/kronos-tw-backtest/actions/runs/42749647713>（commit `3dc5b51`，約 1.5 分鐘）。前一次：<https://github.com/mues990/kronos-tw-backtest/actions/runs/42749197328>。

## 必要命令

```bash
pip install -r research/tw20/requirements.lock        # Python 3.12
python research/tw20/test_tw20.py
python research/tw20/tw20.py events   OUT             # 需連證交所；其餘步驟離線
python research/tw20/tw20.py labels   OUT
python research/tw20/tw20.py features OUT             # 不加 --codes ＝ 全部曾入池的個股（約 611 檔，本機約 1 分鐘）
python research/tw20/tw20.py checks   OUT
python research/tw20/tw20.py hashes   OUT
gh workflow run tw20-data.yml --ref research/tw20     # 只能在此分支執行
```

## `labels.parquet` 欄位

`date`（訊號日 t）、`code`、`label_end_date`（t+20；第 2 段 purge 用）、`R20`、`cls`（UP／DOWN／OTHER）、`status`、`entry_locked_up`、`entry_locked_down`、`vol20`（訊號前 20 日還原日報酬標準差 × √20，ddof=1）、`liq_rank`、`close_raw`、`entry_open_raw`、`exit_close_raw`、`window_factor`。

`status` 只有 `ok` 的樣本可以評分：

| status | 筆數 | 意義 |
|---|---|---|
| ok | 410,762 | 可評分 |
| immature | 3,000 | t+20 超過資料截止日，不評分 |
| feature_event_unreliable | 640 | t−60..t 內有無法還原的跳空（訊號時已知） |
| no_exit_price | 413 | t+20 當日停牌或無成交 |
| label_event_unreliable | 145 | t+1..t+20 內有無法還原的跳空（事後才知道，屬資料品質排除，不是選股規則） |
| no_entry_price | 90 | t+1 停牌或無開盤價 |

`ok` 樣本另有交易可行旗標（沒有刪除）：t+1 一字漲停 `entry_locked_up`（UP 訊號買不到）、t+1 一字跌停 `entry_locked_down`（DOWN 訊號賣不掉）。第 2 段評分時要決定並事前寫明如何處理。

## 檢查結果

1. **特徵與標籤口徑一致。** 還原收盤由還原日報酬累乘；開高低與 VWAP 乘同一調整係數，成交量除以該係數。用 Qlib 讀回還原價重算 R20，與標籤最大差 2.5e-7（float32 誤差）；`$close/$factor` 對原始收盤最大相對差 1.4e-7。
2. **單位。** 成交金額（元）÷ 成交股數（股）＝ 每股均價。STOCK_DAY 與 MI_INDEX 兩個來源的股數與金額，12 檔 10,764 筆完全相同。VWAP 沒有用收盤價替代。
3. **VWAP 落在當日高低價外的比例：全市場 0.38%，股票池樣本 0.11%。** 推測是官方全日量額包含鉅額、盤後等非盤中撮合成交（未核實）。照原值使用，沒有修改。
4. **事件排除與未來資訊。** 舊基準排除「t−20..t+20 內有無法還原跳空」的樣本，其中 t+1..t+20 用到了未來資訊。本研究拆成兩種狀態（見上表），並逐筆保留、列出數量，不當成事前選股規則。
5. **完整歷史先算特徵。** 每檔從第一個成交日到最後一個成交日整段寫入 Qlib，算完 Alpha158 才套用每日股票池。
6. **與舊基準比較。** 舊基準的 411,187 筆全部都在，R20 最大差 4.4e-16。其中 425 筆因特徵視窗由 20 日加長到 60 日，改標為 `feature_event_unreliable`。舊基準默默排除的 3,863 筆現在都有狀態。
7. **抽查 11 檔 13 筆全部一致。** 涵蓋一般日期、窗口內除息、除權、除權息、變更面額前後、三個資料切分邊界、最後一個成熟日與第一個未成熟日。做法是直接讀原始日 K 與事件表獨立重算。
8. **加權指數。** MI_INDEX（`manifest_YYYY.csv`）缺 2019-04-29 一日的指數（第 1 段當時誤記為沒有缺值，第 2 段發現）。現在改用證交所 FMTQIK（`backtest/raw/calendar/`），2,886 個交易日沒有缺值；兩個來源在其餘日期的差為 0。

第 1–7 項的數字來自本機，程式與資料跟 Actions 相同；Actions 上的 `checks.md` 在該次執行的 Summary 與 artifact 裡。樣本狀態筆數已與 Actions 的執行紀錄核對一致。

## Qlib 接入

- `pyqlib==0.9.7`，只用資料層與 `qlib.contrib.data.loader.Alpha158DL.get_feature_config()`（官方 Alpha158 handler 使用同一組設定），共 158 個因子。
- 沒有使用官方預設的短期回歸標籤、label 標準化、任何 processor 或 TopkDropout。標籤與評分都用本研究自己的定義。

## 未解問題

1. **歷史變更面額事件抓不到。** `rwd/zh/change/TWTB7U` 可以連線，欄位有換股率（factor = 1 ÷ 換股率）。但試過 `strDate/endDate`、`startDate/endDate`、`date`、`year` 四種日期參數，都只回傳最新一筆（6949，2026-09-07）。所以 2327（2025-08-25）等歷史事件仍無法還原，相關樣本列在 `feature_event_unreliable` 與 `label_event_unreliable`，合計 785 筆，佔 0.19%。要補救，需要找到正確的查詢參數，或改用其他可驗證的官方來源。
2. **無法還原的跳空共 133 筆，清單在 `jumps.csv`。** 其中約一半是 +10.5%～+20% 的單日漲幅，原因未查明（可能是恢復交易首日等無漲跌幅限制的情況）。
3. **VWAP 超出高低價的原因未核實**（見檢查結果第 3 項）。
4. **「上市滿 120 個交易日」以「有成交的交易日數 ≥ 120」近似**，沒有用實際上市日期。
5. **一字漲跌停的判定是近似。** 條件是 t+1 全日一價，且開盤相對前收達到漲跌幅限制減 0.5%，沒有用官方漲跌停價。
6. **`tw20-data.yml` 的定義也放了一份在 main。** GitHub 規定手動觸發的 workflow 必須存在於預設分支。第一步會拒絕在 `research/tw20` 以外的分支執行。研究程式碼沒有合併到 main。
7. **2025–2026 保留期不是完全沒看過。** 整體基準統計，以及先前 Qlib 二分類研究（`backtest/market/qlib/`）在保留期的結果都已被查看。第 3 段報告必須如實說明。
