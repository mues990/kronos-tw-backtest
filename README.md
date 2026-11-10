# Kronos 台股 5 日方向回測

用 Kronos 預訓練模型（不微調）對 15 檔台股／ETF 做滾動回測，檢驗第 5 個交易日的漲跌方向預測是否勝過簡單基準。

- 結果摘要：[`backtest/results/摘要.md`](backtest/results/摘要.md)
- 方法、資料來源與執行方式：[`backtest/README.md`](backtest/README.md)

## 致謝

本專案使用 [Kronos](https://github.com/shiyu-coder/Kronos)（MIT License）及
[NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)、Kronos-Tokenizer-base 模型。
workflow 執行時抓取 Kronos 指定 commit `67b630e`，本 repo 不包含其程式碼。本專案與 Kronos 作者無關。

論文：Yu Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets*, arXiv:2508.02739

## 聲明

個人研究用途，不構成投資建議。方向命中率是預測準確度，不是交易績效（未計交易成本、滑價與部位）。
