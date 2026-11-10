# 台股短線預測回測

先檢驗 Kronos K 線模型能否預測台股短線方向，再延伸到「條件成立時 20 日 ±10% 的勝率」研究：包含量價規則，以及 Qlib Alpha158 ＋ LightGBM。

**所有輪次的結論、數字與檔案位置：[`研究紀錄.md`](研究紀錄.md)**

- Kronos 5 日方向回測：[`backtest/results/摘要.md`](backtest/results/摘要.md)；方法與資料：[`backtest/README.md`](backtest/README.md)
- 原版 vs 台股微調版：[`backtest/compare/results/摘要.md`](backtest/compare/results/摘要.md)
- 條件研究基準：[`backtest/market/base/summary.md`](backtest/market/base/summary.md)
- 量價條件：[`backtest/market/conditions/summary.md`](backtest/market/conditions/summary.md)
- Qlib：[`backtest/market/qlib/summary.md`](backtest/market/qlib/summary.md)

## 致謝

本專案使用 [Kronos](https://github.com/shiyu-coder/Kronos)（MIT License）及
[NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)、Kronos-Tokenizer-base 模型。
workflow 執行時抓取 Kronos 指定 commit `67b630e`，本 repo 不包含其程式碼。本專案與 Kronos 作者無關。

論文：Yu Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets*, arXiv:2508.02739

台股微調版比較使用 [talant28/Kronos-TW-Predictor](https://huggingface.co/talant28/Kronos-TW-Predictor) 與 [talant28/Kronos-TW-Tokenizer](https://huggingface.co/talant28/Kronos-TW-Tokenizer)（MIT License），模型卡由 workflow 下載保存於 `backtest/compare/runs/*/card/`。

條件研究使用 [Microsoft Qlib](https://github.com/microsoft/qlib)（MIT License）的 Alpha158 因子定義（`pyqlib` 0.9.7），本 repo 不包含其程式碼。行情資料來自臺灣證券交易所公開資料。

## 聲明

個人研究用途，不構成投資建議。方向命中率是預測準確度，不是交易績效（未計交易成本、滑價與部位）。
