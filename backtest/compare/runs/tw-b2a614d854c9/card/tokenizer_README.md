---
language: zh
license: mit
tags:
  - finance
  - time-series
  - taiwan
  - stock
  - kronos
base_model: NeoQuasar/Kronos-Tokenizer-base
---

# Kronos-TW-Tokenizer

Kronos VQ Tokenizer fine-tuned on Taiwan Stock Exchange (TWSE) daily OHLCV data.

## Base Model
[NeoQuasar/Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)

## Training Data
- Source: TWSE listed stocks (613 stocks after quality filtering)
- Period: 2010-01-01 ~ 2024-12-31
- Features: open, high, low, close, volume, amount
- Lookback: 90 trading days

## Training Config
- Epochs: 3
- Batch size: 8 (effective 256 with gradient accumulation)
- Learning rate: 2e-4
- Hardware: RTX 4080 Laptop 12GB
