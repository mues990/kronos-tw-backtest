---
language: zh
license: mit
tags:
  - finance
  - time-series
  - taiwan
  - stock
  - kronos
base_model: NeoQuasar/Kronos-base
---

# Kronos-TW-Predictor

Kronos Autoregressive Predictor fine-tuned on Taiwan Stock Exchange (TWSE) daily data.

## Base Model
[NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

## Training Data
- Source: TWSE listed stocks (613 stocks after quality filtering)
- Period: 2010-01-01 ~ 2024-12-31
- Features: open, high, low, close, volume, amount
- Predict window: 10 trading days

## Training Config
- Epochs: 3
- Batch size: 8 (effective 256 with gradient accumulation)
- Learning rate: 4e-5
- Hardware: RTX 4080 Laptop 12GB

## Usage
Use with [Kronos-TW-Tokenizer](https://huggingface.co/talant28/Kronos-TW-Tokenizer)

## Citation
This model is fine-tuned from Kronos. If you use this work, please cite the original paper:

```bibtex
@article{shi2025kronos,
  title={Kronos: A Foundation Model for Time Series},
  author={Shi, Yu and others},
  journal={arXiv preprint arXiv:2504.03249},
  year={2025}
}
```
