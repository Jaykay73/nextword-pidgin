---
title: Nigerian Pidgin Next-Word Predictor
emoji: 🇳🇬
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Nigerian Pidgin Next-Word Predictor

LSTM Language Model trained on Nigerian Pidgin text (NaijaSenti + BBC Pidgin corpus).

## Usage

Enter Nigerian Pidgin text and get predictions for the next word.

**Example inputs:**
- "i dey" → go, come, work...
- "wetin you" → dey, go, wan...
- "how far" → ?, na, you...

## Model

- **Architecture**: 2-layer LSTM (256 embed, 512 hidden)
- **Training data**: ~10k texts (NaijaSenti + BBC Pidgin)
- **Vocab size**: ~8k words
