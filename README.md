# Nigerian English/Pidgin Next-Word Prediction

A production-grade trigram language model for next-word prediction, trained on Nigerian Pidgin text from the NaijaSenti dataset.

## Overview

**Problem**: Predict the most likely next word(s) given a text context, optimized for Nigerian English and Pidgin.

**Approach**: Trigram statistical baseline with Laplace smoothing.

```
P(word | context) = P(wₙ | wₙ₋₂, wₙ₋₁)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

## Usage

```python
from src.data_loader import load_all_texts
from src.preprocessing import preprocess_corpus
from src.trigram_model import TrigramLM

# Load and preprocess
texts = load_all_texts()
sentences = preprocess_corpus(texts)

# Train
model = TrigramLM(smoothing=1.0)
model.train(sentences)

# Predict
predictions = model.predict_next_words("i dey", top_k=5)
# [('go', 0.12), ('come', 0.08), ('work', 0.06), ...]
```

## Project Structure

```
nextword/
├── src/
│   ├── data_loader.py      # NaijaSenti dataset loading
│   ├── preprocessing.py    # Text cleaning & tokenization
│   ├── trigram_model.py    # Core language model
│   └── utils.py            # Helper functions
├── main.py                 # Demo entry point
└── requirements.txt
```

## Technical Details

### Preprocessing
- Lowercase normalization
- URL/username removal
- Preserves: slang, contractions, code-switching

### Model
- Laplace (add-one) smoothing for unseen trigrams
- Perplexity evaluation for model comparison

### Limitations
| Aspect | Trigram Limitation |
|--------|-------------------|
| Context | Fixed 2-word window |
| Semantics | No understanding, purely statistical |
| Rare phrases | Sparsity even with smoothing |

## Evaluation

```python
# Compute perplexity on test data
perplexity = model.perplexity(test_sentences)
```

Lower perplexity = better model. This baseline establishes the floor for LSTM/Transformer comparisons.

## Next Steps

1. **LSTM model**: Variable-length context
2. **Transformer model**: Full sequence attention
3. **Subword tokenization**: Handle OOV words

## Dataset

[NaijaSenti](https://huggingface.co/datasets/mteb/NaijaSenti) - Nigerian Pidgin (PCM) split
