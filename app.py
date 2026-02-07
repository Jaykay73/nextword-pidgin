"""
Streamlit app for Nigerian Pidgin Next-Word Prediction.
Deploy to Hugging Face Spaces.
"""

import streamlit as st
import torch
import torch.nn as nn
import re
from typing import List, Dict

# Page config
st.set_page_config(
    page_title="Nigerian Pidgin Predictor",
    page_icon="💬",
    layout="centered"
)

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '</EOS>'


def clean_text(text: str) -> str:
    """Clean text while preserving Nigerian Pidgin features."""
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)
    return tokens


class LSTMLanguageModel(nn.Module):
    """LSTM-based language model for next-word prediction."""
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 256, 
        hidden_dim: int = 512, 
        num_layers: int = 2, 
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        logits = self.fc(out)
        return logits


@st.cache_resource
def load_model():
    """Load model (cached)."""
    checkpoint = torch.load('model/lstm_pidgin_model.pt', map_location='cpu')
    word_to_idx = checkpoint['word_to_idx']
    idx_to_word = checkpoint['idx_to_word']
    vocab_size = checkpoint['vocab_size']
    
    model = LSTMLanguageModel(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, word_to_idx, idx_to_word


def predict_next_words(context: str, model, word_to_idx, idx_to_word, top_k: int = 5):
    """Predict next words given context."""
    if not context.strip():
        return []
    
    tokens = tokenize(clean_text(context))
    if not tokens:
        return []
    
    unk_idx = word_to_idx.get(UNK_TOKEN, 1)
    indices = [word_to_idx.get(t, unk_idx) for t in tokens]
    
    x = torch.tensor([indices], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs[0], top_k)
    
    results = []
    for prob, idx in zip(top_probs.numpy(), top_indices.numpy()):
        word = idx_to_word.get(str(idx), idx_to_word.get(idx, UNK_TOKEN))
        if word not in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            results.append((word, float(prob)))
    
    return results


# Load model
model, word_to_idx, idx_to_word = load_model()

# UI
st.title("💬 Nigerian Pidgin Next-Word Predictor")
st.markdown("**LSTM Language Model** trained on Nigerian Pidgin text.")

# Input
context = st.text_input(
    "Enter Nigerian Pidgin text:",
    placeholder="e.g., 'i dey', 'wetin you', 'how far'"
)

top_k = st.slider("Number of predictions:", 1, 10, 5)

# Predict button
if st.button("Predict", type="primary") or context:
    if context:
        predictions = predict_next_words(context, model, word_to_idx, idx_to_word, top_k)
        
        if predictions:
            st.subheader("Predictions:")
            for word, prob in predictions:
                st.markdown(f"**{word}** — {prob:.1%}")
        else:
            st.warning("No predictions available.")
    else:
        st.info("Enter some text to get predictions.")

# Examples
st.markdown("---")
st.markdown("**Try these examples:**")
cols = st.columns(4)
examples = ["i dey", "wetin you", "how far", "e don"]
for col, ex in zip(cols, examples):
    if col.button(ex):
        st.session_state['context'] = ex
        st.rerun()

# Footer
st.markdown("---")
st.caption("Trained on NaijaSenti + BBC Pidgin corpus (~10k texts)")
