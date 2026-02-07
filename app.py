"""
Nigerian Pidgin Next-Word Prediction - Streamlit App
Supports both LSTM and Trigram models for comparison.
"""

import streamlit as st
import torch
import torch.nn as nn
import re
import pickle
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional

# =============================================================================
# Page Config & Custom CSS
# =============================================================================
st.set_page_config(
    page_title="Nigerian Pidgin Predictor",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a5f2a 0%, #2d8a3e 50%, #f4c430 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    
    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2d8a3e;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .prediction-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .word {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a5f2a;
    }
    
    .prob {
        font-size: 1rem;
        color: #666;
    }
    
    /* Model selector */
    .stRadio > div {
        display: flex;
        gap: 1rem;
    }
    
    /* Example buttons */
    .stButton > button {
        border-radius: 20px;
        border: 2px solid #2d8a3e;
        background: white;
        color: #2d8a3e;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: #2d8a3e;
        color: white;
    }
    
    /* Comparison columns */
    .model-column {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Special Tokens
# =============================================================================
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '</EOS>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'

# =============================================================================
# Text Processing
# =============================================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)
    return tokens

# =============================================================================
# LSTM Model
# =============================================================================
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        return self.fc(out)

# =============================================================================
# Trigram Model
# =============================================================================
class TrigramLM:
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.vocab = set()
    
    def probability(self, w3: str, w1: str, w2: str) -> float:
        trigram_count = self.trigram_counts.get((w1, w2, w3), 0)
        bigram_count = self.bigram_counts.get((w1, w2), 0)
        vocab_size = len(self.vocab)
        numerator = trigram_count + self.smoothing
        denominator = bigram_count + (self.smoothing * vocab_size)
        return numerator / denominator if denominator > 0 else 0.0
    
    def predict_next_words(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        words = context.lower().split()
        if len(words) == 0:
            w1, w2 = START_TOKEN, START_TOKEN
        elif len(words) == 1:
            w1, w2 = START_TOKEN, words[0]
        else:
            w1, w2 = words[-2], words[-1]
        
        candidates = []
        for word in self.vocab:
            if word not in (START_TOKEN, END_TOKEN, '<s>', '</s>'):
                prob = self.probability(word, w1, w2)
                candidates.append((word, prob))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

# =============================================================================
# Model Loading
# =============================================================================
@st.cache_resource
def load_lstm_model():
    """Load LSTM model."""
    try:
        checkpoint = torch.load('model/lstm_pidgin_model.pt', map_location='cpu')
        word_to_idx = checkpoint['word_to_idx']
        idx_to_word = checkpoint['idx_to_word']
        vocab_size = checkpoint['vocab_size']
        
        model = LSTMLanguageModel(vocab_size=vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, word_to_idx, idx_to_word, True
    except Exception as e:
        return None, None, None, False

@st.cache_resource
def load_trigram_model():
    """Load or build Trigram model."""
    try:
        # Try to load pre-saved trigram model
        if os.path.exists('model/trigram_model.pkl'):
            with open('model/trigram_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model, True
        else:
            # Build a simple demo trigram with common patterns
            model = TrigramLM(smoothing=1.0)
            # Add some common Nigerian Pidgin patterns
            common_patterns = [
                ['<s>', '<s>', 'i', 'dey', 'go', '</s>'],
                ['<s>', '<s>', 'i', 'dey', 'come', '</s>'],
                ['<s>', '<s>', 'wetin', 'you', 'dey', 'do', '</s>'],
                ['<s>', '<s>', 'how', 'far', '</s>'],
                ['<s>', '<s>', 'e', 'don', 'happen', '</s>'],
                ['<s>', '<s>', 'na', 'the', 'matter', '</s>'],
                ['<s>', '<s>', 'you', 'no', 'sabi', '</s>'],
                ['<s>', '<s>', 'make', 'we', 'go', '</s>'],
            ]
            for sent in common_patterns:
                model.vocab.update(sent)
                for token in sent:
                    model.unigram_counts[token] += 1
                for i in range(len(sent) - 1):
                    model.bigram_counts[(sent[i], sent[i+1])] += 1
                for i in range(len(sent) - 2):
                    model.trigram_counts[(sent[i], sent[i+1], sent[i+2])] += 1
            return model, True
    except Exception as e:
        return None, False

# =============================================================================
# Prediction Functions
# =============================================================================
def predict_lstm(context: str, model, word_to_idx, idx_to_word, top_k: int = 5):
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
    
    top_probs, top_indices = torch.topk(probs[0], top_k + 5)
    
    results = []
    for prob, idx in zip(top_probs.numpy(), top_indices.numpy()):
        word = idx_to_word.get(str(idx), idx_to_word.get(idx, UNK_TOKEN))
        if word not in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            results.append((word, float(prob)))
        if len(results) >= top_k:
            break
    
    return results

def predict_trigram(context: str, model, top_k: int = 5):
    if not context.strip() or model is None:
        return []
    return model.predict_next_words(context, top_k)

# =============================================================================
# UI Components
# =============================================================================
def render_predictions(predictions: List[Tuple[str, float]], model_name: str):
    if not predictions:
        st.warning(f"No predictions from {model_name}")
        return
    
    for word, prob in predictions:
        st.markdown(f"""
        <div class="prediction-card">
            <span class="word">{word}</span>
            <span class="prob"> — {prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Main App
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>💬 Nigerian Pidgin Next-Word Predictor</h1>
    <p>Compare LSTM neural network vs Trigram statistical model</p>
</div>
""", unsafe_allow_html=True)

# Load models
lstm_model, word_to_idx, idx_to_word, lstm_loaded = load_lstm_model()
trigram_model, trigram_loaded = load_trigram_model()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_choice = st.radio(
        "Select Model:",
        ["🤖 LSTM (Neural)", "📊 Trigram (Statistical)", "⚔️ Compare Both"],
        index=2
    )
    
    top_k = st.slider("Number of predictions:", 1, 10, 5)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **LSTM Model**: Neural network that learns patterns from data. Better at capturing complex dependencies.
    
    **Trigram Model**: Statistical model using word co-occurrence counts. Fast and interpretable.
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub](https://github.com/Jaykay73/nextword-pidgin)")

# Main input
st.markdown("### Enter Nigerian Pidgin text:")
context = st.text_input(
    label="Context",
    placeholder="e.g., 'i dey', 'wetin you', 'how far'",
    label_visibility="collapsed"
)

# Example buttons
st.markdown("**Try these examples:**")
example_cols = st.columns(5)
examples = ["i dey", "wetin you", "how far", "e don", "make we"]
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True):
        context = ex

# Predictions
if context:
    st.markdown("---")
    
    if "Compare" in model_choice:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 LSTM Neural Network")
            if lstm_loaded:
                predictions = predict_lstm(context, lstm_model, word_to_idx, idx_to_word, top_k)
                render_predictions(predictions, "LSTM")
            else:
                st.error("LSTM model not loaded")
        
        with col2:
            st.markdown("### 📊 Trigram Statistical")
            if trigram_loaded:
                predictions = predict_trigram(context, trigram_model, top_k)
                render_predictions(predictions, "Trigram")
            else:
                st.error("Trigram model not loaded")
    
    elif "LSTM" in model_choice:
        st.markdown("### 🤖 LSTM Predictions")
        if lstm_loaded:
            predictions = predict_lstm(context, lstm_model, word_to_idx, idx_to_word, top_k)
            render_predictions(predictions, "LSTM")
        else:
            st.error("LSTM model not loaded")
    
    else:
        st.markdown("### 📊 Trigram Predictions")
        if trigram_loaded:
            predictions = predict_trigram(context, trigram_model, top_k)
            render_predictions(predictions, "Trigram")
        else:
            st.error("Trigram model not loaded")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Trained on <strong>NaijaSenti</strong> + <strong>BBC Pidgin</strong> corpus (~10k texts)</p>
    <p>🇳🇬 Nigerian Pidgin Language Model</p>
</div>
""", unsafe_allow_html=True)
