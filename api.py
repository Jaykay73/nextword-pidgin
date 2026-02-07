"""
FastAPI backend for Nigerian Pidgin Next-Word Prediction.
Serves both LSTM and Trigram models as REST API.
Deploy to Hugging Face Spaces with Docker SDK.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import pickle
import re
import os

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Nigerian Pidgin Next-Word Predictor API",
    description="LSTM + Trigram models for Nigerian Pidgin next-word prediction",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}
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
# Global Models (loaded once at startup)
# =============================================================================
lstm_model = None
word_to_idx = None
idx_to_word = None
trigram_model = None

# =============================================================================
# Custom Unpickler to fix the 'src' module error
# =============================================================================
class PatchingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If the pickle creates a dependency on 'src', redirect it to __main__
        if module.startswith("src") and name == "TrigramLM":
            return TrigramLM
        return super().find_class(module, name)

@app.on_event("startup")
async def load_models():
    global lstm_model, word_to_idx, idx_to_word, trigram_model
    
    # 1. Load LSTM
    try:
        checkpoint = torch.load('model/lstm_pidgin_model.pt', map_location='cpu')
        word_to_idx = checkpoint['word_to_idx']
        idx_to_word = checkpoint['idx_to_word']
        vocab_size = checkpoint['vocab_size']
        
        lstm_model = LSTMLanguageModel(vocab_size=vocab_size)
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.eval()
        print(f"LSTM model loaded! Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load LSTM model: {e}")

    # 2. Load Trigram (Using the Custom Unpickler)
    try:
        with open('model/trigram_model.pkl', 'rb') as f:
            # Use PatchingUnpickler instead of standard pickle.load
            trigram_model = PatchingUnpickler(f).load()
        print(f"Trigram model loaded! Vocab size: {len(trigram_model.vocab)}")
    except Exception as e:
        print(f"Failed to load Trigram model: {e}")

# =============================================================================
# Request/Response Models
# =============================================================================
class PredictionRequest(BaseModel):
    context: str
    top_k: int = 5
    model: str = "lstm"  # "lstm", "trigram", or "both"

class Prediction(BaseModel):
    word: str
    probability: float

class PredictionResponse(BaseModel):
    context: str
    model: str
    predictions: List[Prediction]

class BothModelsResponse(BaseModel):
    context: str
    lstm: List[Prediction]
    trigram: List[Prediction]

# =============================================================================
# Prediction Functions
# =============================================================================
def predict_lstm(context: str, top_k: int = 5) -> List[Prediction]:
    if lstm_model is None or not context.strip():
        return []
    
    tokens = tokenize(clean_text(context))
    if not tokens:
        return []
    
    unk_idx = word_to_idx.get(UNK_TOKEN, 1)
    indices = [word_to_idx.get(t, unk_idx) for t in tokens]
    x = torch.tensor([indices], dtype=torch.long)
    
    with torch.no_grad():
        logits = lstm_model(x)
        probs = torch.softmax(logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs[0], top_k + 5)
    
    results = []
    for prob, idx in zip(top_probs.numpy(), top_indices.numpy()):
        word = idx_to_word.get(str(idx), idx_to_word.get(idx, UNK_TOKEN))
        if word not in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            results.append(Prediction(word=word, probability=float(prob)))
        if len(results) >= top_k:
            break
    
    return results

def predict_trigram(context: str, top_k: int = 5) -> List[Prediction]:
    if trigram_model is None or not context.strip():
        return []
    
    preds = trigram_model.predict_next_words(context, top_k)
    return [Prediction(word=w, probability=p) for w, p in preds]

# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/")
async def root():
    return {
        "message": "Nigerian Pidgin Next-Word Predictor API",
        "endpoints": {
            "/predict": "POST - Get predictions",
            "/predict/lstm": "GET - LSTM predictions",
            "/predict/trigram": "GET - Trigram predictions",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "lstm_loaded": lstm_model is not None,
        "trigram_loaded": trigram_model is not None
    }

@app.post("/predict", response_model=BothModelsResponse)
async def predict(request: PredictionRequest):
    """Get predictions from both models."""
    return BothModelsResponse(
        context=request.context,
        lstm=predict_lstm(request.context, request.top_k),
        trigram=predict_trigram(request.context, request.top_k)
    )

@app.get("/predict/lstm")
async def predict_lstm_endpoint(context: str, top_k: int = 5):
    """Get LSTM predictions."""
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")
    
    predictions = predict_lstm(context, top_k)
    return PredictionResponse(
        context=context,
        model="lstm",
        predictions=predictions
    )

@app.get("/predict/trigram")
async def predict_trigram_endpoint(context: str, top_k: int = 5):
    """Get Trigram predictions."""
    if trigram_model is None:
        raise HTTPException(status_code=503, detail="Trigram model not loaded")
    
    predictions = predict_trigram(context, top_k)
    return PredictionResponse(
        context=context,
        model="trigram",
        predictions=predictions
    )

# =============================================================================
# Run with: uvicorn api:app --reload
# =============================================================================
