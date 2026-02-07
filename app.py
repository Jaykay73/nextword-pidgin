"""
Gradio app for Nigerian Pidgin Next-Word Prediction.
Deploy to Hugging Face Spaces.
"""

import gradio as gr
import torch
import torch.nn as nn
import re
from typing import List, Dict, Tuple

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'


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


# Load model
print("Loading model...")
checkpoint = torch.load('model/lstm_pidgin_model.pt', map_location='cpu')
word_to_idx = checkpoint['word_to_idx']
idx_to_word = checkpoint['idx_to_word']
vocab_size = checkpoint['vocab_size']

model = LSTMLanguageModel(vocab_size=vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded! Vocab size: {vocab_size:,}")


def predict_next_words(context: str, top_k: int = 5) -> str:
    """Predict next words given context."""
    if not context.strip():
        return "Please enter some text..."
    
    # Tokenize and convert to indices
    tokens = tokenize(clean_text(context))
    if not tokens:
        return "No valid tokens found in input."
    
    unk_idx = word_to_idx.get(UNK_TOKEN, 1)
    indices = [word_to_idx.get(t, unk_idx) for t in tokens]
    
    # Create input tensor
    x = torch.tensor([indices], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs[0], top_k)
    
    results = []
    for prob, idx in zip(top_probs.numpy(), top_indices.numpy()):
        word = idx_to_word.get(str(idx), idx_to_word.get(idx, UNK_TOKEN))
        if word not in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            results.append(f"**{word}** ({prob:.1%})")
    
    return "\n".join(results) if results else "No predictions available."


# Gradio Interface
demo = gr.Interface(
    fn=predict_next_words,
    inputs=[
        gr.Textbox(
            label="Enter Nigerian Pidgin text",
            placeholder="e.g., 'i dey', 'wetin you', 'how far'",
            lines=2
        ),
        gr.Slider(
            minimum=1, maximum=10, value=5, step=1,
            label="Number of predictions"
        )
    ],
    outputs=gr.Markdown(label="Predicted next words"),
    title="🇳🇬 Nigerian Pidgin Next-Word Predictor",
    description="""
    **LSTM Language Model** trained on Nigerian Pidgin text.
    
    Enter some Pidgin text and get predictions for the next word!
    
    Try: "i dey", "wetin you", "na the", "how far", "e don"
    """,
    examples=[
        ["i dey", 5],
        ["wetin you", 5],
        ["how far", 5],
        ["na the", 5],
        ["e don", 5],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
