"""
Streamlit frontend that calls the FastAPI backend.
Can run locally or connect to HF Spaces API.
"""

import streamlit as st
import requests
from typing import List, Dict, Optional

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Nigerian Pidgin Predictor",
    page_icon="💬",
    layout="wide"
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    
    .main-header {
        background: linear-gradient(135deg, #1a5f2a 0%, #2d8a3e 50%, #f4c430 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 { color: white !important; margin-bottom: 0.5rem; }
    
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
    
    .word { font-size: 1.3rem; font-weight: 600; color: #1a5f2a; }
    .prob { font-size: 1rem; color: #666; }
    
    .stButton > button {
        border-radius: 20px;
        border: 2px solid #2d8a3e;
        background: white;
        color: #2d8a3e;
        transition: all 0.3s;
    }
    
    .stButton > button:hover { background: #2d8a3e; color: white; }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# API Configuration
# =============================================================================
# When deployed on HF Spaces, the API runs on the same server
# When running locally, you can point to the HF Space URL
API_BASE_URL = st.sidebar.text_input(
    "API URL",
    value="https://jaykay73-nextword-pidgin-api.hf.space",
    help="URL of the prediction API"
)

# =============================================================================
# API Functions
# =============================================================================
def get_predictions(context: str, top_k: int = 5) -> Dict:
    """Call the API to get predictions from both models separately for robustness."""
    results = {"lstm": [], "trigram": []}
    
    # 1. Get LSTM predictions
    try:
        response = requests.get(
            f"{API_BASE_URL}/predict/lstm",
            params={"context": context, "top_k": top_k},
            timeout=15
        )
        if response.status_code == 200:
            results["lstm"] = response.json().get("predictions", [])
        else:
            results["lstm_error"] = f"Error {response.status_code}: {response.text}"
    except Exception as e:
        results["lstm_error"] = str(e)

    # 2. Get Trigram predictions
    try:
        response = requests.get(
            f"{API_BASE_URL}/predict/trigram",
            params={"context": context, "top_k": top_k},
            timeout=15
        )
        if response.status_code == 200:
            results["trigram"] = response.json().get("predictions", [])
        else:
            results["trigram_error"] = f"Error {response.status_code}: {response.text}"
    except Exception as e:
        results["trigram_error"] = str(e)
        
    return results

def check_api_health() -> Dict:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except:
        return {"status": "offline"}

# =============================================================================
# UI Components
# =============================================================================
# =============================================================================
# UI Components
# =============================================================================
def render_predictions(predictions: List[Dict], model_name: str):
    if not predictions:
        st.warning(f"No predictions from {model_name}")
        return
    
    for pred in predictions:
        word = pred.get("word", "?")
        prob = pred.get("probability", 0)
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

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Health
    health = check_api_health()
    if health.get("status") == "healthy":
        st.success("✅ API Connected")
        st.caption(f"LSTM: {'✓' if health.get('lstm_loaded') else '✗'} | Trigram: {'✓' if health.get('trigram_loaded') else '✗'}")
    else:
        st.error("❌ API Offline")
        st.caption("Start with: `uvicorn api:app`")
    
    st.markdown("---")
    
    model_choice = st.radio(
        "Display Mode:",
        ["⚔️ Compare Both", "🤖 LSTM Only", "📊 Trigram Only"],
        index=0
    )
    
    top_k = st.slider("Predictions:", 1, 10, 5)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **LSTM**: Neural network trained on ~10k Pidgin texts.
    
    **Trigram**: Statistical model using word co-occurrence.
    """)

# Main input
try:
    from st_keyup import st_keyup
except ImportError:
    st.error("Please install streamlit-keyup: pip install streamlit-keyup")
    def st_keyup(label, value="", key=None, **kwargs):
        return st.text_input(label, value=value, key=key)

st.markdown("### Enter Nigerian Pidgin text:")

# Handle example button clicks updates
if 'context_value' not in st.session_state:
    st.session_state['context_value'] = ""

def set_example(ex):
    st.session_state['context_value'] = ex

# Example buttons
st.markdown("**Try these examples:**")
example_cols = st.columns(5)
examples = ["i dey", "wetin you", "how far", "e don", "make we"]
for col, ex in zip(example_cols, examples):
    col.button(ex, use_container_width=True, on_click=set_example, args=(ex,))

# Input with real-time updates
context = st_keyup(
    label="Context",
    value=st.session_state['context_value'],
    key="keyup_context",
    label_visibility="collapsed",
    placeholder="e.g., 'i dey', 'wetin you', 'how far'"
)
# Sync keyup back to session if typed manually (optional, but good for consistency)
if context != st.session_state['context_value']:
    st.session_state['context_value'] = context

# Predictions
if context:
    st.markdown("---")
    
    with st.spinner("Getting predictions..."):
        result = get_predictions(context, top_k)
    
    if "lstm_error" in result:
        st.error(f"LSTM Error: {result['lstm_error']}")
        
    if "trigram_error" in result:
        st.error(f"Trigram Error: {result['trigram_error']}")

    lstm_preds = result.get("lstm", [])
    trigram_preds = result.get("trigram", [])
    
    if "Compare" in model_choice:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 LSTM Neural Network")
            render_predictions(lstm_preds, "LSTM")
        
        with col2:
            st.markdown("### 📊 Trigram Statistical")
            render_predictions(trigram_preds, "Trigram")
    
    elif "LSTM" in model_choice:
        st.markdown("### 🤖 LSTM Predictions")
        render_predictions(lstm_preds, "LSTM")
    
    else:
        st.markdown("### 📊 Trigram Predictions")
        render_predictions(trigram_preds, "Trigram")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🇳🇬 Nigerian Pidgin Language Model | Trained on NaijaSenti + BBC Pidgin</p>
</div>
""", unsafe_allow_html=True)
