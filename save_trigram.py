"""
Save the trained trigram model for use in the Streamlit app.
"""

import pickle
import os
from src.data_loader import load_all_texts
from src.preprocessing import preprocess_corpus
from src.trigram_model import TrigramLM

def save_trigram_model():
    print("Loading data...")
    texts = load_all_texts(include_bbc=True)
    
    print("Preprocessing...")
    sentences = preprocess_corpus(texts)
    
    print("Training trigram model...")
    model = TrigramLM(smoothing=1.0)
    model.train(sentences)
    
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    print("Saving model...")
    with open('model/trigram_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Done! Saved to model/trigram_model.pkl")
    
    # Test predictions
    print("\nTest predictions:")
    for ctx in ["i dey", "wetin you", "how far"]:
        preds = model.predict_next_words(ctx, top_k=3)
        print(f"  '{ctx}' -> {preds}")

if __name__ == "__main__":
    save_trigram_model()
