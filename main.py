#!/usr/bin/env python3
"""
Next-Word Prediction System: Trigram Baseline

This script demonstrates the complete pipeline for training and using a trigram
language model for Nigerian English/Pidgin next-word prediction.

Usage:
    python main.py
"""

from src.data_loader import load_all_texts, get_dataset_stats
from src.preprocessing import preprocess_corpus
from src.trigram_model import TrigramLM
from src.utils import format_predictions, top_k_accuracy


def main():
    """Main entry point for the trigram language model demo."""
    
    print("=" * 70)
    print("Nigerian English/Pidgin Next-Word Prediction System")
    print("Trigram Language Model Baseline")
    print("=" * 70)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/4] Loading NaijaSenti PCM Dataset...")
    print("-" * 40)
    
    texts = load_all_texts()
    stats = get_dataset_stats(texts)
    
    print(f"\nDataset Statistics:")
    print(f"  Total texts: {stats['num_texts']:,}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Avg words/text: {stats['avg_words_per_text']:.1f}")
    
    # =========================================================================
    # 2. Preprocess
    # =========================================================================
    print("\n[2/4] Preprocessing Corpus...")
    print("-" * 40)
    
    sentences = preprocess_corpus(texts)
    print(f"  Processed sentences: {len(sentences):,}")
    
    # Show sample
    print("\n  Sample preprocessed sentences:")
    for i, sent in enumerate(sentences[:3]):
        display = ' '.join(sent[:10])
        if len(sent) > 10:
            display += ' ...'
        # Handle Unicode safely for Windows console
        try:
            print(f"    {i+1}. {display}")
        except UnicodeEncodeError:
            print(f"    {i+1}. {display.encode('ascii', 'replace').decode('ascii')}")
    
    # =========================================================================
    # 3. Train Model
    # =========================================================================
    print("\n[3/4] Training Trigram Model...")
    print("-" * 40)
    
    model = TrigramLM(smoothing=1.0)
    model.train(sentences)
    
    model_stats = model.get_stats()
    print(f"\nModel Statistics:")
    print(f"  Vocabulary size: {model_stats['vocab_size']:,}")
    print(f"  Unique trigrams: {model_stats['unique_trigrams']:,}")
    
    # =========================================================================
    # 4. Demonstrate Predictions
    # =========================================================================
    print("\n[4/4] Example Predictions")
    print("-" * 40)
    
    # Nigerian Pidgin test contexts
    test_contexts = [
        "i dey",           # Common Pidgin phrase
        "wetin you",       # "What are you..."
        "na the",          # "It's the..."
        "how far",         # "How's it going"
        "e don",           # "It has..."
        "you no",          # "You don't..."
        "make we",         # "Let us..."
        "dem dey",         # "They are..."
    ]
    
    print("\nTop-5 predictions for Nigerian Pidgin contexts:\n")
    for context in test_contexts:
        predictions = model.predict_next_words(context, top_k=5)
        print(f"Context: \"{context}\"")
        print(format_predictions(predictions))
        print()
    
    # =========================================================================
    # 5. Evaluation Notes
    # =========================================================================
    print("=" * 70)
    print("EVALUATION NOTES")
    print("=" * 70)
    
    # Compute perplexity on training data (for demonstration)
    # In practice, use held-out test set
    sample_perplexity = model.perplexity(sentences[:1000])
    
    print(f"""
Perplexity (sample): {sample_perplexity:.2f}

About Perplexity:
  - Measures how well the model predicts held-out data
  - Lower = better predictive performance
  - Baseline for comparison with neural models (LSTM, Transformer)

Trigram Model Limitations:
  1. Fixed 2-word context window
     - Cannot capture: "The man who went to the store bought..."
     - Loses long-range dependencies
  
  2. Data sparsity
     - Many valid trigrams never seen in training
     - Laplace smoothing helps but doesn't solve underlying issue
  
  3. No semantic understanding
     - Purely statistical co-occurrence
     - Can't understand meaning or paraphrase

Comparison Framework:
  | Model      | Context   | Expected Perplexity | Inference |
  |------------|-----------|---------------------|-----------|
  | Trigram    | 2 words   | {sample_perplexity:.0f}+ | Fast |
  | LSTM       | Variable  | Lower               | Medium    |
  | Transformer| Full seq  | Lowest              | Slower    |
""")
    
    print("=" * 70)
    print("Model ready for use. Import and call model.predict_next_words()")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    model = main()
