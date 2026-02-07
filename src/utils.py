"""
Utility functions for the next-word prediction system.
"""

from typing import List, Tuple
import math


def format_predictions(predictions: List[Tuple[str, float]], show_percent: bool = True) -> str:
    """
    Format prediction results for display.
    
    Args:
        predictions: List of (word, probability) tuples.
        show_percent: If True, show as percentage.
        
    Returns:
        Formatted string.
    """
    lines = []
    for word, prob in predictions:
        if show_percent:
            lines.append(f"  {word}: {prob*100:.2f}%")
        else:
            lines.append(f"  {word}: {prob:.6f}")
    return "\n".join(lines)


def calculate_entropy(probabilities: List[float]) -> float:
    """
    Calculate entropy of a probability distribution.
    
    H(X) = -sum(p * log2(p))
    
    Args:
        probabilities: List of probabilities.
        
    Returns:
        Entropy in bits.
    """
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def top_k_accuracy(
    model, 
    test_sentences: List[List[str]], 
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy on test data.
    
    Measures what fraction of true next words appear in top-k predictions.
    
    Args:
        model: Trained TrigramLM instance.
        test_sentences: List of tokenized sentences with markers.
        k: Number of top predictions to consider.
        
    Returns:
        Accuracy as fraction between 0 and 1.
    """
    correct = 0
    total = 0
    
    for sentence in test_sentences:
        if len(sentence) < 3:
            continue
        
        for i in range(2, len(sentence)):
            w1, w2 = sentence[i-2], sentence[i-1]
            true_word = sentence[i]
            
            # Get top-k predictions
            preds = model.get_context_distribution(w1, w2, top_k=k)
            pred_words = [w for w, _ in preds]
            
            if true_word in pred_words:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0
