"""
Trigram Language Model for Next-Word Prediction.

Implements a statistical trigram model with Laplace (add-one) smoothing
for Nigerian English/Pidgin next-word prediction.

Mathematical Foundation:
    P(w_n | w_{n-2}, w_{n-1}) = (C(w_{n-2}, w_{n-1}, w_n) + α) / (C(w_{n-2}, w_{n-1}) + α|V|)
    
Where:
    - C(.) = count of n-gram in training corpus
    - α = smoothing parameter (1.0 for Laplace)
    - |V| = vocabulary size
"""

from collections import Counter
from typing import List, Tuple, Dict, Optional
import math


class TrigramLM:
    """
    Trigram Language Model with Laplace smoothing.
    
    Attributes:
        smoothing: Smoothing parameter (α). Default 1.0 for add-one smoothing.
        unigram_counts: Counter for single word frequencies.
        bigram_counts: Counter for word pair frequencies.
        trigram_counts: Counter for word triple frequencies.
        vocab: Set of all unique words in training corpus.
    """
    
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the trigram model.
        
        Args:
            smoothing: Laplace smoothing parameter. Higher values provide more
                      smoothing for unseen n-grams. Default 1.0 (add-one).
        """
        self.smoothing = smoothing
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()
        self.vocab: set = set()
        self._total_unigrams: int = 0
    
    def train(self, sentences: List[List[str]]) -> None:
        """
        Train the model by counting n-grams from tokenized sentences.
        
        Expects sentences with start/end markers already added:
        ['<s>', '<s>', 'word1', 'word2', ..., '</s>']
        
        Args:
            sentences: List of tokenized sentences with boundary markers.
        """
        for sentence in sentences:
            # Build vocabulary
            self.vocab.update(sentence)
            
            # Count unigrams
            for token in sentence:
                self.unigram_counts[token] += 1
                self._total_unigrams += 1
            
            # Count bigrams
            for i in range(len(sentence) - 1):
                bigram = (sentence[i], sentence[i + 1])
                self.bigram_counts[bigram] += 1
            
            # Count trigrams
            for i in range(len(sentence) - 2):
                trigram = (sentence[i], sentence[i + 1], sentence[i + 2])
                self.trigram_counts[trigram] += 1
        
        print(f"Training complete:")
        print(f"  Vocabulary size: {len(self.vocab):,}")
        print(f"  Unique unigrams: {len(self.unigram_counts):,}")
        print(f"  Unique bigrams: {len(self.bigram_counts):,}")
        print(f"  Unique trigrams: {len(self.trigram_counts):,}")
    
    def probability(self, w3: str, w1: str, w2: str) -> float:
        """
        Compute P(w3 | w1, w2) with Laplace smoothing.
        
        Formula:
            P(w3|w1,w2) = (C(w1,w2,w3) + α) / (C(w1,w2) + α|V|)
        
        Args:
            w3: The word to predict.
            w1: First context word (two positions before w3).
            w2: Second context word (one position before w3).
            
        Returns:
            Conditional probability P(w3 | w1, w2).
        """
        trigram_count = self.trigram_counts.get((w1, w2, w3), 0)
        bigram_count = self.bigram_counts.get((w1, w2), 0)
        vocab_size = len(self.vocab)
        
        # Laplace smoothing
        numerator = trigram_count + self.smoothing
        denominator = bigram_count + (self.smoothing * vocab_size)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def log_probability(self, w3: str, w1: str, w2: str) -> float:
        """
        Compute log P(w3 | w1, w2) for numerical stability.
        
        Args:
            w3: The word to predict.
            w1: First context word.
            w2: Second context word.
            
        Returns:
            Log probability.
        """
        prob = self.probability(w3, w1, w2)
        return math.log(prob) if prob > 0 else float('-inf')
    
    def predict_next_words(
        self, 
        context: str, 
        top_k: int = 5,
        exclude_special: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Predict the top-k most likely next words given a context.
        
        Args:
            context: Input text (will use last two words as context).
            top_k: Number of predictions to return.
            exclude_special: If True, exclude <s> and </s> from predictions.
            
        Returns:
            List of (word, probability) tuples, sorted by probability descending.
        """
        # Tokenize and extract last two words
        words = context.lower().split()
        
        if len(words) == 0:
            w1, w2 = '<s>', '<s>'
        elif len(words) == 1:
            w1, w2 = '<s>', words[0]
        else:
            w1, w2 = words[-2], words[-1]
        
        # Compute probability for each word in vocabulary
        candidates = []
        for word in self.vocab:
            if exclude_special and word in ('<s>', '</s>'):
                continue
            prob = self.probability(word, w1, w2)
            candidates.append((word, prob))
        
        # Sort by probability descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]
    
    def sentence_probability(self, tokens: List[str]) -> float:
        """
        Compute the probability of a sentence.
        
        Args:
            tokens: Tokenized sentence WITH start/end markers.
            
        Returns:
            Log probability of the sentence.
        """
        if len(tokens) < 3:
            return float('-inf')
        
        log_prob = 0.0
        for i in range(2, len(tokens)):
            log_prob += self.log_probability(tokens[i], tokens[i-2], tokens[i-1])
        
        return log_prob
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """
        Compute perplexity on a set of sentences.
        
        Perplexity = exp(-1/N * sum(log P(w_i | w_{i-2}, w_{i-1})))
        
        Lower perplexity = better model fit.
        
        Args:
            sentences: List of tokenized sentences with boundary markers.
            
        Returns:
            Perplexity score.
        """
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in sentences:
            if len(sentence) < 3:
                continue
            for i in range(2, len(sentence)):
                total_log_prob += self.log_probability(
                    sentence[i], sentence[i-2], sentence[i-1]
                )
                total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_words
        return math.exp(-avg_log_prob)
    
    def get_context_distribution(
        self, 
        w1: str, 
        w2: str, 
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get the probability distribution for a specific bigram context.
        
        Args:
            w1: First context word.
            w2: Second context word.
            top_k: If provided, return only top-k predictions.
            
        Returns:
            List of (word, probability) tuples.
        """
        candidates = []
        for word in self.vocab:
            if word not in ('<s>', '</s>'):
                prob = self.probability(word, w1, w2)
                candidates.append((word, prob))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            return candidates[:top_k]
        return candidates
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get model statistics.
        
        Returns:
            Dictionary of statistics.
        """
        return {
            'vocab_size': len(self.vocab),
            'unique_unigrams': len(self.unigram_counts),
            'unique_bigrams': len(self.bigram_counts),
            'unique_trigrams': len(self.trigram_counts),
            'total_tokens': self._total_unigrams,
        }


if __name__ == "__main__":
    # Quick test with sample data
    sample_sentences = [
        ['<s>', '<s>', 'i', 'dey', 'go', 'market', '</s>'],
        ['<s>', '<s>', 'i', 'dey', 'come', 'back', '</s>'],
        ['<s>', '<s>', 'you', 'dey', 'go', 'where', '?', '</s>'],
        ['<s>', '<s>', 'how', 'far', '?', '</s>'],
        ['<s>', '<s>', 'e', 'don', 'happen', '</s>'],
    ]
    
    model = TrigramLM(smoothing=1.0)
    model.train(sample_sentences)
    
    print("\nTest Predictions:")
    contexts = ["i dey", "you dey", "how"]
    for ctx in contexts:
        preds = model.predict_next_words(ctx, top_k=3)
        print(f"  '{ctx}' -> {preds}")
