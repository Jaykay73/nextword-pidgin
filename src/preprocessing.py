"""
Text preprocessing pipeline for Nigerian English/Pidgin.

Design principles:
- Preserve linguistic features of Nigerian Pidgin (slang, contractions, code-switching)
- Remove noise (URLs, usernames) that don't contribute to language modeling
- Minimal normalization to avoid losing dialectal patterns
"""

import re
from typing import List


# Special tokens for sentence boundaries
START_TOKEN = "<s>"
END_TOKEN = "</s>"


def clean_text(text: str) -> str:
    """
    Clean text while preserving Nigerian Pidgin features.
    
    Operations:
    1. Lowercase (case doesn't matter for prediction)
    2. Remove URLs
    3. Remove @usernames (Twitter-style)
    4. Normalize whitespace
    
    Preserved:
    - Contractions (don't, I'm, na'm)
    - Slang (abi, sha, sef)
    - Code-switching patterns
    - Pidgin grammar structures
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned text string.
    """
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove @usernames
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """
    Word-level tokenization for Nigerian Pidgin.
    
    Handles:
    - Standard word boundaries
    - Punctuation as separate tokens
    - Preserves contractions as single tokens
    
    Args:
        text: Cleaned text string.
        
    Returns:
        List of tokens.
    """
    # Split on whitespace first
    words = text.split()
    
    tokens = []
    for word in words:
        # Handle punctuation attached to words
        # Keep contractions together (don't, I'm, etc.)
        
        # Strip leading punctuation
        while word and word[0] in '.,!?;:"\'-([{':
            if word[0] not in "'":  # Keep leading apostrophe for contractions
                tokens.append(word[0])
            word = word[1:]
        
        # Strip trailing punctuation
        trailing = []
        while word and word[-1] in '.,!?;:"\'-)]}"':
            if word[-1] not in "'":  # Keep trailing apostrophe for contractions
                trailing.insert(0, word[-1])
            word = word[:-1]
        
        if word:
            tokens.append(word)
        
        tokens.extend(trailing)
    
    return tokens


def preprocess_text(text: str) -> List[str]:
    """
    Full preprocessing pipeline: clean + tokenize.
    
    Args:
        text: Raw text string.
        
    Returns:
        List of tokens.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return tokens


def add_sentence_markers(tokens: List[str]) -> List[str]:
    """
    Add start/end markers for sentence boundary modeling.
    
    For trigram models, we need context at sentence boundaries.
    We add two start tokens to provide full context for the first word.
    
    Args:
        tokens: List of tokens from a sentence.
        
    Returns:
        Tokens with boundary markers.
    """
    if not tokens:
        return []
    return [START_TOKEN, START_TOKEN] + tokens + [END_TOKEN]


def preprocess_corpus(texts: List[str]) -> List[List[str]]:
    """
    Preprocess entire corpus for language model training.
    
    Args:
        texts: List of raw text strings.
        
    Returns:
        List of tokenized sentences with boundary markers.
    """
    processed = []
    for text in texts:
        tokens = preprocess_text(text)
        if tokens:  # Skip empty results
            marked = add_sentence_markers(tokens)
            processed.append(marked)
    return processed


if __name__ == "__main__":
    # Test preprocessing on Nigerian Pidgin examples
    test_texts = [
        "I dey go market, you wan follow?",
        "That guy na correct person sha @handle https://example.com",
        "Wetin you dey do? Abi you no sabi?",
        "E don happen before, no be today matter",
        "How far? Everything dey go well?",
    ]
    
    print("Preprocessing Examples:\n")
    for text in test_texts:
        tokens = preprocess_text(text)
        marked = add_sentence_markers(tokens)
        print(f"Original: {text}")
        print(f"Tokens:   {tokens}")
        print(f"Marked:   {marked}")
        print()
