"""
Data loading utilities for NaijaSenti and BBC Pidgin datasets.

Loads Nigerian Pidgin text from multiple sources for language modeling.
Sentiment/category labels are ignored.
"""

from datasets import load_dataset
from typing import List, Dict, Any, Optional
import csv
import os

# Path to BBC Pidgin corpus (relative to project root)
BBC_PIDGIN_CORPUS_PATH = "bbc_pidgin_scraper/data/pidgin_corpus.csv"


def load_naijasenti_pcm() -> Dict[str, List[str]]:
    """
    Load the NaijaSenti PCM (Nigerian Pidgin) dataset.
    
    Returns:
        Dict with keys 'train', 'test', 'validation' containing text lists.
    """
    dataset = load_dataset("mteb/NaijaSenti", "pcm")
    
    result = {}
    for split in dataset.keys():
        # Extract text field, ignore sentiment labels
        result[split] = [example['text'] for example in dataset[split]]
    
    return result


def load_bbc_pidgin(limit: Optional[int] = None, project_root: Optional[str] = None) -> List[str]:
    """
    Load BBC Pidgin articles from the scraped corpus.
    
    The corpus contains headlines and article texts scraped from BBC Pidgin.
    We concatenate headline + text for each article.
    
    Args:
        limit: Maximum number of articles to load. None for all.
        project_root: Path to project root. Defaults to current working directory.
        
    Returns:
        List of article texts (headline + body combined).
    """
    if project_root is None:
        project_root = os.getcwd()
    
    corpus_path = os.path.join(project_root, BBC_PIDGIN_CORPUS_PATH)
    
    if not os.path.exists(corpus_path):
        print(f"Warning: BBC Pidgin corpus not found at {corpus_path}")
        return []
    
    texts = []
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                # Combine headline and text
                headline = row.get('headline', '').strip()
                text = row.get('text', '').strip()
                if headline and text:
                    combined = f"{headline}. {text}"
                    texts.append(combined)
                elif text:
                    texts.append(text)
    except Exception as e:
        print(f"Error loading BBC Pidgin corpus: {e}")
        return []
    
    print(f"Loaded {len(texts):,} BBC Pidgin articles")
    return texts


def load_all_texts(include_bbc: bool = True, bbc_limit: Optional[int] = None) -> List[str]:
    """
    Load all text from all sources combined.
    
    Combines NaijaSenti PCM dataset with BBC Pidgin articles
    for maximum data coverage.
    
    Args:
        include_bbc: Whether to include BBC Pidgin articles.
        bbc_limit: Maximum number of BBC articles to include.
        
    Returns:
        List of all text strings from all sources.
    """
    all_texts = []
    
    # Load NaijaSenti
    print("Loading NaijaSenti PCM dataset...")
    splits = load_naijasenti_pcm()
    for split_name, texts in splits.items():
        all_texts.extend(texts)
        print(f"  Loaded {len(texts):,} texts from {split_name} split")
    
    naija_total = len(all_texts)
    print(f"  NaijaSenti total: {naija_total:,} texts")
    
    # Load BBC Pidgin
    if include_bbc:
        print(f"\nLoading BBC Pidgin corpus (limit={bbc_limit})...")
        bbc_texts = load_bbc_pidgin(limit=bbc_limit)
        all_texts.extend(bbc_texts)
    
    print(f"\nCombined total: {len(all_texts):,} texts")
    return all_texts


def get_dataset_stats(texts: List[str]) -> Dict[str, Any]:
    """
    Compute basic statistics about the dataset.
    
    Args:
        texts: List of text strings.
        
    Returns:
        Dictionary of statistics.
    """
    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)
    
    return {
        'num_texts': len(texts),
        'total_characters': total_chars,
        'total_words': total_words,
        'avg_words_per_text': total_words / len(texts) if texts else 0,
        'avg_chars_per_text': total_chars / len(texts) if texts else 0,
    }


if __name__ == "__main__":
    # Quick test
    texts = load_all_texts(include_bbc=True)  # Loads all BBC articles by default
    stats = get_dataset_stats(texts)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:,}")

