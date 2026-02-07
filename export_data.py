"""
Export combined training data to CSV.
Combines NaijaSenti PCM + BBC Pidgin corpus.
"""

import csv
import os
from src.data_loader import load_all_texts

def export_to_csv(output_path: str = "data/combined_corpus.csv"):
    """Export all training data to CSV."""
    
    # Create data directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Loading all data...")
    texts = load_all_texts(include_bbc=True)
    
    print(f"Writing {len(texts):,} texts to {output_path}...")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'source'])
        
        # NaijaSenti texts come first
        naija_count = 9275  # Known count from earlier
        for i, text in enumerate(texts):
            source = 'naijasenti' if i < naija_count else 'bbc_pidgin'
            writer.writerow([i, text, source])
    
    print(f"Done! Saved to {output_path}")
    
    # Show file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_to_csv()
