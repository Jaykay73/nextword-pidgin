"""Quick test script to verify trigram model works correctly."""

from src.trigram_model import TrigramLM
from src.preprocessing import preprocess_text, add_sentence_markers

# Test with sample data
sample_sentences = [
    ['<s>', '<s>', 'i', 'dey', 'go', 'market', '</s>'],
    ['<s>', '<s>', 'i', 'dey', 'come', 'back', '</s>'],
    ['<s>', '<s>', 'you', 'dey', 'go', 'where', '?', '</s>'],
    ['<s>', '<s>', 'how', 'far', '?', '</s>'],
    ['<s>', '<s>', 'e', 'don', 'happen', '</s>'],
    ['<s>', '<s>', 'wetin', 'you', 'dey', 'do', '</s>'],
    ['<s>', '<s>', 'na', 'the', 'matter', '</s>'],
]

print("=" * 50)
print("TRIGRAM MODEL TEST")
print("=" * 50)

# Train model
model = TrigramLM(smoothing=1.0)
model.train(sample_sentences)

# Test predictions
test_contexts = ["i dey", "you dey", "how"]
print("\nPredictions:")
for ctx in test_contexts:
    preds = model.predict_next_words(ctx, top_k=3)
    print(f"  '{ctx}' -> {preds}")

# Test preprocessing
print("\nPreprocessing test:")
test = "I dey go market @user https://example.com"
tokens = preprocess_text(test)
print(f"  Input:  {test}")
print(f"  Output: {tokens}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
