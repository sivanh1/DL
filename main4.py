# Install required libraries
# pip install tensorflow numpy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = [
    "I love this movie!", "This is an amazing product", "Worst experience ever",
    "I have this", "Fantastic service", "So bad I want my money back"
]
labels = np.array([1, 1, 0, 1, 1, 0])

# Tokenization and padding
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, padding='post')
y = labels

# Build model
model = Sequential([
    Embedding(input_dim=1000, output_dim=64),
    SimpleRNN(64, activation='tanh'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x, y, epochs=5, batch_size=2)

# Evaluate model
loss, accuracy = model.evaluate(x, y)
print(f"\nTest accuracy on training data: {accuracy:.2f}")

# Test predictions
test_texts = ["I really enjoyed this", "Worst experience ever"]
test_seq = tokenizer.texts_to_sequences(test_texts)
test_x = pad_sequences(test_seq, padding='post')
predictions = model.predict(test_x)

# Show predictions
for text, pred in zip(test_texts, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"{text} => Sentiment: {sentiment} (Confidence: {pred[0]:.2f})")