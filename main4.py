# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load IMDB sentiment dataset
# Keep only the 10,000 most frequent words
vocab_size = 10000
max_length = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Step 2: Pad all sequences to the same length
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Step 3: Build the RNN model
model = tf.keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),  # Word embedding
    layers.SimpleRNN(32),                                                            # RNN layer
    layers.Dense(1, activation='sigmoid')                                            # Output: 0 (neg) or 1 (pos)
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Step 6: Evaluate on test data
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)
