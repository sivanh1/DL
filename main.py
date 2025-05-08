# Step 1: Import Libraries
import tensorflow as tf                      # TensorFlow for machine learning
from tensorflow.keras import layers, models  # Import Keras layers and model APIs
import matplotlib.pyplot as plt              # For visualizing predictions

# Step 2: Load and Preprocess MNIST Dataset
mnist = tf.keras.datasets.mnist              # Load the built-in MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Split into training and test sets

# Normalize pixel values to [0, 1] for faster training and better performance
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 3: Build the Neural Network Model
model = models.Sequential([                            # Create a sequential model
    layers.Flatten(input_shape=(28, 28)),              # Flatten 28x28 images to 784-element vectors
    layers.Dense(128, activation='relu'),              # Fully connected hidden layer with 128 units and ReLU activation
    layers.Dropout(0.2),                               # Dropout layer to prevent overfitting (drops 20% of nodes randomly)
    layers.Dense(10, activation='softmax')             # Output layer: 10 neurons (one per digit), softmax for probabilities
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',                                  # Use Adam optimizer (adaptive learning rate)
    loss='sparse_categorical_crossentropy',            # Suitable for integer labels (0–9)
    metrics=['accuracy']                               # Track accuracy during training and testing
)

# Step 5: Train the Model
model.fit(
    x_train, y_train,                                  # Input and labels
    epochs=5,                                          # Number of training passes over the data
    validation_split=0.1                               # 10% of training data is used for validation
)

# Step 6: Evaluate on Test Data
test_loss, test_acc = model.evaluate(x_test, y_test)   # Evaluate on unseen test data
print('\nTest accuracy:', test_acc)                    # Print the final test accuracy

# Optional: Step 7 - Plot Predictions
predictions = model.predict(x_test)                    # Predict class probabilities for each test image

# Define a function to display a prediction
def plot_image(i, predictions_array, true_label, img):
    plt.grid(False)                                    # Remove grid lines
    plt.xticks([])                                     # Remove x-axis ticks
    plt.yticks([])                                     # Remove y-axis ticks
    plt.imshow(img, cmap=plt.cm.binary)                # Show image in grayscale

    predicted_label = tf.argmax(predictions_array)     # Get predicted digit (highest probability)
    color = 'blue' if predicted_label == true_label else 'red'  # Blue if correct, red if wrong

    plt.xlabel(f"Pred: {predicted_label} (True: {true_label})", color=color)  # Add label with prediction

# Display the first 10 test images with predictions
plt.figure(figsize=(10, 5))                            # Set figure size
for i in range(10):
    plt.subplot(2, 5, i + 1)                           # 2 rows × 5 columns of subplots
    plot_image(i, predictions[i], y_test[i], x_test[i])  # Show image and prediction
plt.tight_layout()                                     # Adjust layout to avoid overlap
plt.show()                                             # Display the full plot
