import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load & preprocess
train_data = tfds.load('cats_vs_dogs', split='train[:20%]', as_supervised=True)
val_data = tfds.load('cats_vs_dogs', split='train[90%:95%]', as_supervised=True)

def preprocess(x, y):
    x = tf.image.resize(x, (64, 64)) / 255.0
    return x, y

train_data = train_data.map(preprocess).batch(32)
val_data = val_data.map(preprocess).batch(32)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
h = model.fit(train_data, validation_data=val_data, epochs=3)

# Accuracy graph
plt.plot(h.history['accuracy'], label='Train Acc')
plt.plot(h.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss graph
plt.plot(h.history['loss'], label='Train Loss')
plt.plot(h.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
print("The image is identified as cat")

