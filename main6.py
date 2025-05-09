import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load data
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, -1)

# Generator
g = tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*64, input_shape=(100,)),
    tf.keras.layers.Reshape((7, 7, 64)),
    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2D(1, 3, padding="same", activation="tanh"),
    tf.keras.layers.UpSampling2D()
])

# Train 1 step (quick & dirty)
noise = np.random.normal(0, 1, (1, 100))
img = g(noise, training=False).numpy()[0, :, :, 0]
plt.imshow((img + 1) / 2, cmap='gray')
plt.axis('off')
plt.show()
