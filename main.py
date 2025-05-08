# Install the necessary libraries (run these in the terminal or notebook if not installed)
# pip install tensorflow
# pip install numpy==1.26.0
# pip install matplotlib

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST data
(x_train, y_train), (_, _) = mnist.load_data()

# Plotting the first 4 images
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()