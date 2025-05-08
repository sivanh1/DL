# Simple GAN to Generate Handwritten Digits

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(256)

# Generator model: turns noise into an image
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(), layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(),
        layers.Conv2DTranspose(1, 5, strides=2, padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Discriminator model: tells real from fake images
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=[28,28,1]),
        layers.LeakyReLU(), layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(), layers.Dropout(0.3),
        layers.Flatten(), layers.Dense(1)
    ])
    return model

# Losses and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator = build_generator()
discriminator = build_discriminator()
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([256, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_imgs = generator(noise, training=True)
        real_out = discriminator(images, training=True)
        fake_out = discriminator(gen_imgs, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_out), fake_out)
        disc_loss = cross_entropy(tf.ones_like(real_out), real_out) + \
                    cross_entropy(tf.zeros_like(fake_out), fake_out)
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

# Generate and show images
def generate_images(model, test_input):
    preds = model(test_input, training=False)
    plt.figure(figsize=(4,4))
    for i in range(preds.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((preds[i, :, :, 0] + 1) / 2.0, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Train loop
def train(dataset, epochs):
    seed = tf.random.normal([16, 100])
    for epoch in range(epochs):
        for batch in dataset:
            train_step(batch)
        print(f'Epoch {epoch+1} done')
        generate_images(generator, seed)

# Run training
train(train_ds, epochs=3)  # You can increase epochs for better results
