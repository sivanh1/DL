# Step 1: Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 2: Preprocess the images (rescale pixel values to [0, 1])
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

# Step 3: Load the training and validation images
train_data = train_gen.flow_from_directory(
    'data/train',               # Folder should contain 'cats' and 'dogs' subfolders
    target_size=(150, 150),     # Resize all images to 150x150
    batch_size=20,              # Process 20 images at a time
    class_mode='binary'         # 0 = cat, 1 = dog
)

val_data = val_gen.flow_from_directory(
    'data/validation',          # Validation folder with same structure
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Step 4: Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# Step 5: Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Step 6: Train the model
model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)
