# Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Preprocess image data
# Rescale pixel values to [0, 1]
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

# Load training images from folders
train_data = train_gen.flow_from_directory(
    'data/train',               # Folder with class subfolders (e.g., cats, dogs)
    target_size=(160, 160),     # Resize images
    batch_size=32,
    class_mode='binary'         # For 2-class classification
)

# Load validation images
val_data = val_gen.flow_from_directory(
    'data/validation',
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary'
)

# Step 2: Load pre-trained MobileNetV2 (without top layer)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model layers

# Step 3: Add custom classification layers on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),     # Reduces tensor shape
    tf.keras.layers.Dense(1, activation='sigmoid') # Output: 0 or 1
])

# Step 4: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train top layers (feature extractor phase)
model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

# Step 6: Unfreeze some base model layers for fine-tuning
base_model.trainable = True
fine_tune_at = 100  # Freeze all layers before this

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Step 7: Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 8: Continue training (fine-tuning phase)
model.fit(
    train_data,
    epochs=10,
    initial_epoch=5,
    validation_data=val_data
)
