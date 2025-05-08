import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

train_path = r'C:\Users\MRC\Downloads\cats_and_dogs_filtered\train'
val_path = r'C:\Users\MRC\Downloads\cats_and_dogs_filtered\validation'

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = datagen.flow_from_directory(val_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'), plt.xlabel('Epochs'), plt.ylabel('Accuracy')
plt.legend(), plt.show()