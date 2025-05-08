# Step 1: Install required packages (run this in your terminal or Jupyter notebook)
# !pip install tensorflow tensorflow-hub opencv-python

# Step 2: Import libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 3: Load and preprocess one image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (384, 384)) / 255.0  # Resize and normalize
    return img

img_path = 'cow.jpg'  # Replace with your image file
image = load_image(img_path)
input_tensor = tf.expand_dims(image, 0)  # Add batch dimension

# Step 4: Load pre-trained object detection model
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

# Step 5: Run detection
output = model(input_tensor)
boxes = output["detection_boxes"][0].numpy()
classes = output["detection_classes"][0].numpy().astype(np.int32)
scores = output["detection_scores"][0].numpy()

# Step 6: Draw boxes on the image
labels = {1: 'person', 17: 'cat', 18: 'dog', 20: 'cow'}  # Add more as needed

img_np = np.array(image * 255, dtype=np.uint8)
h, w = img_np.shape[:2]

for box, cls, score in zip(boxes, classes, scores):
    if score < 0.3:
        continue
    y1, x1, y2, x2 = box
    start = (int(x1 * w), int(y1 * h))
    end = (int(x2 * w), int(y2 * h))
    cv2.rectangle(img_np, start, end, (0, 255, 0), 2)
    label = f"{labels.get(cls, 'ID:'+str(cls))} ({score:.2f})"
    cv2.putText(img_np, label, (start[0], start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Step 7: Show result
plt.imshow(img_np)
plt.axis("off")
plt.show()
