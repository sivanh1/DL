import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained object detection model from TF Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load and preprocess image
img = cv2.imread("/content/Blog_cover-52-scaled.jpeg")  # Replace with your image path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension

# Run detection
outputs = model(img_tensor)
boxes = outputs["detection_boxes"][0].numpy()
scores = outputs["detection_scores"][0].numpy()

# Draw results
h, w, _ = img.shape
for i in range(len(scores)):
    if scores[i] > 0.5:
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)

# Show image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
