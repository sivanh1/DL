import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
    'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors'
]

def detect_objects(image_path, threshold=0.7):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    img = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(img)
    
    with torch.no_grad():
        preds = model([tensor])[0]

    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']

    keep = scores > threshold

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
        xmin, ymin, xmax, ymax = box
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='red', linewidth=2, fill=False))
        ax.text(xmin, ymin - 10, f"{COCO_CLASSES[label]}: {score:.2f}", color='white', fontsize=8,
                bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

# Call the function with your image path
image_path = r"C:\Users\MAC\Downloads\chatGPT image Apr 8, 2025, 02_30_55 PM.jpg"
detect_objects(image_path)