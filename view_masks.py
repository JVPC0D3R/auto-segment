from IPython.display import display, Image
from roboflow import Roboflow
import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from utils import show_box, show_mask, show_points, get_edge_coordinates, get_edge_mask
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Image folder
folder_path = "./images/"

# Output path
output_folder = "./output/"

# Load the YOLOv8 model
yolo = YOLO('yolov8n.pt')

# Set up SAM

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
sam.to(device = "cuda")

segmentator = SamPredictor(sam)

image_path = "./images/bus.jpg"
        
# Load image
img = cv2.imread(image_path)

# Conver from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Predict bounding boxes
results = yolo.predict(source = img)

# Set image for SAM
segmentator.set_image(img)

plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

for result in results:
    
    boxes = result.boxes
    class_id = result#.index(max(result.probs))
    
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(img)
    axs[1, 0].imshow(img)
    axs[1, 1].imshow(img)

    for bbox, cls in zip(boxes.xyxy.tolist(), boxes.cls):

        input_box = np.array(bbox)
        mask, _, _ = segmentator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        show_box(bbox, axs[0, 1])
        show_mask(mask, axs[1, 0])
        show_mask(get_edge_mask(mask), axs[1, 1], edge = True, times = 5)

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    
    axs[0, 0].set_title('Raw Image')
    axs[0, 1].set_title('Bounding boxes [YOLOv8]')
    axs[1, 0].set_title('Segmentation mask [SAM]')
    axs[1, 1].set_title('Mask edges for YOLO format')
    
    plt.show()

