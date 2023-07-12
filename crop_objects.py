from IPython.display import display, Image
from roboflow import Roboflow
import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Image folder
folder_path = "./images/"

# Output path
output_folder = "./crops/"

# Load the YOLOv8 model
yolo = YOLO('yolov8n.pt')

# Ask user for inout and output filename
input_filename = input("Enter your inout filename (add extension): ")
output_filename = input("Enter your output filename: ")


# Set up SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
sam.to(device = "cuda")

segmentator = SamPredictor(sam)

image_path = f"./images/{input_filename}"
        
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

# Convert to BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for result in results:
    
    boxes = result.boxes
    class_id = result#.index(max(result.probs))

    for i, bbox in enumerate(boxes.xyxy.tolist()):

        input_box = np.array(bbox)
        mask, _, _ = segmentator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        segmentation_mask = mask[0]

        # Convert the segmentation mask to a binary mask
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

        # Create the alpha channel scaled to 255
        alpha_channel = (binary_mask * 255).astype(np.uint8)

        # Stack the RGB image with the alpha channel
        object = np.dstack((img, alpha_channel))

        # Save the image in PNG format (to preserve transparency)
        output_path = f'{output_folder}{output_filename}_{i}.png'
        cv2.imwrite(output_path, object)