from IPython.display import display, Image
from roboflow import Roboflow
import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from utils import show_box, show_mask, show_points, get_edge_coordinates, write_line_to_file
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

for filename in os.listdir(folder_path):
    
    if filename.endswith((".jpg", ".jpeg", ".png")):

        # Get image path
        image_path = os.path.join(folder_path, filename)

        # Build txt label filename
        label_filename = os.path.splitext(filename)[0] + '.txt'

        # Get label path
        label_path = os.path.join(output_folder, label_filename)
        
        # Load image
        img = cv2.imread(image_path)

        # Conver from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Predict bounding boxes
        results = yolo.predict(source = img)

        # Set image for SAM
        segmentator.set_image(img)

        for result in results:
            
            boxes = result.boxes
            class_id = result#.index(max(result.probs))
            

            for bbox, cls in zip(boxes.xyxy.tolist(), boxes.cls):

                input_box = np.array(bbox)
                mask, _, _ = segmentator.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                write_line_to_file(label_path, f'{int(cls)} {get_edge_coordinates(mask)}')


