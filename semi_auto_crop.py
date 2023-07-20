import cv2
import numpy as np

from IPython.display import display, Image
from roboflow import Roboflow
import numpy as np
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# global variables
point1 = ()
point2 = ()
drawn = False
done = False
drawing = False
img = None

def resize_img(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global point1, point2, drawing, img, drawn, done  # Added done as a global variable.

    img_temp = resized_img.copy()

    if not drawn:
        if event == cv2.EVENT_LBUTTONDOWN:
            if not drawing:
                drawing = True
                point1 = (x, y)
            else:
                drawing = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                
                cv2.rectangle(img_temp, point1, (x, y), (0,255,0), 1)
                cv2.imshow('image', img_temp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            point2 = (x, y)
            cv2.rectangle(img_temp, point1, point2, (0,255,0), 1)
            cv2.imshow('image', img_temp)
            drawn = True
    else:
        drawn = False  # We set drawn to False here. The question to the user is moved to main loop.

def main():
    # Set up SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam.to(device = "cuda")
    segmentator = SamPredictor(sam)

    global img, resized_img, drawn, done  # Now we declare done as a global variable.
    img = cv2.imread('./images/zidane.jpg')
    segmentator.set_image(img)
    scale = 50
    resized_img = resize_img(img, scale)
    

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_rectangle)

    while(not done):
        cv2.imshow('image', resized_img)  # Display img instead of resized_img
        if drawn:
            if input("Do you want to select again? (S,n)").lower() != "s":  # Moved input function here from draw_rectangle
                done = True
        if cv2.waitKey(1) & 0xFF == ord('q'):  # exit on pressing 'q'
            break
    cv2.destroyAllWindows()

    
    input_box = np.array([point1[0]*2, point1[1]*2, point2[0]*2, point2[1]*2])

    mask, _, _ = segmentator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
    segmentation_mask = mask[0]
    binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
    alpha_channel = (binary_mask * 255).astype(np.uint8)
    object = np.dstack((img, alpha_channel))
    output_path = f'./crops/prueba.png'
    cv2.imwrite(output_path, object)

if __name__ == "__main__":
    main()
