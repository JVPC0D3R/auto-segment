import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_opening, binary_dilation
import cv2

def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def get_edge_mask(mask, sigma = 1, structure=np.ones((3,3))):
    

    if mask.ndim > 2:
        mask = mask[0, :, :]  


    # Apply Gaussian blur
    mask = gaussian_filter(mask, sigma=sigma)

    # Apply opening operation
    mask = binary_opening(mask, structure=structure)

    # Apply dilation operation
    for i in range(0,3):
        mask = binary_dilation(mask, structure=structure)


    mask = (mask * 255).astype(np.uint8)

    edges = cv2.Canny(mask, threshold1=254, threshold2=255)

    # Convert back to original mask format
    edges = edges == 255

    # Reshape to maintain original mask shape
    edge_mask = edges[np.newaxis, :, :]
    
    
    return edge_mask

def get_edge_coordinates(mask):

    _, y_indices, x_indices = np.nonzero(get_edge_mask(mask))

    h, w = mask.shape[-2:]

    edge_coords = ' '.join([f'{x / w} {y / h}' for x, y in zip(x_indices, y_indices)])
    return edge_coords

def write_line_to_file(filename, line):
    with open(filename, 'a') as file:
        file.write(line + '\n')

