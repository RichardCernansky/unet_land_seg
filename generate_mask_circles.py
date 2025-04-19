# this script is for generating the mask that consists of CIRCLES with radius=RADIUS and
# center from ./data/centers

import os
import json
import numpy as np
import cv2

def parse_tfw(tfw_path):
    """Parse .tfw file for geospatial transformation information."""
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    x_scale = float(lines[0].strip())
    y_scale = float(lines[3].strip())
    upper_left_x = float(lines[4].strip())
    upper_left_y = float(lines[5].strip())
    return x_scale, y_scale, upper_left_x, upper_left_y


def generate_mask(tif_path, tfw_path, json_path, output_mask_path, radius):
    """Generate a mask based on tree centers and save as a PNG image."""
    x_scale, y_scale, upper_left_x, upper_left_y = parse_tfw(tfw_path)

    tif_image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if tif_image is None:
        raise ValueError(f"Unable to read TIF file: {tif_path}")

    mask = np.zeros(tif_image.shape[:2], dtype=np.uint8)

    with open(json_path, 'r') as f:
        center_data = json.load(f)
    center_list = center_data["center_list"]

    for center in center_list:
        longitude, latitude, _ = center
        pixel_x = int((longitude - upper_left_x) / x_scale)
        pixel_y = int((latitude - upper_left_y) / y_scale)

        if 0 <= pixel_x < mask.shape[1] and 0 <= pixel_y < mask.shape[0]:
            cv2.circle(mask, (pixel_x, pixel_y), radius, 255, thickness=-1)

    cv2.imwrite(output_mask_path, mask)
    print(f"Mask saved to {output_mask_path}")
    return mask


def process_all_images(image_dir, mask_dir, centers_dir, tfw_dir, radius):
    os.makedirs(mask_dir, exist_ok=True)

    for file in os.listdir(image_dir):
        if file.endswith(".tif"):
            tif_path = os.path.join(image_dir, file)
            tfw_path = os.path.join(tfw_dir, file.replace(".tif", ".tfw"))
            json_path = os.path.join(centers_dir, file.replace(".tif", ".json"))
            mask_path = os.path.join(mask_dir, file.replace(".tif", ".png"))

            if not os.path.exists(tfw_path) or not os.path.exists(json_path):
                print(f"Skipping {file}: Missing .tfw or JSON file.")
                continue

            generate_mask(tif_path, tfw_path, json_path, mask_path, radius)


# Directories
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
TFW_DIR = "data/tfws"
CENTERS_DIR = "data/centers"
RADIUS = 20

# Process all images
process_all_images(IMAGE_DIR, MASK_DIR, CENTERS_DIR, TFW_DIR, RADIUS)
