import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import normalize
from PIL import Image
from osgeo import gdal, osr

# Map class-id -> RGB for the PNG visualization
COLOR_MAP = {(0, 0, 0): 0, (128, 128, 128): 1, (255, 0, 0): 2}

def predict_and_save_rgb_png(image_folder, model_path, patch_size, output_png_path):
    # Build a fast lookup table: class_id -> [R,G,B]
    lut = np.zeros((max(COLOR_MAP.values()) + 1, 3), dtype=np.uint8)
    for rgb, cid in COLOR_MAP.items():
        lut[cid] = rgb

    # Load trained model once for all tiles
    model = load_model(model_path)

    # List tiles; assume names end with "_x_y.tif" where x=row-index, y=col-index
    files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".tif")])

    # Derive stitch grid from filenames
    coords = [tuple(map(int, os.path.splitext(f)[0].split("_")[-2:])) for f in files]
    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)
    H = (max_x + 1) * patch_size                     # full mosaic height in pixels
    W = (max_y + 1) * patch_size                     # full mosaic width in pixels

    # Allocate class-id mosaic (grayscale labels)
    class_mask = np.zeros((H, W), dtype=np.uint8)

    # Predict each tile and place its class labels into the mosaic
    for fn in files:
        x_idx, y_idx = map(int, os.path.splitext(fn)[0].split("_")[-2:])
        img = cv2.imread(os.path.join(image_folder, fn), cv2.IMREAD_COLOR)
        img = normalize(img, axis=1)                 # same normalization used at train time
        img = np.expand_dims(img, axis=0)            # add batch dimension: (1,H,W,C)
        pred = model.predict(img)                    # network output per-pixel class probabilities
        lab = np.argmax(pred, axis=3).squeeze(0).astype(np.uint8)  # convert to class ids

        # Compute paste window in the big mosaic
        ys, ye = x_idx * patch_size, (x_idx + 1) * patch_size
        xs, xe = y_idx * patch_size, (y_idx + 1) * patch_size

        class_mask[ys:ye, xs:xe] = lab               # stitch class ids

    # Colorize via LUT for human-friendly PNG
    rgb_mask = lut[class_mask]                        # shape: (H,W,3), dtype=uint8
    Image.fromarray(rgb_mask).save(output_png_path)   # write visualization PNG

    return class_mask                                 # return class-id mosaic for GeoTIFF step


# ---------- Usage ----------
CASE_NAME = "Buriny"
PATCH_SIZE = 256
MODEL_PATH = "./data/models/unet_model_multiclass_buriny.keras"
TILE_DIR = f"data/predicting_images/{CASE_NAME}_tiles"
PNG_OUT = f"data/predicted_masks/{CASE_NAME}.png"

class_mask = predict_and_save_rgb_png(TILE_DIR, MODEL_PATH, PATCH_SIZE, PNG_OUT)   # writes RGB PNG
