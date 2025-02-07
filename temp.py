import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from keras.utils import normalize
import random

PATCH_SIZE = 256
NUM_PROCESSED = 100

COLOR_MAP = {
    (0, 0, 0): 0,        # Black → Background (Class 0)
    (128, 128, 128): 1,  # Gray → Roads (Class 1)
    (255, 0, 0): 2       # Red → Rodents (Class 2)
}

def decode_mask(mask):
    h, w = mask.shape  # Get height & width
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)  # Initialize RGB mask

    # Assign RGB colors correctly using broadcasting
    for rgb, class_id in COLOR_MAP.items():
        rgb_mask[mask == class_id] = np.array(rgb, dtype=np.uint8)  # Assign RGB color

    return rgb_mask

# define paths
model = load_model("colab_storage/unet_model_multiclass.keras")
case_name = "HL2"
image_folder = f"data/predicting_images/{case_name}_tiles"  # Folder containing .tif tiles
output_mask_path = f"data/predicted_masks/{case_name}_multiclass.png"  # Path for the final stitched mask

print("Loading the file...")
# Get sorted list of image tiles
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".tif")])
# print([f.split("_")[-2:] for f in image_files])


# Extract dimensions from filename (assuming standard naming like tile_x_y.tif)
# like: image_0_0.tif, image_0_1.tif, image_1_0.tif, etc.
coords = [tuple(map(int, f.replace(".tif", "").split("_")[-2:])) for f in image_files]
max_x = max(c[0] for c in coords) + 1
max_y = max(c[1] for c in coords) + 1

# Initialize an empty array for the final mask for inscribing
final_mask = np.zeros((max_x * PATCH_SIZE, max_y * PATCH_SIZE), dtype=np.uint8)

# Shuffle images randomly
random.shuffle(image_files)
# Select only 2000 images
image_files = image_files[:NUM_PROCESSED]

# Process each tile
print("Model started predicting...")

for file_name in image_files:
    # Extract x, y position from the filename
    parts = file_name.replace(".tif", "").split("_")
    x_idx, y_idx = int(parts[-2]), int(parts[-1])

    # Read and preprocess the image
    img_path = os.path.join(image_folder, file_name)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = normalize(image, axis=1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict mask
    prediction = model.predict(image)
    predicted_mask= np.argmax(prediction, axis=3)  # Convert softmax to class labels
    predicted_mask= predicted_mask.squeeze(0)  # Now shape is (256, 256)

    # Store the predicted patch in the correct position
    final_mask[x_idx * PATCH_SIZE: (x_idx + 1) * PATCH_SIZE,
               y_idx * PATCH_SIZE: (y_idx + 1) * PATCH_SIZE] = predicted_mask

decoded_mask = decode_mask(final_mask)  # Converts (H, W) → (H, W, 3)
final_image = Image.fromarray(decoded_mask)  # Convert to RGB image
final_image.save(output_mask_path)

print(f"Final stitched mask saved as {output_mask_path}")
