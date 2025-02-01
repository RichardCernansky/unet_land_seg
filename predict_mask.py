import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from unet_model_specific_functions import *

PATCH_SIZE = 256

def bce_dice_loss(y_true, y_pred):
    """
    Combines Binary Cross-Entropy (BCE) loss and Dice loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)  # Use TensorFlow's BCE
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice  # Adjust weights as needed

# define paths
model = load_model("colab_storage/unet_model_1.keras", custom_objects={"bce_dice_loss": bce_dice_loss})
case_name = "HL1"
image_folder = f"data/predicting_images/{case_name}_tiles"  # Folder containing .tif tiles
output_mask_path = f"data/predicted_masks/{case_name}_mask.png"  # Path for the final stitched mask

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

# Process each tile
print("Model started predicting...")
for file_name in image_files:
    # Extract x, y position from the filename
    parts = file_name.replace(".tif", "").split("_")
    x_idx, y_idx = int(parts[-2]), int(parts[-1])

    # Read and preprocess the image
    img_path = os.path.join(image_folder, file_name)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict mask
    predicted_mask = model.predict(image)[0]  # Remove batch dimension
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Thresholding

    # Store the predicted patch in the correct position
    final_mask[x_idx * PATCH_SIZE: (x_idx + 1) * PATCH_SIZE,
               y_idx * PATCH_SIZE: (y_idx + 1) * PATCH_SIZE] = predicted_mask[..., 0]

# Save the final stitched mask as an image
final_image = Image.fromarray(final_mask * 255)  # Convert to 8-bit image
final_image.save(output_mask_path)

print(f"Final stitched mask saved as {output_mask_path}")
