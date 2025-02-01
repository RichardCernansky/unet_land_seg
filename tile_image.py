import os
import numpy as np
from patchify import patchify
import tifffile as tiff
import shutil

# Patch size
PATCH_SIZE = 256

# Input file (single .tif image)
input_file = "data/images/HL2.tif"  # Replace with actual file path
output_folder = "data/predicting_images/HL2_tiles"  # Folder where patches will be saved

# Deletes entire folder and its contents
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)  # Recreate empty output folder

def tile_single_tif_image(input_file, output_folder):
    """
    Splits a single .tif image into 256x256 patches and saves them in `output_folder`.
    """

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the input image (supports multi-band TIFFs)
    image = tiff.imread(input_file)  # Reads as a NumPy array

    # Debugging: Print shape of the image
    print(f"✅ Loaded image: {input_file}")
    print(f"Image shape before processing: {image.shape}")

    # Ensure it's at least a 3D array for compatibility
    if len(image.shape) == 2:  # Convert grayscale to 3D (H, W, 1)
        image = np.expand_dims(image, axis=-1)

    # Get dimensions and crop to nearest multiple of 256
    SIZE_Y = (image.shape[0] // PATCH_SIZE) * PATCH_SIZE
    SIZE_X = (image.shape[1] // PATCH_SIZE) * PATCH_SIZE
    image = image[:SIZE_Y, :SIZE_X]  # Crop

    # Debugging: Check new image size after cropping
    print(f"Image shape after cropping: {image.shape}")

    # Extract patches
    patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, image.shape[-1]), step=PATCH_SIZE)

    # Debugging: Print the number of patches
    print(f"🔹 Extracted patches shape: {patches_img.shape}")

    # Save patches
    patch_count = 0
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :, :]  # Extract patch

            # Drop unnecessary dimensions if single-channel
            if single_patch_img.shape[-1] == 1:
                single_patch_img = np.squeeze(single_patch_img, axis=-1)

            # Define filename
            patch_filename = f"{os.path.basename(input_file).replace('.tif', '')}_patch_{i}_{j}.tif"
            patch_path = os.path.join(output_folder, patch_filename)

            # Save patch
            tiff.imwrite(patch_path, single_patch_img)
            patch_count += 1

    print(f"✅ {patch_count} patches saved in: {output_folder}")

# Run the function
tile_single_tif_image(input_file, output_folder)
