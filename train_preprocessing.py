import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
import albumentations as A
import splitfolders

Image.MAX_IMAGE_PIXELS = None
# Define PATCH_SIZE and augmentation count
PATCH_SIZE = 256
NUM_AUGMENTATIONS = 2  # Number of augmented versions per patch
USEFUL_PERC = 0.05  # Threshold for useful masks

# Define Augmentation Pipeline (Only for Images)
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),  # Minimal contrast change
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),  # Smooths road edges, reducing false positives
    A.MedianBlur(blur_limit=3, p=0.3),  # Removes small noise and thin lines
    A.Equalize(p=0.3),  # Helps balance pixel intensities without boosting edges
])

# Define Paths
root_directory = 'data/'
img_dir = os.path.join(root_directory, "images")
mask_dir = os.path.join(root_directory, "masks")

output_img_dir = os.path.join(root_directory, "256_patches/images")
output_mask_dir = os.path.join(root_directory, "256_patches/masks")
useful_info_img_dir = os.path.join(root_directory, "256_patches/images_with_useful_info/images")
useful_info_mask_dir = os.path.join(root_directory, "256_patches/images_with_useful_info/masks")

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(useful_info_img_dir, exist_ok=True)
os.makedirs(useful_info_mask_dir, exist_ok=True)

# Initialize ideal patch count
ideal_patch_count = 0
# Iterate through images and masks
for image_name in sorted(os.listdir(img_dir)):
    if image_name.endswith(".tif"):
        mask_name = image_name.replace(".tif", ".png")  # Assuming masks have the same name but in .png format

        image_path = os.path.join(img_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):  # Ensure corresponding mask exists
            print(f"Skipping {image_name}, mask not found.")
            continue

        # Read image and mask
        image = cv2.imread(image_path, 1)  # Load image as BGR
        mask = cv2.imread(mask_path, 0)  # Load mask as grayscale

        # Ensure both image and mask are the same size
        SIZE_X = (image.shape[1] // PATCH_SIZE) * PATCH_SIZE
        SIZE_Y = (image.shape[0] // PATCH_SIZE) * PATCH_SIZE
        image = Image.fromarray(image).crop((0, 0, SIZE_X, SIZE_Y))
        mask = Image.fromarray(mask).crop((0, 0, SIZE_X, SIZE_Y))

        image = np.array(image)
        mask = np.array(mask)

        # Calculate ideal patch count
        y_num_patches = image.shape[0] // PATCH_SIZE
        x_num_patches = image.shape[1] // PATCH_SIZE
        ideal_patch_count += y_num_patches * x_num_patches

        # Patchify image and mask
        patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
        patches_mask = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE)

        for pi in range(patches_img.shape[0]):
            for pj in range(patches_img.shape[1]):
                single_patch_img = patches_img[pi, pj, :, :]
                single_patch_mask = patches_mask[pi, pj, :, :]

                single_patch_img = single_patch_img[0]  # Remove extra dimension from patchify

                # Save original patch
                patch_img_name = f"{image_name}_patch_{pi}_{pj}.tif"
                patch_mask_name = f"{mask_name}_patch_{pi}_{pj}.png"

                cv2.imwrite(os.path.join(output_img_dir, patch_img_name), single_patch_img)
                cv2.imwrite(os.path.join(output_mask_dir, patch_mask_name), single_patch_mask)

                # Generate multiple augmented versions for the image (mask remains the same)
                for aug_idx in range(NUM_AUGMENTATIONS):
                    augmented_patch = augment(image=single_patch_img)['image']

                    aug_patch_img_name = f"{image_name}_patch_{pi}_{pj}_aug_{aug_idx}.tif"
                    aug_patch_mask_name = f"{mask_name}_patch_{pi}_{pj}_aug_{aug_idx}.png"

                    cv2.imwrite(os.path.join(output_img_dir, aug_patch_img_name), augmented_patch)
                    cv2.imwrite(os.path.join(output_mask_dir, aug_patch_mask_name), single_patch_mask)  # Copy same mask

        print(f"Processed {y_num_patches * x_num_patches} patches for: {image_name}")

# Print final patch count
num_files = len([f for f in os.listdir(output_img_dir) if os.path.isfile(os.path.join(output_img_dir, f))])
print(f"\nTotal/ideal patch count: {num_files}/{ideal_patch_count*NUM_AUGMENTATIONS+1}")

# Filter images with real information (non-empty masks)
print("Now preparing USEFUL images and masks.")
useless = 0  # Useless image counter
for img_name in sorted(os.listdir(output_img_dir)):  # Iterate through processed images
    mask_name = img_name.replace(".tif", ".png")  # Ensure mask filename matches

    img_path = os.path.join(output_img_dir, img_name)
    mask_path = os.path.join(output_mask_dir, mask_name)

    if not os.path.exists(mask_path):
        continue

    temp_image = cv2.imread(img_path, 1)
    temp_mask = cv2.imread(mask_path, 0)

    val, counts = np.unique(temp_mask, return_counts=True)

    if len(counts) > 1 and (1 - (counts[0] / counts.sum())) > USEFUL_PERC:  # At least 5% useful area
        cv2.imwrite(os.path.join(useful_info_img_dir, img_name), temp_image)
        cv2.imwrite(os.path.join(useful_info_mask_dir, mask_name), temp_mask)
    else:
        useless += 1

print(f"Total useful images: {len(os.listdir(output_img_dir)) - useless}")
print(f"Total useless images: {useless}")

# Now split the dataset into training and validation sets
print("#####...SPLITTING...#####")
input_folder = os.path.join(root_directory, "256_patches/images_with_useful_info/")
output_folder = os.path.join(root_directory, "dataset/aug")

# Ensure the dataset is properly structured for segmentation
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.75, 0.25), group_prefix=None)

print(f"Dataset split into train and val in path: {output_folder}")

"""
Final dataset structure:
dataset/
    aug/
        train/
            images/
                img1.tif, img2.tif, ...
            masks/
                mask1.tif, mask2.tif, ...
        val/
            images/
                img1.tif, img2.tif, ...
            masks/
                mask1.tif, mask2.tif, ...
"""
