import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
import splitfolders

Image.MAX_IMAGE_PIXELS = None

PATCH_SIZE = 256
USEFUL_PERC = 0.05

root_directory = 'data/'
img_dir = "data/images"
mask_dir = "data/masks"

output_img_dir = os.path.join(root_directory, "256_patches/images")
output_mask_dir = os.path.join(root_directory, "256_patches/masks")
useful_info_img_dir = os.path.join(root_directory, "256_patches/images_with_useful_info/images")
useful_info_mask_dir = os.path.join(root_directory, "256_patches/images_with_useful_info/masks")

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(useful_info_img_dir, exist_ok=True)
os.makedirs(useful_info_mask_dir, exist_ok=True)

ideal_patch_count = 0
for image_name in sorted(os.listdir(img_dir)):
    if image_name.endswith(".tif"):
        mask_name = image_name

        image_path = os.path.join(img_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Skipping {image_name}, mask not found.")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        print(f"Image and mask opened for: {image_name}")

        SIZE_X = (image.shape[1] // PATCH_SIZE) * PATCH_SIZE
        SIZE_Y = (image.shape[0] // PATCH_SIZE) * PATCH_SIZE
        image = Image.fromarray(image).crop((0, 0, SIZE_X, SIZE_Y))
        mask = Image.fromarray(mask).crop((0, 0, SIZE_X, SIZE_Y))

        image = np.array(image)
        mask = np.array(mask)

        y_num_patches = image.shape[0] // PATCH_SIZE
        x_num_patches = image.shape[1] // PATCH_SIZE
        ideal_patch_count += y_num_patches * x_num_patches

        patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
        patches_mask = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE)

        for pi in range(patches_img.shape[0]):
            for pj in range(patches_img.shape[1]):
                single_patch_img = patches_img[pi, pj, 0, :, :, :]
                single_patch_mask = patches_mask[pi, pj, :, :] # one-channel, patchify doesnt add SINGLETON dimension

                if single_patch_img.shape[-1] != 3 or single_patch_mask.ndim != 2:
                    raise ValueError(
                        f"Invalid shape detected! Image: {single_patch_img.shape}, Mask: {single_patch_mask.shape}"
                    )

                patch_img_name = f"{image_name}_patch_{pi}_{pj}.tif"
                patch_mask_name = f"{mask_name}_patch_{pi}_{pj}.tif"

                cv2.imwrite(os.path.join(output_img_dir, patch_img_name), single_patch_img)
                cv2.imwrite(os.path.join(output_mask_dir, patch_mask_name), single_patch_mask)

        print(f"Processed {y_num_patches * x_num_patches} patches for: {image_name}")

num_files = len([f for f in os.listdir(output_img_dir) if os.path.isfile(os.path.join(output_img_dir, f))])
print(f"\nTotal/ideal patch count: {num_files}/{ideal_patch_count+1}")

print("Now preparing USEFUL images and masks.")
useless = 0
for img_name in sorted(os.listdir(output_img_dir)):
    mask_name = img_name

    img_path = os.path.join(output_img_dir, img_name)
    mask_path = os.path.join(output_mask_dir, mask_name)

    if not os.path.exists(mask_path):
        continue

    temp_image = cv2.imread(img_path, 1)
    temp_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    if temp_mask is None or temp_image is None:
        print(f"Warning: Could not load {mask_name} or {img_name}")
        continue

    non_black_pixels = np.count_nonzero(temp_mask)
    total_pixels = temp_mask.size
    useful_ratio = non_black_pixels / total_pixels

    if useful_ratio > USEFUL_PERC:
        cv2.imwrite(os.path.join(useful_info_img_dir, img_name), temp_image)
        cv2.imwrite(os.path.join(useful_info_mask_dir, mask_name), temp_mask)
    else:
        useless += 1

print(f"Total useful images: {len(os.listdir(output_img_dir)) - useless}")
print(f"Total useless images: {useless}")

print("#####...SPLITTING...#####")
input_folder = os.path.join(root_directory, "256_patches/images_with_useful_info/")
output_folder = os.path.join(root_directory, "dataset/")

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.75, 0.25), group_prefix=None)

print(f"Dataset split into train and val in path: {output_folder}")
"""
Final dataset structure:
dataset/
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
