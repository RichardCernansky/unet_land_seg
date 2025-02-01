# THIS SCRIPT IS FOR PREPROCESSING THE ./data/images AND ./data/masks
# AND PARSING THEM INTO  W * H  sized PATCHES EVENTUALLY SAVED IN THE DATASET FOLDER
# TODO: solve the non_aug missing issue

# https://youtu.be/0W6MKZqSke8
"""
Author: Dr. Sreenivas Bhattiprolu

The following code performs these tasks - relevant to work with landcover dataset
from here: https://landcover.ai/

Code can be modified to work with any other dataset.

Tasks achieved.

1. Read large images and corresponding masks, divide them into smaller patches.
And write the patches as images to the local drive.

2. Save only images and masks where masks have some decent amount of labels other than 0.
Using blank images with label=0 is a waste of time and may bias the model towards
unlabeled pixels.

3. Divide the sorted dataset from above into train and validation datasets.

4. You have to manually move some folders and rename appropriately if you want to use
ImageDataGenerator from keras.

"""

import os
import cv2
import numpy as np
import glob
import math

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image
import tensorflow as tf
import keras
from tensorflow.keras.metrics import MeanIoU
import splitfolders
import random
from PIL import Image
import splitfolders  # or import split_folders

PATCH_SIZE = 256
USEFUL_PERC = 0.05
Image.MAX_IMAGE_PIXELS = None

# Quick understanding of the dataset
temp_img = cv2.imread("data/images/HL1.tif")  # 3 channels / spectral bands
plt.imshow(temp_img[:, :, 2])  # View each channel...
temp_mask = cv2.imread("data/masks/HL1.png")  # 3 channels but all same.
labels, count = np.unique(temp_mask[:, :, 0], return_counts=True)  # Check for each channel. All chanels are identical
print("Labels are: ", labels, " and the counts are: ", count)

# Now, crop each large image into patches of 256x256. Save them into a directory
# so we can use data augmentation and read directly from the drive.
root_directory = 'data/'

# Read images from repsective 'images' subdirectory
# As all images are of different size we have 2 options, either resize or crop
# But, some images are too large and some small. Resizing will change the size of real objects.
# Therefore, we will crop them to a nearest size divisible by 256 and then
# divide all images into patches of 256x256x3.
img_dir = root_directory + "images/"
ideal_patch_count = 0
for path, subdirs, files in os.walk(img_dir):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    # print(dirname)
    images = os.listdir(path)  # List of all image names in this subdirectory
    # print(images)
    for i, image_name in enumerate(images):
        if image_name.endswith(".tif"):
            # print(image_name)
            image = cv2.imread(path + "/" + image_name, 1)  # Read each image as BGR
            SIZE_X = (image.shape[1] // PATCH_SIZE) * PATCH_SIZE  # Nearest size divisible by our patch size
            SIZE_Y = (image.shape[0] // PATCH_SIZE) * PATCH_SIZE  # Nearest size divisible by our patch size
            image = Image.fromarray(image)
            image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            image = np.array(image)

            y_num_patches = temp_img.shape[0] // PATCH_SIZE
            x_num_patches = temp_img.shape[1] // PATCH_SIZE
            ideal_patch_count += y_num_patches * x_num_patches

            # Extract patches from each image
            print("Now patchifying image:", path + "/" + image_name)
            patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, 3), step=256)  # Step=256 for 256 patches means no overlap

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :]
                    # single_patch_img = (single_patch_img.astype('float32')) / 255. #We will preprocess using one of the backbones
                    single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.

                    cv2.imwrite(os.path.join(root_directory, "256_patches/images",
                                             f"{image_name}_patch_{i}_{j}.tif"), single_patch_img)
                    # image_dataset.append(single_patch_img)

path_patched_images = os.path.join(root_directory, "256_patches/images/")
file_count_img_tiles = len([f for f in os.listdir(path_patched_images) if os.path.isfile(os.path.join(path_patched_images, f))])
print(f"{file_count_img_tiles}/{ideal_patch_count} .tif tiles patched successfully.")

# Now do the same as above for masks
# For this specific dataset we could have added masks to the above code as masks have extension png
mask_dir = root_directory + "masks/"
for path, subdirs, files in os.walk(mask_dir):
    # print(path)
    dirname = path.split(os.path.sep)[-1]

    masks = os.listdir(path)  # List of all image names in this subdirectory
    for i, mask_name in enumerate(masks):
        if mask_name.endswith(".png"):
            mask = cv2.imread(path + "/" + mask_name,
                              0)  # Read each image as Grey (or color but remember to map each color to an integer)
            SIZE_X = (mask.shape[1] // PATCH_SIZE) * PATCH_SIZE  # Nearest size divisible by our patch size
            SIZE_Y = (mask.shape[0] // PATCH_SIZE) * PATCH_SIZE  # Nearest size divisible by our patch size
            mask = Image.fromarray(mask)
            mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            mask = np.array(mask)

            # Extract patches from each image
            print("Now patchifying mask:", path + "/" + mask_name)
            patches_mask = patchify(mask, (256, 256), step=256)  # Step=256 for 256 patches means no overlap

            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i, j, :, :]
                    # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                    # single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
                    cv2.imwrite(os.path.join(root_directory, "256_patches/masks",
                                             f"{mask_name}_patch_{i}_{j}.tif"), single_patch_mask)

path_patched_masks = os.path.join(root_directory, "256_patches/masks/")
file_count_mask_tiles = len([f for f in os.listdir(path_patched_masks) if os.path.isfile(os.path.join(path_patched_masks, f))])
print(f"{file_count_mask_tiles}/{ideal_patch_count} mask tiles patched successfully.")

train_img_dir = "data/256_patches/images/"
train_mask_dir = "data/256_patches/masks/"

img_list = sorted(os.listdir(train_img_dir))
msk_list = sorted(os.listdir(train_mask_dir))

num_images = len(os.listdir(train_img_dir))

###########################################################################
# Now, let us copy images and masks with real information to a new folder.
# real information => if mask has decent amount of labels other than 0.

print("Now preparing USEFUL images and masks.")
useless = 0  # Useless image counter
for img in range(len(img_list)):  # Using t1_list as all lists are of same size
    img_name = img_list[img]
    mask_name = msk_list[img]
    temp_image = cv2.imread(train_img_dir + img_list[img], 1)

    temp_mask = cv2.imread(train_mask_dir + msk_list[img], 0)
    # temp_mask=temp_mask.astype(np.uint8)
    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > USEFUL_PERC:  # At least 5% useful area with labels that are not 0
        cv2.imwrite('data/256_patches/images_with_useful_info/images/' + img_name, temp_image)
        cv2.imwrite('data/256_patches/images_with_useful_info/masks/' + mask_name, temp_mask)

    else:
        useless += 1

print("Total useful images are: ", len(img_list) - useless)  # 20,075
print("Total useless images are: ", useless)  # 21,571
###############################################################
# Now split the data into training, validation and testing.

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 

pip install split-folders
"""

print("#####...SPLITTING...#####")
input_folder = 'data/256_patches/images_with_useful_info/'
output_folder = 'data/dataset/non_aug'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)  # default values
print("Dataset split into train and val in path: 'data/dataset/non_aug'")
########################################

# Now manually move folders around to bring them to the following structure.
"""
The current directory structure:
dataset/
    non_aug/
        train/
            images/
                img1, img2, ...
            masks/
                msk1, msk2, ....
        val/
            images/
                img1, img2, ...
            masks/
                msk1, msk2, ....

"""