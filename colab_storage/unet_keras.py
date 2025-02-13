# -*- coding: utf-8 -*-
"""unet_keras.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GL9RJUqI5FBPyt93AhXwhRZTZg1X3KUX

# **U-NET Data modeling**

# Preparation
"""

from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/MyDrive/unet_land_seg/dataset.zip

!pip install keras-tuner

import os
import shutil
import numpy as np
import cv2
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
os.makedirs("files/non_aug", exist_ok=True)

!ls /content/dataset/

# Set the seed values
os.environ["PYTHONHASHSEED"] = str(42)  # Ensures hash-based randomness is fixed
np.random.seed(42)  # NumPy random seed
tf.random.set_seed(42)  # TensorFlow random seed

# Define hyperparameters
batch_size = 8       # Number of images per batch
lr = 1e-4            # Learning rate (0.0001)
epochs = 100         # Total number of training epochs
height = 256         # Input image height
width = 256          # Input image width

dataset_path = os.path.join("/content/dataset", "aug")
files_dir = os.path.join("files", "non_aug")
model_file = os.path.join(files_dir, "unet-non-aug.keras")
log_file = os.path.join(files_dir, "log-non-aug.csv")

train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "val")

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

"""# U-NET definition"""

def conv_block(inputs, num_filters):
    """ Convolutional block: Two Conv2D layers with Batch Normalization & ReLU activation. """
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    """ Encoder Block: Convolutional block + MaxPooling """
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip, num_filters):
    """ Decoder Block: Upsampling + Skip Connection + Convolutional Block """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)  # Upsample
    x = Concatenate()([x, skip])  # Skip connection
    x = conv_block(x, num_filters)  # Apply conv block
    return x

def build_unet(input_shape=(256, 256, 3), num_classes=3):
    """ U-Net Model for Multi-Class Segmentation """

    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """ Bridge """
    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output Layer for Multi-Class Segmentation """
    outputs = Conv2D(num_classes, 1, padding="same")(d4)  # ❌ No activation


    model = Model(inputs, outputs, name="UNET_MultiClass")
    return model

"""# **Dataset pipeline**"""

# Define RGB values for each class
BACKGROUND_COLOR = [0, 0, 0]  # Black (Background)
ROAD_COLOR = [128, 128, 128]  # Gray (Roads)
RODENT_COLOR = [255, 0, 0]  # Red (Rodents)

# Dictionary for easy access
CLASS_COLORS = {
    0: BACKGROUND_COLOR,  # Background
    1: ROAD_COLOR,  # Roads
    2: RODENT_COLOR  # Rodents
}

def encode_mask(mask):
    # Create an empty mask with class labels
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Assign class labels based on color
    new_mask[np.all(mask == ROAD_COLOR, axis=-1)] = 1  # Roads -> Class 1
    new_mask[np.all(mask == RODENT_COLOR, axis=-1)] = 2  # Rodents -> Class 2
    # Background remains 0

    return new_mask

def decode_mask(mask):
    # Create an empty RGB mask
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Assign RGB colors based on class labels
    for class_id, color in CLASS_COLORS.items():
        rgb_mask[mask == class_id] = color

    return rgb_mask

def load_data(path):
    # Load training images and masks
    train_x = sorted(glob(os.path.join(path, "train", "images", "*")))
    train_y = sorted(glob(os.path.join(path, "train", "masks", "*")))

    # Load validation images and masks
    valid_x = sorted(glob(os.path.join(path, "val", "images", "*")))
    valid_y = sorted(glob(os.path.join(path, "val", "masks", "*")))

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()  # Decode bytes to string (for TensorFlow datasets)
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # Read image in color mode
    x = x / 255.0  # Normalize pixel values to range [0,1]
    return x

def read_mask(path):
    path = path.decode()  # Decode bytes to string (for TensorFlow dataset pipelines)
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # Read mask as a color image
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) #invert because of open cv
    x = encode_mask(x)  # Convert to class labels {0,1,2}
    x  = np.expand_dims(x, axis=-1)
    x= x.astype(np.double)
    return x

"""**tf.dataset**"""

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)  # Read and preprocess the image
        y = read_mask(y)   # Read and preprocess the mask
        return x, y

    # Convert Python function into a TensorFlow operation
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])

    # Set explicit shapes for TensorFlow tensors
    x.set_shape([height, width, 3])  # Image shape (H, W, 3 channels)
    y.set_shape([height, width, 1])  # Mask shape (H, W, 1 channel)
    return x, y

def tf_dataset(x, y, batch=8):
  dataset = tf.data.Dataset.from_tensor_slices((x, y))  # Create dataset from file paths
  dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)  # Preprocess images/masks
  dataset = dataset.batch(batch)  # Group data into batches
  dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize data loading
  return dataset

"""## **Loss definitions**"""

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice coefficient loss function.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])  # Adjust axis for your data format
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_loss(y_true, y_pred):
    """
    Dice loss (1 - dice_coef).  Minimizing this is equivalent to maximizing Dice.
    """
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combines Binary Cross-Entropy (BCE) loss and Dice loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)  # Use TensorFlow's BCE
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice  # Adjust weights as needed

"""## **Training**"""

# Load dataset
(train_x, train_y), (valid_x, valid_y) = load_data(dataset_path)

# Print dataset size
print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")

# Create training and validation datasets
train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

"""**Check shape of train dataset**"""

for x,y in train_dataset:
  print(x.shape, y.shape)

"""**Model summary and callback definition**"""

input_shape = (height, width, 3)
model = build_unet(input_shape)
model.summary()

"""**Model compile and fit**"""

# Create U-Net model for 3-class segmentation
num_classes = 3  # Background, Roads, Rodents
model = build_unet(num_classes=num_classes)

# Compile model with Sparse Categorical Crossentropy loss
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # ✅ Handles logits correctly
    metrics=["accuracy"]
)

# Define callbacks
callbacks = [
    ModelCheckpoint(model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, mode='max', verbose=1),
    CSVLogger(log_file),
    EarlyStopping(monitor='val_loss', patience=15, mode='max', restore_best_weights=True, verbose=1)
]

# Train the model with increased epochs
history = model.fit(
    train_dataset,                  # Replace with your training dataset
    validation_data=valid_dataset,  # Replace with your validation dataset
    epochs=50,                     # Increased number of epochs
    callbacks=callbacks,
    verbose=1                       # Display detailed training logs
)

"""**Save model and hyperparameters**"""

model.save('/content/drive/MyDrive/unet_land_seg/unet_model_multiclass.keras')  # Saves in Google Drive
# tuner_path = "/content/hyperband_tuning"
# drive_path = "/content/drive/MyDrive/unet_land_seg/hyperband_tuning"

# # Copy the tuner directory to Google Drive
# shutil.copytree(tuner_path, drive_path)

"""**Model fit**"""

# model.fit(
#     train_dataset,
#     validation_data=valid_dataset,
#     epochs=epochs,
#     callbacks=callbacks
# )

"""**Load model from file**"""

# tf.keras.utils.get_custom_objects()["bce_dice_loss"] = bce_dice_loss
# model_path = "files/non_aug/unet-non-aug.keras"
# model = tf.keras.models.load_model(model_path, custom_objects={"bce_dice_loss": bce_dice_loss})

# # Print model summary to verify
# model.summary()

"""# Results

**Plot metrics**
"""

# Load training log
log_file = "files/non_aug/log-non-aug.csv"
history_df = pd.read_csv(log_file)

# Check available columns
print("Available columns:", history_df.columns)

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history_df["loss"], label="Training Loss", color="blue", linestyle="dashed")
plt.plot(history_df["val_loss"], label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history_df["accuracy"], label="Training Accuracy", color="green", linestyle="dashed")
plt.plot(history_df["val_accuracy"], label="Validation Accuracy", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()

"""**Plot prediciton**"""

# Function to read and normalize an image for visualization
def read_image_for_plot(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # Read image in color mode (H, W, 3)
    x = x / 255.0  # Normalize pixel values to range [0,1]
    return x

# Function to read and preprocess a multi-class mask for visualization
def read_mask_for_plot(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # Read mask in color mode
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x.astype(np.uint8)  # Ensure integer labels

# Select a sample validation image and mask
sample_image_path = valid_x[5]  # Select an image path
sample_mask_path = valid_y[5]  # Select corresponding mask path
sample_image_path = "/content/dataset/aug/val/images/HL1.tif_patch_25_69.tif"
sample_mask_path = "/content/dataset/aug/val/masks/HL1.png_patch_25_69.png"


sample_image = read_image_for_plot(sample_image_path)  # Load image

sample_mask = read_mask_for_plot(sample_mask_path)


# Ensure correct shape for model input
input_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

# Predict the mask
pred_mask = model.predict(input_image)  # Model outputs (H, W, num_classes)


# Apply softmax if model outputs logits
pred_mask = tf.nn.softmax(pred_mask[0], axis=-1)  # Convert logits to probabilities
pred_mask = np.argmax(pred_mask, axis=-1)  # Convert probabilities to class indices
decoded_mask = decode_mask(pred_mask)  # Convert to RGB mask for visualization

# Plot the original image, ground truth, and predicted mask
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(sample_image)  # Original image
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(sample_mask)  # Ground truth mask (converted to RGB)
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")

axes[2].imshow(decoded_mask)  # Predicted mask
axes[2].set_title("Predicted Mask")
axes[2].axis("off")

plt.show()