#!/bin/bash

# Define directories to remove and recreate
directories=(
    "256_patches/images"
    "256_patches/masks"
    "256_patches/images_with_useful_info/images"
    "256_patches/images_with_useful_info/masks"
    "dataset/train/images"
    "dataset/train/masks"
    "dataset/val/images"
    "dataset/val/masks"
)

# Remove and recreate each directory
for dir in "${directories[@]}"; do
    echo "Removing $dir..."
    rm -rf "$dir"
    echo "Recreating $dir..."
    mkdir -p "$dir"
done

echo "✅ All directories cleared and recreated!"