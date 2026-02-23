# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from PIL import Image

# Directories for input TIFFs and output patches
input_image_path = "UAV_ORTHO.tif"   # Path to the input TIFF image
input_mask_path = "LS_Poly.tif"     # Path to the input TIFF mask

# Create directories to store the patches
os.makedirs("/train_images", exist_ok=True)
os.makedirs("/train_masks", exist_ok=True)

# Patch size
PATCH_SIZE = 128

def slice_image(image, patch_size, save_dir, prefix):
    """Slices the image into (patch_size x patch_size) patches and saves them."""
    width, height = image.size
    num_patches = 0

    # Loop through the image in steps of patch_size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # Crop the patch
            patch = image.crop((j, i, j + patch_size, i + patch_size))

            # Ensure the patch size matches the desired size (handle borders)
            if patch.size != (patch_size, patch_size):
                patch = patch.resize((patch_size, patch_size))

            # Save the patch
            patch.save(os.path.join(save_dir, f"{prefix}_{num_patches}.png"))
            num_patches += 1

# Open the input TIFF files
image = Image.open(input_image_path).convert("RGB")  # Ensure it's 3-channel
mask = Image.open(input_mask_path).convert("L")      # Single channel for masks

# Slice and save the patches
slice_image(image, PATCH_SIZE, "train_images", "image")
slice_image(mask, PATCH_SIZE, "train_masks", "image")

print("Slicing completed! Check 'train_images/' and 'train_masks/'.")

