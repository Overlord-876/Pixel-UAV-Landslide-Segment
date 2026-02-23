# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:57:21 2024

@author: ALOK BHARDWAJ
"""

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os



img_path = "D:/TMI-102/ATALI_1/UAV_ORTHO.tif"
model_path = "D:/TMI-102/ATALI_1/saved_model/U-Net.keras"
plot_path = "D:/TMI-102/ATALI_1/Plots/Prediction_Atali.tiff"
output_path = "D:/TMI-102/ATALI_1/LS_Poly.png"
mask_path = "D:/TMI-102/ATALI_1/LS_Poly.tif"

model = load_model(model_path)

def load_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img_array = np.array(img)/255.0 
    
    return np.expand_dims(img_array, axis=0)

def segment_predict(image_path):
    input_image = load_preprocess_image(image_path)
    prediction = model.predict(input_image)
    
    predicted_mask = prediction.reshape(128, 128)
    
    return prediction, predicted_mask

def plot(predicted_mask, image_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        
    axes[0].imshow(Image.open(image_path))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(predicted_mask, cmap='jet', alpha = 0.6)
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    plt.show()

prediction, predicted_mask = segment_predict(img_path)

plot(predicted_mask, img_path)

def iou_score(y_true, y_pred, threshold=0.0001, epsilon=1e-7):
    """
    Calculates the Intersection over Union (IoU) score.
    
    Parameters:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        threshold (float): Threshold to binarize `y_pred`.
        epsilon (float): Small value to prevent division by zero.
        
    Returns:
        float: IoU score.
    """
    # Binarize predictions based on threshold
    y_pred = (y_pred > threshold).astype(np.float32)
    y_true = (y_true > 0).astype(np.float32)
    # Calculate intersection and union
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    
    # Compute IoU with a small epsilon to avoid division by zero
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou

def load_ground_truth_mask(mask_path):
    # Load and preprocess the ground truth mask to match the dimensions (128, 128)
    mask = Image.open(mask_path)
    mask = mask.resize((128, 128))
    mask_array = np.array(mask) / 255.0  # Normalize if needed
   
    return mask_array

mask_arr = load_ground_truth_mask(mask_path)

iou = iou_score(mask_arr, predicted_mask)
print(f"IoU Score: {iou:.4f}")

