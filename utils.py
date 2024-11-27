import torch
import torchvision
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def save_augmented_samples(original_images, augmented_images_list, augmentation_names):
    """Save original and augmented images for visualization"""
    
    # Create directory if it doesn't exist
    save_dir = 'augmentation_samples'
    os.makedirs(save_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # For each image in the batch
    for idx in range(original_images.shape[0]):
        plt.figure(figsize=(15, 3))
        
        # Plot original
        plt.subplot(1, 4, 1)
        plt.imshow(original_images[idx].squeeze(), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Plot each augmentation
        for aug_idx, (aug_image, aug_name) in enumerate(zip(augmented_images_list, augmentation_names)):
            plt.subplot(1, 4, aug_idx + 2)
            plt.imshow(aug_image[idx].squeeze(), cmap='gray')
            plt.title(aug_name)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/augmented_sample_{timestamp}_{idx}.png')
        plt.close() 