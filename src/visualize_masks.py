"""
Visualize sample masks from each percentage level (20%, 40%, 60%, 80%)
"""
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def visualize_masks(num_samples=4):
    """
    Visualize sample masks from each percentage level.
    
    Args:
        num_samples: Number of mask samples to show per percentage level
    """
    mask_dirs = {
        20: "masks/20",
        40: "masks/40",
        60: "masks/60",
        80: "masks/80"
    }
    
    fig, axes = plt.subplots(len(mask_dirs), num_samples, figsize=(15, 12))
    fig.suptitle('Mask Samples at Different Coverage Percentages', fontsize=16, fontweight='bold')
    
    for row_idx, (percent, mask_dir) in enumerate(mask_dirs.items()):
        # Get random mask files
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        selected_masks = random.sample(mask_files, min(num_samples, len(mask_files)))
        
        for col_idx, mask_file in enumerate(selected_masks):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            
            # Calculate actual coverage percentage
            masked_pixels = np.sum(mask_array == 255)
            total_pixels = mask_array.size
            actual_percent = (masked_pixels / total_pixels) * 100
            
            # Display mask
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[row_idx]
            ax.imshow(mask_array, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'{percent}% (actual: {actual_percent:.1f}%)', fontsize=10)
            ax.axis('off')
    
    # Add row labels
    for row_idx, percent in enumerate(mask_dirs.keys()):
        if num_samples > 1:
            axes[row_idx, 0].set_ylabel(f'{percent}% Coverage', fontsize=12, fontweight='bold')
        else:
            axes[row_idx].set_ylabel(f'{percent}% Coverage', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mask_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'mask_visualization.png'")
    plt.show()

if __name__ == "__main__":
    print("Generating mask visualization...")
    print("Showing 4 random samples from each percentage level (20%, 40%, 60%, 80%)")
    visualize_masks(num_samples=4)

