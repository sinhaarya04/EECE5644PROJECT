"""
Visualize faces with masks applied at different coverage percentages (20%, 40%, 60%, 80%)
"""
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def visualize_masked_faces(num_samples=4):
    """
    Visualize sample faces with masks applied at different percentages.
    
    Args:
        num_samples: Number of face samples to show per percentage level
    """
    img_dir = "CelebAMask-HQ/CelebA-HQ-img"
    mask_dirs = {
        20: "masks/20",
        40: "masks/40",
        60: "masks/60",
        80: "masks/80"
    }
    
    # Get random image files
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(img_files, min(num_samples, len(img_files)))
    
    fig, axes = plt.subplots(len(mask_dirs), num_samples, figsize=(15, 12))
    fig.suptitle('Faces with Masks at Different Coverage Percentages', fontsize=16, fontweight='bold')
    
    for row_idx, (percent, mask_dir) in enumerate(mask_dirs.items()):
        # Get random mask files
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        for col_idx, img_file in enumerate(selected_images):
            # Load image
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Load random mask
            mask_file = random.choice(mask_files)
            mask_path = os.path.join(mask_dir, mask_file)
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            
            # Resize mask to match image if needed
            if mask_array.shape != img_array.shape[:2]:
                mask = mask.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
                mask_array = np.array(mask)
            
            # Apply mask: white (255) = masked, black (0) = visible
            # Convert mask to 0-1 range where 1 = masked
            mask_normalized = (mask_array == 255).astype(np.float32)
            
            # Create masked image: masked regions become black
            masked_img = img_array.copy().astype(np.float32)
            for c in range(3):  # Apply mask to each color channel
                masked_img[:, :, c] = masked_img[:, :, c] * (1 - mask_normalized)
            masked_img = masked_img.astype(np.uint8)
            
            # Calculate actual coverage percentage
            masked_pixels = np.sum(mask_array == 255)
            total_pixels = mask_array.size
            actual_percent = (masked_pixels / total_pixels) * 100
            
            # Display masked face
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[row_idx]
            ax.imshow(masked_img)
            ax.set_title(f'{percent}% (actual: {actual_percent:.1f}%)', fontsize=10)
            ax.axis('off')
    
    # Add row labels
    for row_idx, percent in enumerate(mask_dirs.keys()):
        if num_samples > 1:
            axes[row_idx, 0].set_ylabel(f'{percent}% Coverage', fontsize=12, fontweight='bold')
        else:
            axes[row_idx].set_ylabel(f'{percent}% Coverage', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('masked_faces_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'masked_faces_visualization.png'")
    plt.show()

if __name__ == "__main__":
    print("Generating visualization of faces with masks...")
    print("Showing 4 random faces with masks at each percentage level (20%, 40%, 60%, 80%)")
    visualize_masked_faces(num_samples=4)

