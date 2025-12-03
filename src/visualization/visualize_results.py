"""
Visualize RePaint results: Original, Masked, and Inpainted images side by side
"""
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def visualize_repaint_results(result_dir, num_samples=6):
    """
    Visualize RePaint results showing original, masked, and inpainted images.
    
    Args:
        result_dir: Path to RePaint log directory (e.g., 'RePaint/log/celeba_256_20')
        num_samples: Number of samples to visualize
    """
    # Define paths
    gt_dir = os.path.join(result_dir, 'gt')
    masked_dir = os.path.join(result_dir, 'gt_masked')
    inpainted_dir = os.path.join(result_dir, 'inpainted')
    
    # Check if directories exist
    if not os.path.exists(gt_dir):
        print(f"Error: {gt_dir} does not exist. Run RePaint first!")
        return
    
    # Get image files (use inpainted as reference since it's the output)
    img_files = sorted([f for f in os.listdir(inpainted_dir) if f.endswith(('.png', '.jpg'))])
    
    if len(img_files) == 0:
        print(f"Error: No images found in {inpainted_dir}")
        return
    
    # Select samples
    num_samples = min(num_samples, len(img_files))
    selected_files = img_files[:num_samples]  # Take first N samples
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('RePaint Results: Original | Masked Input | Inpainted Output', 
                  fontsize=16, fontweight='bold')
    
    for row_idx, img_file in enumerate(selected_files):
        # Load images
        gt_path = os.path.join(gt_dir, img_file)
        masked_path = os.path.join(masked_dir, img_file)
        inpainted_path = os.path.join(inpainted_dir, img_file)
        
        # Load and display original
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB')
            axes[row_idx, 0].imshow(gt_img)
            axes[row_idx, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        else:
            axes[row_idx, 0].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[row_idx, 0].set_title('Original Image (missing)', fontsize=12)
        
        axes[row_idx, 0].axis('off')
        
        # Load and display masked
        if os.path.exists(masked_path):
            masked_img = Image.open(masked_path).convert('RGB')
            axes[row_idx, 1].imshow(masked_img)
            axes[row_idx, 1].set_title('Masked Input', fontsize=12, fontweight='bold')
        else:
            axes[row_idx, 1].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[row_idx, 1].set_title('Masked Input (missing)', fontsize=12)
        
        axes[row_idx, 1].axis('off')
        
        # Load and display inpainted
        if os.path.exists(inpainted_path):
            inpainted_img = Image.open(inpainted_path).convert('RGB')
            axes[row_idx, 2].imshow(inpainted_img)
            axes[row_idx, 2].set_title('Inpainted Output', fontsize=12, fontweight='bold')
        else:
            axes[row_idx, 2].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[row_idx, 2].set_title('Inpainted Output (missing)', fontsize=12)
        
        axes[row_idx, 2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(result_dir, 'visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as '{output_path}'")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        # Default: look for most recent result directory
        log_base = "RePaint/log"
        if os.path.exists(log_base):
            # Find all result directories
            result_dirs = [d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))]
            if result_dirs:
                # Use the first one found (or you can modify to use most recent)
                result_dir = os.path.join(log_base, result_dirs[0])
                print(f"Using result directory: {result_dir}")
            else:
                print(f"Error: No result directories found in {log_base}")
                print("Usage: python visualize_results.py <path_to_result_dir>")
                sys.exit(1)
        else:
            print(f"Error: {log_base} does not exist")
            print("Usage: python visualize_results.py <path_to_result_dir>")
            sys.exit(1)
    
    print(f"Visualizing results from: {result_dir}")
    visualize_repaint_results(result_dir, num_samples=6)

