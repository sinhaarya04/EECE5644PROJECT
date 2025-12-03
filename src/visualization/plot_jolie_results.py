"""
Create comprehensive comparison plot for Angelina Jolie's reconstruction results.
Shows both results and error heatmaps in one plot.
"""

import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Default configuration (can be overridden via command-line arguments)
DEFAULT_BASE_PATH = "default path"
DEFAULT_RESULTS_PATH = os.path.join(DEFAULT_BASE_PATH, "jolie_results")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_BASE_PATH, "jolie_comparisons")

MASK_LEVELS = [20, 40, 60, 80]
TARGET_SIZE = (256, 256)

# File structure
FILES = {
    'original': 'Angelina_Jolie_original.jpg',
    'masks': {
        20: 'maskjolie20.png',
        40: 'maskjolie40.png',
        60: 'maskjolie60.png',
        80: 'maskjolie80.png'
    },
    'methods': {
        'knn_distance': {
            20: 'Distance_knn/Angelina_Jolie_20.jpg',
            40: 'Distance_knn/Angelina_Jolie_40.jpg',
            60: 'Distance_knn/Angelina_Jolie_60.jpg',
            80: 'Distance_knn/Angelina_Jolie_80.jpg'
        },
        'navier_stokes': {
            20: 'naive_stokes/Angelina_Jolie_20.jpg',
            40: 'naive_stokes/Angelina_Jolie_40.jpg',
            60: 'naive_stokes/Angelina_Jolie_60.jpg',
            80: 'naive_stokes/Angelina_Jolie_80.jpg'
        },
        'convcp': {
            20: 'convcp/Angelina_Jolie_20.png',
            40: 'convcp/Angelina_Jolie_40.png',
            60: 'convcp/Angelina_Jolie_60.png',
            80: 'convcp/Angelina_Jolie_80.png'
        },
        'diffusion': {
            20: 'diffusion/Angelina_Jolie_20.png',
            40: 'diffusion/Angelina_Jolie_40.png',
            60: 'diffusion/Angelina_Jolie_60.png',
            80: 'diffusion/Angelina_Jolie_80.png'
        }
    }
}


def load_and_resize(image_path, target_size=(256, 256)):
    """Load image and resize if needed."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size)
    
    return img


def load_mask_from_png(mask_path, target_size=(256, 256), threshold=127):
    """Load binary mask from PNG file."""
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Could not read mask from {mask_path}")
    
    if mask_img.shape[:2] != target_size:
        mask_img = cv2.resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask_img > threshold
    return binary_mask


def apply_mask_to_image(original_img, mask):
    """Create masked image by setting masked pixels to black."""
    masked_img = original_img.copy()
    masked_img[mask] = 0
    return masked_img


def create_comprehensive_comparison(results_path, output_path):
    """
    Create comprehensive comparison plot with results AND error heatmaps.
    
    Layout:
    - Rows: mask levels (20%, 40%, 60%, 80%)
    - Columns: Original, Masked, k-NN Distance, Navier-Stokes, ConvCP, Diffusion,
               k-NN Error, NS Error, ConvCP Error, Diffusion Error
    """
    n_rows = len(MASK_LEVELS)
    n_cols = 10  # Original, Masked, 4 methods, 4 errors
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, n_rows * 2.5))
    
    fig.suptitle('Inpainting Methods Comparison - Angelina Jolie', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Load original image
    original_path = os.path.join(results_path, FILES['original'])
    original_img = load_and_resize(original_path, TARGET_SIZE)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    last_im = None
    
    for row_idx, mask_level in enumerate(MASK_LEVELS):
        # Load mask
        mask_path = os.path.join(results_path, FILES['masks'][mask_level])
        mask = load_mask_from_png(mask_path, TARGET_SIZE)
        
        # Create masked image
        masked_img = apply_mask_to_image(original_img, mask)
        masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        
        # Load all method results
        knn_path = os.path.join(results_path, FILES['methods']['knn_distance'][mask_level])
        ns_path = os.path.join(results_path, FILES['methods']['navier_stokes'][mask_level])
        convcp_path = os.path.join(results_path, FILES['methods']['convcp'][mask_level])
        diffusion_path = os.path.join(results_path, FILES['methods']['diffusion'][mask_level])
        
        knn_rgb = cv2.cvtColor(load_and_resize(knn_path, TARGET_SIZE), cv2.COLOR_BGR2RGB)
        ns_rgb = cv2.cvtColor(load_and_resize(ns_path, TARGET_SIZE), cv2.COLOR_BGR2RGB)
        convcp_rgb = cv2.cvtColor(load_and_resize(convcp_path, TARGET_SIZE), cv2.COLOR_BGR2RGB)
        diffusion_rgb = cv2.cvtColor(load_and_resize(diffusion_path, TARGET_SIZE), cv2.COLOR_BGR2RGB)
        
        # Mask level label
        axes[row_idx, 0].text(-0.25, 0.5, f'{mask_level}%', 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=15, fontweight='bold', 
                             verticalalignment='center',
                             rotation=90)
        
        # Column 0: Original
        axes[row_idx, 0].imshow(original_rgb)
        if row_idx == 0:
            axes[row_idx, 0].set_title('Original', fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Column 1: Masked
        axes[row_idx, 1].imshow(masked_rgb)
        if row_idx == 0:
            axes[row_idx, 1].set_title('Masked', fontsize=11, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Column 2: k-NN Distance
        axes[row_idx, 2].imshow(knn_rgb)
        if row_idx == 0:
            axes[row_idx, 2].set_title('k-NN\nDistance', fontsize=10, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Column 3: Navier-Stokes
        axes[row_idx, 3].imshow(ns_rgb)
        if row_idx == 0:
            axes[row_idx, 3].set_title('Navier-\nStokes', fontsize=10, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
        # Column 4: ConvCP
        axes[row_idx, 4].imshow(convcp_rgb)
        if row_idx == 0:
            axes[row_idx, 4].set_title('ConvCP', fontsize=10, fontweight='bold')
        axes[row_idx, 4].axis('off')
        
        # Column 5: Diffusion
        axes[row_idx, 5].imshow(diffusion_rgb)
        if row_idx == 0:
            axes[row_idx, 5].set_title('Diffusion', fontsize=10, fontweight='bold')
        axes[row_idx, 5].axis('off')
        
        # Column 6: k-NN Error
        error_knn = np.abs(original_rgb.astype(float) - knn_rgb.astype(float)).mean(axis=2)
        last_im = axes[row_idx, 6].imshow(error_knn, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 6].set_title('k-NN\nError', fontsize=10, fontweight='bold')
        axes[row_idx, 6].axis('off')
        
        # Column 7: NS Error
        error_ns = np.abs(original_rgb.astype(float) - ns_rgb.astype(float)).mean(axis=2)
        axes[row_idx, 7].imshow(error_ns, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 7].set_title('NS\nError', fontsize=10, fontweight='bold')
        axes[row_idx, 7].axis('off')
        
        # Column 8: ConvCP Error
        error_convcp = np.abs(original_rgb.astype(float) - convcp_rgb.astype(float)).mean(axis=2)
        axes[row_idx, 8].imshow(error_convcp, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 8].set_title('ConvCP\nError', fontsize=10, fontweight='bold')
        axes[row_idx, 8].axis('off')
        
        # Column 9: Diffusion Error
        error_diffusion = np.abs(original_rgb.astype(float) - diffusion_rgb.astype(float)).mean(axis=2)
        axes[row_idx, 9].imshow(error_diffusion, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 9].set_title('Diffusion\nError', fontsize=10, fontweight='bold')
        axes[row_idx, 9].axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.96, 0.99])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    fig.colorbar(last_im, cax=cbar_ax, label='Error (pixel intensity)')
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    plot_path = os.path.join(output_path, "jolie_comprehensive_comparison.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive comparison to: {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create comprehensive comparison plot for Angelina Jolie results")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_PATH,
                       help=f"Directory containing results (default: {DEFAULT_RESULTS_PATH})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_PATH,
                       help=f"Output directory for plots (default: {DEFAULT_OUTPUT_PATH})")
    
    args = parser.parse_args()
    
    results_path = args.results_dir
    output_path = args.output_dir
    
    print("=" * 80)
    print("Creating Angelina Jolie Comprehensive Comparison Plot")
    print("=" * 80)
    print(f"Results directory: {results_path}")
    print(f"Output directory: {output_path}")
    print("=" * 80)
    print()
    
    # Verify files exist
    print("Verifying files...")
    
    original_path = os.path.join(results_path, FILES['original'])
    if os.path.exists(original_path):
        print(f"✓ Original: {FILES['original']}")
    else:
        print(f"✗ Original NOT FOUND: {FILES['original']}")
        return
    
    all_files_exist = True
    
    for mask_level in MASK_LEVELS:
        mask_path = os.path.join(results_path, FILES['masks'][mask_level])
        exists = os.path.exists(mask_path)
        status = "✓" if exists else "✗"
        print(f"{status} Mask {mask_level}%: {FILES['masks'][mask_level]}")
        if not exists:
            all_files_exist = False
        
        for method_name in FILES['methods'].keys():
            method_path = os.path.join(results_path, FILES['methods'][method_name][mask_level])
            exists = os.path.exists(method_path)
            status = "✓" if exists else "✗"
            print(f"  {status} {method_name} {mask_level}%")
            if not exists:
                all_files_exist = False
    
    if not all_files_exist:
        print("\n⚠ Warning: Some files are missing, but will proceed with available data")
    
    print("\n" + "=" * 80)
    print("Creating comprehensive comparison plot...")
    print("=" * 80)
    
    try:
        create_comprehensive_comparison(results_path, output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Plot saved to: {output_path}/jolie_comprehensive_comparison.png")
    print("=" * 80)


if __name__ == "__main__":
    main()