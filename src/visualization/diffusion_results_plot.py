"""
Create comprehensive comparison plots for diffusion model results.
Uses only files in the diffusion_results directory.
"""

import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Default configuration (can be overridden via command-line arguments)
DEFAULT_BASE_PATH = "default path"
DEFAULT_DIFFUSION_RESULTS_PATH = os.path.join(DEFAULT_BASE_PATH, "diffusion_results")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_BASE_PATH, "diffusion_comparisons")

MASK_LEVELS = [20, 40, 60, 80]
TARGET_SIZE = (256, 256)

# Map files to their data - EXACT FILENAMES
FILE_MAPPING = {
    'celeba': {
        'original': 'CelebA_face00002.jpg',
        20: {
            'inpainted': 'CelebA_face00002_mask20_inpainted.png',
            'mask': 'mask20_29999.png'
        },
        40: {
            'inpainted': 'CelebA_face00002_mask40_inpainted.png',
            'mask': 'mask40_29999.png'
        },
        60: {
            'inpainted': 'CelebA_face00002_mask60_inpainted.png',
            'mask': 'mask60_29999.png'
        },
        80: {
            'inpainted': 'CelebA_face00002_mask80_inpainted.png',
            'mask': 'mask80_29999.png'
        }
    },
    'Mauresmo': {
        'original': 'Amelie Mauresmo_3_original_image.jpg',
        20: {
            'inpainted': 'Mauresmo_Masked20inpainted.png',
            'mask': 'mask20_forMauresmo.png'
        },
        40: {
            'inpainted': 'Moresmo_Masked40inpainted.png',
            'mask': 'mask40_forMauresmo.png'
        },
        60: {
            'inpainted': 'Mauresmo_Masked60inpainted.png',
            'mask': 'mask60_forMauresmo.png'
        },
        80: {
            'inpainted': 'Mauresmo_Masked80inpainted.png',
            'mask': 'mask80_forMauresmo.png'
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
    
    # White = masked, Black = valid
    binary_mask = mask_img > threshold
    return binary_mask


def apply_mask_to_image(original_img, mask):
    """Create masked image by setting masked pixels to black."""
    masked_img = original_img.copy()
    masked_img[mask] = 0
    return masked_img


def create_comprehensive_comparison_for_person(person_name, person_files, results_path, output_path):
    """
    Create comprehensive comparison plot for one person across all mask levels.
    
    Layout:
    - Rows: mask levels (20%, 40%, 60%, 80%)
    - Columns: Original, Masked, Diffusion Output, Error Heatmap
    """
    n_rows = len(MASK_LEVELS)
    n_cols = 4  # Original, Masked, Inpainted, Error
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    
    fig.suptitle(f'Diffusion Model Inpainting CelebA dataset', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    valid_rows = 0
    last_error_im = None
    
    for row_idx, mask_level in enumerate(MASK_LEVELS):
        level_data = person_files[mask_level]
        
        # Check if we have all required files
        if person_files['original'] is None or 'inpainted' not in level_data or 'mask' not in level_data:
            print(f"  Warning: Missing files for {person_name} at {mask_level}%")
            print(f"    Original: {person_files.get('original', 'MISSING')}")
            print(f"    Inpainted: {level_data.get('inpainted', 'MISSING')}")
            print(f"    Mask: {level_data.get('mask', 'MISSING')}")
            # Hide this row
            for col in range(n_cols):
                axes[row_idx, col].axis('off')
            continue
        
        try:
            # Build full paths
            original_path = os.path.join(results_path, person_files['original'])
            inpainted_path = os.path.join(results_path, level_data['inpainted'])
            mask_path = os.path.join(results_path, level_data['mask'])
            
            # Check if files exist
            if not os.path.exists(original_path):
                print(f"  Error: Original file not found: {original_path}")
                for col in range(n_cols):
                    axes[row_idx, col].axis('off')
                continue
            
            if not os.path.exists(inpainted_path):
                print(f"  Error: Inpainted file not found: {inpainted_path}")
                for col in range(n_cols):
                    axes[row_idx, col].axis('off')
                continue
            
            if not os.path.exists(mask_path):
                print(f"  Error: Mask file not found: {mask_path}")
                for col in range(n_cols):
                    axes[row_idx, col].axis('off')
                continue
            
            print(f"  {mask_level}%: Loading files...")
            print(f"    Original:  {person_files['original']}")
            print(f"    Inpainted: {level_data['inpainted']}")
            print(f"    Mask:      {level_data['mask']}")
            
            # Load images
            original_img = load_and_resize(original_path, TARGET_SIZE)
            inpainted_img = load_and_resize(inpainted_path, TARGET_SIZE)
            mask = load_mask_from_png(mask_path, TARGET_SIZE)
            
            # Create masked image
            masked_img = apply_mask_to_image(original_img, mask)
            
            # Convert to RGB
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            inpainted_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
            
            # Column 0: Original
            axes[row_idx, 0].imshow(original_rgb)
            if row_idx == 0:
                axes[row_idx, 0].set_title('Original', fontsize=14, fontweight='bold')
            axes[row_idx, 0].axis('off')
            
            # Mask level label
            axes[row_idx, 0].text(-0.25, 0.5, f'{mask_level}%', 
                                 transform=axes[row_idx, 0].transAxes,
                                 fontsize=16, fontweight='bold', 
                                 verticalalignment='center',
                                 rotation=90)
            
            # Column 1: Masked
            axes[row_idx, 1].imshow(masked_rgb)
            if row_idx == 0:
                axes[row_idx, 1].set_title('Masked', fontsize=14, fontweight='bold')
            axes[row_idx, 1].axis('off')
            
            # Column 2: Diffusion Output
            axes[row_idx, 2].imshow(inpainted_rgb)
            if row_idx == 0:
                axes[row_idx, 2].set_title('Diffusion', 
                                          fontsize=14, fontweight='bold')
            axes[row_idx, 2].axis('off')
            
            # Column 3: Error Heatmap
            error = np.abs(original_rgb.astype(float) - inpainted_rgb.astype(float)).mean(axis=2)
            last_error_im = axes[row_idx, 3].imshow(error, cmap='hot', vmin=0, vmax=50)
            if row_idx == 0:
                axes[row_idx, 3].set_title('Diffusion Error', 
                                          fontsize=14, fontweight='bold')
            axes[row_idx, 3].axis('off')
            
            valid_rows += 1
            
        except Exception as e:
            print(f"  Error processing {person_name} at {mask_level}%: {str(e)}")
            import traceback
            traceback.print_exc()
            # Hide this row
            for col in range(n_cols):
                axes[row_idx, col].axis('off')
            continue
    
    if valid_rows == 0:
        print(f"  No valid data for {person_name}, skipping plot")
        plt.close()
        return
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.94, 0.99])
    
    # Add colorbar if we have at least one error heatmap
    if last_error_im is not None:
        cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
        fig.colorbar(last_error_im, cax=cbar_ax, label='Error (pixel intensity)')
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    plot_path = os.path.join(output_path, f"diffusion_comparison_{person_name}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plot: {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create comprehensive comparison plots for diffusion model results")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_DIFFUSION_RESULTS_PATH,
                       help=f"Directory containing diffusion results (default: {DEFAULT_DIFFUSION_RESULTS_PATH})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_PATH,
                       help=f"Output directory for plots (default: {DEFAULT_OUTPUT_PATH})")
    
    args = parser.parse_args()
    
    results_path = args.results_dir
    output_path = args.output_dir
    
    print("=" * 80)
    print("Diffusion Model Results - Comprehensive Comparison")
    print("=" * 80)
    print(f"Results directory: {results_path}")
    print(f"Output directory: {output_path}")
    print("=" * 80)
    print()
    
    # Check if directory exists
    if not os.path.exists(results_path):
        print(f"Error: Directory not found: {results_path}")
        return
    
    # Verify all files exist
    print("Verifying files...")
    all_files_found = True
    
    for person_name, person_files in FILE_MAPPING.items():
        print(f"\n{person_name}:")
        
        # Check original
        original_path = os.path.join(results_path, person_files['original'])
        if os.path.exists(original_path):
            print(f"  ✓ Original: {person_files['original']}")
        else:
            print(f"  ✗ Original NOT FOUND: {person_files['original']}")
            all_files_found = False
        
        # Check each mask level
        for level in MASK_LEVELS:
            level_data = person_files[level]
            
            inpainted_path = os.path.join(results_path, level_data['inpainted'])
            mask_path = os.path.join(results_path, level_data['mask'])
            
            inpainted_exists = os.path.exists(inpainted_path)
            mask_exists = os.path.exists(mask_path)
            
            status = "✓" if (inpainted_exists and mask_exists) else "✗"
            print(f"  {status} {level}%: inpainted={inpainted_exists}, mask={mask_exists}")
            
            if not inpainted_exists:
                print(f"      Missing: {level_data['inpainted']}")
            if not mask_exists:
                print(f"      Missing: {level_data['mask']}")
    
    if not all_files_found:
        print("\n⚠ Warning: Some files are missing, but will proceed with available data")
    
    print("\n" + "=" * 80)
    print("Creating comparison plots...")
    print("=" * 80)
    
    # Create comparison plots
    for person_name, person_files in FILE_MAPPING.items():
        print(f"\nProcessing {person_name}...")
        try:
            create_comprehensive_comparison_for_person(person_name, person_files, results_path, output_path)
        except Exception as e:
            print(f"  Error processing {person_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"Comparison plots saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()