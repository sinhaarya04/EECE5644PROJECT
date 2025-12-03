"""
Run inpainting methods with given hyperparameters on specified images.
Computes mean metrics and timing (in milliseconds) for LFW dataset.
Saves reconstruction images only for Angelina Jolie.
"""

import numpy as np
import cv2
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.utils import (
    load_mask, apply_mask_to_image, knn_interpolation, 
    navier_stokes_inpainting, compute_metrics, 
    get_mask_filename, load_image, format_time
)

# Default configuration (can be overridden via command-line arguments)
DEFAULT_BASE_PATH = "default path"
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_BASE_PATH, "dataset")
DEFAULT_INPUT_ORIGINAL_ROOT = os.path.join(DEFAULT_DATASET_PATH, "lfw_100_people")
DEFAULT_MASK_ROOT = DEFAULT_DATASET_PATH
DEFAULT_OUTPUT_ROOT = os.path.join(DEFAULT_DATASET_PATH, "inpainting_results_lfw")

MASK_LEVELS = [20, 40, 60, 80]
TARGET_SIZE = (256, 256)

# Fixed hyperparameters (set these based on CV results)
K_UNIFORM = 4
K_DISTANCE = 6

# Default values
DEFAULT_N_IMAGES = 5
DEFAULT_N_IMAGES_TO_VISUALIZE = 7
DEFAULT_PERSON_TO_SAVE = None


def create_output_dirs(output_root):
    """Create output directory structure."""
    for mask_level in MASK_LEVELS:
        level_str = f"{mask_level}%"
        for method in ['knn_uniform', 'knn_distance', 'navier_stokes']:
            method_dir = os.path.join(output_root, level_str, method)
            os.makedirs(method_dir, exist_ok=True)
        masked_dir = os.path.join(output_root, level_str, "masked_images")
        os.makedirs(masked_dir, exist_ok=True)
    
    comp_dir = os.path.join(output_root, "comprehensive_comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    
    print(f"Created output directories in: {output_root}")


def should_save_image(image_filename, person_filter=None):
    """
    Check if image should be saved based on person filter.
    
    Parameters:
    - image_filename: name of the image file
    - person_filter: person name to filter (e.g., "Angelina_Jolie")
    
    Returns:
    - True if should save, False otherwise
    """
    if person_filter is None:
        return True
    
    # Check if filename contains the person's name
    return person_filter.lower() in image_filename.lower()


def process_image(original_path, mask_path, mask_level):
    """
    Process a single image with all interpolation methods.
    
    Returns:
    - original_img: original image
    - masked_img: masked image
    - results: dictionary with results from each method
    - mask: the binary mask used
    - metrics: dictionary with PSNR and SSIM values
    - timings: dictionary with execution times IN MILLISECONDS
    """
    original_img = load_image(original_path, TARGET_SIZE)
    mask = load_mask(mask_path, TARGET_SIZE)
    masked_img = apply_mask_to_image(original_img, mask)
    
    results = {}
    metrics = {}
    timings = {}
    
    # 1. k-NN uniform
    start_time = time.time()
    results['knn_uniform'] = knn_interpolation(masked_img, mask, k=K_UNIFORM, weighted=False)
    timings['knn_uniform'] = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    psnr_u, ssim_u = compute_metrics(original_img, results['knn_uniform'])
    metrics['knn_uniform'] = {'psnr': psnr_u, 'ssim': ssim_u}
    
    # 2. k-NN distance
    start_time = time.time()
    results['knn_distance'] = knn_interpolation(masked_img, mask, k=K_DISTANCE, weighted=True)
    timings['knn_distance'] = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    psnr_d, ssim_d = compute_metrics(original_img, results['knn_distance'])
    metrics['knn_distance'] = {'psnr': psnr_d, 'ssim': ssim_d}
    
    # 3. Navier-Stokes
    start_time = time.time()
    results['navier_stokes'] = navier_stokes_inpainting(masked_img, mask)
    timings['navier_stokes'] = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    psnr_ns, ssim_ns = compute_metrics(original_img, results['navier_stokes'])
    metrics['navier_stokes'] = {'psnr': psnr_ns, 'ssim': ssim_ns}
    
    return original_img, masked_img, results, mask, metrics, timings


def save_results(original_img, masked_img, results, image_name, mask_level, output_root):
    """Save all results."""
    level_str = f"{mask_level}%"
    
    masked_path = os.path.join(output_root, level_str, "masked_images", image_name)
    cv2.imwrite(masked_path, masked_img)
    
    for method_name, result_img in results.items():
        method_path = os.path.join(output_root, level_str, method_name, image_name)
        cv2.imwrite(method_path, result_img)


def create_comprehensive_comparison(all_data, image_filename, output_root):
    """Create comprehensive comparison plot WITHOUT metrics in titles."""
    n_rows = len(MASK_LEVELS)
    n_cols = 8
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2.5))
    
    fig.suptitle(f'Baseline Inpainting Methods - LFW Dataset', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    for row_idx, mask_level in enumerate(MASK_LEVELS):
        data = all_data[mask_level]
        
        # Convert to RGB
        original_rgb = cv2.cvtColor(data['original'], cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(data['masked'], cv2.COLOR_BGR2RGB)
        knn_uniform_rgb = cv2.cvtColor(data['results']['knn_uniform'], cv2.COLOR_BGR2RGB)
        knn_distance_rgb = cv2.cvtColor(data['results']['knn_distance'], cv2.COLOR_BGR2RGB)
        navier_stokes_rgb = cv2.cvtColor(data['results']['navier_stokes'], cv2.COLOR_BGR2RGB)
        
        # Column 0: Original
        axes[row_idx, 0].imshow(original_rgb)
        if row_idx == 0:
            axes[row_idx, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Mask level label
        axes[row_idx, 0].text(-0.3, 0.5, f'{mask_level}%', 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=14, fontweight='bold', 
                             verticalalignment='center',
                             rotation=90)
        
        # Column 1: Masked
        axes[row_idx, 1].imshow(masked_rgb)
        if row_idx == 0:
            axes[row_idx, 1].set_title('Masked', fontsize=12, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Column 2: k-NN Uniform
        axes[row_idx, 2].imshow(knn_uniform_rgb)
        if row_idx == 0:
            axes[row_idx, 2].set_title(f'k-NN Uniform\n(k={K_UNIFORM})', 
                                      fontsize=11, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Column 3: k-NN Distance
        axes[row_idx, 3].imshow(knn_distance_rgb)
        if row_idx == 0:
            axes[row_idx, 3].set_title(f'Distance Weighted k-NN\n(k={K_DISTANCE})', 
                                      fontsize=11, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
        # Column 4: Navier-Stokes
        axes[row_idx, 4].imshow(navier_stokes_rgb)
        if row_idx == 0:
            axes[row_idx, 4].set_title('Navier-Stokes', 
                                      fontsize=11, fontweight='bold')
        axes[row_idx, 4].axis('off')
        
        # Column 5: Uniform Error
        error_uniform = np.abs(original_rgb.astype(float) - knn_uniform_rgb.astype(float)).mean(axis=2)
        im1 = axes[row_idx, 5].imshow(error_uniform, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 5].set_title('Uniform k-NN \nError', fontsize=11, fontweight='bold')
        axes[row_idx, 5].axis('off')
        
        # Column 6: Distance Error
        error_distance = np.abs(original_rgb.astype(float) - knn_distance_rgb.astype(float)).mean(axis=2)
        axes[row_idx, 6].imshow(error_distance, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 6].set_title('Distance Weighted k-NN \nError', fontsize=11, fontweight='bold')
        axes[row_idx, 6].axis('off')
        
        # Column 7: NS Error
        error_ns = np.abs(original_rgb.astype(float) - navier_stokes_rgb.astype(float)).mean(axis=2)
        axes[row_idx, 7].imshow(error_ns, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 7].set_title('Navier-StokesS\nError', fontsize=11, fontweight='bold')
        axes[row_idx, 7].axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.96, 0.99])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Error (pixel intensity)')
    
    # Save
    comp_dir = os.path.join(output_root, "comprehensive_comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    
    stem = Path(image_filename).stem
    plot_path = os.path.join(comp_dir, f"comparison_{stem}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved comprehensive comparison")


def save_summary_statistics(all_metrics_by_level, all_timings_by_level, output_root):
    """Save summary statistics for all mask levels."""
    summary_path = os.path.join(output_root, "summary_statistics.txt")
    
    method_names = ['knn_uniform', 'knn_distance', 'navier_stokes']
    method_labels = [
        f'k-NN Uniform (k={K_UNIFORM})',
        f'k-NN Distance (k={K_DISTANCE})',
        'Navier-Stokes'
    ]
    
    with open(summary_path, 'w') as f:
        f.write("INPAINTING METHODS - SUMMARY STATISTICS (LFW Dataset)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total images processed: {len(all_metrics_by_level[MASK_LEVELS[0]]) if MASK_LEVELS and len(all_metrics_by_level[MASK_LEVELS[0]]) > 0 else 0}\n")
        f.write(f"Mask levels: {MASK_LEVELS}\n")
        f.write("\n")
        
        for mask_level in MASK_LEVELS:
            if mask_level not in all_metrics_by_level or len(all_metrics_by_level[mask_level]) == 0:
                continue
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"MASK LEVEL: {mask_level}%\n")
            f.write("=" * 80 + "\n\n")
            
            metrics_list = all_metrics_by_level[mask_level]
            timings_list = all_timings_by_level[mask_level]
            
            n_images = len(metrics_list)
            f.write(f"Number of images processed: {n_images}\n\n")
            
            for method_name, method_label in zip(method_names, method_labels):
                # Collect metrics
                psnr_values = [m[method_name]['psnr'] for m in metrics_list]
                ssim_values = [m[method_name]['ssim'] for m in metrics_list]
                time_values = [t[method_name] for t in timings_list]  # Already in milliseconds
                
                # Compute statistics
                psnr_mean = np.mean(psnr_values)
                psnr_std = np.std(psnr_values)
                psnr_min = np.min(psnr_values)
                psnr_max = np.max(psnr_values)
                
                ssim_mean = np.mean(ssim_values)
                ssim_std = np.std(ssim_values)
                ssim_min = np.min(ssim_values)
                ssim_max = np.max(ssim_values)
                
                time_mean = np.mean(time_values)
                time_std = np.std(time_values)
                time_total = np.sum(time_values)
                
                f.write(f"{method_label}:\n")
                f.write(f"  PSNR:\n")
                f.write(f"    Mean: {psnr_mean:.2f} dB\n")
                f.write(f"    Std:  {psnr_std:.2f} dB\n")
                f.write(f"    Range: [{psnr_min:.2f}, {psnr_max:.2f}] dB\n")
                f.write(f"  SSIM:\n")
                f.write(f"    Mean: {ssim_mean:.4f}\n")
                f.write(f"    Std:  {ssim_std:.4f}\n")
                f.write(f"    Range: [{ssim_min:.4f}, {ssim_max:.4f}]\n")
                f.write(f"  Time:\n")
                f.write(f"    Mean per image: {time_mean:.2f} ms\n")
                f.write(f"    Std:  {time_std:.2f} ms\n")
                f.write(f"    Total for {n_images} images: {time_total:.2f} ms ({time_total/1000:.2f} seconds)\n\n")
    
    print(f"\nSaved summary statistics to: {summary_path}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS - LFW DATASET")
    print("=" * 80)
    
    for mask_level in MASK_LEVELS:
        if mask_level not in all_metrics_by_level or len(all_metrics_by_level[mask_level]) == 0:
            continue
        
        print(f"\nMask Level: {mask_level}%")
        print("-" * 80)
        
        metrics_list = all_metrics_by_level[mask_level]
        timings_list = all_timings_by_level[mask_level]
        n_images = len(metrics_list)
        
        print(f"Images processed: {n_images}")
        print()
        
        for method_name, method_label in zip(method_names, method_labels):
            psnr_values = [m[method_name]['psnr'] for m in metrics_list]
            ssim_values = [m[method_name]['ssim'] for m in metrics_list]
            time_values = [t[method_name] for t in timings_list]  # In milliseconds
            
            psnr_mean = np.mean(psnr_values)
            ssim_mean = np.mean(ssim_values)
            time_mean = np.mean(time_values)
            
            print(f"  {method_label:30s}: PSNR={psnr_mean:.2f}dB, SSIM={ssim_mean:.4f}, Time={time_mean:.2f}ms/image")


def plot_summary_statistics(all_metrics_by_level, all_timings_by_level, output_root):
    """Create plots showing mean metrics and timing across mask levels."""
    method_names = ['knn_uniform', 'knn_distance', 'navier_stokes']
    method_labels = ['k-NN Uniform', 'k-NN Distance', 'Navier-Stokes']
    colors = ['blue', 'orange', 'green']
    markers = ['o', 's', '^']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Prepare data
    mask_levels_with_data = []
    psnr_by_method = {m: [] for m in method_names}
    psnr_std_by_method = {m: [] for m in method_names}
    ssim_by_method = {m: [] for m in method_names}
    ssim_std_by_method = {m: [] for m in method_names}
    time_by_method = {m: [] for m in method_names}
    time_std_by_method = {m: [] for m in method_names}
    
    for mask_level in MASK_LEVELS:
        if mask_level not in all_metrics_by_level or len(all_metrics_by_level[mask_level]) == 0:
            continue
        
        mask_levels_with_data.append(mask_level)
        metrics_list = all_metrics_by_level[mask_level]
        timings_list = all_timings_by_level[mask_level]
        
        for method_name in method_names:
            psnr_values = [m[method_name]['psnr'] for m in metrics_list]
            ssim_values = [m[method_name]['ssim'] for m in metrics_list]
            time_values = [t[method_name] for t in timings_list]  # In milliseconds
            
            psnr_by_method[method_name].append(np.mean(psnr_values))
            psnr_std_by_method[method_name].append(np.std(psnr_values))
            ssim_by_method[method_name].append(np.mean(ssim_values))
            ssim_std_by_method[method_name].append(np.std(ssim_values))
            time_by_method[method_name].append(np.mean(time_values))
            time_std_by_method[method_name].append(np.std(time_values))
    
    # Plot 1: PSNR vs Mask Level
    for method_name, method_label, color, marker in zip(method_names, method_labels, colors, markers):
        axes[0].errorbar(mask_levels_with_data, psnr_by_method[method_name],
                        yerr=psnr_std_by_method[method_name],
                        marker=marker, linewidth=2, markersize=8, 
                        label=method_label, color=color, capsize=4)
    
    axes[0].set_xlabel('Mask Level (%)', fontsize=12)
    axes[0].set_ylabel('Mean PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR vs Mask Level', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: SSIM vs Mask Level
    for method_name, method_label, color, marker in zip(method_names, method_labels, colors, markers):
        axes[1].errorbar(mask_levels_with_data, ssim_by_method[method_name],
                        yerr=ssim_std_by_method[method_name],
                        marker=marker, linewidth=2, markersize=8, 
                        label=method_label, color=color, capsize=4)
    
    axes[1].set_xlabel('Mask Level (%)', fontsize=12)
    axes[1].set_ylabel('Mean SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Mask Level', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Time vs Mask Level (IN MILLISECONDS)
    for method_name, method_label, color, marker in zip(method_names, method_labels, colors, markers):
        axes[2].errorbar(mask_levels_with_data, time_by_method[method_name],
                        yerr=time_std_by_method[method_name],
                        marker=marker, linewidth=2, markersize=8, 
                        label=method_label, color=color, capsize=4)
    
    axes[2].set_xlabel('Mask Level (%)', fontsize=12)
    axes[2].set_ylabel('Mean Time per Image (ms)', fontsize=12)  # Changed to ms
    axes[2].set_title('Execution Time vs Mask Level', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_root, "summary_statistics.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary statistics plot to: {plot_path}")


def main():
    """Main function to run inpainting with fixed hyperparameters."""
    parser = argparse.ArgumentParser(description="Run inpainting methods on LFW dataset")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_ORIGINAL_ROOT,
                       help=f"Directory containing original images (default: {DEFAULT_INPUT_ORIGINAL_ROOT})")
    parser.add_argument("--mask_root", type=str, default=DEFAULT_MASK_ROOT,
                       help=f"Root directory containing mask subdirectories (default: {DEFAULT_MASK_ROOT})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_ROOT,
                       help=f"Output directory for results (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--n_images", type=int, default=DEFAULT_N_IMAGES,
                       help=f"Number of images to process (default: {DEFAULT_N_IMAGES})")
    parser.add_argument("--n_visualize", type=int, default=DEFAULT_N_IMAGES_TO_VISUALIZE,
                       help=f"Number of images to create comprehensive comparisons for (default: {DEFAULT_N_IMAGES_TO_VISUALIZE})")
    parser.add_argument("--person_filter", type=str, default=DEFAULT_PERSON_TO_SAVE,
                       help="Only save reconstruction images for this person (default: None, saves all)")
    
    args = parser.parse_args()
    
    # Use arguments or defaults
    input_original_root = args.input_dir
    mask_root = args.mask_root
    output_root = args.output_dir
    n_images = args.n_images
    n_images_to_visualize = args.n_visualize
    person_to_save = args.person_filter
    
    print("=" * 80)
    print("Inpainting Evaluation - LFW Dataset")
    print("=" * 80)
    print(f"Input directory: {input_original_root}")
    print(f"Mask root: {mask_root}")
    print(f"Output directory: {output_root}")
    print(f"k-NN Uniform:  k = {K_UNIFORM}")
    print(f"k-NN Distance: k = {K_DISTANCE}")
    print(f"Total images to process: {n_images}")
    print(f"Comprehensive visualizations: First {n_images_to_visualize} images")
    if person_to_save:
        print(f"Saving reconstruction images: Only for '{person_to_save}'")
    else:
        print(f"Saving reconstruction images: All images")
    print("=" * 80)
    print()
    
    total_start_time = time.time()
    
    create_output_dirs(output_root)
    
    # Get all images
    original_images = sorted([f for f in os.listdir(input_original_root) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(original_images) == 0:
        print(f"Error: No images found in {input_original_root}")
        return
    
    print(f"Found {len(original_images)} total images in dataset\n")
    
    # Limit to n_images
    images_to_process = original_images[:n_images]
    print(f"Processing first {len(images_to_process)} images\n")
    
    # Storage for metrics and timings per mask level
    all_metrics_by_level = {level: [] for level in MASK_LEVELS}
    all_timings_by_level = {level: [] for level in MASK_LEVELS}
    
    # Track how many images saved
    images_saved = 0
    
    # Process images
    for img_idx, image_file in enumerate(images_to_process):
        print(f"\n{'='*80}")
        print(f"Processing image {img_idx+1}/{len(images_to_process)}: {image_file}")
        
        # Progress estimation
        if img_idx > 0:
            elapsed = time.time() - total_start_time
            avg_time_per_image = elapsed / img_idx
            remaining_images = len(images_to_process) - img_idx
            eta = avg_time_per_image * remaining_images
            print(f"Progress: {img_idx}/{len(images_to_process)} | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")
        
        print(f"{'='*80}")
        
        # Check if this image should be saved
        save_this_image = should_save_image(image_file, person_to_save)
        if save_this_image:
            print(f"  ✓ Will save reconstruction images for this person")
        
        # Process across all mask levels
        image_data_all_levels = {}
        
        for mask_level in MASK_LEVELS:
            print(f"\n  Mask level: {mask_level}%")
            
            mask_dir = os.path.join(mask_root, str(mask_level))
            original_path = os.path.join(input_original_root, image_file)
            mask_filename = get_mask_filename(img_idx)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                print(f"    ERROR: Mask not found: {mask_path}")
                continue
            
            try:
                original_img, masked_img, results, mask, metrics, timings = \
                    process_image(original_path, mask_path, mask_level)
                
                # Print metrics for this image (time in milliseconds)
                print(f"    - k-NN uniform:  PSNR={metrics['knn_uniform']['psnr']:.2f}dB, SSIM={metrics['knn_uniform']['ssim']:.4f}, Time={timings['knn_uniform']:.2f}ms")
                print(f"    - k-NN distance: PSNR={metrics['knn_distance']['psnr']:.2f}dB, SSIM={metrics['knn_distance']['ssim']:.4f}, Time={timings['knn_distance']:.2f}ms")
                print(f"    - Navier-Stokes: PSNR={metrics['navier_stokes']['psnr']:.2f}dB, SSIM={metrics['navier_stokes']['ssim']:.4f}, Time={timings['navier_stokes']:.2f}ms")
                
                # Store data for visualization (only for first n_images_to_visualize)
                if img_idx < n_images_to_visualize:
                    image_data_all_levels[mask_level] = {
                        'original': original_img,
                        'masked': masked_img,
                        'results': results,
                        'mask': mask,
                        'metrics': metrics
                    }
                
                # Collect metrics and timings for ALL images
                all_metrics_by_level[mask_level].append(metrics)
                all_timings_by_level[mask_level].append(timings)
                
                # Save individual results ONLY if person matches filter
                if save_this_image:
                    save_results(original_img, masked_img, results, image_file, mask_level, output_root)
                    if mask_level == MASK_LEVELS[0]:  # Only increment once per image
                        images_saved += 1
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create comprehensive comparison only for first n_images_to_visualize
        if img_idx < n_images_to_visualize and len(image_data_all_levels) == len(MASK_LEVELS):
            print(f"\n  Creating comprehensive comparison...")
            create_comprehensive_comparison(image_data_all_levels, image_file, output_root)
    
    total_time = time.time() - total_start_time
    
    # Save and display summary statistics
    print("\n" + "=" * 80)
    print("Computing summary statistics...")
    print("=" * 80)
    save_summary_statistics(all_metrics_by_level, all_timings_by_level, output_root)
    
    # Create summary plots
    print("\nCreating summary plots...")
    plot_summary_statistics(all_metrics_by_level, all_timings_by_level, output_root)
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total images processed: {len(images_to_process)}")
    if person_to_save:
        print(f"Images saved to disk: {images_saved} (filtered for '{person_to_save}')")
    else:
        print(f"Images saved to disk: {len(images_to_process)} (all images)")
    print(f"Total execution time: {format_time(total_time)} ({total_time:.2f}s)")
    print(f"Average time per image: {total_time/len(images_to_process)*1000:.2f}ms")  # Show in ms
    print()
    print(f"Results saved to: {output_root}")
    print(f"  - Summary statistics: {output_root}/summary_statistics.txt")
    print(f"  - Summary plots: {output_root}/summary_statistics.png")
    print(f"  - Comprehensive comparisons: {output_root}/comprehensive_comparisons/ (first {n_images_to_visualize} images)")
    if person_to_save:
        print(f"  - Reconstruction images: Only saved for '{person_to_save}'")
    print("=" * 80)


if __name__ == "__main__":
    main()