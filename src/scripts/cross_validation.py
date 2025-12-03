"""
Cross-validation script for finding optimal k values.
"""

import numpy as np
import cv2
import os
import sys
import argparse
import time
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.utils import (
    load_mask, apply_mask_to_image, knn_interpolation, 
    compute_metrics, get_mask_filename, load_image, format_time
)

# Default configuration (can be overridden via command-line arguments)
DEFAULT_BASE_PATH = "default path"
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_BASE_PATH, "dataset")
DEFAULT_INPUT_ORIGINAL_ROOT = os.path.join(DEFAULT_DATASET_PATH, "lfw_100_people")
DEFAULT_MASK_ROOT = DEFAULT_DATASET_PATH
DEFAULT_OUTPUT_ROOT = os.path.join(DEFAULT_DATASET_PATH, "cv_results")

MASK_LEVELS = [20, 40, 60, 80]

# Cross-validation parameters
K_VALUES_TO_TEST = [2, 4, 6, 8, 10, 12, 16, 20, 24]
N_FOLDS = 5
RANDOM_SEED = 42

TARGET_SIZE = (256, 256)


def evaluate_k_on_images(image_indices, all_images, mask_levels, k_values, input_root, mask_root):
    """
    Evaluate all k values on a given set of images.
    
    Parameters:
    - image_indices: list of image indices to evaluate
    - all_images: list of all image filenames
    - mask_levels: list of mask levels to test
    - k_values: list of k values to test
    - input_root: directory containing original images
    - mask_root: root directory containing mask subdirectories
    
    Returns:
    - k_results: dictionary with PSNR/SSIM for each k value
    """
    k_results = {
        'uniform': {k: {'psnr': [], 'ssim': []} for k in k_values},
        'distance': {k: {'psnr': [], 'ssim': []} for k in k_values}
    }
    
    for img_idx in image_indices:
        image_file = all_images[img_idx]
        
        try:
            original_img = load_image(os.path.join(input_root, image_file), TARGET_SIZE)
        except:
            continue
        
        # Test on all mask levels
        for mask_level in mask_levels:
            mask_dir = os.path.join(mask_root, str(mask_level))
            mask_filename = get_mask_filename(img_idx)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                continue
            
            mask = load_mask(mask_path, TARGET_SIZE)
            masked_img = apply_mask_to_image(original_img, mask)
            
            # Test each k value
            for k in k_values:
                # Uniform weighting
                result_uniform = knn_interpolation(masked_img, mask, k=k, weighted=False)
                psnr_uniform, ssim_uniform = compute_metrics(original_img, result_uniform)
                k_results['uniform'][k]['psnr'].append(psnr_uniform)
                k_results['uniform'][k]['ssim'].append(ssim_uniform)
                
                # Distance weighting
                result_distance = knn_interpolation(masked_img, mask, k=k, weighted=True)
                psnr_distance, ssim_distance = compute_metrics(original_img, result_distance)
                k_results['distance'][k]['psnr'].append(psnr_distance)
                k_results['distance'][k]['ssim'].append(ssim_distance)
    
    return k_results


def optimize_k_with_cross_validation(original_images, mask_levels, input_root, mask_root, output_root):
    """
    Find optimal k value using k-fold cross-validation.
    
    Returns:
    - optimal_k_uniform: best k for uniform weighting
    - optimal_k_distance: best k for distance weighting
    - cv_results: detailed cross-validation results
    """
    print("=" * 80)
    print("K-VALUE OPTIMIZATION WITH CROSS-VALIDATION")
    print("=" * 80)
    print(f"Cross-validation: {N_FOLDS}-fold")
    print(f"Testing k values: {K_VALUES_TO_TEST}")
    print(f"Total images: {len(original_images)}")
    print()
    
    cv_start_time = time.time()
    
    # Create fold indices
    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(original_images))
    np.random.shuffle(indices)
    
    fold_size = len(original_images) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        if i < N_FOLDS - 1:
            fold = indices[i*fold_size:(i+1)*fold_size]
        else:
            fold = indices[i*fold_size:]
        folds.append(fold)
    
    cv_results = {
        'fold_optimal_k': {'uniform': [], 'distance': []},
        'fold_times': [],
        'fold_results': []
    }
    
    # Perform cross-validation
    for fold_idx in range(N_FOLDS):
        fold_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*80}")
        
        val_indices = folds[fold_idx]
        print(f"Validation set: {len(val_indices)} images")
        
        # Evaluate all k values on validation set
        fold_k_results = evaluate_k_on_images(val_indices, original_images, mask_levels, K_VALUES_TO_TEST, input_root, mask_root)
        
        # Find best k for this fold
        best_k_uniform_fold = max(K_VALUES_TO_TEST, 
                                  key=lambda k: np.mean(fold_k_results['uniform'][k]['psnr']))
        best_k_distance_fold = max(K_VALUES_TO_TEST, 
                                   key=lambda k: np.mean(fold_k_results['distance'][k]['psnr']))
        
        print(f"  Best k (uniform):  {best_k_uniform_fold}")
        print(f"  Best k (distance): {best_k_distance_fold}")
        
        cv_results['fold_optimal_k']['uniform'].append(best_k_uniform_fold)
        cv_results['fold_optimal_k']['distance'].append(best_k_distance_fold)
        cv_results['fold_results'].append(fold_k_results)
        
        fold_elapsed = time.time() - fold_start_time
        cv_results['fold_times'].append(fold_elapsed)
        
        print(f"  Fold completed in {format_time(fold_elapsed)}")
    
    total_cv_time = time.time() - cv_start_time
    
    # Aggregate results
    optimal_k_uniform = Counter(cv_results['fold_optimal_k']['uniform']).most_common(1)[0][0]
    optimal_k_distance = Counter(cv_results['fold_optimal_k']['distance']).most_common(1)[0][0]
    
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total CV time: {format_time(total_cv_time)}")
    print(f"Optimal k per fold:")
    print(f"  Uniform:  {cv_results['fold_optimal_k']['uniform']}")
    print(f"  Distance: {cv_results['fold_optimal_k']['distance']}")
    print(f"\nFinal optimal k (mode):")
    print(f"  Uniform:  k = {optimal_k_uniform}")
    print(f"  Distance: k = {optimal_k_distance}")
    
    # Save results
    save_cv_results(cv_results, optimal_k_uniform, optimal_k_distance, total_cv_time, output_root)
    plot_cv_results(cv_results, optimal_k_uniform, optimal_k_distance, output_root)
    
    return optimal_k_uniform, optimal_k_distance


def save_cv_results(cv_results, optimal_k_uniform, optimal_k_distance, total_time, output_root):
    """Save cross-validation results to file."""
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "cv_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("CROSS-VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Number of folds: {N_FOLDS}\n")
        f.write(f"  K values tested: {K_VALUES_TO_TEST}\n")
        f.write(f"  Random seed: {RANDOM_SEED}\n\n")
        
        f.write(f"Timing:\n")
        f.write(f"  Total CV time: {format_time(total_time)}\n")
        f.write(f"  Average per fold: {format_time(np.mean(cv_results['fold_times']))}\n\n")
        
        f.write("Fold times:\n")
        for i, t in enumerate(cv_results['fold_times'], 1):
            f.write(f"  Fold {i}: {format_time(t)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Results:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Optimal k per fold:\n")
        f.write(f"  Uniform:  {cv_results['fold_optimal_k']['uniform']}\n")
        f.write(f"  Distance: {cv_results['fold_optimal_k']['distance']}\n\n")
        
        f.write("Final selection (mode):\n")
        f.write(f"  Uniform:  k = {optimal_k_uniform}\n")
        f.write(f"  Distance: k = {optimal_k_distance}\n")
        
        # Detailed results per k
        f.write("\n" + "=" * 80 + "\n")
        f.write("Average PSNR/SSIM per k-value across all folds:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Uniform Weighting:\n")
        for k in K_VALUES_TO_TEST:
            all_psnr = []
            all_ssim = []
            for fold_results in cv_results['fold_results']:
                all_psnr.extend(fold_results['uniform'][k]['psnr'])
                all_ssim.extend(fold_results['uniform'][k]['ssim'])
            
            avg_psnr = np.mean(all_psnr)
            std_psnr = np.std(all_psnr)
            avg_ssim = np.mean(all_ssim)
            std_ssim = np.std(all_ssim)
            
            f.write(f"  k={k:2d}: PSNR={avg_psnr:.2f}±{std_psnr:.2f} dB, SSIM={avg_ssim:.4f}±{std_ssim:.4f}\n")
        
        f.write("\nDistance Weighting:\n")
        for k in K_VALUES_TO_TEST:
            all_psnr = []
            all_ssim = []
            for fold_results in cv_results['fold_results']:
                all_psnr.extend(fold_results['distance'][k]['psnr'])
                all_ssim.extend(fold_results['distance'][k]['ssim'])
            
            avg_psnr = np.mean(all_psnr)
            std_psnr = np.std(all_psnr)
            avg_ssim = np.mean(all_ssim)
            std_ssim = np.std(all_ssim)
            
            f.write(f"  k={k:2d}: PSNR={avg_psnr:.2f}±{std_psnr:.2f} dB, SSIM={avg_ssim:.4f}±{std_ssim:.4f}\n")
    
    print(f"\nSaved CV report to: {report_path}")


def plot_cv_results(cv_results, optimal_k_uniform, optimal_k_distance, output_root):
    """Plot cross-validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    k_values = K_VALUES_TO_TEST
    
    # Aggregate PSNR across all folds for uniform
    uniform_psnr_by_k = {k: [] for k in k_values}
    for fold_results in cv_results['fold_results']:
        for k in k_values:
            avg_psnr = np.mean(fold_results['uniform'][k]['psnr'])
            uniform_psnr_by_k[k].append(avg_psnr)
    
    uniform_psnr_means = [np.mean(uniform_psnr_by_k[k]) for k in k_values]
    uniform_psnr_stds = [np.std(uniform_psnr_by_k[k]) for k in k_values]
    
    axes[0, 0].errorbar(k_values, uniform_psnr_means, yerr=uniform_psnr_stds,
                        marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0, 0].axvline(optimal_k_uniform, color='r', linestyle='--',
                      label=f'Optimal k={optimal_k_uniform}', linewidth=2)
    axes[0, 0].set_xlabel('k value', fontsize=12)
    axes[0, 0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0, 0].set_title('k-NN Uniform - PSNR', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Aggregate SSIM for uniform
    uniform_ssim_by_k = {k: [] for k in k_values}
    for fold_results in cv_results['fold_results']:
        for k in k_values:
            avg_ssim = np.mean(fold_results['uniform'][k]['ssim'])
            uniform_ssim_by_k[k].append(avg_ssim)
    
    uniform_ssim_means = [np.mean(uniform_ssim_by_k[k]) for k in k_values]
    uniform_ssim_stds = [np.std(uniform_ssim_by_k[k]) for k in k_values]
    
    axes[0, 1].errorbar(k_values, uniform_ssim_means, yerr=uniform_ssim_stds,
                        marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0, 1].axvline(optimal_k_uniform, color='r', linestyle='--',
                      label=f'Optimal k={optimal_k_uniform}', linewidth=2)
    axes[0, 1].set_xlabel('k value', fontsize=12)
    axes[0, 1].set_ylabel('SSIM', fontsize=12)
    axes[0, 1].set_title('k-NN Uniform - SSIM', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Distance PSNR
    distance_psnr_by_k = {k: [] for k in k_values}
    for fold_results in cv_results['fold_results']:
        for k in k_values:
            avg_psnr = np.mean(fold_results['distance'][k]['psnr'])
            distance_psnr_by_k[k].append(avg_psnr)
    
    distance_psnr_means = [np.mean(distance_psnr_by_k[k]) for k in k_values]
    distance_psnr_stds = [np.std(distance_psnr_by_k[k]) for k in k_values]
    
    axes[1, 0].errorbar(k_values, distance_psnr_means, yerr=distance_psnr_stds,
                        marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
    axes[1, 0].axvline(optimal_k_distance, color='r', linestyle='--',
                      label=f'Optimal k={optimal_k_distance}', linewidth=2)
    axes[1, 0].set_xlabel('k value', fontsize=12)
    axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1, 0].set_title('k-NN Distance - PSNR', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Distance SSIM
    distance_ssim_by_k = {k: [] for k in k_values}
    for fold_results in cv_results['fold_results']:
        for k in k_values:
            avg_ssim = np.mean(fold_results['distance'][k]['ssim'])
            distance_ssim_by_k[k].append(avg_ssim)
    
    distance_ssim_means = [np.mean(distance_ssim_by_k[k]) for k in k_values]
    distance_ssim_stds = [np.std(distance_ssim_by_k[k]) for k in k_values]
    
    axes[1, 1].errorbar(k_values, distance_ssim_means, yerr=distance_ssim_stds,
                        marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
    axes[1, 1].axvline(optimal_k_distance, color='r', linestyle='--',
                      label=f'Optimal k={optimal_k_distance}', linewidth=2)
    axes[1, 1].set_xlabel('k value', fontsize=12)
    axes[1, 1].set_ylabel('SSIM', fontsize=12)
    axes[1, 1].set_title('k-NN Distance - SSIM', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    os.makedirs(output_root, exist_ok=True)
    plot_path = os.path.join(output_root, "cv_results.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved CV plots to: {plot_path}")
    
    return optimal_k_uniform, optimal_k_distance


def main():
    """Main cross-validation function."""
    parser = argparse.ArgumentParser(description="Cross-validation for k-value optimization")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_ORIGINAL_ROOT,
                       help=f"Directory containing original images (default: {DEFAULT_INPUT_ORIGINAL_ROOT})")
    parser.add_argument("--mask_root", type=str, default=DEFAULT_MASK_ROOT,
                       help=f"Root directory containing mask subdirectories (default: {DEFAULT_MASK_ROOT})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_ROOT,
                       help=f"Output directory for results (default: {DEFAULT_OUTPUT_ROOT})")
    
    args = parser.parse_args()
    
    input_root = args.input_dir
    mask_root = args.mask_root
    output_root = args.output_dir
    
    print("Starting cross-validation for k-value optimization...\n")
    print(f"Input directory: {input_root}")
    print(f"Mask root: {mask_root}")
    print(f"Output directory: {output_root}\n")
    
    original_images = sorted([f for f in os.listdir(input_root) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(original_images) == 0:
        print(f"Error: No images found in {input_root}")
        return
    
    print(f"Found {len(original_images)} images\n")
    
    optimal_k_uniform, optimal_k_distance = optimize_k_with_cross_validation(
        original_images, MASK_LEVELS, input_root, mask_root, output_root
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Optimal k (uniform):  {optimal_k_uniform}")
    print(f"Optimal k (distance): {optimal_k_distance}")
    print(f"\nResults saved to: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()