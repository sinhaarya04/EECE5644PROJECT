import numpy as np
import cv2
import os
import sys
import argparse
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from datetime import timedelta
from collections import Counter

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
DEFAULT_OUTPUT_ROOT = os.path.join(DEFAULT_DATASET_PATH, "output_interpolations")

MASK_LEVELS = [20, 40, 60, 80]

# Cross-validation parameters
K_VALUES_TO_TEST = [2, 4, 6, 8, 10, 12, 16, 20, 24]
N_FOLDS = 5  # 5-fold cross-validation
RANDOM_SEED = 42  # For reproducibility

TARGET_SIZE = (256, 256)
MASK_THRESHOLD = 10

METHODS = ['knn_uniform', 'knn_distance', 'navier_stokes']


def create_output_dirs(output_root):
    """Create output directory structure."""
    for mask_level in MASK_LEVELS:
        level_str = f"{mask_level}%"
        for method in METHODS:
            method_dir = os.path.join(output_root, level_str, method)
            os.makedirs(method_dir, exist_ok=True)
        masked_dir = os.path.join(output_root, level_str, "masked_images")
        os.makedirs(masked_dir, exist_ok=True)
    
    comp_dir = os.path.join(output_root, "comprehensive_comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    
    k_opt_dir = os.path.join(output_root, "k_optimization")
    os.makedirs(k_opt_dir, exist_ok=True)
    
    print(f"Created output directories in: {output_root}")


def load_mask(mask_path, threshold=127):
    """Load binary mask from PNG file."""
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Could not read mask from {mask_path}")
    
    if mask_img.shape[:2] != TARGET_SIZE:
        mask_img = cv2.resize(mask_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask_img > threshold
    return binary_mask


def apply_mask_to_image(img, mask, mask_threshold=MASK_THRESHOLD):
    """Apply mask to image by setting masked pixels to black."""
    masked_img = img.copy()
    masked_img[mask] = 0
    return masked_img


def knn_interpolation(img, mask, k=8, weighted=True):
    """Fill masked regions using k-nearest neighbors interpolation."""
    valid_mask = ~mask
    h, w = mask.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    valid_points = np.column_stack((y_coords[valid_mask], x_coords[valid_mask]))
    target_points = np.column_stack((y_coords[mask], x_coords[mask]))
    
    if len(valid_points) == 0 or len(target_points) == 0:
        return img.copy()
    
    tree = KDTree(valid_points)
    result = img.copy()
    
    if len(img.shape) == 3:
        for c in range(img.shape[2]):
            valid_values = img[:, :, c][valid_mask]
            
            if len(valid_values) > 0:
                distances, indices = tree.query(target_points, k=min(k, len(valid_values)))
                
                if k == 1:
                    distances = distances.reshape(-1, 1)
                    indices = indices.reshape(-1, 1)
                
                if weighted:
                    weights = 1.0 / (distances + 1e-10)
                    weights = weights / np.sum(weights, axis=1, keepdims=True)
                else:
                    weights = np.ones_like(distances) / distances.shape[1]
                
                interpolated = np.sum(valid_values[indices] * weights, axis=1)
                result[mask, c] = interpolated
    else:
        valid_values = img[valid_mask]
        
        if len(valid_values) > 0:
            distances, indices = tree.query(target_points, k=min(k, len(valid_values)))
            
            if k == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)
            
            if weighted:
                weights = 1.0 / (distances + 1e-10)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
            else:
                weights = np.ones_like(distances) / distances.shape[1]
            
            interpolated = np.sum(valid_values[indices] * weights, axis=1)
            result[mask] = interpolated
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def navier_stokes_inpainting(img, mask, inpaint_radius=3):
    """Fill masked regions using Navier-Stokes based inpainting."""
    inpaint_mask = mask.astype(np.uint8) * 255
    result = cv2.inpaint(img, inpaint_mask, inpaintRadius=inpaint_radius, 
                        flags=cv2.INPAINT_NS)
    return result


def compute_metrics(original, reconstructed):
    """Compute PSNR and SSIM between original and reconstructed images."""
    psnr_value = psnr(original, reconstructed, data_range=255)
    
    if len(original.shape) == 3:
        ssim_value = ssim(original, reconstructed, 
                         channel_axis=2, 
                         data_range=255)
    else:
        ssim_value = ssim(original, reconstructed, 
                         data_range=255)
    
    return psnr_value, ssim_value


def format_time(seconds):
    """Format seconds into human-readable string."""
    return str(timedelta(seconds=int(seconds)))


def evaluate_k_on_images(image_indices, all_images, mask_levels, k_values, input_root, mask_root):
    """
    Evaluate all k values on a given set of images.
    
    Returns:
    - k_results: dictionary with PSNR/SSIM for each k value
    """
    k_results = {
        'uniform': {k: {'psnr': [], 'ssim': []} for k in k_values},
        'distance': {k: {'psnr': [], 'ssim': []} for k in k_values}
    }
    
    for img_idx in image_indices:
        image_file = all_images[img_idx]
        
        original_path = os.path.join(input_root, image_file)
        original_img = cv2.imread(original_path)
        
        if original_img is None:
            continue
        
        if original_img.shape[:2] != TARGET_SIZE:
            original_img = cv2.resize(original_img, TARGET_SIZE)
        
        # Test on all mask levels
        for mask_level in mask_levels:
            mask_dir = os.path.join(mask_root, str(mask_level))
            mask_filename = f"mask_{img_idx:05d}.png"
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                continue
            
            mask = load_mask(mask_path)
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
    - timing_info: timing information
    """
    print("\n" + "=" * 80)
    print("STEP 1: K-VALUE OPTIMIZATION WITH CROSS-VALIDATION")
    print("=" * 80)
    print(f"Cross-validation: {N_FOLDS}-fold")
    print(f"Testing k values: {K_VALUES_TO_TEST}")
    print(f"Total images: {len(original_images)}")
    print(f"Images per fold: ~{len(original_images) // N_FOLDS}")
    print(f"Total evaluations: {N_FOLDS} folds × {len(original_images) // N_FOLDS} images × {len(mask_levels)} masks × {len(K_VALUES_TO_TEST)} k-values × 2 methods")
    print(f"                 ≈ {N_FOLDS * (len(original_images) // N_FOLDS) * len(mask_levels) * len(K_VALUES_TO_TEST) * 2} interpolations")
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
            # Last fold gets all remaining images
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
        
        # Validation set for this fold
        val_indices = folds[fold_idx]
        
        # FIXED DISPLAY: Show fold position and sample of actual indices
        fold_start = fold_idx * fold_size
        fold_end = fold_start + len(val_indices) - 1
        sample_indices = sorted(val_indices)[:5]  # Show first 5 sorted indices
        
        print(f"Validation set: {len(val_indices)} images")
        print(f"  Fold position: images {fold_start}-{fold_end} in shuffled order")
        print(f"  Sample original indices: {sample_indices[0]}, {sample_indices[1]}, {sample_indices[2]}, ...")
        
        
        # print(f"Validation set: {len(val_indices)} images (indices {val_indices[0]}-{val_indices[-1]})")
        
        # Evaluate all k values on validation set
        print(f"Evaluating {len(K_VALUES_TO_TEST)} k-values on validation set...")
        
        fold_k_results = evaluate_k_on_images(val_indices, original_images, mask_levels, K_VALUES_TO_TEST, input_root, mask_root)
        
        # Find best k for this fold
        print("\nResults for this fold:")
        
        # Uniform
        best_k_uniform_fold = None
        best_psnr_uniform_fold = -np.inf
        
        print("  Uniform weighting:")
        for k in K_VALUES_TO_TEST:
            avg_psnr = np.mean(fold_k_results['uniform'][k]['psnr'])
            avg_ssim = np.mean(fold_k_results['uniform'][k]['ssim'])
            print(f"    k={k:2d}: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
            
            if avg_psnr > best_psnr_uniform_fold:
                best_psnr_uniform_fold = avg_psnr
                best_k_uniform_fold = k
        
        print(f"  --> Best k (uniform) for fold {fold_idx+1}: {best_k_uniform_fold}")
        
        # Distance
        best_k_distance_fold = None
        best_psnr_distance_fold = -np.inf
        
        print("  Distance weighting:")
        for k in K_VALUES_TO_TEST:
            avg_psnr = np.mean(fold_k_results['distance'][k]['psnr'])
            avg_ssim = np.mean(fold_k_results['distance'][k]['ssim'])
            print(f"    k={k:2d}: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
            
            if avg_psnr > best_psnr_distance_fold:
                best_psnr_distance_fold = avg_psnr
                best_k_distance_fold = k
        
        print(f"  --> Best k (distance) for fold {fold_idx+1}: {best_k_distance_fold}")
        
        # Store fold results
        cv_results['fold_optimal_k']['uniform'].append(best_k_uniform_fold)
        cv_results['fold_optimal_k']['distance'].append(best_k_distance_fold)
        cv_results['fold_results'].append(fold_k_results)
        
        fold_elapsed = time.time() - fold_start_time
        cv_results['fold_times'].append(fold_elapsed)
        
        print(f"\nFold {fold_idx+1} completed in {format_time(fold_elapsed)}")
    
    total_cv_time = time.time() - cv_start_time
    
    # Aggregate results across folds
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total CV time: {format_time(total_cv_time)} ({total_cv_time:.2f}s)")
    print(f"Average time per fold: {format_time(np.mean(cv_results['fold_times']))}")
    print()
    
    print("Optimal k per fold:")
    print(f"  Uniform:  {cv_results['fold_optimal_k']['uniform']}")
    print(f"  Distance: {cv_results['fold_optimal_k']['distance']}")
    print()
    
    # Use mode (most common k) or mean
    optimal_k_uniform = Counter(cv_results['fold_optimal_k']['uniform']).most_common(1)[0][0]
    optimal_k_distance = Counter(cv_results['fold_optimal_k']['distance']).most_common(1)[0][0]
    
    print("Final optimal k (most common across folds):")
    print(f"  Uniform:  k = {optimal_k_uniform}")
    print(f"  Distance: k = {optimal_k_distance}")
    
    timing_info = {
        'total_cv_time': total_cv_time,
        'fold_times': cv_results['fold_times'],
        'avg_fold_time': np.mean(cv_results['fold_times']),
        'optimal_k_uniform': optimal_k_uniform,
        'optimal_k_distance': optimal_k_distance
    }
    
    # Save CV results
    save_cv_results(cv_results, timing_info, output_root)
    
    # Plot CV results
    plot_cv_results(cv_results, optimal_k_uniform, optimal_k_distance, output_root)
    
    return optimal_k_uniform, optimal_k_distance, cv_results, timing_info


def save_cv_results(cv_results, timing_info, output_root):
    """Save cross-validation results to file."""
    k_opt_dir = os.path.join(output_root, "k_optimization")
    report_path = os.path.join(k_opt_dir, "cv_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("CROSS-VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Number of folds: {N_FOLDS}\n")
        f.write(f"  K values tested: {K_VALUES_TO_TEST}\n")
        f.write(f"  Random seed: {RANDOM_SEED}\n\n")
        
        f.write(f"Timing:\n")
        f.write(f"  Total CV time: {format_time(timing_info['total_cv_time'])} ({timing_info['total_cv_time']:.2f}s)\n")
        f.write(f"  Average time per fold: {format_time(timing_info['avg_fold_time'])}\n\n")
        
        f.write("Fold times:\n")
        for i, t in enumerate(timing_info['fold_times'], 1):
            f.write(f"  Fold {i}: {format_time(t)} ({t:.2f}s)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Results:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Optimal k per fold:\n")
        f.write(f"  Uniform:  {cv_results['fold_optimal_k']['uniform']}\n")
        f.write(f"  Distance: {cv_results['fold_optimal_k']['distance']}\n\n")
        
        f.write("Final selection (mode):\n")
        f.write(f"  Uniform:  k = {timing_info['optimal_k_uniform']}\n")
        f.write(f"  Distance: k = {timing_info['optimal_k_distance']}\n")
    
    print(f"\n  Saved CV report to: {report_path}")


def plot_cv_results(cv_results, optimal_k_uniform, optimal_k_distance, output_root):
    """Plot cross-validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Aggregate results across all folds
    k_values = K_VALUES_TO_TEST
    
    # Uniform - PSNR across folds
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
                      label=f'Optimal k={optimal_k_uniform}')
    axes[0, 0].set_xlabel('k value', fontsize=12)
    axes[0, 0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0, 0].set_title('k-NN Uniform - PSNR (CV)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Distance - PSNR across folds
    distance_psnr_by_k = {k: [] for k in k_values}
    for fold_results in cv_results['fold_results']:
        for k in k_values:
            avg_psnr = np.mean(fold_results['distance'][k]['psnr'])
            distance_psnr_by_k[k].append(avg_psnr)
    
    distance_psnr_means = [np.mean(distance_psnr_by_k[k]) for k in k_values]
    distance_psnr_stds = [np.std(distance_psnr_by_k[k]) for k in k_values]
    
    axes[0, 1].errorbar(k_values, distance_psnr_means, yerr=distance_psnr_stds,
                        marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
    axes[0, 1].axvline(optimal_k_distance, color='r', linestyle='--',
                      label=f'Optimal k={optimal_k_distance}')
    axes[0, 1].set_xlabel('k value', fontsize=12)
    axes[0, 1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0, 1].set_title('k-NN Distance - PSNR (CV)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Fold consistency - Uniform
    fold_numbers = list(range(1, N_FOLDS + 1))
    axes[1, 0].plot(fold_numbers, cv_results['fold_optimal_k']['uniform'],
                   marker='o', linewidth=2, markersize=10)
    axes[1, 0].axhline(optimal_k_uniform, color='r', linestyle='--',
                      label=f'Final k={optimal_k_uniform}')
    axes[1, 0].set_xlabel('Fold number', fontsize=12)
    axes[1, 0].set_ylabel('Optimal k', fontsize=12)
    axes[1, 0].set_title('Uniform - Optimal k per Fold', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(fold_numbers)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Fold consistency - Distance
    axes[1, 1].plot(fold_numbers, cv_results['fold_optimal_k']['distance'],
                   marker='s', linewidth=2, markersize=10, color='orange')
    axes[1, 1].axhline(optimal_k_distance, color='r', linestyle='--',
                      label=f'Final k={optimal_k_distance}')
    axes[1, 1].set_xlabel('Fold number', fontsize=12)
    axes[1, 1].set_ylabel('Optimal k', fontsize=12)
    axes[1, 1].set_title('Distance - Optimal k per Fold', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(fold_numbers)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    k_opt_dir = os.path.join(output_root, "k_optimization")
    plot_path = os.path.join(k_opt_dir, "cv_results.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved CV plots to: {plot_path}")


def process_image(original_path, mask_path, mask_level, k_uniform, k_distance):
    """Process a single image with optimized k values."""
    original_img = cv2.imread(original_path)
    if original_img is None:
        raise ValueError(f"Could not read image from {original_path}")
    
    if original_img.shape[:2] != TARGET_SIZE:
        original_img = cv2.resize(original_img, TARGET_SIZE)
    
    mask = load_mask(mask_path, threshold=127)
    masked_img = apply_mask_to_image(original_img, mask)
    
    print(f"  Processing: {os.path.basename(original_path)}")
    print(f"    - Mask pixels: {np.sum(mask)} / {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
    
    results = {}
    metrics = {}
    
    # 1. k-NN with uniform weights
    print(f"    - k-NN (uniform, k={k_uniform})...")
    results['knn_uniform'] = knn_interpolation(masked_img, mask, k=k_uniform, weighted=False)
    knn_uniform_psnr, knn_uniform_ssim = compute_metrics(original_img, results['knn_uniform'])
    metrics['knn_uniform'] = {'psnr': knn_uniform_psnr, 'ssim': knn_uniform_ssim}
    print(f"      PSNR: {knn_uniform_psnr:.2f} dB, SSIM: {knn_uniform_ssim:.4f}")
    
    # 2. k-NN with distance weights
    print(f"    - k-NN (distance, k={k_distance})...")
    results['knn_distance'] = knn_interpolation(masked_img, mask, k=k_distance, weighted=True)
    knn_distance_psnr, knn_distance_ssim = compute_metrics(original_img, results['knn_distance'])
    metrics['knn_distance'] = {'psnr': knn_distance_psnr, 'ssim': knn_distance_ssim}
    print(f"      PSNR: {knn_distance_psnr:.2f} dB, SSIM: {knn_distance_ssim:.4f}")
    
    # 3. Navier-Stokes
    print("    - Navier-Stokes...")
    results['navier_stokes'] = navier_stokes_inpainting(masked_img, mask, inpaint_radius=3)
    ns_psnr, ns_ssim = compute_metrics(original_img, results['navier_stokes'])
    metrics['navier_stokes'] = {'psnr': ns_psnr, 'ssim': ns_ssim}
    print(f"      PSNR: {ns_psnr:.2f} dB, SSIM: {ns_ssim:.4f}")
    
    return original_img, masked_img, results, mask, metrics


def save_results(original_img, masked_img, results, image_name, mask_level, output_root):
    """Save all results."""
    level_str = f"{mask_level}%"
    
    masked_path = os.path.join(output_root, level_str, "masked_images", image_name)
    cv2.imwrite(masked_path, masked_img)
    
    for method_name, result_img in results.items():
        method_path = os.path.join(output_root, level_str, method_name, image_name)
        cv2.imwrite(method_path, result_img)


def create_comprehensive_comparison(all_data, image_index, image_filename, k_uniform, k_distance, output_root):
    """Create comprehensive comparison plot."""
    n_rows = len(MASK_LEVELS)
    n_cols = 8
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2.5))
    
    fig.suptitle(f'Inpainting Methods Comparison - {image_filename}\n(CV-Optimal k: uniform={k_uniform}, distance={k_distance})', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    for row_idx, mask_level in enumerate(MASK_LEVELS):
        data = all_data[mask_level]
        
        original_rgb = cv2.cvtColor(data['original'], cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(data['masked'], cv2.COLOR_BGR2RGB)
        knn_uniform_rgb = cv2.cvtColor(data['results']['knn_uniform'], cv2.COLOR_BGR2RGB)
        knn_distance_rgb = cv2.cvtColor(data['results']['knn_distance'], cv2.COLOR_BGR2RGB)
        navier_stokes_rgb = cv2.cvtColor(data['results']['navier_stokes'], cv2.COLOR_BGR2RGB)
        
        axes[row_idx, 0].imshow(original_rgb)
        if row_idx == 0:
            axes[row_idx, 0].set_title('Original', fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 0].text(-0.3, 0.5, f'{mask_level}%', 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=14, fontweight='bold', 
                             verticalalignment='center',
                             rotation=90)
        
        axes[row_idx, 1].imshow(masked_rgb)
        if row_idx == 0:
            axes[row_idx, 1].set_title('Masked', fontsize=11, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(knn_uniform_rgb)
        if row_idx == 0:
            psnr_val = data['metrics']['knn_uniform']['psnr']
            title = f'k-NN Uniform\n(k={k_uniform})\nPSNR: {psnr_val:.1f}dB'
            axes[row_idx, 2].set_title(title, fontsize=9, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        axes[row_idx, 3].imshow(knn_distance_rgb)
        if row_idx == 0:
            psnr_val = data['metrics']['knn_distance']['psnr']
            title = f'k-NN Distance\n(k={k_distance})\nPSNR: {psnr_val:.1f}dB'
            axes[row_idx, 3].set_title(title, fontsize=9, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
        axes[row_idx, 4].imshow(navier_stokes_rgb)
        if row_idx == 0:
            psnr_val = data['metrics']['navier_stokes']['psnr']
            title = f'Navier-Stokes\nPSNR: {psnr_val:.1f}dB'
            axes[row_idx, 4].set_title(title, fontsize=9, fontweight='bold')
        axes[row_idx, 4].axis('off')
        
        error_uniform = np.abs(original_rgb.astype(float) - knn_uniform_rgb.astype(float)).mean(axis=2)
        im1 = axes[row_idx, 5].imshow(error_uniform, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 5].set_title('Uniform\nError', fontsize=9, fontweight='bold')
        axes[row_idx, 5].axis('off')
        
        error_distance = np.abs(original_rgb.astype(float) - knn_distance_rgb.astype(float)).mean(axis=2)
        im2 = axes[row_idx, 6].imshow(error_distance, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 6].set_title('Distance\nError', fontsize=9, fontweight='bold')
        axes[row_idx, 6].axis('off')
        
        error_ns = np.abs(original_rgb.astype(float) - navier_stokes_rgb.astype(float)).mean(axis=2)
        im3 = axes[row_idx, 7].imshow(error_ns, cmap='hot', vmin=0, vmax=50)
        if row_idx == 0:
            axes[row_idx, 7].set_title('NS\nError', fontsize=9, fontweight='bold')
        axes[row_idx, 7].axis('off')
    
    plt.tight_layout(rect=[0, 0, 0.96, 0.99])
    
    cbar_ax1 = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    fig.colorbar(im1, cax=cbar_ax1, label='Error (pixel intensity)')
    
    comp_dir = os.path.join(output_root, "comprehensive_comparisons")
    stem = Path(image_filename).stem
    plot_path = os.path.join(comp_dir, f"comparison_{stem}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"    - Saved comprehensive comparison plot")


def save_metrics_summary(all_metrics, mask_level, k_uniform, k_distance, output_root):
    """Save summary statistics."""
    level_str = f"{mask_level}%"
    summary_path = os.path.join(output_root, level_str, "metrics_summary.txt")
    
    method_names = ['knn_uniform', 'knn_distance', 'navier_stokes']
    method_labels = [
        f'k-NN Uniform (k={k_uniform})', 
        f'k-NN Distance Weighted (k={k_distance})',
        'Navier-Stokes Inpainting'
    ]
    
    with open(summary_path, 'w') as f:
        f.write(f"Metrics Summary for Mask Level {mask_level}%\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of images: {len(all_metrics)}\n")
        f.write(f"CV-Optimal k (uniform): {k_uniform}\n")
        f.write(f"CV-Optimal k (distance): {k_distance}\n\n")
        
        for method_name, method_label in zip(method_names, method_labels):
            psnr_values = [m[method_name]['psnr'] for m in all_metrics]
            ssim_values = [m[method_name]['ssim'] for m in all_metrics]
            
            psnr_avg = np.mean(psnr_values)
            psnr_std = np.std(psnr_values)
            ssim_avg = np.mean(ssim_values)
            ssim_std = np.std(ssim_values)
            
            f.write(f"{method_label}:\n")
            f.write(f"  PSNR: {psnr_avg:.2f} ± {psnr_std:.2f} dB\n")
            f.write(f"  SSIM: {ssim_avg:.4f} ± {ssim_std:.4f}\n\n")
        
        f.write("\nIndividual Results:\n")
        f.write("-" * 80 + "\n")
        for i, m in enumerate(all_metrics, 1):
            f.write(f"\nImage {i}:\n")
            for method_name, method_label in zip(method_names, method_labels):
                psnr_val = m[method_name]['psnr']
                ssim_val = m[method_name]['ssim']
                f.write(f"  {method_label:35s}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}\n")
    
    print(f"\n  Summary Statistics for {level_str}:")
    for method_name, method_label in zip(method_names, method_labels):
        psnr_values = [m[method_name]['psnr'] for m in all_metrics]
        ssim_values = [m[method_name]['ssim'] for m in all_metrics]
        psnr_avg = np.mean(psnr_values)
        ssim_avg = np.mean(ssim_values)
        print(f"    {method_label:35s} -> Avg PSNR: {psnr_avg:.2f} dB, Avg SSIM: {ssim_avg:.4f}")


def get_mask_filename(image_index):
    """Get the corresponding mask filename."""
    return f"mask_{image_index:05d}.png"


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Inpainting pipeline with cross-validation for k optimization")
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
    
    print("=" * 80)
    print("Inpainting Pipeline with Cross-Validation")
    print("=" * 80)
    print(f"Input directory: {input_root}")
    print(f"Mask root: {mask_root}")
    print(f"Output directory: {output_root}")
    print(f"Cross-validation: {N_FOLDS}-fold")
    print()
    
    pipeline_start_time = time.time()
    
    create_output_dirs(output_root)
    
    original_images = sorted([f for f in os.listdir(input_root) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(original_images) == 0:
        print(f"Error: No images found in {input_root}")
        return
    
    print(f"Found {len(original_images)} original images")
    
    # STEP 1: Cross-validation for k optimization
    optimal_k_uniform, optimal_k_distance, cv_results, timing_info = \
        optimize_k_with_cross_validation(original_images, MASK_LEVELS, input_root, mask_root, output_root)
    
    # STEP 2: Evaluate on all images with optimal k (for comprehensive evaluation)
    print("\n" + "=" * 80)
    print("STEP 2: FULL DATASET EVALUATION")
    print("=" * 80)
    print(f"Using CV-optimal k: uniform={optimal_k_uniform}, distance={optimal_k_distance}")
    print(f"Evaluating on all {len(original_images)} images")
    print()
    
    evaluation_start = time.time()
    
    all_level_metrics = {level: [] for level in MASK_LEVELS}
    
    # Process subset for visualization (e.g., first 10)
    n_visualize = min(10, len(original_images))
    
    for img_idx, image_file in enumerate(original_images[:n_visualize]):
        print(f"\n{'='*80}")
        print(f"Processing image {img_idx+1}/{n_visualize}: {image_file}")
        print(f"{'='*80}")
        
        image_data_all_levels = {}
        
        for mask_level in MASK_LEVELS:
            print(f"\n  Mask level: {mask_level}%")
            
            mask_dir = os.path.join(mask_root, str(mask_level))
            
            if not os.path.exists(mask_dir):
                print(f"  Warning: Mask directory not found: {mask_dir}")
                continue
            
            original_path = os.path.join(input_root, image_file)
            mask_filename = get_mask_filename(img_idx)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                print(f"  ERROR: Mask file not found: {mask_path}")
                continue
            
            try:
                original_img, masked_img, results, mask, metrics = \
                    process_image(original_path, mask_path, mask_level, 
                                optimal_k_uniform, optimal_k_distance)
                
                image_data_all_levels[mask_level] = {
                    'original': original_img,
                    'masked': masked_img,
                    'results': results,
                    'mask': mask,
                    'metrics': metrics
                }
                
                all_level_metrics[mask_level].append(metrics)
                save_results(original_img, masked_img, results, image_file, mask_level, output_root)
                
            except Exception as e:
                print(f"  ERROR processing: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(image_data_all_levels) == len(MASK_LEVELS):
            print(f"\n  Creating comprehensive comparison plot...")
            create_comprehensive_comparison(image_data_all_levels, img_idx, image_file,
                                          optimal_k_uniform, optimal_k_distance, output_root)
    
    evaluation_time = time.time() - evaluation_start
    
    # Save metrics summaries
    print(f"\n{'='*80}")
    print("Saving metrics summaries...")
    print(f"{'='*80}")
    for mask_level in MASK_LEVELS:
        if all_level_metrics[mask_level]:
            save_metrics_summary(all_level_metrics[mask_level], mask_level, 
                               optimal_k_uniform, optimal_k_distance, output_root)
    
    total_pipeline_time = time.time() - pipeline_start_time
    
    # Final timing summary
    print("\n" + "=" * 80)
    print("PIPELINE TIMING SUMMARY")
    print("=" * 80)
    print(f"Cross-validation time: {format_time(timing_info['total_cv_time'])} ({timing_info['total_cv_time']:.2f}s)")
    print(f"  Average per fold:    {format_time(timing_info['avg_fold_time'])} ({timing_info['avg_fold_time']:.2f}s)")
    print(f"Evaluation time:       {format_time(evaluation_time)} ({evaluation_time:.2f}s)")
    print(f"Total pipeline time:   {format_time(total_pipeline_time)} ({total_pipeline_time:.2f}s)")
    print()
    print(f"CV-Optimal k values:")
    print(f"  Uniform:  k = {optimal_k_uniform}")
    print(f"  Distance: k = {optimal_k_distance}")
    print()
    print(f"Results saved to: {output_root}")
    print(f"CV results saved to: {output_root}/k_optimization/")
    print(f"Comprehensive comparisons saved to: {output_root}/comprehensive_comparisons/")
    print("=" * 80)


if __name__ == "__main__":
    main()