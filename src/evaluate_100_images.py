"""
Evaluate first 100 images for 20% and 40% mask levels
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
import json

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import calculate_psnr, calculate_ssim
from evaluate import extract_images_from_combined, is_combined_image, image_to_tensor

def evaluate_100_images(results_dir, mask_levels=[20, 40], num_images=100):
    """
    Evaluate first 100 images for specified mask levels
    
    Args:
        results_dir: Directory containing results (e.g., "results_hf")
        mask_levels: List of mask percentages to evaluate
        num_images: Number of images to evaluate (default: 100)
    """
    all_results = {}
    
    print("Evaluating First 100 Images for 20% and 40% Mask Levels")
    print("=" * 70)
    print()
    
    for mask_level in mask_levels:
        level_dir = os.path.join(results_dir, f"{mask_level}percent")
        
        if not os.path.exists(level_dir):
            print(f"Warning: Skipping {mask_level}% - directory not found: {level_dir}")
            continue
        
        # Get all result files and sort them
        result_files = sorted([f for f in os.listdir(level_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(result_files) == 0:
            print(f"Warning: Skipping {mask_level}% - no results found")
            continue
        
        # Take first num_images files
        files_to_evaluate = result_files[:num_images]
        print(f"Evaluating {mask_level}% masks ({len(files_to_evaluate)} images)...")
        
        psnr_scores = []
        ssim_scores = []
        per_image_results = []
        
        for idx, result_file in enumerate(files_to_evaluate):
            result_path = os.path.join(level_dir, result_file)
            
            # Skip if not a combined image
            if not is_combined_image(result_path):
                print(f"  Warning: {result_file} is not a combined image, skipping...")
                continue
            
            try:
                # Extract images from combined PNG
                original, masked, inpainted = extract_images_from_combined(result_path)
                
                # Convert to tensors
                orig_tensor = image_to_tensor(original)
                inpainted_tensor = image_to_tensor(inpainted)
                
                # Calculate metrics
                psnr = calculate_psnr(orig_tensor, inpainted_tensor, max_val=1.0)
                ssim_val = calculate_ssim(orig_tensor, inpainted_tensor, max_val=1.0)
                
                psnr_scores.append(psnr)
                ssim_scores.append(ssim_val)
                
                # Store per-image results
                img_name = os.path.splitext(result_file)[0]
                per_image_results.append({
                    'image': img_name,
                    'psnr': float(psnr),
                    'ssim': float(ssim_val)
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(files_to_evaluate)} images...")
                
            except Exception as e:
                print(f"  Warning: Error processing {result_file}: {e}")
                continue
        
        if len(psnr_scores) > 0:
            all_results[mask_level] = {
                'num_images': len(psnr_scores),
                'psnr_mean': float(np.mean(psnr_scores)),
                'psnr_std': float(np.std(psnr_scores)),
                'psnr_min': float(np.min(psnr_scores)),
                'psnr_max': float(np.max(psnr_scores)),
                'ssim_mean': float(np.mean(ssim_scores)),
                'ssim_std': float(np.std(ssim_scores)),
                'ssim_min': float(np.min(ssim_scores)),
                'ssim_max': float(np.max(ssim_scores)),
                'per_image': per_image_results
            }
            
            print(f"  Processed {len(psnr_scores)} images")
            print(f"     PSNR: {all_results[mask_level]['psnr_mean']:.2f} ± {all_results[mask_level]['psnr_std']:.2f} dB")
            print(f"           Range: {all_results[mask_level]['psnr_min']:.2f} - {all_results[mask_level]['psnr_max']:.2f} dB")
            print(f"     SSIM: {all_results[mask_level]['ssim_mean']:.4f} ± {all_results[mask_level]['ssim_std']:.4f}")
            print(f"           Range: {all_results[mask_level]['ssim_min']:.4f} - {all_results[mask_level]['ssim_max']:.4f}")
        else:
            print(f"  Error: No valid results for {mask_level}%")
        
        print()
    
    return all_results

def print_summary(all_results):
    """Print summary of evaluation results"""
    print("=" * 70)
    print("EVALUATION SUMMARY - First 100 Images")
    print("=" * 70)
    print()
    
    if len(all_results) == 0:
        print("No results to display.")
        return
    
    # Print table header
    print(f"{'Mask %':<10} {'Images':<10} {'PSNR (dB)':<25} {'SSIM':<25}")
    print("-" * 70)
    
    for mask_level in sorted(all_results.keys()):
        r = all_results[mask_level]
        print(f"{mask_level}%{'':<7} {r['num_images']:<10} "
              f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}{'':<15} "
              f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
    
    print()
    print("Detailed Statistics:")
    print("-" * 70)
    
    for mask_level in sorted(all_results.keys()):
        r = all_results[mask_level]
        print(f"\n{mask_level}% Mask Level:")
        print(f"  Images evaluated: {r['num_images']}")
        print(f"  PSNR: {r['psnr_mean']:.2f} ± {r['psnr_std']:.2f} dB")
        print(f"        Range: {r['psnr_min']:.2f} - {r['psnr_max']:.2f} dB")
        print(f"  SSIM: {r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
        print(f"        Range: {r['ssim_min']:.4f} - {r['ssim_max']:.4f}")

def save_results(all_results, output_file):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate first 100 images for 20% and 40% mask levels")
    parser.add_argument("--results_dir", type=str, default="results_hf",
                       help="Directory containing results")
    parser.add_argument("--mask_levels", type=int, nargs="+", default=[20, 40],
                       help="Mask levels to evaluate")
    parser.add_argument("--num_images", type=int, default=100,
                       help="Number of images to evaluate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (default: metrics_100_images.json in results directory)")
    
    args = parser.parse_args()
    
    # Evaluate
    all_results = evaluate_100_images(
        results_dir=args.results_dir,
        mask_levels=args.mask_levels,
        num_images=args.num_images
    )
    
    if len(all_results) == 0:
        print("Error: No results to evaluate!")
        return
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    output_file = args.output or os.path.join(args.results_dir, "metrics_100_images.json")
    save_results(all_results, output_file)

if __name__ == "__main__":
    main()

