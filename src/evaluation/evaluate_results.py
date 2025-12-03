"""
Evaluate PSNR and SSIM metrics on RePaint results
Compares original images with inpainted results
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import json

# Add parent directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.metrics import calculate_psnr, calculate_ssim

def extract_images_from_combined(combined_path):
    """
    Extract original, masked, and inpainted images from combined PNG
    Combined image format: [Original | Masked | Inpainted]
    """
    combined = Image.open(combined_path).convert("RGB")
    width, height = combined.size
    img_width = width // 3
    
    # Extract three parts
    original = combined.crop((0, 0, img_width, height))
    masked = combined.crop((img_width, 0, img_width * 2, height))
    inpainted = combined.crop((img_width * 2, 0, width, height))
    
    return original, masked, inpainted

def image_to_tensor(img):
    """Convert PIL Image to tensor in [0, 1] range"""
    img_array = np.array(img).astype(np.float32) / 255.0
    # Convert to [C, H, W] format
    if len(img_array.shape) == 3:
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    else:
        img_tensor = torch.from_numpy(img_array)
    return img_tensor

def evaluate_results(results_dir="results_hf", mask_levels=[20, 40, 60, 80]):
    """
    Evaluate PSNR and SSIM metrics for all results
    
    Args:
        results_dir: Directory containing results
        mask_levels: List of mask percentages to evaluate
    """
    all_results = {}
    
    print("Evaluating RePaint Results")
    print("=" * 70)
    print()
    
    for mask_level in mask_levels:
        level_dir = os.path.join(results_dir, f"{mask_level}percent")
        
        if not os.path.exists(level_dir):
            print(f"Warning: Skipping {mask_level}% - directory not found: {level_dir}")
            continue
        
        # Get all result files
        result_files = sorted([f for f in os.listdir(level_dir) 
                              if f.endswith('_inpainted.png')])
        
        if len(result_files) == 0:
            print(f"Warning: Skipping {mask_level}% - no results found")
            continue
        
        print(f"Evaluating {mask_level}% masks ({len(result_files)} images)...")
        
        psnr_scores = []
        ssim_scores = []
        
        for result_file in result_files:
            result_path = os.path.join(level_dir, result_file)
            
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
            }
            
            print(f"  Processed {len(psnr_scores)} images")
            print(f"     PSNR: {all_results[mask_level]['psnr_mean']:.2f} ± {all_results[mask_level]['psnr_std']:.2f} dB")
            print(f"     SSIM: {all_results[mask_level]['ssim_mean']:.4f} ± {all_results[mask_level]['ssim_std']:.4f}")
        else:
            print(f"  Error: No valid results for {mask_level}%")
        
        print()
    
    # Print summary table
    print("=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Mask %':<10} {'Images':<10} {'PSNR (dB)':<20} {'SSIM':<20}")
    print("-" * 70)
    
    for mask_level in mask_levels:
        if mask_level in all_results:
            r = all_results[mask_level]
            print(f"{mask_level}%{'':<7} {r['num_images']:<10} "
                  f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}{'':<10} "
                  f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
        else:
            print(f"{mask_level}%{'':<7} {'N/A':<10} {'N/A':<20} {'N/A':<20}")
    
    print()
    print("Detailed Statistics:")
    print("-" * 70)
    for mask_level in mask_levels:
        if mask_level in all_results:
            r = all_results[mask_level]
            print(f"\n{mask_level}% Mask Level:")
            print(f"  PSNR: {r['psnr_mean']:.2f} dB (range: {r['psnr_min']:.2f} - {r['psnr_max']:.2f})")
            print(f"  SSIM: {r['ssim_mean']:.4f} (range: {r['ssim_min']:.4f} - {r['ssim_max']:.4f})")
            print(f"  Images evaluated: {r['num_images']}")
    
    # Save results to JSON
    results_file = os.path.join(results_dir, "metrics_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PSNR and SSIM metrics on RePaint results")
    parser.add_argument("--results_dir", type=str, default="results_hf",
                       help="Directory containing results")
    parser.add_argument("--mask_levels", type=int, nargs="+", default=[20, 40, 60, 80],
                       help="Mask levels to evaluate")
    
    args = parser.parse_args()
    
    evaluate_results(
        results_dir=args.results_dir,
        mask_levels=args.mask_levels
    )

