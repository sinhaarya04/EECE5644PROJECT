"""
Unified evaluation script for PSNR and SSIM metrics
Works with any inpainting model (RePaint, CNN, k-NN, etc.)

Supports two modes:
1. RePaint format: Combined PNGs with [Original | Masked | Inpainted] side-by-side
2. General format: Separate directories for original and inpainted images
"""
import os
import numpy as np
from PIL import Image
import torch
import sys

# Ensure `src` directory is in the import path when run from project root
# so local modules like `metrics` resolve consistently both in interactive
# sessions and when executed with `python src/evaluate.py`.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Normal import when running inside the src package folder
    from metrics import calculate_psnr, calculate_ssim
except Exception:
    # Fallback import (when running python from repo root with src in sys.path)
    from src.metrics import calculate_psnr, calculate_ssim
import json
import argparse

import sys, os
#sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def image_to_tensor(img):
    """Convert PIL Image to tensor in [0, 1] range"""
    img_array = np.array(img).astype(np.float32) / 255.0
    # Convert to [C, H, W] format
    if len(img_array.shape) == 3:
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    else:
        img_tensor = torch.from_numpy(img_array)
    return img_tensor

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

def is_combined_image(image_path):
    """Check if image is a combined RePaint format (3x width)"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        # Combined images should be roughly 3x width (with some tolerance)
        return width > height * 2.5
    except:
        return False

def find_matching_inpainted(original_file, inpainted_dir):
    """Find matching inpainted image for an original file"""
    # Try same filename first
    inpainted_path = os.path.join(inpainted_dir, original_file)
    if os.path.exists(inpainted_path):
        return inpainted_path
    
    # Try common suffixes (same extension)
    name, ext = os.path.splitext(original_file)
    for suffix in ['_inpainted', '_result', '_output', '_predicted']:
        alt_path = os.path.join(inpainted_dir, f"{name}{suffix}{ext}")
        if os.path.exists(alt_path):
            return alt_path

    # Try same name but with alternate common extensions
    alt_exts = ['.png', '.jpg', '.jpeg']
    for ae in alt_exts:
        if ae == ext.lower():
            continue
        alt = os.path.join(inpainted_dir, f"{name}{ae}")
        if os.path.exists(alt):
            return alt
        for suffix in ['_inpainted', '_result', '_output', '_predicted']:
            alt2 = os.path.join(inpainted_dir, f"{name}{suffix}{ae}")
            if os.path.exists(alt2):
                return alt2

    # Try to find by numeric index within filename (handles filenames like 'Abdullah Gul_0_inpainted.png')
    import re
    # Extract numeric index at the end of basename (handles '0.png' and 'Name_0.png')
    name_no_ext = os.path.splitext(original_file)[0]
    m = re.search(r"(\d+)$", name_no_ext)
    if m:
        idx = m.group(1)
        # Prefer *_<idx>_inpainted.*
        for entry in os.listdir(inpainted_dir):
            if not entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            # Find patterns that contain the index
            if re.search(rf"(?:_|-){idx}(?:_|\.|$)", entry):
                # Prefer a file with _inpainted in its name
                if '_inpainted' in entry:
                    return os.path.join(inpainted_dir, entry)
        # If no _inpainted variants, return any file ending with _<idx>.* or containing _<idx> before extension
        for entry in os.listdir(inpainted_dir):
            if not entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if re.search(rf"(?:_|-){idx}(?:\.|$)", entry):
                return os.path.join(inpainted_dir, entry)
    
    return None

def evaluate_repaint_format(results_dir, mask_levels=[20, 40, 60, 80]):
    """
    Evaluate RePaint results (combined PNG format)
    """
    all_results = {}
    
    print("Evaluating RePaint Results (Combined PNG Format)")
    print("=" * 70)
    print()
    
    for mask_level in mask_levels:
        level_dir = os.path.join(results_dir, f"{mask_level}percent")
        
        if not os.path.exists(level_dir):
            print(f"Warning: Skipping {mask_level}% - directory not found: {level_dir}")
            continue
        
        # Get all result files
        result_files = sorted([f for f in os.listdir(level_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(result_files) == 0:
            print(f"Warning: Skipping {mask_level}% - no results found")
            continue
        
        print(f"Evaluating {mask_level}% masks ({len(result_files)} images)...")
        
        psnr_scores = []
        ssim_scores = []
        
        for result_file in result_files:
            result_path = os.path.join(level_dir, result_file)
            
            # Skip if not a combined image
            if not is_combined_image(result_path):
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
    
    return all_results

def evaluate_general_format(original_dir, inpainted_dir, show_pairs=False):
    """
    Evaluate general model results (separate original and inpainted directories)
    """
    # Get all image files
    original_files = sorted([f for f in os.listdir(original_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(original_files) == 0:
        print(f"Error: No images found in {original_dir}")
        return {}
    
    print("Evaluating Model Results (Separate Directories)")
    print("=" * 70)
    print(f"Original images: {original_dir}")
    print(f"Inpainted images: {inpainted_dir}")
    print(f"Found {len(original_files)} original images")
    print("=" * 70)
    print()
    
    psnr_scores = []
    ssim_scores = []
    processed = 0
    
    for orig_file in original_files:
        orig_path = os.path.join(original_dir, orig_file)
        
        # Find matching inpainted file
        inpainted_path = find_matching_inpainted(orig_file, inpainted_dir)
        if show_pairs:
            print(f"Match check: {orig_file} -> {inpainted_path}")
        
        if inpainted_path is None:
            print(f"Warning: No matching inpainted image for {orig_file}, skipping...")
            continue
        
        try:
            # Load images
            orig_img = Image.open(orig_path).convert("RGB")
            inpainted_img = Image.open(inpainted_path).convert("RGB")
            
            # Resize inpainted to match original if needed
            if orig_img.size != inpainted_img.size:
                print(f"Warning: Size mismatch for {orig_file}, resizing inpainted to match...")
                inpainted_img = inpainted_img.resize(orig_img.size, Image.BICUBIC)
            
            # Convert to tensors
            orig_tensor = image_to_tensor(orig_img)
            inpainted_tensor = image_to_tensor(inpainted_img)
            
            # Calculate metrics
            psnr = calculate_psnr(orig_tensor, inpainted_tensor, max_val=1.0)
            ssim_val = calculate_ssim(orig_tensor, inpainted_tensor, max_val=1.0)
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim_val)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {orig_file}: {e}")
            continue
    
    if len(psnr_scores) == 0:
        print("Error: No valid image pairs found!")
        return {}
    
    # Calculate statistics
    results = {
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
    
    return {'all': results}

def print_results(all_results, results_dir=None):
    """Print evaluation results in a formatted table"""
    print("=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    
    # Check if results are organized by mask level or single set
    if 'all' in all_results:
        # General format - single result set
        r = all_results['all']
        print(f"{'Images':<10} {'PSNR (dB)':<20} {'SSIM':<20}")
        print("-" * 70)
        print(f"{r['num_images']:<10} "
              f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}{'':<10} "
              f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
        print()
        print("Detailed Statistics:")
        print("-" * 70)
        print(f"  PSNR: {r['psnr_mean']:.2f} dB (range: {r['psnr_min']:.2f} - {r['psnr_max']:.2f})")
        print(f"  SSIM: {r['ssim_mean']:.4f} (range: {r['ssim_min']:.4f} - {r['ssim_max']:.4f})")
        print(f"  Images evaluated: {r['num_images']}")
    else:
        # RePaint format - organized by mask level
        print(f"{'Mask %':<10} {'Images':<10} {'PSNR (dB)':<20} {'SSIM':<20}")
        print("-" * 70)
        
        mask_levels = sorted([k for k in all_results.keys() if isinstance(k, int)])
        for mask_level in mask_levels:
            r = all_results[mask_level]
            print(f"{mask_level}%{'':<7} {r['num_images']:<10} "
                  f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}{'':<10} "
                  f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
        
        print()
        print("Detailed Statistics:")
        print("-" * 70)
        for mask_level in mask_levels:
            r = all_results[mask_level]
            print(f"\n{mask_level}% Mask Level:")
            print(f"  PSNR: {r['psnr_mean']:.2f} dB (range: {r['psnr_min']:.2f} - {r['psnr_max']:.2f})")
            print(f"  SSIM: {r['ssim_mean']:.4f} (range: {r['ssim_min']:.4f} - {r['ssim_max']:.4f})")
            print(f"  Images evaluated: {r['num_images']}")

def save_results(all_results, output_file):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR and SSIM metrics for any inpainting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RePaint format (combined PNGs)
  python evaluate.py --results_dir results_hf --mask_levels 20 40 60 80
  
  # General format (separate directories)
  python evaluate.py --original_dir CelebAMask-HQ/CelebA-HQ-img --inpainted_dir cnn_results
  
  # General format with custom output
  python evaluate.py --original_dir originals --inpainted_dir cnn_results --output results.json
        """
    )
    
    # RePaint format arguments
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Directory containing RePaint results (combined PNGs)")
    parser.add_argument("--mask_levels", type=int, nargs="+", default=[20, 40, 60, 80],
                       help="Mask levels to evaluate (for RePaint format)")
    
    # General format arguments
    parser.add_argument("--original_dir", type=str, default=None,
                       help="Directory containing original (ground truth) images")
    parser.add_argument("--inpainted_dir", type=str, default=None,
                       help="Directory containing inpainted images")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (default: metrics_results.json in results directory)")
    parser.add_argument("--show_pairs", action='store_true',
                       help="Print matched original->inpainted pairs for debugging")
    
    args = parser.parse_args()
    
    # Determine which format to use
    if args.results_dir is not None:
        # RePaint format
        all_results = evaluate_repaint_format(args.results_dir, args.mask_levels)
        output_file = args.output or os.path.join(args.results_dir, "metrics_results.json")
    elif args.original_dir is not None and args.inpainted_dir is not None:
        # General format
        all_results = evaluate_general_format(args.original_dir, args.inpainted_dir, show_pairs=args.show_pairs)
        output_file = args.output or os.path.join(args.inpainted_dir, "evaluation_results.json")
    else:
        print("Error: Must specify either:")
        print("  --results_dir (for RePaint format)")
        print("  OR --original_dir and --inpainted_dir (for general format)")
        return
    
    if len(all_results) == 0:
        print("Error: No results to evaluate!")
        return
    
    # Print and save results
    print_results(all_results, args.results_dir or args.inpainted_dir)
    save_results(all_results, output_file)

if __name__ == "__main__":
    main()

