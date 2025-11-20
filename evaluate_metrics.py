import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import CelebAInpaintingDataset
from src.metrics import calculate_psnr, calculate_ssim
import numpy as np

def evaluate_metrics(mask_levels=[20, 40, 60, 80], num_samples=1000, batch_size=32):
    """
    Evaluate PSNR and SSIM metrics for different mask levels.
    
    For now, this tests the metrics by comparing:
    - Original images vs themselves (should give perfect scores)
    - Masked images vs originals (shows degradation)
    
    Args:
        mask_levels: List of mask percentages to test
        num_samples: Number of samples to evaluate per level
        batch_size: Batch size for data loading
    """
    results = {}
    
    for percent in mask_levels:
        print(f"\n{'='*60}")
        print(f"Evaluating mask level: {percent}%")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = CelebAInpaintingDataset(
            img_dir="celeba_processed_128",
            mask_dir=f"masks/{percent}",
            transform=transforms.ToTensor()
        )
        
        # Limit number of samples
        num_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        
        # Metrics storage
        psnr_scores = []
        ssim_scores = []
        psnr_masked_scores = []  # Masked vs original (should be lower)
        ssim_masked_scores = []
        
        print(f"Processing {num_samples} samples...")
        
        for batch_idx, (masked_img, mask, original_img) in enumerate(dataloader):
            # Test 1: Original vs Original (should be perfect)
            for i in range(original_img.shape[0]):
                orig = original_img[i]
                psnr = calculate_psnr(orig, orig)
                ssim_val = calculate_ssim(orig, orig)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim_val)
                
                # Test 2: Masked vs Original (shows degradation)
                masked = masked_img[i]
                psnr_masked = calculate_psnr(original_img[i], masked)
                ssim_masked = calculate_ssim(original_img[i], masked)
                psnr_masked_scores.append(psnr_masked)
                ssim_masked_scores.append(ssim_masked)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{num_samples} samples...")
        
        # Calculate statistics
        results[percent] = {
            'original_vs_original': {
                'psnr_mean': np.mean(psnr_scores),
                'psnr_std': np.std(psnr_scores),
                'ssim_mean': np.mean(ssim_scores),
                'ssim_std': np.std(ssim_scores)
            },
            'masked_vs_original': {
                'psnr_mean': np.mean(psnr_masked_scores),
                'psnr_std': np.std(psnr_masked_scores),
                'ssim_mean': np.mean(ssim_masked_scores),
                'ssim_std': np.std(ssim_masked_scores)
            }
        }
        
        # Print results
        print(f"\nResults for {percent}% mask level:")
        print(f"  Original vs Original (baseline - should be perfect):")
        print(f"    PSNR: {results[percent]['original_vs_original']['psnr_mean']:.2f} ± {results[percent]['original_vs_original']['psnr_std']:.4f} dB")
        print(f"    SSIM: {results[percent]['original_vs_original']['ssim_mean']:.4f} ± {results[percent]['original_vs_original']['ssim_std']:.4f}")
        print(f"  Masked vs Original (shows degradation):")
        print(f"    PSNR: {results[percent]['masked_vs_original']['psnr_mean']:.2f} ± {results[percent]['masked_vs_original']['psnr_std']:.4f} dB")
        print(f"    SSIM: {results[percent]['masked_vs_original']['ssim_mean']:.4f} ± {results[percent]['masked_vs_original']['ssim_std']:.4f}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Mask %':<10} {'PSNR (dB)':<15} {'SSIM':<15}")
    print(f"{'-'*60}")
    for percent in mask_levels:
        psnr = results[percent]['masked_vs_original']['psnr_mean']
        ssim_val = results[percent]['masked_vs_original']['ssim_mean']
        print(f"{percent}%{'':<7} {psnr:<15.2f} {ssim_val:<15.4f}")
    
    return results

if __name__ == "__main__":
    print("Starting evaluation of PSNR and SSIM metrics...")
    print("Testing with mask levels: 20%, 40%, 60%, 80%")
    results = evaluate_metrics(mask_levels=[20, 40, 60, 80], num_samples=1000)

