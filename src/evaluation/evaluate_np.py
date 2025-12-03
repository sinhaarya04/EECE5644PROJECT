"""
Evaluation script for Neural Processes (CNP and ConvCNP) for image inpainting
Uses existing metrics (PSNR and SSIM)
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# Import models and utilities
import sys
from pathlib import Path

# Add parent directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.cnp import CNP
from models.convcnp import ConvCNP
from data.datasets_np import NPImageDataset, collate_fn_cnp, collate_fn_convcnp
from core.metrics import calculate_psnr, calculate_ssim


def evaluate_cnp(model, test_loader, device, output_dir=None):
    """Evaluate CNP model"""
    model.eval()
    
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating CNP")):
            context_x = batch['context_x'].to(device)
            context_y = batch['context_y'].to(device)
            target_x = batch['target_x'].to(device)
            images = batch['image'].to(device)  # [B, 3, H, W]
            masks = batch['mask'].to(device)    # [B, 1, H, W]
            
            # Forward pass
            pred_y, _, _ = model(context_x, context_y, target_x)
            
            # Reshape predictions to images
            B, H, W = images.shape[0], images.shape[2], images.shape[3]
            pred_images = pred_y.view(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
            
            # Calculate metrics on masked regions only
            for b in range(B):
                mask_b = masks[b]  # [1, H, W]
                pred_b = pred_images[b]  # [3, H, W]
                gt_b = images[b]  # [3, H, W]
                
                # Only evaluate on masked regions
                if mask_b.sum() > 0:
                    masked_pred = pred_b * mask_b
                    masked_gt = gt_b * mask_b
                    
                    psnr = calculate_psnr(masked_pred, masked_gt, max_val=1.0)
                    ssim_val = calculate_ssim(masked_pred, masked_gt, max_val=1.0)
                    
                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim_val)
                    
                    # Save sample images
                    if output_dir and batch_idx == 0 and b == 0:
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Convert tensors to PIL Images
                        def tensor_to_image(tensor):
                            # [C, H, W] -> [H, W, C] -> PIL
                            img_np = tensor.cpu().permute(1, 2, 0).numpy()
                            img_np = np.clip(img_np, 0, 1)
                            img_np = (img_np * 255).astype(np.uint8)
                            return Image.fromarray(img_np)
                        
                        # Create full prediction (context + predicted)
                        full_pred = pred_b * mask_b + gt_b * (1 - mask_b)
                        
                        # Save images
                        gt_img = tensor_to_image(gt_b)
                        pred_img = tensor_to_image(full_pred)
                        masked_img = tensor_to_image(gt_b * (1 - mask_b))
                        
                        # Create combined image
                        width, height = gt_img.size
                        combined = Image.new('RGB', (width * 3, height))
                        combined.paste(gt_img, (0, 0))
                        combined.paste(masked_img, (width, 0))
                        combined.paste(pred_img, (width * 2, 0))
                        
                        combined.save(os.path.join(output_dir, 'cnp_sample.png'))
    
    return psnr_scores, ssim_scores


def evaluate_convcnp(model, test_loader, device, output_dir=None):
    """Evaluate ConvCNP model"""
    model.eval()
    
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating ConvCNP")):
            masked_images = batch['masked_image'].to(device)
            masks = batch['mask'].to(device)
            images = batch['image'].to(device)
            
            # Forward pass
            pred_image, pred_masked = model(masked_images, masks)
            
            # Calculate metrics on masked regions
            for b in range(images.shape[0]):
                mask_b = masks[b]
                pred_b = pred_image[b]
                gt_b = images[b]
                
                if mask_b.sum() > 0:
                    masked_pred_b = pred_b * mask_b
                    masked_gt_b = gt_b * mask_b
                    
                    psnr = calculate_psnr(masked_pred_b, masked_gt_b, max_val=1.0)
                    ssim_val = calculate_ssim(masked_pred_b, masked_gt_b, max_val=1.0)
                    
                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim_val)
                    
                    # Save sample images
                    if output_dir and batch_idx == 0 and b == 0:
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Convert tensors to PIL Images
                        def tensor_to_image(tensor):
                            img_np = tensor.cpu().permute(1, 2, 0).numpy()
                            img_np = np.clip(img_np, 0, 1)
                            img_np = (img_np * 255).astype(np.uint8)
                            return Image.fromarray(img_np)
                        
                        # Create full prediction
                        full_pred = pred_b * mask_b + gt_b * (1 - mask_b)
                        
                        # Save images
                        gt_img = tensor_to_image(gt_b)
                        pred_img = tensor_to_image(full_pred)
                        masked_img = tensor_to_image(masked_images[b])
                        
                        # Create combined image
                        width, height = gt_img.size
                        combined = Image.new('RGB', (width * 3, height))
                        combined.paste(gt_img, (0, 0))
                        combined.paste(masked_img, (width, 0))
                        combined.paste(pred_img, (width * 2, 0))
                        
                        combined.save(os.path.join(output_dir, 'convcnp_sample.png'))
    
    return psnr_scores, ssim_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural Processes for image inpainting")
    parser.add_argument("--model", type=str, choices=['cnp', 'convcnp', 'both'], default='both',
                       help="Model to evaluate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_np",
                       help="Directory containing checkpoints")
    parser.add_argument("--image_dir", type=str, default="celeba_hq_256",
                       help="Directory containing images")
    parser.add_argument("--mask_dir", type=str, default="mask_coords/40",
                       help="Directory containing mask coordinate files (e.g., mask_coords/40)")
    parser.add_argument("--output_dir", type=str, default="results_np",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--max_context_points", type=int, default=5000,
                       help="Maximum context points for CNP")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Image size")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    all_results = {}
    
    # Evaluate CNP
    if args.model in ['cnp', 'both']:
        print("\n" + "="*60)
        print("Evaluating CNP")
        print("="*60)
        
        # Load model
        checkpoint_path = os.path.join(args.checkpoint_dir, 'cnp_best.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Skipping CNP evaluation.")
        else:
            model = CNP(
                input_dim=5,
                hidden_dim=128,
                encoder_layers=4,
                decoder_layers=3,
                output_dim=3
            ).to(device)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            
            # Create test dataset
            test_dataset = NPImageDataset(
                img_dir=args.image_dir,
                mask_dir=args.mask_dir,
                transform=transform,
                max_context_points=args.max_context_points,
                model_type='cnp'
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn_cnp,
                num_workers=2
            )
            
            # Evaluate
            output_dir_cnp = os.path.join(args.output_dir, 'cnp')
            psnr_scores, ssim_scores = evaluate_cnp(model, test_loader, device, output_dir_cnp)
            
            if len(psnr_scores) > 0:
                all_results['cnp'] = {
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
                
                print(f"\nCNP Results:")
                print(f"  Images evaluated: {len(psnr_scores)}")
                print(f"  PSNR: {all_results['cnp']['psnr_mean']:.2f} ± {all_results['cnp']['psnr_std']:.2f} dB")
                print(f"  SSIM: {all_results['cnp']['ssim_mean']:.4f} ± {all_results['cnp']['ssim_std']:.4f}")
    
    # Evaluate ConvCNP
    if args.model in ['convcnp', 'both']:
        print("\n" + "="*60)
        print("Evaluating ConvCNP")
        print("="*60)
        
        # Load model
        checkpoint_path = os.path.join(args.checkpoint_dir, 'convcnp_best.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Skipping ConvCNP evaluation.")
        else:
            model = ConvCNP(
                in_channels=3,
                hidden_channels=64,
                num_layers=4,
                kernel_size=3
            ).to(device)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            
            # Create test dataset
            test_dataset = NPImageDataset(
                img_dir=args.image_dir,
                mask_dir=args.mask_dir,
                transform=transform,
                model_type='convcnp'
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn_convcnp,
                num_workers=2
            )
            
            # Evaluate
            output_dir_convcnp = os.path.join(args.output_dir, 'convcnp')
            psnr_scores, ssim_scores = evaluate_convcnp(model, test_loader, device, output_dir_convcnp)
            
            if len(psnr_scores) > 0:
                all_results['convcnp'] = {
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
                
                print(f"\nConvCNP Results:")
                print(f"  Images evaluated: {len(psnr_scores)}")
                print(f"  PSNR: {all_results['convcnp']['psnr_mean']:.2f} ± {all_results['convcnp']['psnr_std']:.2f} dB")
                print(f"  SSIM: {all_results['convcnp']['ssim_mean']:.4f} ± {all_results['convcnp']['ssim_std']:.4f}")
    
    # Save results
    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, "metrics_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Model':<15} {'Images':<10} {'PSNR (dB)':<20} {'SSIM':<20}")
        print("-" * 60)
        for model_name, results in all_results.items():
            print(f"{model_name:<15} {results['num_images']:<10} "
                  f"{results['psnr_mean']:.2f} ± {results['psnr_std']:.2f}{'':<10} "
                  f"{results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

