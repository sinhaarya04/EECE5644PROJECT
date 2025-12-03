"""
Comprehensive training and evaluation script for Neural Processes
Trains and evaluates both CNP and ConvCNP for all mask levels (20, 40, 60, 80)
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json

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


def train_cnp_for_mask(model, train_loader, val_loader, device, epochs, lr, checkpoint_dir, mask_level):
    """Train CNP model for a specific mask level"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            context_x = batch['context_x'].to(device)
            context_y = batch['context_y'].to(device)
            target_x = batch['target_x'].to(device)
            target_y = batch['target_y'].to(device)
            
            # Forward pass
            pred_y, _, _ = model(context_x, context_y, target_x)
            
            # Compute loss
            loss = criterion(pred_y, target_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                context_x = batch['context_x'].to(device)
                context_y = batch['context_y'].to(device)
                target_x = batch['target_x'].to(device)
                target_y = batch['target_y'].to(device)
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                pred_y, _, _ = model(context_x, context_y, target_x)
                
                # Loss
                loss = criterion(pred_y, target_y)
                val_loss += loss.item()
                
                # Reshape predictions to images for metrics
                B, H, W = images.shape[0], images.shape[2], images.shape[3]
                pred_images = pred_y.view(B, H, W, 3).permute(0, 3, 1, 2)
                
                # Calculate metrics on masked regions only
                for b in range(B):
                    mask_b = masks[b]
                    pred_b = pred_images[b]
                    gt_b = images[b]
                    
                    if mask_b.sum() > 0:
                        masked_pred = pred_b * mask_b
                        masked_gt = gt_b * mask_b
                        
                        psnr = calculate_psnr(masked_pred, masked_gt, max_val=1.0)
                        ssim_val = calculate_ssim(masked_pred, masked_gt, max_val=1.0)
                        
                        val_psnr_sum += psnr
                        val_ssim_sum += ssim_val
                        val_count += 1
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr_sum / val_count if val_count > 0 else 0.0
        avg_val_ssim = val_ssim_sum / val_count if val_count > 0 else 0.0
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        val_ssims.append(avg_val_ssim)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'cnp_mask{mask_level}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
                'mask_level': mask_level,
            }, checkpoint_path)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims,
        'best_val_loss': best_val_loss
    }


def train_convcnp_for_mask(model, train_loader, val_loader, device, epochs, lr, checkpoint_dir, mask_level):
    """Train ConvCNP model for a specific mask level"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            masked_images = batch['masked_image'].to(device)
            masks = batch['mask'].to(device)
            images = batch['image'].to(device)
            
            # Forward pass
            pred_image, pred_masked = model(masked_images, masks)
            
            # Compute loss only on masked regions
            masked_pred = pred_image * masks
            masked_gt = images * masks
            
            loss = criterion(masked_pred, masked_gt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                masked_images = batch['masked_image'].to(device)
                masks = batch['mask'].to(device)
                images = batch['image'].to(device)
                
                # Forward pass
                pred_image, pred_masked = model(masked_images, masks)
                
                # Loss on masked regions
                masked_pred = pred_image * masks
                masked_gt = images * masks
                loss = criterion(masked_pred, masked_gt)
                val_loss += loss.item()
                
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
                        
                        val_psnr_sum += psnr
                        val_ssim_sum += ssim_val
                        val_count += 1
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr_sum / val_count if val_count > 0 else 0.0
        avg_val_ssim = val_ssim_sum / val_count if val_count > 0 else 0.0
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        val_ssims.append(avg_val_ssim)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'convcnp_mask{mask_level}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
                'mask_level': mask_level,
            }, checkpoint_path)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims,
        'best_val_loss': best_val_loss
    }


def evaluate_model_for_mask(model, test_loader, device, model_type, mask_level):
    """Evaluate a model for a specific mask level"""
    model.eval()
    
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'cnp':
                context_x = batch['context_x'].to(device)
                context_y = batch['context_y'].to(device)
                target_x = batch['target_x'].to(device)
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                pred_y, _, _ = model(context_x, context_y, target_x)
                
                # Reshape predictions to images
                B, H, W = images.shape[0], images.shape[2], images.shape[3]
                pred_images = pred_y.view(B, H, W, 3).permute(0, 3, 1, 2)
                
                # Calculate metrics on masked regions
                for b in range(B):
                    mask_b = masks[b]
                    pred_b = pred_images[b]
                    gt_b = images[b]
                    
                    if mask_b.sum() > 0:
                        masked_pred = pred_b * mask_b
                        masked_gt = gt_b * mask_b
                        
                        psnr = calculate_psnr(masked_pred, masked_gt, max_val=1.0)
                        ssim_val = calculate_ssim(masked_pred, masked_gt, max_val=1.0)
                        
                        psnr_scores.append(psnr)
                        ssim_scores.append(ssim_val)
            
            else:  # convcnp
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
    
    if len(psnr_scores) == 0:
        return None
    
    return {
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


def get_project_root():
    """Get the project root directory (parent of src/)"""
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)
    return project_root


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Neural Processes for all mask levels"
    )
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Directory containing images")
    parser.add_argument("--mask_base_dir", type=str, default=None,
                       help="Base directory containing mask coordinate folders (e.g., mask_coords)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Directory to save checkpoints")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--max_context_points", type=int, default=5000,
                       help="Maximum context points for CNP")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Image size")
    parser.add_argument("--mask_levels", type=int, nargs="+", default=[20, 40, 60, 80],
                       help="Mask levels to process")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only evaluate existing models")
    
    args = parser.parse_args()
    
    # Get project root and set default paths
    project_root = get_project_root()
    if args.image_dir is None:
        args.image_dir = os.path.join(project_root, "celeba_hq_256")
    elif not os.path.isabs(args.image_dir):
        args.image_dir = os.path.join(project_root, args.image_dir)
    
    if args.mask_base_dir is None:
        args.mask_base_dir = os.path.join(project_root, "mask_coords")
    elif not os.path.isabs(args.mask_base_dir):
        args.mask_base_dir = os.path.join(project_root, args.mask_base_dir)
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(project_root, "checkpoints_np")
    elif not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.join(project_root, args.checkpoint_dir)
    
    if args.results_dir is None:
        args.results_dir = os.path.join(project_root, "results_np")
    elif not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(project_root, args.results_dir)
    
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
    
    print("=" * 80)
    print("Neural Processes - Full Training and Evaluation Pipeline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Image directory: {args.image_dir}")
    print(f"Mask levels: {args.mask_levels}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)
    print()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    # Store all results
    all_results = {}
    training_histories = {}
    
    # Process each mask level
    for mask_level in args.mask_levels:
        print("\n" + "=" * 80)
        print(f"Processing Mask Level: {mask_level}%")
        print("=" * 80)
        
        # Construct mask directory path
        mask_dir = os.path.join(args.mask_base_dir, f"{mask_level}")
        
        if not os.path.exists(mask_dir):
            print(f"Warning: Mask directory not found: {mask_dir}")
            print(f"Skipping mask level {mask_level}%")
            continue
        
        mask_results = {}
        
        # ========== Train and Evaluate CNP ==========
        print(f"\n--- CNP for {mask_level}% masks ---")
        
        # Create dataset
        full_dataset = NPImageDataset(
            img_dir=args.image_dir,
            mask_dir=mask_dir,
            transform=transform,
            max_context_points=args.max_context_points,
            model_type='cnp'
        )
        
        if len(full_dataset) == 0:
            print(f"Warning: No data found for mask level {mask_level}%")
            continue
        
        # Split dataset
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Use all data for final evaluation
        test_dataset = full_dataset
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_cnp,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_cnp,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_cnp,
            num_workers=0
        )
        
        # Create and train model
        if not args.skip_training:
            print(f"Training CNP...")
            model_cnp = CNP(
                input_dim=5,
                hidden_dim=128,
                encoder_layers=4,
                decoder_layers=3,
                output_dim=3
            ).to(device)
            
            cnp_history = train_cnp_for_mask(
                model_cnp, train_loader, val_loader, device,
                args.epochs, args.learning_rate, args.checkpoint_dir, mask_level
            )
            training_histories[f'cnp_mask{mask_level}'] = cnp_history
            print(f"CNP training completed. Best val loss: {cnp_history['best_val_loss']:.6f}")
        else:
            # Load existing model
            checkpoint_path = os.path.join(args.checkpoint_dir, f'cnp_mask{mask_level}_best.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found: {checkpoint_path}")
                print("Skipping CNP evaluation for this mask level.")
            else:
                model_cnp = CNP(
                    input_dim=5,
                    hidden_dim=128,
                    encoder_layers=4,
                    decoder_layers=3,
                    output_dim=3
                ).to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model_cnp.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded CNP checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Evaluate CNP
        if 'model_cnp' in locals():
            print(f"Evaluating CNP on test set...")
            cnp_results = evaluate_model_for_mask(
                model_cnp, test_loader, device, 'cnp', mask_level
            )
            if cnp_results:
                mask_results['cnp'] = cnp_results
                print(f"  PSNR: {cnp_results['psnr_mean']:.2f} ± {cnp_results['psnr_std']:.2f} dB")
                print(f"  SSIM: {cnp_results['ssim_mean']:.4f} ± {cnp_results['ssim_std']:.4f}")
        
        # ========== Train and Evaluate ConvCNP ==========
        print(f"\n--- ConvCNP for {mask_level}% masks ---")
        
        # Create dataset
        full_dataset_conv = NPImageDataset(
            img_dir=args.image_dir,
            mask_dir=mask_dir,
            transform=transform,
            model_type='convcnp'
        )
        
        # Split dataset
        val_size = int(len(full_dataset_conv) * args.val_split)
        train_size = len(full_dataset_conv) - val_size
        train_dataset_conv, val_dataset_conv = torch.utils.data.random_split(
            full_dataset_conv, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        test_dataset_conv = full_dataset_conv
        
        train_loader_conv = DataLoader(
            train_dataset_conv,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_convcnp,
            num_workers=0
        )
        val_loader_conv = DataLoader(
            val_dataset_conv,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_convcnp,
            num_workers=0
        )
        test_loader_conv = DataLoader(
            test_dataset_conv,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_convcnp,
            num_workers=0
        )
        
        # Create and train model
        if not args.skip_training:
            print(f"Training ConvCNP...")
            model_convcnp = ConvCNP(
                in_channels=3,
                hidden_channels=64,
                num_layers=4,
                kernel_size=3
            ).to(device)
            
            convcnp_history = train_convcnp_for_mask(
                model_convcnp, train_loader_conv, val_loader_conv, device,
                args.epochs, args.learning_rate, args.checkpoint_dir, mask_level
            )
            training_histories[f'convcnp_mask{mask_level}'] = convcnp_history
            print(f"ConvCNP training completed. Best val loss: {convcnp_history['best_val_loss']:.6f}")
        else:
            # Load existing model
            checkpoint_path = os.path.join(args.checkpoint_dir, f'convcnp_mask{mask_level}_best.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found: {checkpoint_path}")
                print("Skipping ConvCNP evaluation for this mask level.")
            else:
                model_convcnp = ConvCNP(
                    in_channels=3,
                    hidden_channels=64,
                    num_layers=4,
                    kernel_size=3
                ).to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model_convcnp.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded ConvCNP checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Evaluate ConvCNP
        if 'model_convcnp' in locals():
            print(f"Evaluating ConvCNP on test set...")
            convcnp_results = evaluate_model_for_mask(
                model_convcnp, test_loader_conv, device, 'convcnp', mask_level
            )
            if convcnp_results:
                mask_results['convcnp'] = convcnp_results
                print(f"  PSNR: {convcnp_results['psnr_mean']:.2f} ± {convcnp_results['psnr_std']:.2f} dB")
                print(f"  SSIM: {convcnp_results['ssim_mean']:.4f} ± {convcnp_results['ssim_std']:.4f}")
        
        # Store results for this mask level
        if mask_results:
            all_results[mask_level] = mask_results
    
    # Save all results
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    if all_results:
        # Print summary table
        print(f"\n{'Mask %':<10} {'Model':<10} {'Images':<10} {'PSNR (dB)':<25} {'SSIM':<25}")
        print("-" * 80)
        
        for mask_level in args.mask_levels:
            if mask_level in all_results:
                for model_name in ['cnp', 'convcnp']:
                    if model_name in all_results[mask_level]:
                        r = all_results[mask_level][model_name]
                        print(f"{mask_level}%{'':<6} {model_name:<10} {r['num_images']:<10} "
                              f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}{'':<15} "
                              f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
        
        # Save results to JSON
        results_file = os.path.join(args.results_dir, "all_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save training histories if available
        if training_histories:
            history_file = os.path.join(args.results_dir, "training_histories.json")
            with open(history_file, 'w') as f:
                json.dump(training_histories, f, indent=2)
            print(f"Training histories saved to: {history_file}")
    else:
        print("No results to save.")
    
    print("\n" + "=" * 80)
    print("Pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

