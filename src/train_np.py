"""
Training script for Neural Processes (CNP and ConvCNP) for image inpainting
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
from datetime import datetime

# Import models
from models.cnp import CNP
from models.convcnp import ConvCNP
from dataset_np import NPImageDataset, collate_fn_cnp, collate_fn_convcnp
from metrics import calculate_psnr, calculate_ssim


def train_cnp(model, train_loader, val_loader, device, args):
    """Train CNP model"""
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in train_pbar:
            context_x = batch['context_x'].to(device)  # [B, N_c, 5]
            context_y = batch['context_y'].to(device)  # [B, N_c, 3]
            target_x = batch['target_x'].to(device)   # [B, H*W, 2]
            target_y = batch['target_y'].to(device)   # [B, H*W, 3]
            
            # Forward pass
            pred_y, _, _ = model(context_x, context_y, target_x)
            
            # Compute loss (only on target points)
            loss = criterion(pred_y, target_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in val_pbar:
                context_x = batch['context_x'].to(device)
                context_y = batch['context_y'].to(device)
                target_x = batch['target_x'].to(device)
                target_y = batch['target_y'].to(device)
                images = batch['image'].to(device)  # [B, 3, H, W]
                masks = batch['mask'].to(device)    # [B, 1, H, W]
                
                # Forward pass
                pred_y, _, _ = model(context_x, context_y, target_x)
                
                # Loss
                loss = criterion(pred_y, target_y)
                val_loss += loss.item()
                
                # Reshape predictions to images for metrics
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
                        
                        val_psnr_sum += psnr
                        val_ssim_sum += ssim_val
                        val_count += 1
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr_sum / val_count if val_count > 0 else 0.0
        avg_val_ssim = val_ssim_sum / val_count if val_count > 0 else 0.0
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        val_ssims.append(avg_val_ssim)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f} dB")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'cnp_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'cnp_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims
    }


def train_convcnp(model, train_loader, val_loader, device, args):
    """Train ConvCNP model"""
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in train_pbar:
            masked_images = batch['masked_image'].to(device)  # [B, 3, H, W]
            masks = batch['mask'].to(device)                  # [B, 1, H, W]
            images = batch['image'].to(device)                 # [B, 3, H, W]
            
            # Forward pass
            pred_image, pred_masked = model(masked_images, masks)
            
            # Compute loss only on masked regions
            # mask: 1 = masked (predict), 0 = context (use original)
            masked_pred = pred_image * masks
            masked_gt = images * masks
            
            loss = criterion(masked_pred, masked_gt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in val_pbar:
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
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr_sum / val_count if val_count > 0 else 0.0
        avg_val_ssim = val_ssim_sum / val_count if val_count > 0 else 0.0
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        val_ssims.append(avg_val_ssim)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f} dB")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'convcnp_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'convcnp_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims
    }


def main():
    parser = argparse.ArgumentParser(description="Train Neural Processes for image inpainting")
    parser.add_argument("--model", type=str, choices=['cnp', 'convcnp', 'both'], default='both',
                       help="Model to train")
    parser.add_argument("--image_dir", type=str, default="celeba_hq_256",
                       help="Directory containing images")
    parser.add_argument("--mask_dir", type=str, default="celeba_mask40_faces",
                       help="Directory containing masks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_np",
                       help="Directory to save checkpoints")
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
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Image size (will be resized)")
    
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
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    if args.model in ['cnp', 'both']:
        print("\n" + "="*60)
        print("Training CNP")
        print("="*60)
        
        full_dataset = NPImageDataset(
            img_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=transform,
            max_context_points=args.max_context_points,
            model_type='cnp'
        )
        
        # Split dataset
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_cnp,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_cnp,
            num_workers=2
        )
        
        # Create model
        model = CNP(
            input_dim=5,
            hidden_dim=128,
            encoder_layers=4,
            decoder_layers=3,
            output_dim=3
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        cnp_history = train_cnp(model, train_loader, val_loader, device, args)
        
        # Save training history
        history_path = os.path.join(args.checkpoint_dir, 'cnp_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(cnp_history, f, indent=2)
        print(f"\nTraining history saved to {history_path}")
    
    if args.model in ['convcnp', 'both']:
        print("\n" + "="*60)
        print("Training ConvCNP")
        print("="*60)
        
        full_dataset = NPImageDataset(
            img_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=transform,
            model_type='convcnp'
        )
        
        # Split dataset
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_convcnp,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_convcnp,
            num_workers=2
        )
        
        # Create model
        model = ConvCNP(
            in_channels=3,
            hidden_channels=64,
            num_layers=4,
            kernel_size=3
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        convcnp_history = train_convcnp(model, train_loader, val_loader, device, args)
        
        # Save training history
        history_path = os.path.join(args.checkpoint_dir, 'convcnp_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(convcnp_history, f, indent=2)
        print(f"\nTraining history saved to {history_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

