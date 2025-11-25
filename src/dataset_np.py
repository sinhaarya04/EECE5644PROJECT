"""
Dataset adapter for Neural Processes
Extracts context points from non-black pixels in masked images
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np


class NPImageDataset(Dataset):
    """
    Dataset for Neural Processes that extracts context points from images.
    
    For CNP: extracts (x, y, r, g, b) tuples for context points
    For ConvCNP: uses full image tensors
    """
    
    def __init__(
        self,
        img_dir,
        mask_dir,
        transform=None,
        normalize_coords=True,
        max_context_points=None,
        model_type='cnp'  # 'cnp' or 'convcnp'
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize_coords = normalize_coords
        self.max_context_points = max_context_points
        self.model_type = model_type
        
        self.images = sorted([f for f in os.listdir(img_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self):
        return len(self.images)
    
    def extract_context_points(self, img, mask):
        """
        Extract context points from non-black pixels.
        
        Args:
            img: Image tensor [C, H, W] in range [0, 1]
            mask: Mask tensor [1, H, W] in range [0, 1]
        
        Returns:
            context_points: [N_c, 5] where columns are (x, y, r, g, b)
            context_values: [N_c, 3] RGB values
        """
        # Convert to numpy for easier manipulation
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        mask_np = mask.squeeze().cpu().numpy()  # [H, W]
        
        H, W = img_np.shape[:2]
        
        # Find non-black pixels (context points)
        # A pixel is context if it's not masked AND not black
        # In our setup: masked_img = img * (1 - mask), so black pixels are masked
        # We want pixels where img_np is not all zeros
        
        # Check if pixel is non-black (has any non-zero channel)
        non_black = np.any(img_np > 0.01, axis=2)  # Threshold to account for noise
        
        # Get coordinates of non-black pixels
        y_coords, x_coords = np.where(non_black)
        
        if len(y_coords) == 0:
            # If no context points, use a few random points
            y_coords = np.random.randint(0, H, size=min(100, H*W))
            x_coords = np.random.randint(0, W, size=min(100, H*W))
        
        # Get RGB values at these coordinates
        rgb_values = img_np[y_coords, x_coords]  # [N, 3]
        
        # Normalize coordinates to [0, 1] or [-1, 1]
        if self.normalize_coords:
            x_norm = x_coords.astype(np.float32) / (W - 1)  # [0, 1]
            y_norm = y_coords.astype(np.float32) / (H - 1)  # [0, 1]
        else:
            x_norm = x_coords.astype(np.float32)
            y_norm = y_coords.astype(np.float32)
        
        # Combine into context points: [N, 5] = (x, y, r, g, b)
        context_points = np.column_stack([x_norm, y_norm, rgb_values])
        
        # Subsample if too many points
        if self.max_context_points and len(context_points) > self.max_context_points:
            indices = np.random.choice(len(context_points), self.max_context_points, replace=False)
            context_points = context_points[indices]
            rgb_values = rgb_values[indices]
        
        return context_points, rgb_values
    
    def get_target_points(self, H, W):
        """
        Generate target points (all pixel locations).
        
        Returns:
            target_points: [H*W, 2] with (x, y) coordinates
        """
        # Create grid of all pixel locations
        y_coords, x_coords = np.meshgrid(
            np.arange(H), np.arange(W), indexing='ij'
        )
        
        # Normalize coordinates
        if self.normalize_coords:
            x_norm = x_coords.astype(np.float32) / (W - 1)
            y_norm = y_coords.astype(np.float32) / (H - 1)
        else:
            x_norm = x_coords.astype(np.float32)
            y_norm = y_coords.astype(np.float32)
        
        # Flatten and combine
        target_points = np.column_stack([
            x_norm.flatten(),
            y_norm.flatten()
        ])
        
        return target_points
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Pick a random mask file
        mask_name = random.choice(self.masks)
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            img = self.transform(img)  # [3, H, W] in range [0, 1]
            mask = self.transform(mask)  # [1, H, W] in range [0, 1]
        
        # Ensure mask is normalized to [0, 1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Create masked image
        masked_img = img * (1 - mask)
        
        if self.model_type == 'cnp':
            # Extract context points for CNP
            context_points, context_values = self.extract_context_points(masked_img, mask)
            
            # Get target points (all pixels)
            H, W = img.shape[1], img.shape[2]
            target_points = self.get_target_points(H, W)
            
            # Get ground truth values for target points
            img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            target_values = img_np.reshape(-1, 3)  # [H*W, 3]
            
            # Convert to tensors
            context_points = torch.from_numpy(context_points).float()
            context_values = torch.from_numpy(context_values).float()
            target_points = torch.from_numpy(target_points).float()
            target_values = torch.from_numpy(target_values).float()
            
            return {
                'context_x': context_points,  # [N_c, 5]
                'context_y': context_values,  # [N_c, 3]
                'target_x': target_points,    # [H*W, 2]
                'target_y': target_values,    # [H*W, 3]
                'image': img,                 # [3, H, W]
                'masked_image': masked_img,   # [3, H, W]
                'mask': mask                  # [1, H, W]
            }
        
        else:  # convcnp
            # For ConvCNP, return full image tensors
            return {
                'masked_image': masked_img,  # [3, H, W]
                'mask': mask,                 # [1, H, W]
                'image': img,                 # [3, H, W]
                'masked_image_orig': masked_img  # Keep for reference
            }


def collate_fn_cnp(batch):
    """
    Custom collate function for CNP that handles variable-sized context points.
    """
    # Pad context points to same length
    max_context = max([item['context_x'].shape[0] for item in batch])
    
    batch_size = len(batch)
    H, W = batch[0]['image'].shape[1], batch[0]['image'].shape[2]
    
    context_x_list = []
    context_y_list = []
    target_x_list = []
    target_y_list = []
    images = []
    masked_images = []
    masks = []
    
    for item in batch:
        n_context = item['context_x'].shape[0]
        
        # Pad context points
        if n_context < max_context:
            padding_size = max_context - n_context
            context_x_padded = torch.cat([
                item['context_x'],
                torch.zeros(padding_size, 5)
            ], dim=0)
            context_y_padded = torch.cat([
                item['context_y'],
                torch.zeros(padding_size, 3)
            ], dim=0)
        else:
            context_x_padded = item['context_x']
            context_y_padded = item['context_y']
        
        context_x_list.append(context_x_padded)
        context_y_list.append(context_y_padded)
        target_x_list.append(item['target_x'])
        target_y_list.append(item['target_y'])
        images.append(item['image'])
        masked_images.append(item['masked_image'])
        masks.append(item['mask'])
    
    return {
        'context_x': torch.stack(context_x_list),      # [B, max_context, 5]
        'context_y': torch.stack(context_y_list),      # [B, max_context, 3]
        'target_x': torch.stack(target_x_list),        # [B, H*W, 2]
        'target_y': torch.stack(target_y_list),        # [B, H*W, 3]
        'image': torch.stack(images),                  # [B, 3, H, W]
        'masked_image': torch.stack(masked_images),    # [B, 3, H, W]
        'mask': torch.stack(masks)                     # [B, 1, H, W]
    }


def collate_fn_convcnp(batch):
    """
    Custom collate function for ConvCNP.
    """
    masked_images = torch.stack([item['masked_image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    
    return {
        'masked_image': masked_images,  # [B, 3, H, W]
        'mask': masks,                  # [B, 1, H, W]
        'image': images                 # [B, 3, H, W]
    }

