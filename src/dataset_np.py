"""
Dataset adapter for Neural Processes.
Derives true binary masks by comparing original and masked images.
"""
import os
import re
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        model_type='cnp',  # 'cnp' or 'convcnp'
        mask_diff_threshold=1e-3
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize_coords = normalize_coords
        self.max_context_points = max_context_points
        self.model_type = model_type
        self.mask_diff_threshold = mask_diff_threshold
        
        self.images = sorted([f for f in os.listdir(img_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.masks = sorted(
            [
                f
                for f in os.listdir(mask_dir)
                if f.lower().endswith((".npz", ".npy", ".json"))
            ],
            key=self._mask_sort_key,
        )
        self.mask_pairs = self._build_pairs(self.images, self.masks)
        
        if len(self.mask_pairs) == 0:
            raise ValueError(f"No mask coordinate files found in {mask_dir}")

        missing = [img for img in self.images if img not in self.mask_pairs]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                "Mask directory is missing coordinates for some images. "
                f"First missing entries: {preview}"
            )
    
    def __len__(self):
        return len(self.images)

    @staticmethod
    def _mask_sort_key(filename: str, width: int = 5):
        """Return zero-padded numeric suffix for consistent ordering."""
        stem = os.path.splitext(filename)[0]
        match = re.search(r'(\d+)$', stem)
        if match:
            return match.group(1).zfill(width)
        return stem
    
    @staticmethod
    def _build_pairs(images, masks) -> Dict[str, str]:
        """Map each image filename to its corresponding mask file by stem."""
        mask_lookup = {}
        for mask_name in masks:
            mask_lookup[os.path.splitext(mask_name)[0]] = mask_name

        paired = {}
        for image_name in images:
            stem = os.path.splitext(image_name)[0]
            if stem in mask_lookup:
                paired[image_name] = mask_lookup[stem]
        return paired

    @staticmethod
    def _load_mask_coords(mask_path: str) -> np.ndarray:
        ext = os.path.splitext(mask_path)[1].lower()
        if ext == ".npz":
            data = np.load(mask_path)
            if "coords" not in data:
                raise ValueError(f"'coords' array missing in {mask_path}")
            coords = data["coords"]
        elif ext == ".npy":
            coords = np.load(mask_path)
        elif ext == ".json":
            import json

            with open(mask_path, "r") as f:
                payload = json.load(f)
            coords = np.array(payload["coords"], dtype=np.int64)
        else:
            raise ValueError(f"Unsupported mask format: {mask_path}")

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Mask coordinates must have shape [N, 2], got {coords.shape} for {mask_path}"
            )
        return coords.astype(np.int64, copy=False)
    
    def extract_context_points(self, img, mask):
        """
        Extract context points using the true mask map.
        
        Args:
            img: Original image tensor [C, H, W]
            mask: Mask tensor [1, H, W] where 1 = masked
        
        Returns:
            context_points: [N_c, 5] where columns are (x, y, r, g, b)
            context_values: [N_c, 3] RGB values
        """
        # Convert to numpy for easier manipulation
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        mask_np = mask.squeeze().cpu().numpy()  # [H, W]
        
        H, W = img_np.shape[:2]
        
        # Context pixels are where mask == 0 (known region)
        context_mask = mask_np < 0.5
        
        y_coords, x_coords = np.where(context_mask)
        
        if len(y_coords) == 0:
            # If no context points, use a few random points
            y_coords = np.random.randint(0, H, size=min(100, H*W))
            x_coords = np.random.randint(0, W, size=min(100, H*W))
        
        # Get RGB values at these coordinates from the original image
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
        
        # Load mask coordinates for this image
        mask_name = self.mask_pairs.get(self.images[idx])
        if mask_name is None:
            raise KeyError(f"No mask coordinates found for {self.images[idx]}")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask_coords = self._load_mask_coords(mask_path)
        
        if self.transform:
            img = self.transform(img)        # [3, H, W]
        else:
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)

        masked_img = img.clone()
        H, W = img.shape[1], img.shape[2]

        mask = torch.zeros((1, H, W), dtype=torch.float32)
        if mask_coords.size > 0:
            coords_tensor = torch.from_numpy(mask_coords)
            y_idx = torch.clamp(coords_tensor[:, 0], 0, H - 1)
            x_idx = torch.clamp(coords_tensor[:, 1], 0, W - 1)
            mask[0, y_idx, x_idx] = 1.0
            masked_img[:, y_idx, x_idx] = 0.0
        
        if self.model_type == 'cnp':
            # Extract context points for CNP
            context_points, context_values = self.extract_context_points(img, mask)
            
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

