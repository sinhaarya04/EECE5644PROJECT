import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr_skimage


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images using scikit-image.
    
    Args:
        img1: First image tensor [C, H, W] or [B, C, H, W] in range [0, max_val]
        img2: Second image tensor [C, H, W] or [B, C, H, W] in range [0, max_val]
        max_val: Maximum pixel value (default 1.0 for normalized images)
    
    Returns:
        PSNR value in dB (higher is better)
    """
    # Convert to numpy and handle channel dimension
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Handle different tensor shapes
    if img1.ndim == 4:  # [B, C, H, W]
        img1 = img1[0]  # Take first batch
    if img2.ndim == 4:
        img2 = img2[0]
    
    # Convert from [C, H, W] to [H, W, C] for skimage
    if img1.ndim == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Handle grayscale
    if img1.ndim == 2:
        img1 = img1[..., np.newaxis]
    if img2.ndim == 2:
        img2 = img2[..., np.newaxis]
    
    # Convert to [0, 255] range for skimage
    data_range = max_val * 255.0 if max_val <= 1.0 else max_val
    img1_uint8 = (img1 * 255.0 / max_val).astype(np.uint8) if max_val <= 1.0 else img1.astype(np.uint8)
    img2_uint8 = (img2 * 255.0 / max_val).astype(np.uint8) if max_val <= 1.0 else img2.astype(np.uint8)
    
    # Calculate PSNR using scikit-image
    if img1_uint8.shape[2] == 1:  # Grayscale
        psnr = calculate_psnr_skimage(img1_uint8[:, :, 0], img2_uint8[:, :, 0], data_range=data_range)
    else:  # RGB
        psnr = calculate_psnr_skimage(img1_uint8, img2_uint8, data_range=data_range)
    
    return float(psnr)


def calculate_ssim(img1, img2, max_val=1.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image tensor [C, H, W] in range [0, max_val]
        img2: Second image tensor [C, H, W] in range [0, max_val]
        max_val: Maximum pixel value (default 1.0 for normalized images)
    
    Returns:
        SSIM value in range [0, 1] (higher is better, 1.0 is perfect)
    """
    # Convert to numpy and handle channel dimension
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Handle different tensor shapes
    if img1.ndim == 4:  # [B, C, H, W]
        img1 = img1[0]  # Take first batch
    if img2.ndim == 4:
        img2 = img2[0]
    
    # Convert from [C, H, W] to [H, W, C] for skimage
    if img1.ndim == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Handle grayscale
    if img1.ndim == 2:
        img1 = img1[..., np.newaxis]
    if img2.ndim == 2:
        img2 = img2[..., np.newaxis]
    
    # Convert to [0, 255] range for skimage
    img1 = (img1 * 255.0 / max_val).astype(np.uint8)
    img2 = (img2 * 255.0 / max_val).astype(np.uint8)
    
    # Calculate SSIM
    if img1.shape[2] == 1:  # Grayscale
        ssim_val = ssim(img1[:, :, 0], img2[:, :, 0], data_range=255)
    else:  # RGB
        ssim_val = ssim(img1, img2, data_range=255, channel_axis=2)
    
    return ssim_val

