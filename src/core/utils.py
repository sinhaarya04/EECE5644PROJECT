"""
Utility functions for image inpainting and evaluation.
"""

import numpy as np
import cv2
from scipy.spatial import KDTree
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import timedelta


def format_time(seconds):
    """Format seconds into human-readable string."""
    return str(timedelta(seconds=int(seconds)))


def load_mask(mask_path, target_size=(256, 256), threshold=127):
    """
    Load binary mask from PNG file.
    
    Parameters:
    - mask_path: path to the mask PNG file
    - target_size: target size to resize to
    - threshold: threshold for binarizing the mask (white=255 is masked)
    
    Returns:
    - mask: binary mask (True = needs interpolation, False = valid pixel)
    """
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Could not read mask from {mask_path}")
    
    if mask_img.shape[:2] != target_size:
        mask_img = cv2.resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
    
    binary_mask = mask_img > threshold
    return binary_mask


def apply_mask_to_image(img, mask):
    """
    Apply mask to image by setting masked pixels to black.
    
    Parameters:
    - img: original image
    - mask: binary mask (True = mask this pixel)
    
    Returns:
    - masked_img: image with masked pixels set to black
    """
    masked_img = img.copy()
    masked_img[mask] = 0
    return masked_img


def knn_interpolation(img, mask, k=8, weighted=True):
    """
    Fill masked regions using k-nearest neighbors interpolation.
    
    IMPORTANT: Only uses KNOWN (valid/unmasked) pixels as neighbors.
    
    Parameters:
    - img: input image (numpy array)
    - mask: binary mask (True = needs interpolation, False = valid pixel)
    - k: number of nearest neighbors to use
    - weighted: if True, use inverse distance weighting; if False, use uniform weights
    
    Returns:
    - result: interpolated image
    """
    valid_mask = ~mask
    h, w = mask.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    valid_points = np.column_stack((y_coords[valid_mask], x_coords[valid_mask]))
    target_points = np.column_stack((y_coords[mask], x_coords[mask]))
    
    if len(valid_points) == 0 or len(target_points) == 0:
        return img.copy()
    
    tree = KDTree(valid_points)
    result = img.copy()
    
    if len(img.shape) == 3:
        for c in range(img.shape[2]):
            valid_values = img[:, :, c][valid_mask]
            
            if len(valid_values) > 0:
                distances, indices = tree.query(target_points, k=min(k, len(valid_values)))
                
                if k == 1:
                    distances = distances.reshape(-1, 1)
                    indices = indices.reshape(-1, 1)
                
                if weighted:
                    weights = 1.0 / (distances + 1e-10)
                    weights = weights / np.sum(weights, axis=1, keepdims=True)
                else:
                    weights = np.ones_like(distances) / distances.shape[1]
                
                interpolated = np.sum(valid_values[indices] * weights, axis=1)
                result[mask, c] = interpolated
    else:
        valid_values = img[valid_mask]
        
        if len(valid_values) > 0:
            distances, indices = tree.query(target_points, k=min(k, len(valid_values)))
            
            if k == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)
            
            if weighted:
                weights = 1.0 / (distances + 1e-10)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
            else:
                weights = np.ones_like(distances) / distances.shape[1]
            
            interpolated = np.sum(valid_values[indices] * weights, axis=1)
            result[mask] = interpolated
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def navier_stokes_inpainting(img, mask, inpaint_radius=3):
    """
    Fill masked regions using Navier-Stokes based inpainting.
    
    Parameters:
    - img: input image (numpy array)
    - mask: binary mask (True = needs interpolation, False = valid pixel)
    - inpaint_radius: radius of circular neighborhood for inpainting
    
    Returns:
    - result: inpainted image
    """
    inpaint_mask = mask.astype(np.uint8) * 255
    result = cv2.inpaint(img, inpaint_mask, inpaintRadius=inpaint_radius, 
                        flags=cv2.INPAINT_NS)
    return result


def telea_inpainting(img, mask, inpaint_radius=3):
    """
    Fill masked regions using Telea's fast marching method.
    
    Parameters:
    - img: input image (numpy array)
    - mask: binary mask (True = needs interpolation, False = valid pixel)
    - inpaint_radius: radius of circular neighborhood for inpainting
    
    Returns:
    - result: inpainted image
    """
    inpaint_mask = mask.astype(np.uint8) * 255
    result = cv2.inpaint(img, inpaint_mask, inpaintRadius=inpaint_radius, 
                        flags=cv2.INPAINT_TELEA)
    return result


def compute_metrics(original, reconstructed):
    """
    Compute PSNR and SSIM between original and reconstructed images.
    
    Parameters:
    - original: original image (numpy array)
    - reconstructed: reconstructed image (numpy array)
    
    Returns:
    - psnr_value: Peak Signal-to-Noise Ratio
    - ssim_value: Structural Similarity Index
    """
    psnr_value = psnr(original, reconstructed, data_range=255)
    
    if len(original.shape) == 3:
        ssim_value = ssim(original, reconstructed, 
                         channel_axis=2, 
                         data_range=255)
    else:
        ssim_value = ssim(original, reconstructed, 
                         data_range=255)
    
    return psnr_value, ssim_value


def get_mask_filename(image_index):
    """
    Get the corresponding mask filename for an image index.
    
    Parameters:
    - image_index: zero-based image index
    
    Returns:
    - filename: mask filename in format mask_XXXXX.png
    """
    return f"mask_{image_index:05d}.png"


def load_image(image_path, target_size=(256, 256)):
    """
    Load and resize image.
    
    Parameters:
    - image_path: path to image file
    - target_size: target size to resize to
    
    Returns:
    - img: loaded and resized image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size)
    
    return img