"""
Create visualization with error heatmap, masked image, and inpainted image
for 20% and 40% mask levels
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import extract_images_from_combined

def compute_error_heatmap(original, inpainted):
    """
    Compute error heatmap showing pixel-wise difference between original and inpainted images
    
    Args:
        original: PIL Image (original)
        inpainted: PIL Image (inpainted result)
    
    Returns:
        tuple: (PIL Image of error heatmap, error_map array, min_error, max_error)
    """
    # Convert to numpy arrays
    orig_array = np.array(original).astype(np.float32)
    inpainted_array = np.array(inpainted).astype(np.float32)
    
    # Compute absolute difference per channel
    diff = np.abs(orig_array - inpainted_array)
    
    # Convert to grayscale error map (L2 norm across channels)
    error_map = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Store min/max for colorbar
    min_error = error_map.min()
    max_error = error_map.max()
    
    # Normalize to [0, 255] range for colormap
    error_map_normalized = (error_map - min_error) / (max_error - min_error + 1e-8) * 255.0
    error_map_normalized = error_map_normalized.astype(np.uint8)
    
    # Apply colormap for better visualization
    # Use 'hot' colormap: black (low error) -> red -> yellow -> white (high error)
    try:
        # For newer matplotlib versions (3.7+)
        import matplotlib
        colormap = matplotlib.colormaps.get_cmap('hot')
    except (AttributeError, ImportError):
        # Fallback for older versions
        colormap = cm.get_cmap('hot')
    error_colored = colormap(error_map_normalized / 255.0)
    error_colored = (error_colored[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel
    
    return Image.fromarray(error_colored), error_map, min_error, max_error

def create_colorbar_index(min_error, max_error, height, width=50):
    """
    Create a colorbar/legend showing the error scale
    
    Args:
        min_error: Minimum error value
        max_error: Maximum error value
        height: Height of the colorbar (should match image height)
        width: Width of the colorbar
    
    Returns:
        PIL Image of the colorbar with labels
    """
    try:
        import matplotlib
        colormap = matplotlib.colormaps.get_cmap('hot')
    except (AttributeError, ImportError):
        colormap = cm.get_cmap('hot')
    
    # Create gradient
    gradient = np.linspace(0, 1, height)
    gradient = np.tile(gradient[:, np.newaxis], (1, width))
    
    # Apply colormap
    colored_gradient = colormap(gradient)
    colored_gradient = (colored_gradient[:, :, :3] * 255).astype(np.uint8)
    
    # Create image with extra width for labels
    label_width = 80
    total_width = width + label_width
    colorbar_img = Image.new('RGB', (total_width, height), color=(0, 0, 0))
    colorbar_img.paste(Image.fromarray(colored_gradient), (0, 0))
    
    # Add text labels using PIL ImageDraw
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(colorbar_img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font = ImageFont.load_default()
    
    # Add labels on the right side
    # Top label (max error) - high error
    max_label = f"{max_error:.1f}"
    draw.text((width + 8, 5), max_label, fill=(255, 255, 255), font=font)
    
    # Bottom label (min error) - low error
    min_label = f"{min_error:.1f}"
    bbox = draw.textbbox((0, 0), min_label, font=font)
    text_height = bbox[3] - bbox[1]
    draw.text((width + 8, height - text_height - 5), min_label, fill=(255, 255, 255), font=font)
    
    # Middle label
    mid_error = (min_error + max_error) / 2
    mid_label = f"{mid_error:.1f}"
    bbox = draw.textbbox((0, 0), mid_label, font=font)
    text_height = bbox[3] - bbox[1]
    draw.text((width + 8, (height - text_height) // 2), mid_label, fill=(255, 255, 255), font=font)
    
    return colorbar_img

def create_visualization(result_path, output_path, mask_level):
    """
    Create visualization with error heatmap, masked image, and inpainted image
    
    Args:
        result_path: Path to combined result image
        output_path: Path to save visualization
        mask_level: Mask level (20 or 40) for title
    """
    # Extract images from combined PNG
    original, masked, inpainted = extract_images_from_combined(result_path)
    
    # Compute error heatmap (returns heatmap image, error map, min, max)
    error_heatmap, error_map, min_error, max_error = compute_error_heatmap(original, inpainted)
    
    # Ensure all images are the same size
    width, height = original.size
    original = original.resize((width, height), Image.BICUBIC)
    error_heatmap = error_heatmap.resize((width, height), Image.BICUBIC)
    masked = masked.resize((width, height), Image.BICUBIC)
    inpainted = inpainted.resize((width, height), Image.BICUBIC)
    
    # Create colorbar index
    colorbar_width = 50
    colorbar_label_width = 80
    colorbar = create_colorbar_index(min_error, max_error, height, width=colorbar_width)
    
    # Create combined visualization: Original | Error Heatmap | Masked | Inpainted | Colorbar
    combined_width = width * 4 + colorbar_width + colorbar_label_width + 10  # Extra space for colorbar and labels
    combined_image = Image.new('RGB', (combined_width, height + 20), color=(0, 0, 0))  # Black background with extra height for title
    
    # Paste images side by side
    combined_image.paste(original, (0, 0))
    combined_image.paste(error_heatmap, (width, 0))
    combined_image.paste(masked, (width * 2, 0))
    combined_image.paste(inpainted, (width * 3, 0))
    
    # Paste colorbar
    combined_image.paste(colorbar, (width * 4 + 5, 0))
    
    # Add labels using PIL ImageDraw
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined_image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Add column labels below images
    label_y = height + 5
    draw.text((width // 2 - 30, label_y), "Original", fill=(255, 255, 255), font=small_font)
    draw.text((width + width // 2 - 30, label_y), "Error Heatmap", fill=(255, 255, 255), font=small_font)
    draw.text((width * 2 + width // 2 - 30, label_y), "Masked Image", fill=(255, 255, 255), font=small_font)
    draw.text((width * 3 + width // 2 - 30, label_y), "Inpainted", fill=(255, 255, 255), font=small_font)
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_image.save(output_path)
    print(f"Saved visualization to: {output_path}")
    print(f"  Layout: Original | Error Heatmap | Masked Image | Inpainted Image | Colorbar")
    print(f"  Error range: {min_error:.2f} - {max_error:.2f}")
    
    return combined_image

def main():
    results_dir = "results_hf"
    output_dir = "visualizations"
    num_images = 5  # Process first 5 images
    
    # Process 20% and 40% mask levels
    for mask_level in [20, 40]:
        level_dir = os.path.join(results_dir, f"{mask_level}percent")
        
        if not os.path.exists(level_dir):
            print(f"Warning: Directory not found: {level_dir}")
            continue
        
        # Get the first result files (sorted)
        result_files = sorted([f for f in os.listdir(level_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(result_files) == 0:
            print(f"Warning: No results found in {level_dir}")
            continue
        
        # Process first num_images files
        files_to_process = result_files[:num_images]
        print(f"\nProcessing {mask_level}% mask level ({len(files_to_process)} images)...")
        
        for idx, result_file in enumerate(files_to_process):
            result_path = os.path.join(level_dir, result_file)
            
            # Extract image number from filename (e.g., "0_inpainted.png" -> "0")
            img_name = os.path.splitext(result_file)[0].replace('_inpainted', '')
            output_path = os.path.join(output_dir, f"{mask_level}_{img_name}.png")
            
            print(f"  Processing image {idx+1}/{len(files_to_process)}: {result_file}")
            create_visualization(result_path, output_path, mask_level)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()

