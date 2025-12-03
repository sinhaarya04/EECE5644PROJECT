"""
RePaint using Hugging Face Diffusers library
Faster and easier to use than the original RePaint implementation
"""
import os
import random
from PIL import Image
import torch
from diffusers import RePaintPipeline, RePaintScheduler
import numpy as np

def load_image(image_path):
    """Load image from local path"""
    return Image.open(image_path).convert("RGB").resize((256, 256))

def load_mask(mask_path):
    """Load mask from local path"""
    mask = Image.open(mask_path).convert("L").resize((256, 256))
    # Convert mask to binary (0 or 255)
    mask_array = np.array(mask)
    mask_array = (mask_array > 127).astype(np.uint8) * 255
    return Image.fromarray(mask_array)

def run_repaint_hf(
    image_path,
    mask_path,
    output_path,
    num_inference_steps=250,
    jump_length=10,
    jump_n_sample=10,
    device="auto"
):
    """
    Run RePaint using Hugging Face diffusers
    
    Args:
        image_path: Path to input image
        mask_path: Path to mask image
        output_path: Path to save inpainted result
        num_inference_steps: Number of diffusion steps (default: 250)
        jump_length: RePaint jump length (default: 10)
        jump_n_sample: RePaint jump n sample (default: 10)
        device: Device to use ("auto", "mps", "cuda", "cpu")
    """
    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load images
    print(f"Loading image: {image_path}")
    original_image = load_image(image_path)
    
    print(f"Loading mask: {mask_path}")
    mask_image = load_mask(mask_path)
    
    # Load the RePaint scheduler and pipeline
    print("Loading RePaint model from Hugging Face...")
    scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
    pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)
    pipe = pipe.to(device)
    
    # Set generator for reproducibility
    if device == "cuda":
        generator = torch.Generator(device=device).manual_seed(0)
    else:
        generator = torch.Generator().manual_seed(0)
    
    # Run inpainting
    print(f"Running inpainting (steps: {num_inference_steps}, jump_length: {jump_length}, jump_n_sample: {jump_n_sample})...")
    output = pipe(
        image=original_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        eta=0.0,
        jump_length=jump_length,
        jump_n_sample=jump_n_sample,
        generator=generator,
    )
    
    # Save results
    inpainted_image = output.images[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create masked image (original with mask applied as black pixels)
    original_array = np.array(original_image)
    mask_array = np.array(mask_image.convert("L"))
    # Our masks: white (255) = masked, black (0) = known
    # For display: where mask is white, make image black
    mask_binary = (mask_array > 127).astype(np.uint8)
    masked_array = original_array.copy()
    masked_array[mask_binary == 1] = 0  # Apply mask as black pixels
    masked_image = Image.fromarray(masked_array)
    
    # Create combined image (side by side: Original | Masked | Inpainted)
    width, height = original_image.size
    combined_width = width * 3
    combined_image = Image.new('RGB', (combined_width, height))
    
    # Paste images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(masked_image, (width, 0))
    combined_image.paste(inpainted_image, (width * 2, 0))
    
    # Save combined image
    combined_image.save(output_path)
    print(f"Saved combined result to: {output_path}")
    print(f"  Layout: Original | Masked | Inpainted")
    
    return inpainted_image, masked_image, original_image

def process_dataset(
    image_dir,
    mask_dir,
    output_dir,
    mask_levels=[20, 40, 60, 80],
    num_images=1,
    num_inference_steps=250,
    jump_length=10,
    jump_n_sample=10,
    device="auto"
):
    """
    Process multiple images with different mask levels
    
    Args:
        image_dir: Directory containing images
        mask_dir: Base directory for masks (will look in mask_dir/20, mask_dir/40, etc.)
        output_dir: Directory to save results
        mask_levels: List of mask percentages to test
        num_images: Number of images to process per mask level
        num_inference_steps: Number of diffusion steps
        jump_length: RePaint jump length
        jump_n_sample: RePaint jump n sample
        device: Device to use
    """
    # Get image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    # Process each mask level
    for mask_level in mask_levels:
        print(f"\n{'='*60}")
        print(f"Processing {mask_level}% masks")
        print(f"{'='*60}")
        
        mask_level_dir = os.path.join(mask_dir, str(mask_level))
        if not os.path.exists(mask_level_dir):
            print(f"Warning: Mask directory {mask_level_dir} not found, skipping...")
            continue
        
        # Get mask files
        mask_files = sorted([f for f in os.listdir(mask_level_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(mask_files) == 0:
            print(f"Warning: No masks found in {mask_level_dir}, skipping...")
            continue
        
        # Process images
        selected_images = image_files[:num_images]
        
        for img_file in selected_images:
            img_path = os.path.join(image_dir, img_file)
            img_name = os.path.splitext(img_file)[0]
            
            # Randomly select a mask
            mask_file = random.choice(mask_files)
            mask_path = os.path.join(mask_level_dir, mask_file)
            
            # Output paths
            output_dir_level = os.path.join(output_dir, f"{mask_level}percent")
            output_path = os.path.join(output_dir_level, f"{img_name}_inpainted.png")
            
            try:
                run_repaint_hf(
                    image_path=img_path,
                    mask_path=mask_path,
                    output_path=output_path,
                    num_inference_steps=num_inference_steps,
                    jump_length=jump_length,
                    jump_n_sample=jump_n_sample,
                    device=device
                )
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RePaint using Hugging Face diffusers")
    parser.add_argument("--image_dir", type=str, default="CelebAMask-HQ/CelebA-HQ-img",
                       help="Directory containing input images")
    parser.add_argument("--mask_dir", type=str, default="masks",
                       help="Base directory for masks")
    parser.add_argument("--output_dir", type=str, default="results_hf",
                       help="Directory to save results")
    parser.add_argument("--mask_levels", type=int, nargs="+", default=[20, 40, 60, 80],
                       help="Mask levels to process")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images to process per mask level")
    parser.add_argument("--num_inference_steps", type=int, default=250,
                       help="Number of diffusion steps")
    parser.add_argument("--jump_length", type=int, default=10,
                       help="RePaint jump length")
    parser.add_argument("--jump_n_sample", type=int, default=10,
                       help="RePaint jump n sample")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "mps", "cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    print("RePaint using Hugging Face Diffusers")
    print("=" * 60)
    print(f"Image directory: {args.image_dir}")
    print(f"Mask directory: {args.mask_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mask levels: {args.mask_levels}")
    print(f"Images per level: {args.num_images}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Jump length: {args.jump_length}")
    print(f"Jump n sample: {args.jump_n_sample}")
    print("=" * 60)
    
    process_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        mask_levels=args.mask_levels,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        jump_length=args.jump_length,
        jump_n_sample=args.jump_n_sample,
        device=args.device
    )
    
    print("\nProcessing complete!")

