"""
Generate random pixel masks at different coverage percentages (20%, 40%, 60%, 80%)
for 256x256 images from CelebA-MaskHQ dataset.
"""
import os
import numpy as np
from PIL import Image

output_root = "masks"
size = 256  # 256x256 for CelebA-MaskHQ

levels = {
    20: "20",
    40: "40",
    60: "60",
    80: "80"
}

num_masks = 30000  # Generate one mask per image (30,000 images in CelebA-MaskHQ)

for percent, folder in levels.items():
    folder_path = os.path.join(output_root, folder)
    os.makedirs(folder_path, exist_ok=True)

    num_pixels_to_mask = int((percent / 100) * size * size)

    print(f"Generating {num_masks} masks at {percent}% coverage...")
    
    for i in range(num_masks):
        mask = np.zeros((size, size), dtype=np.uint8)

        # Get all pixel positions
        total_pixels = size * size
        pixel_indices = np.arange(total_pixels)
        
        # Randomly select pixels to mask
        masked_indices = np.random.choice(
            pixel_indices, 
            size=num_pixels_to_mask, 
            replace=False
        )
        
        # Convert flat indices to 2D coordinates
        y_coords = masked_indices // size
        x_coords = masked_indices % size
        
        # Set selected pixels as missing (white = 255)
        # This format: 255 = masked (unknown), 0 = known
        mask[y_coords, x_coords] = 255

        img = Image.fromarray(mask)
        img.save(os.path.join(folder_path, f"mask_{i:05d}.png"))

    print(f"{percent}% masks generated in {folder_path}/")

print("\nAll random pixel masks created!")
print(f"Mask format: White (255) = masked region, Black (0) = known region")
print(f"Total masks: {num_masks} per percentage level")

