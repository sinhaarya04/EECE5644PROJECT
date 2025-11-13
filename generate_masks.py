import os
import numpy as np
from PIL import Image

output_root = "masks"
size = 128

levels = {
    20: "20",
    40: "40",
    60: "60",
    80: "80"
}

num_masks = 5000  # plenty for training/testing

for percent, folder in levels.items():
    folder_path = os.path.join(output_root, folder)
    os.makedirs(folder_path, exist_ok=True)

    num_pixels_to_mask = int((percent / 100) * size * size)

    for i in range(num_masks):
        mask = np.zeros((size, size), dtype=np.uint8)

        # Get all pixel positions
        total_pixels = size * size
        pixel_indices = np.arange(total_pixels)
        
        # Randomly select pixels to mask
        masked_indices = np.random.choice(pixel_indices, size=num_pixels_to_mask, replace=False)
        
        # Convert flat indices to 2D coordinates
        y_coords = masked_indices // size
        x_coords = masked_indices % size
        
        # Set selected pixels as missing (white)
        mask[y_coords, x_coords] = 255

        img = Image.fromarray(mask)
        img.save(os.path.join(folder_path, f"mask_{i}.png"))

    print(f"✅ {percent}% masks generated.")

print("✅ All random pixel masks created!")
