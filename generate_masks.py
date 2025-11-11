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

    missing_area = int((percent / 100) * size * size)
    side = int(np.sqrt(missing_area))  # side length of square block

    for i in range(num_masks):
        mask = np.zeros((size, size), dtype=np.uint8)

        # random top-left corner of the block
        x = np.random.randint(0, size - side)
        y = np.random.randint(0, size - side)

        # fill the block as missing (white)
        mask[y:y+side, x:x+side] = 255

        img = Image.fromarray(mask)
        img.save(os.path.join(folder_path, f"mask_{i}.png"))

    print(f"✅ {percent}% masks generated.")

print("✅ All block masks created!")
