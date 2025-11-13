import matplotlib.pyplot as plt
from torchvision import transforms
from src.dataset import CelebAInpaintingDataset
import numpy as np

# Get one image to use for all visualizations
base_dataset = CelebAInpaintingDataset(
    img_dir="celeba_processed_128",
    mask_dir="masks/20",
    transform=transforms.ToTensor()
)
_, _, base_img = base_dataset[0]
base_img_np = base_img.permute(1, 2, 0).cpu().numpy()

# Create visualization for all mask percentages
mask_levels = [20, 40, 60, 80]
fig, axes = plt.subplots(len(mask_levels), 4, figsize=(16, 4 * len(mask_levels)))

for idx, percent in enumerate(mask_levels):
    dataset = CelebAInpaintingDataset(
        img_dir="celeba_processed_128",
        mask_dir=f"masks/{percent}",
        transform=transforms.ToTensor()
    )
    
    # Get one sample
    masked_img, mask, img = dataset[0]
    
    # Convert tensors to numpy for plotting
    masked_img_np = masked_img.permute(1, 2, 0).cpu().numpy()
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.permute(1, 2, 0).cpu().numpy()
    
    # Original image
    axes[idx, 0].imshow(img_np)
    axes[idx, 0].set_title(f"Original Image ({percent}% level)")
    axes[idx, 0].axis("off")
    
    # Mask
    axes[idx, 1].imshow(mask_np.squeeze(), cmap="gray")
    axes[idx, 1].set_title(f"Random Pixel Mask ({percent}%)")
    axes[idx, 1].axis("off")
    
    # Masked image
    axes[idx, 2].imshow(masked_img_np)
    axes[idx, 2].set_title(f"Masked Image ({percent}%)")
    axes[idx, 2].axis("off")
    
    # Overlay (mask on original)
    overlay = img_np.copy()
    mask_binary = mask_np.squeeze() > 0.5
    overlay[mask_binary] = [1.0, 0.0, 0.0]  # Red overlay for masked pixels
    axes[idx, 3].imshow(overlay)
    axes[idx, 3].set_title(f"Overlay ({percent}%)")
    axes[idx, 3].axis("off")

plt.tight_layout()
plt.savefig("dataset_sample.png", dpi=150, bbox_inches='tight')
print("✅ Visualization saved to dataset_sample.png")
plt.close()

