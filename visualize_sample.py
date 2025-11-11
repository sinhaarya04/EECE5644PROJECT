import matplotlib.pyplot as plt
from torchvision import transforms
from src.dataset import CelebAInpaintingDataset

# Create dataset object
dataset = CelebAInpaintingDataset(
    img_dir="celeba_processed_128",
    mask_dir="masks/20",   # change to 40/60/80 to test others
    transform=transforms.ToTensor()
)

# Get one sample
masked_img, mask, img = dataset[0]

# Convert tensors to numpy for plotting
masked_img = masked_img.permute(1, 2, 0).cpu().numpy()
img = img.permute(1, 2, 0).cpu().numpy()
mask = mask.permute(1, 2, 0).cpu().numpy()

# Plot the images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask.squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Masked Image")
plt.imshow(masked_img)
plt.axis("off")

plt.tight_layout()
plt.savefig("dataset_sample.png", dpi=150, bbox_inches='tight')
print("✅ Visualization saved to dataset_sample.png")
plt.close()

