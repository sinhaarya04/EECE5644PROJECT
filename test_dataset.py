from src.dataset import CelebAInpaintingDataset
from torchvision import transforms

dataset = CelebAInpaintingDataset(
    img_dir="celeba_processed_128",
    mask_dir="masks/20",
    transform=transforms.ToTensor()
)

masked_img, mask, img = dataset[0]
print(masked_img.shape, mask.shape, img.shape)

