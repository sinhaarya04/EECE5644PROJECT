# CelebA-MaskHQ Inpainting with RePaint

This project implements image inpainting on the CelebA-MaskHQ dataset using the RePaint diffusion model.

## Project Structure

```
.
├── CelebAMask-HQ/              # CelebA-MaskHQ dataset
│   ├── CelebA-HQ-img/          # 256x256 face images
│   └── CelebAMask-HQ-mask-anno/ # Feature masks (hair, eyes, etc.)
├── RePaint/                    # RePaint implementation
│   ├── confs/                  # Configuration files
│   │   ├── celeba_256_20.yml  # 20% mask level
│   │   ├── celeba_256_40.yml  # 40% mask level
│   │   ├── celeba_256_60.yml  # 60% mask level
│   │   └── celeba_256_80.yml  # 80% mask level
│   ├── guided_diffusion/       # Core RePaint code
│   │   └── custom_dataset.py   # Custom dataset loader for CelebA-MaskHQ
│   └── test.py                 # Main inference script
├── src/                        # Utility code
│   ├── dataset.py              # Dataset utilities
│   └── metrics.py              # Evaluation metrics (PSNR, SSIM)
├── evaluate_metrics.py         # Evaluation script
└── requirements.txt           # Python dependencies
```

## Features

- **256x256 Resolution**: Direct use of CelebA-MaskHQ 256x256 images
- **Feature Masks**: Uses semantic feature masks (hair, eyes, brows, mouth, nose, etc.)
- **Multiple Mask Levels**: Configurations for 20%, 40%, 60%, and 80% mask coverage
- **RePaint Integration**: Custom dataset loader that handles CelebA-MaskHQ structure

## Setup

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download CelebA-MaskHQ Dataset

Download the CelebA-MaskHQ dataset and place it in the project root:
- `CelebAMask-HQ/CelebA-HQ-img/` - Face images (256x256)
- `CelebAMask-HQ/CelebAMask-HQ-mask-anno/` - Feature masks

### 3. Download Pre-trained Model

```bash
cd RePaint
bash download.sh
```

This downloads the pre-trained CelebA 256x256 diffusion model to `RePaint/data/pretrained/celeba256_250000.pt`


### Evaluate Results

```bash
python evaluate_metrics.py
```

## Configuration

Each config file (`celeba_256_XX.yml`) contains:

- **Model settings**: Architecture parameters for 256x256 images
- **Data paths**: 
  - `gt_path`: Path to CelebA-HQ-img directory
  - `mask_path`: Path to CelebAMask-HQ-mask-anno directory
- **Output paths**: Where to save inpainted results
- **Processing settings**: Batch size, max images, etc.

### Key Settings

- `image_size: 256` - Image resolution
- `use_celeba_maskhq: true` - Enable CelebA-MaskHQ feature mask loading
- `random_mask: true` - Randomly select feature masks per image
- `max_len: 100` - Process only first 100 images (for testing)

## Dataset Format

### Images
- Location: `CelebAMask-HQ/CelebA-HQ-img/`
- Format: JPG files named `0.jpg`, `1.jpg`, etc.
- Resolution: 256x256

## Mask Format

RePaint expects masks where:
- **White (255) = KEEP** (known regions)
- **Black (0) = GENERATE** (unknown regions)


## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pillow
- PyYAML
- blobfile

See `requirements.txt` for complete list.

## License

This project uses the RePaint implementation. Please refer to the original RePaint repository for licensing information.
