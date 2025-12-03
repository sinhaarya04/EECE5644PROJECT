# Image Inpainting Project

A comprehensive image inpainting project implementing multiple methods including baseline techniques (k-NN, Navier-Stokes), Neural Processes (CNP, ConvCNP), and diffusion models (RePaint). This project also includes support for CelebA-MaskHQ dataset with RePaint integration.

## Project Structure

```
.
├── src/                          # Source code
│   ├── core/                     # Core utilities and metrics
│   │   ├── utils.py             # Image utilities, inpainting methods
│   │   └── metrics.py           # PSNR, SSIM calculations
│   ├── models/                   # Model definitions
│   │   ├── cnp.py               # Conditional Neural Process
│   │   └── convcnp.py           # Convolutional Neural Process
│   ├── data/                     # Dataset classes
│   │   ├── datasets.py          # CelebA dataset
│   │   └── datasets_np.py       # Neural Process dataset
│   ├── training/                  # Training scripts
│   │   ├── train_np.py          # Train CNP/ConvCNP
│   │   └── train_evaluate_all_masks.py  # Train for all mask levels
│   ├── evaluation/               # Evaluation scripts
│   │   ├── evaluate_np.py       # Evaluate Neural Processes
│   │   └── evaluate_results.py  # Evaluate RePaint results
│   ├── inpainting/               # Inpainting implementations
│   │   ├── run_inpainting.py    # Baseline methods (LFW dataset)
│   │   ├── run_inpainting_celeba.py  # Baseline methods (CelebA)
│   │   └── run_repaint_hf.py    # RePaint diffusion model
│   ├── masks/                    # Mask generation utilities
│   │   ├── generate_masks.py     # Generate mask images
│   │   ├── generate_masks_np.py # Generate mask coordinates
│   │   ├── rename_masks.py       # Rename mask files
│   │   └── check_mask_coverage.py  # Verify mask coverage
│   ├── visualization/            # Visualization scripts
│   │   ├── plot_jolie_results.py # Plot Jolie comparison
│   │   ├── diffusion_results_plot.py  # Plot diffusion results
│   │   ├── visualize_results.py  # Visualize inpainting results
│   │   ├── visualize_masks.py   # Visualize masks
│   │   └── visualize_masked_faces.py  # Visualize masked faces
│   └── scripts/                  # Utility scripts
│       ├── cross_validation.py   # Cross-validation for k optimization
│       └── nearest_neighbors.py  # k-NN inpainting pipeline
├── CelebAMask-HQ/              # CelebA-MaskHQ dataset (optional)
│   ├── CelebA-HQ-img/          # 256x256 face images
│   └── CelebAMask-HQ-mask-anno/ # Feature masks (hair, eyes, etc.)
├── RePaint/                    # RePaint implementation (optional)
│   ├── confs/                  # Configuration files
│   │   ├── celeba_256_20.yml  # 20% mask level
│   │   ├── celeba_256_40.yml  # 40% mask level
│   │   ├── celeba_256_60.yml  # 60% mask level
│   │   └── celeba_256_80.yml  # 80% mask level
│   ├── guided_diffusion/       # Core RePaint code
│   │   └── custom_dataset.py   # Custom dataset loader for CelebA-MaskHQ
│   └── test.py                 # Main inference script
├── dataset/                      # Dataset directory
│   ├── lfw_100_people/          # LFW dataset images
│   ├── celeba_hq_256/          # CelebA-HQ images
│   ├── mask_coords/            # Mask coordinate files (.npz)
│   └── [mask_levels]/          # Mask directories (20, 40, 60, 80)
├── checkpoints_np/              # Neural Process model checkpoints
├── results_np/                   # Neural Process results
├── results_hf/                   # RePaint (HuggingFace) results
└── requirements.txt             # Python dependencies
```

## Features

- **Baseline Methods**: k-NN interpolation (uniform and distance-weighted), Navier-Stokes inpainting
- **Neural Processes**: Conditional Neural Process (CNP) and Convolutional Neural Process (ConvCNP)
- **Diffusion Models**: RePaint using Hugging Face Diffusers
- **256x256 Resolution**: Direct use of CelebA-MaskHQ 256x256 images
- **Feature Masks**: Uses semantic feature masks (hair, eyes, brows, mouth, nose, etc.)
- **Multiple Mask Levels**: Configurations for 20%, 40%, 60%, and 80% mask coverage
- **RePaint Integration**: Custom dataset loader that handles CelebA-MaskHQ structure
- **Cross-Validation**: Automatic k-value optimization for k-NN methods
- **Comprehensive Evaluation**: PSNR, SSIM metrics with detailed statistics
- **Visualization**: Comparison plots, error heatmaps, comprehensive visualizations

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Organize your dataset in the following structure:

```
dataset/
├── lfw_100_people/          # LFW dataset images
├── celeba_hq_256/          # CelebA-HQ images (256x256)
├── mask_coords/            # Mask coordinate files for CelebA
│   ├── 20/                # 20% mask level
│   ├── 40/                # 40% mask level
│   ├── 60/                # 60% mask level
│   └── 80/                # 80% mask level
├── 20/                     # Mask images for LFW (20% level)
├── 40/                     # Mask images for LFW (40% level)
├── 60/                     # Mask images for LFW (60% level)
└── 80/                     # Mask images for LFW (60% level)
```

### 4. Download CelebA-MaskHQ Dataset (Optional)

If using the CelebA-MaskHQ dataset, download it and place it in the project root:
- `CelebAMask-HQ/CelebA-HQ-img/` - Face images (256x256)
- `CelebAMask-HQ/CelebAMask-HQ-mask-anno/` - Feature masks

### 5. Download Pre-trained Model (Optional)

If using the RePaint implementation directly:

```bash
cd RePaint
bash download.sh
```

This downloads the pre-trained CelebA 256x256 diffusion model to `RePaint/data/pretrained/celeba256_250000.pt`

## Usage

### Baseline Methods

#### LFW Dataset

```bash
# Run baseline inpainting on LFW dataset
python src/inpainting/run_inpainting.py \
    --input_dir dataset/lfw_100_people \
    --mask_root dataset \
    --output_dir dataset/inpainting_results_lfw \
    --n_images 100 \
    --n_visualize 10

# Filter for specific person
python src/inpainting/run_inpainting.py \
    --input_dir dataset/lfw_100_people \
    --mask_root dataset \
    --output_dir dataset/inpainting_results_lfw \
    --person_filter "Angelina_Jolie"
```

#### CelebA Dataset

```bash
# Run baseline inpainting on CelebA dataset
python src/inpainting/run_inpainting_celeba.py \
    --input_dir dataset/celeba_hq_256 \
    --mask_coords_dir dataset/mask_coords \
    --output_dir dataset/inpainting_results \
    --n_images 100 \
    --n_visualize 100
```

### Cross-Validation for k Optimization

```bash
# Find optimal k values using cross-validation
python src/scripts/cross_validation.py \
    --input_dir dataset/lfw_100_people \
    --mask_root dataset \
    --output_dir dataset/cv_results
```

### Neural Processes (CNP/ConvCNP)

#### Training

```bash
# Train ConvCNP
python src/training/train_np.py \
    --model convcnp \
    --image_dir dataset/celeba_hq_256 \
    --mask_dir dataset/mask_coords/40 \
    --checkpoint_dir checkpoints_np \
    --batch_size 4 \
    --epochs 50 \
    --learning_rate 1e-4

# Train for all mask levels
python src/training/train_evaluate_all_masks.py \
    --image_dir dataset/celeba_hq_256 \
    --mask_coords_root dataset/mask_coords \
    --checkpoint_dir checkpoints_np \
    --output_dir results_np \
    --epochs 50
```

#### Evaluation

```bash
# Evaluate ConvCNP
python src/evaluation/evaluate_np.py \
    --model convcnp \
    --checkpoint_dir checkpoints_np \
    --image_dir dataset/celeba_hq_256 \
    --mask_dir dataset/mask_coords/40 \
    --output_dir results_np/convcnp \
    --batch_size 4
```

### Diffusion Model (RePaint)

#### Using Hugging Face Diffusers

```bash
# Run RePaint on CelebA dataset
python src/inpainting/run_repaint_hf.py \
    --image_dir dataset/celeba_hq_256 \
    --mask_dir dataset \
    --output_dir results_hf \
    --mask_levels 20 40 60 80 \
    --num_images 10 \
    --num_inference_steps 250

# Quick test with fewer steps
python src/inpainting/run_repaint_hf.py \
    --image_dir dataset/celeba_hq_256 \
    --mask_dir dataset \
    --output_dir results_hf \
    --mask_levels 20 40 \
    --num_images 2 \
    --num_inference_steps 50
```

#### Using RePaint Implementation Directly

```bash
# Evaluate results
python evaluate_metrics.py
```

### Visualization

```bash
# Plot Jolie comparison results
python src/visualization/plot_jolie_results.py \
    --results_dir dataset/jolie_results \
    --output_dir dataset/jolie_comparisons

# Plot diffusion results
python src/visualization/diffusion_results_plot.py \
    --results_dir diffusion_results \
    --output_dir diffusion_comparisons
```

## Methods Overview

### Baseline Methods

1. **k-NN Uniform**: k-nearest neighbors with uniform weighting
2. **k-NN Distance**: k-nearest neighbors with inverse distance weighting
3. **Navier-Stokes**: OpenCV's Navier-Stokes inpainting algorithm

### Neural Processes

1. **CNP (Conditional Neural Process)**: Point-based neural process model
2. **ConvCNP (Convolutional Neural Process)**: Convolutional variant for images

### Diffusion Models

1. **RePaint**: Denoising diffusion model with resampling for inpainting

## Output Structure

### Baseline Methods Output

```
output_dir/
├── 20%/                    # 20% mask level results
│   ├── knn_uniform/       # k-NN uniform results
│   ├── knn_distance/      # k-NN distance results
│   ├── navier_stokes/     # Navier-Stokes results
│   └── masked_images/     # Masked input images
├── 40%/                    # 40% mask level results
├── 60%/                    # 60% mask level results
├── 80%/                    # 80% mask level results
├── comprehensive_comparisons/  # Comparison plots
├── summary_statistics.txt      # Summary metrics
└── summary_statistics.png      # Summary plots
```

### Neural Process Output

```
results_np/
├── convcnp/               # ConvCNP results
│   ├── convcnp_sample.png # Sample results
│   └── metrics_results.json  # Metrics
└── cnp/                   # CNP results
    └── ...
```

## Configuration

### RePaint Configuration Files

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
- Location: `CelebAMask-HQ/CelebA-HQ-img/` or `dataset/celeba_hq_256/`
- Format: JPG/PNG files
- Resolution: 256x256

### Mask Format

RePaint expects masks where:
- **White (255) = KEEP** (known regions)
- **Black (0) = GENERATE** (unknown regions)

## Metrics

All methods are evaluated using:
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (dB)
- **SSIM** (Structural Similarity Index): Higher is better (0-1)

## Command-Line Arguments

All scripts support command-line arguments for paths and parameters. Use `--help` to see available options:

```bash
python src/inpainting/run_inpainting.py --help
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- OpenCV 4.7+
- NumPy, SciPy
- Matplotlib, scikit-image
- Pillow
- tqdm
- diffusers, transformers, accelerate (for RePaint)
- PyYAML, blobfile (for RePaint implementation)

See `requirements.txt` for complete list with versions.

## License

This project implements various inpainting methods for research purposes. Please refer to individual method implementations for their respective licenses.
