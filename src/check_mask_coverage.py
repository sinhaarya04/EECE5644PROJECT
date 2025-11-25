"""
Utility script to sanity-check mask coverage.

For each requested mask level directory (e.g., mask_coords/40),
we sample a few masks, compute the percentage of masked pixels, and
compare it to the nominal value (20, 40, 60, 80).
"""
import argparse
import os
import random
from typing import List

import numpy as np
import torch
from torchvision import transforms

from dataset_np import NPImageDataset


def inspect_dataset_samples(
    dataset: NPImageDataset,
    expected_ratio: float,
    sample_count: int,
    tolerance: float = 1e-5
) -> List[float]:
    """
    Sample items from NPImageDataset and report the masked coverage and mask fidelity.
    """
    num_samples = min(sample_count, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    ratios = []

    for idx in indices:
        sample = dataset[idx]
        mask = sample["mask"]  # [1, H, W]
        image = sample["image"]
        masked_image = sample["masked_image"]

        coverage = mask.mean().item()
        ratios.append(coverage)

        reconstructed_masked = image * (1.0 - mask)
        max_diff = torch.abs(masked_image - reconstructed_masked).max().item()

        image_name = dataset.images[idx]
        mask_name = dataset.mask_pairs.get(image_name)
        if mask_name is None:
            mask_name = dataset.masks[idx % len(dataset.masks)]

        print(
            f"  mask:{mask_name:<20} | img:{image_name:<8} "
            f"masked: {coverage*100:6.2f}% "
            f"(expected ~{expected_ratio*100:.0f}%) | "
            f"reconstruct Δmax: {max_diff:.2e}"
        )

        if max_diff > tolerance:
            print("    [!] Warning: masked_image does not match image*(1-mask) within tolerance.")

    return ratios


def main():
    parser = argparse.ArgumentParser(description="Check mask coverage percentages.")
    parser.add_argument(
        "--mask_base_dir",
        type=str,
        default="mask_coords",
        help="Base directory containing mask coordinate folders (e.g., mask_coords/40)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="celeba_hq_256",
        help="Directory containing the original images (paired by order).",
    )
    parser.add_argument(
        "--mask_levels",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80],
        help="Mask percentage levels to inspect.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="How many masks to sample per level.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size used for resizing before feeding the dataset.",
    )
    parser.add_argument(
        "--reconstruct_tolerance",
        type=float,
        default=1e-5,
        help="Allowed max per-pixel difference between masked_image and image*(1-mask).",
    )

    args = parser.parse_args()

    print("Mask coverage sanity check")
    print("=" * 60)
    print(f"Base directory : {os.path.abspath(args.mask_base_dir)}")
    print(f"Mask levels    : {args.mask_levels}")
    print(f"Image dir      : {os.path.abspath(args.image_dir)}")
    print(f"Masks per level: {args.samples}")
    print("=" * 60)

    for level in args.mask_levels:
        folder = os.path.join(args.mask_base_dir, f"{level}")
        if not os.path.isdir(folder):
            print(f"\n[!] Skipping {level}% - directory not found: {folder}")
            continue

        expected = level / 100.0
        print(f"\n{level}% masks ({folder})")
        print("-" * 60)
        try:
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor()
            ])

            dataset = NPImageDataset(
                img_dir=args.image_dir,
                mask_dir=folder,
                transform=transform,
                model_type='convcnp'
            )

            if len(dataset) == 0:
                print("  Warning: Dataset returned zero samples.")
                continue

            ratios = inspect_dataset_samples(
                dataset,
                expected_ratio=expected,
                sample_count=args.samples,
                tolerance=args.reconstruct_tolerance
            )

            avg = np.mean(ratios) * 100
            std = np.std(ratios) * 100
            print(f"Average masked: {avg:6.2f}% ± {std:4.2f}% (n={len(ratios)})")
        except Exception as exc:
            print(f"  Error processing {folder}: {exc}")


if __name__ == "__main__":
    main()

