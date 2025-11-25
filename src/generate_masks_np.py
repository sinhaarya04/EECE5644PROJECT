"""
Generate reproducible binary mask coordinates for every CelebA-HQ image.

Instead of saving JPEG-masked images (which introduce compression artifacts),
this script records the exact pixel coordinates that should be masked at each
coverage level. Downstream datasets can then rebuild the binary mask and
masked image deterministically at load time.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mask coordinate archives for CelebA-HQ images."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="celeba_hq_256",
        help="Directory containing the original CelebA-HQ (256x256) images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mask_coords",
        help="Root directory where per-level subfolders (20/40/60/80/...) are created.",
    )
    parser.add_argument(
        "--mask_levels",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80],
        help="Mask coverage percentages to generate.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Square image resolution used for coordinate sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base RNG seed. Per-image seeds are derived from this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate coordinate files even if they already exist.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on number of images to process (debug / smoke tests).",
    )
    return parser.parse_args()


def gather_images(image_dir: Path) -> List[str]:
    images = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    return images


def per_image_seed(base_seed: int, level: int, idx: int) -> int:
    """Derive a deterministic, unique seed per (mask level, image index)."""
    return (base_seed + level * 1_000_003 + idx * 97) & 0xFFFFFFFF


def save_mask(coords: np.ndarray, meta: Dict, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, coords=coords.astype(np.uint16), meta=json.dumps(meta))


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    images = gather_images(image_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
        print(f"Limiting run to first {len(images)} images (max_images={args.max_images})")

    total_pixels = args.image_size * args.image_size
    pixel_indices = np.arange(total_pixels, dtype=np.int64)

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mask levels     : {args.mask_levels}")

    for level in args.mask_levels:
        coverage = level / 100.0
        mask_count = int(round(coverage * total_pixels))
        level_dir = output_dir / f"{level}"
        print(f"\n[{level}%] Masked pixels per image: {mask_count}")

        for idx, image_name in enumerate(images):
            seed = per_image_seed(args.seed, level, idx)
            rng = np.random.default_rng(seed)
            masked_indices = rng.choice(pixel_indices, size=mask_count, replace=False)
            y_coords = masked_indices // args.image_size
            x_coords = masked_indices % args.image_size
            coords = np.stack((y_coords, x_coords), axis=1)

            meta = {
                "image_name": image_name,
                "mask_level": level,
                "coverage": coverage,
                "image_size": args.image_size,
                "seed": int(seed),
            }

            stem = Path(image_name).stem
            out_path = level_dir / f"{stem}.npz"
            save_mask(coords, meta, out_path, args.overwrite)

            if (idx + 1) % 1000 == 0 or idx == len(images) - 1:
                print(f"  Processed {idx + 1:5d}/{len(images)} images", end="\r")

        print(f"\n[{level}%] Completed. Files in {level_dir}")

    print("\nDone! Mask coordinate archives generated for all requested levels.")


if __name__ == "__main__":
    main()

