"""
Dataset classes for image inpainting.
"""
from .datasets import CelebAInpaintingDataset
from .datasets_np import (
    NPImageDataset,
    collate_fn_cnp,
    collate_fn_convcnp,
)

__all__ = [
    'CelebAInpaintingDataset',
    'NPImageDataset',
    'collate_fn_cnp',
    'collate_fn_convcnp',
]

