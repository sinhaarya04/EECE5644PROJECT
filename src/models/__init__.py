"""
Neural Process models for image inpainting
"""
from .cnp import CNP, CNPWithUncertainty
from .convcnp import ConvCNP, ConvCNPWithUncertainty

__all__ = ['CNP', 'CNPWithUncertainty', 'ConvCNP', 'ConvCNPWithUncertainty']

