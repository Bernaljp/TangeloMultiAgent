"""
Tangelo Velocity: Multi-modal Single-Cell Velocity Estimation

A modular package for RNA velocity estimation that integrates spatial 
transcriptomics, RNA velocity, and ATAC-seq data using graph neural networks 
and ordinary differential equation modeling.
"""

# Core modules
from . import preprocessing
from . import models  

# Tools and utilities
from . import tools

# Configuration
from .config import TangeloConfig

__version__ = "0.1.0"
__author__ = "Tangelo Velocity Team"

__all__ = [
    "preprocessing",
    "models", 
    "tools",
    "TangeloConfig",
]