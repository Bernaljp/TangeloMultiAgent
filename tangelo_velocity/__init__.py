"""
Tangelo Velocity: Multi-modal Single-Cell Velocity Estimation

A novel computational method for RNA velocity estimation that integrates spatial 
transcriptomics, RNA velocity, and ATAC-seq data using graph neural networks 
and ordinary differential equation modeling.
"""

from .api import TangeloVelocity
from .config import TangeloConfig
from . import preprocessing
from . import models
from . import analysis
from . import plotting

__version__ = "0.1.0"
__author__ = "Tangelo Velocity Team"

__all__ = [
    "TangeloVelocity",
    "TangeloConfig", 
    "preprocessing",
    "models",
    "analysis",
    "plotting",
]