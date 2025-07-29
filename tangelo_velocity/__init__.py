"""
Tangelo Velocity: Multi-modal Single-Cell Velocity Estimation

A novel computational method for RNA velocity estimation that integrates spatial 
transcriptomics, RNA velocity, and ATAC-seq data using graph neural networks 
and ordinary differential equation modeling.
"""

# Core configuration always available
from .config import TangeloConfig

# Conditional imports for modules with dependencies
try:
    from . import preprocessing
    _has_preprocessing = True
except ImportError:
    preprocessing = None
    _has_preprocessing = False

try:
    from . import models  
    _has_models = True
except ImportError:
    models = None
    _has_models = False

# Analysis and plotting (currently empty, always available)
from . import analysis
from . import plotting

# Conditional API import (requires muon dependency)
try:
    from .api import TangeloVelocity
    _has_api = True
except ImportError:
    TangeloVelocity = None
    _has_api = False

__version__ = "0.1.0"
__author__ = "Tangelo Velocity Team"

# Build __all__ dynamically based on available imports
__all__ = ["TangeloConfig", "analysis", "plotting"]

if _has_preprocessing:
    __all__.append("preprocessing")
if _has_models:
    __all__.append("models") 
if _has_api:
    __all__.append("TangeloVelocity")