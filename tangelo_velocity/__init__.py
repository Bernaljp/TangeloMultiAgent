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
    
    # Import available components dynamically 
    _model_components = {}
    if hasattr(models, 'list_available_components'):
        available = models.list_available_components()
        for component_name in available:
            if hasattr(models, component_name):
                _model_components[component_name] = getattr(models, component_name)
                globals()[component_name] = _model_components[component_name]
    
    # Always try to import the factory function
    if hasattr(models, 'get_velocity_model'):
        get_velocity_model = models.get_velocity_model
        _model_components['get_velocity_model'] = get_velocity_model
        
except ImportError:
    models = None
    _has_models = False
    _model_components = {}

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
    __all__.extend(_model_components.keys())
if _has_api:
    __all__.append("TangeloVelocity")