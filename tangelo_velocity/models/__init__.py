"""Neural network models for Tangelo Velocity."""

# Base classes - implemented
from .base import BaseVelocityModel, BaseEncoder, MLP

# Stage models - implemented stages only
from .stage0 import Stage0Model
from .stage1 import Stage1RegulatoryModel
from .stage2 import Stage2GraphModel
from .stage3 import Stage3IntegratedModel
from .stage4 import Stage4AdvancedModel

# Component modules - implemented
from .regulatory import SigmoidFeatureModule, LinearInteractionNetwork
from .encoders import SpatialGraphEncoder, ExpressionGraphEncoder, FusionModule
from .ode_dynamics import VelocityODE, ODEParameterPredictor

# Loss functions - implemented
from .loss_functions import (
    ReconstructionLoss,
    VelocityConsistencyLoss,
    RegularizationLoss,
    RegulatoryNetworkLoss,
    ELBOLoss,
    TangentSpaceLoss,
    Stage1TotalLoss,
    Stage2TotalLoss
)

# Multiscale training - Stage 4 advanced feature
from .multiscale import (
    MultiscaleConfig,
    MultiscaleSampler,
    MultiscaleLoss,
    MultiscaleTrainer,
    create_multiscale_config
)

# Factory function
def get_velocity_model(config, gene_dim: int, atac_dim: int = None):
    """Factory function to get the appropriate velocity model for the given stage."""
    stage = config.development_stage
    
    if stage == 0:
        return Stage0Model(config, gene_dim, atac_dim)
    elif stage == 1:
        return Stage1RegulatoryModel(config, gene_dim, atac_dim)
    elif stage == 2:
        return Stage2GraphModel(config, gene_dim, atac_dim)
    elif stage == 3:
        return Stage3IntegratedModel(config, gene_dim, atac_dim)
    elif stage == 4:
        return Stage4AdvancedModel(config, gene_dim, atac_dim)
    else:
        raise ValueError(f"Invalid development stage: {stage}. Must be 0, 1, 2, 3, or 4.")


__all__ = [
    # Base classes
    "BaseVelocityModel",
    "BaseEncoder", 
    "MLP",
    
    # Stage models
    "Stage0Model",
    "Stage1RegulatoryModel",
    "Stage2GraphModel",
    "Stage3IntegratedModel",
    "Stage4AdvancedModel",
    
    # Regulatory components
    "SigmoidFeatureModule",
    "LinearInteractionNetwork",
    
    # Graph encoders
    "SpatialGraphEncoder",
    "ExpressionGraphEncoder",
    "FusionModule",
    
    # ODE dynamics
    "VelocityODE",
    "ODEParameterPredictor",
    
    # Loss functions
    "ReconstructionLoss",
    "VelocityConsistencyLoss", 
    "RegularizationLoss",
    "RegulatoryNetworkLoss",
    "ELBOLoss",
    "TangentSpaceLoss",
    "Stage1TotalLoss",
    "Stage2TotalLoss",
    
    # Multiscale training
    "MultiscaleConfig",
    "MultiscaleSampler",
    "MultiscaleLoss", 
    "MultiscaleTrainer",
    "create_multiscale_config",
    
    # Factory
    "get_velocity_model",
]