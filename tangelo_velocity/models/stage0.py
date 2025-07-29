"""Stage 0: Foundation model (preprocessing only)."""

from typing import Dict, Any, Optional
import torch

from .base import BaseVelocityModel


class Stage0Model(BaseVelocityModel):
    """
    Stage 0 foundation model - preprocessing and basic graph construction only.
    
    This model serves as the foundation, providing data preprocessing
    and graph construction capabilities but no velocity estimation.
    It's used to validate the preprocessing pipeline.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object with Stage 0 settings.
    gene_dim : int
        Number of genes in the dataset.
    atac_dim : int, optional
        Number of ATAC features (optional for Stage 0).
    """
    
    def __init__(
        self,
        config,
        gene_dim: int,
        atac_dim: Optional[int] = None
    ):
        super().__init__(config, gene_dim, atac_dim)
        
        # Validate configuration
        if config.development_stage != 0:
            raise ValueError(f"Expected stage 0, got stage {config.development_stage}")
    
    def _initialize_components(self) -> None:
        """Initialize Stage 0 components (minimal)."""
        # Stage 0 has no trainable components - only preprocessing
        pass
    
    def forward(
        self,
        spliced: torch.Tensor,
        unspliced: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Stage 0 (identity operation).
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts.
        unspliced : torch.Tensor
            Unspliced RNA counts.
        **kwargs
            Additional arguments (ignored).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with identity transformations.
        """
        # Stage 0 performs no transformations
        outputs = {
            'pred_spliced': spliced,
            'pred_unspliced': unspliced,
            'velocity': torch.zeros_like(spliced)  # No velocity estimation
        }
        
        self._last_outputs = outputs
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Stage 0 loss (identity loss for testing).
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs.
        targets : Dict[str, torch.Tensor]
            Target values.
            
        Returns
        -------
        torch.Tensor
            Zero loss (Stage 0 has no parameters to optimize).
        """
        return torch.tensor(0.0, device=outputs['pred_spliced'].device)
    
    def _get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """Stage 0 has no ODE parameters."""
        raise NotImplementedError("ODE parameters not available in Stage 0.")
    
    def _get_interaction_network(self) -> torch.Tensor:
        """Stage 0 has no interaction network."""
        raise NotImplementedError("Interaction network not available in Stage 0.")