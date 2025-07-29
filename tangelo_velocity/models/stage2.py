"""Stage 2: Core Graph Model (MVP) implementation."""

import warnings
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("torch_geometric not available. Stage 2 functionality will be limited.")

from .base import BaseVelocityModel
from .encoders import SpatialGraphEncoder, ExpressionGraphEncoder, FusionModule
from .ode_dynamics import ODEParameterPredictor, VelocityODE
from .loss_functions import Stage2TotalLoss


class Stage2GraphModel(BaseVelocityModel):
    """
    Stage 2 graph-based velocity model using dual GraphSAGE encoders.
    
    This model implements the graph-based architecture (Route 1) using:
    - SpatialGraphEncoder for spatial neighborhood learning
    - ExpressionGraphEncoder for expression similarity learning  
    - FusionModule for latent space combination
    - Graph-aware ODE parameter prediction
    - Tangent space loss for biological plausibility
    
    The model processes spatial and expression graphs to learn cell-specific
    dynamics using graph neural networks, providing a foundation for 
    understanding cellular trajectories through graph-based representations.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object with Stage 2 settings.
    gene_dim : int
        Number of genes in the dataset.
    atac_dim : int, optional
        Number of ATAC features (not used in Stage 2).
    """
    
    def __init__(
        self,
        config,
        gene_dim: int,
        atac_dim: Optional[int] = None
    ):
        super().__init__(config, gene_dim, atac_dim)
        
        # Validate configuration
        if config.development_stage != 2:
            raise ValueError(
                f"Stage2GraphModel requires development_stage = 2, "
                f"got {config.development_stage}"
            )
        
        # Validate torch_geometric availability
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for Stage 2. "
                "Install with: pip install torch_geometric"
            )
    
    def _initialize_components(self) -> None:
        """Initialize Stage 2 specific components."""
        # Get configuration parameters
        encoder_config = getattr(self.config, 'encoder', None)
        if encoder_config is None:
            # Create default encoder config if not provided
            from types import SimpleNamespace
            encoder_config = SimpleNamespace(
                hidden_dims=(256, 128, 64),
                latent_dim=32,
                dropout=0.1,
                batch_norm=True,
                aggregator='mean',
                fusion_method='sum',
                spatial_feature_dim=2
            )
        
        latent_dim = getattr(encoder_config, 'latent_dim', 32)
        spatial_feature_dim = getattr(encoder_config, 'spatial_feature_dim', 2)
        
        # Initialize dual GraphSAGE encoders
        self.spatial_encoder = SpatialGraphEncoder(
            input_dim=self.gene_dim,  # Will add spatial coords internally
            latent_dim=latent_dim,
            config=encoder_config
        )
        
        self.expression_encoder = ExpressionGraphEncoder(
            input_dim=self.gene_dim,
            latent_dim=latent_dim,
            config=encoder_config
        )
        
        # Initialize fusion module for combining latent representations
        fusion_method = getattr(encoder_config, 'fusion_method', 'sum')
        self.fusion_module = FusionModule(
            latent_dim=latent_dim,
            fusion_method=fusion_method
        )
        
        # Enhanced ODE parameter predictor using fused latent variables
        ode_config = getattr(self.config, 'ode', None)
        self.ode_predictor = ODEParameterPredictor(
            input_dim=latent_dim,  # Uses fused latent variables
            config=ode_config or self.config
        )
        
        # ODE dynamics for velocity computation
        self.velocity_ode = VelocityODE(ode_config or self.config)
        
        # Combined loss function for training
        self.loss_function = Stage2TotalLoss(self.config)
        
        # Initialize tracking variables
        self._last_outputs = None
    
    def forward(
        self,
        spliced: torch.Tensor,
        unspliced: torch.Tensor,
        spatial_graph: Optional[Data] = None,
        expression_graph: Optional[Data] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stage 2 model.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (n_cells, n_genes).
        unspliced : torch.Tensor  
            Unspliced RNA counts of shape (n_cells, n_genes).
        spatial_graph : torch_geometric.Data, optional
            Spatial k-NN graph with edge_index and optional node features.
        expression_graph : torch_geometric.Data, optional
            Expression similarity graph with edge_index.
        spatial_coords : torch.Tensor, optional
            Spatial coordinates of shape (n_cells, 2) for x_pixel, y_pixel.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing model outputs including 'velocity'.
        """
        batch_size = spliced.shape[0]
        device = spliced.device
        
        # Validate inputs
        if spliced.shape != unspliced.shape:
            raise ValueError(
                f"Spliced and unspliced shapes must match. "
                f"Got {spliced.shape} and {unspliced.shape}"
            )
        
        # Handle missing graphs by creating dummy graphs
        if spatial_graph is None:
            warnings.warn("No spatial graph provided. Creating identity graph.")
            spatial_graph = self._create_identity_graph(batch_size, device)
        
        if expression_graph is None:
            warnings.warn("No expression graph provided. Creating identity graph.")
            expression_graph = self._create_identity_graph(batch_size, device)
        
        # Encode spatial and expression graphs
        try:
            spatial_mean, spatial_log_var = self.spatial_encoder(
                spliced, spatial_graph.edge_index, spatial_coords
            )
        except Exception as e:
            raise RuntimeError(f"Error in spatial encoder: {e}")
        
        try:
            expr_mean, expr_log_var = self.expression_encoder(
                spliced, expression_graph.edge_index
            )
        except Exception as e:
            raise RuntimeError(f"Error in expression encoder: {e}")
        
        # Fuse latent representations
        try:
            fused_mean, fused_log_var = self.fusion_module(
                spatial_mean, spatial_log_var, expr_mean, expr_log_var
            )
        except Exception as e:
            raise RuntimeError(f"Error in fusion module: {e}")
        
        # Sample from latent distribution using reparameterization trick
        fused_latent = self.fusion_module.reparameterize(fused_mean, fused_log_var)
        
        # Predict ODE parameters from fused latent variables
        try:
            ode_params = self.ode_predictor(fused_latent)
        except Exception as e:
            raise RuntimeError(f"Error in ODE parameter prediction: {e}")
        
        # Compute velocity using ODE dynamics
        try:
            velocity = self.velocity_ode(
                spliced=spliced,
                unspliced=unspliced,
                **ode_params
            )
        except Exception as e:
            raise RuntimeError(f"Error in velocity computation: {e}")
        
        # Prepare comprehensive outputs
        outputs = {
            'velocity': velocity,
            'latent_mean': fused_mean,
            'latent_log_var': fused_log_var,
            'spatial_latent': spatial_mean,
            'expression_latent': expr_mean,
            'fused_latent': fused_latent,
            'ode_parameters': ode_params,
            'spatial_mean': spatial_mean,
            'spatial_log_var': spatial_log_var,
            'expr_mean': expr_mean,
            'expr_log_var': expr_log_var
        }
        
        # Store for later access
        self._last_outputs = outputs
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Stage 2 comprehensive loss.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from forward pass.
        targets : Dict[str, torch.Tensor]
            Target values for loss computation.
            
        Returns
        -------
        torch.Tensor
            Total loss value for optimization.
        """
        total_loss, loss_components = self.loss_function(outputs, targets)
        
        # Store loss components for analysis
        self._last_loss_components = loss_components
        
        return total_loss
    
    def _get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract ODE parameters from last forward pass."""
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError(
                "Model must be run through forward pass before accessing ODE parameters."
            )
        return self._last_outputs['ode_parameters']
    
    def _get_interaction_network(self) -> torch.Tensor:
        """
        Get interaction network representation.
        
        Stage 2 doesn't use explicit interaction networks like Stage 1.
        Instead, it learns interactions through graph neural networks.
        
        Returns
        -------
        torch.Tensor
            Placeholder tensor indicating no explicit interaction network.
        """
        warnings.warn(
            "Stage 2 uses graph-based representations instead of explicit interaction networks."
        )
        # Return identity matrix as placeholder
        return torch.eye(self.gene_dim)
    
    def _get_transcription_rates(self, spliced: torch.Tensor) -> torch.Tensor:
        """
        Get transcription rates from ODE parameters.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts.
            
        Returns
        -------
        torch.Tensor
            Transcription rates derived from graph-based parameter prediction.
        """
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError(
                "Model must be run through forward pass before accessing transcription rates."
            )
        
        ode_params = self._last_outputs['ode_parameters']
        
        # Transcription rates from ODE parameters (if available)
        if 'alpha' in ode_params:
            return ode_params['alpha']
        else:
            # Compute from beta and velocity if alpha not directly available
            if 'beta' in ode_params and 'velocity' in self._last_outputs:
                # α = β * u + dS/dt where dS/dt is velocity
                beta = ode_params['beta']
                velocity = self._last_outputs['velocity']
                
                # This is a simplified computation
                # Full implementation would consider the ODE dynamics more carefully
                return beta.unsqueeze(1) * spliced + velocity
            else:
                raise RuntimeError("Cannot compute transcription rates from available parameters.")
    
    def _set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """
        Set ATAC accessibility mask.
        
        Stage 2 doesn't use explicit ATAC masking like Stage 1,
        as it learns from graph structure directly.
        
        Parameters
        ----------
        atac_mask : torch.Tensor
            ATAC accessibility mask (ignored in Stage 2).
        """
        warnings.warn(
            "ATAC masking not applicable to Stage 2 graph model. "
            "Graph structure encodes accessibility implicitly."
        )
    
    def _get_regulatory_loss(self) -> torch.Tensor:
        """
        Get regulatory loss components.
        
        Stage 2 uses graph-based regularization through the loss function
        rather than explicit regulatory network losses.
        
        Returns
        -------
        torch.Tensor
            Zero tensor as Stage 2 doesn't have explicit regulatory losses.
        """
        # Stage 2 regularization is handled through ELBO and tangent space losses
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)
    
    def _create_identity_graph(self, n_nodes: int, device: torch.device) -> Data:
        """
        Create identity graph for missing graph inputs.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes (cells).
        device : torch.device
            Device for tensor creation.
            
        Returns
        -------
        Data
            PyTorch Geometric Data object with identity edges.
        """
        # Create self-loops (identity graph)
        edge_index = torch.stack([
            torch.arange(n_nodes, device=device),
            torch.arange(n_nodes, device=device)
        ])
        
        return Data(edge_index=edge_index, num_nodes=n_nodes)
    
    def get_latent_representations(self) -> Dict[str, torch.Tensor]:
        """
        Get learned latent representations from last forward pass.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing spatial, expression, and fused latent variables.
        """
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError(
                "Model must be run through forward pass before accessing latent representations."
            )
        
        return {
            'spatial_latent': self._last_outputs['spatial_latent'],
            'expression_latent': self._last_outputs['expression_latent'],
            'fused_latent': self._last_outputs['fused_latent'],
            'spatial_mean': self._last_outputs['spatial_mean'],
            'spatial_log_var': self._last_outputs['spatial_log_var'],
            'expr_mean': self._last_outputs['expr_mean'],
            'expr_log_var': self._last_outputs['expr_log_var'],
            'latent_mean': self._last_outputs['latent_mean'],
            'latent_log_var': self._last_outputs['latent_log_var']
        }
    
    def get_loss_components(self) -> Dict[str, torch.Tensor]:
        """
        Get detailed loss components from last loss computation.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing individual loss component values.
        """
        if not hasattr(self, '_last_loss_components'):
            raise RuntimeError(
                "Model must compute loss before accessing loss components."
            )
        
        return self._last_loss_components