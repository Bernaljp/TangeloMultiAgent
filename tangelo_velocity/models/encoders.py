"""Graph encoder implementations for Stage 2 graph-based velocity modeling."""

import warnings
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("torch_geometric not available. GraphSAGE functionality will be limited.")

from .base import BaseEncoder, MLP, initialize_weights


class GraphSAGEEncoder(BaseEncoder):
    """
    Base GraphSAGE encoder for graph-based feature learning.
    
    Implements GraphSAGE message passing with configurable aggregation
    and multiple layers for hierarchical representation learning.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimensionality.
    latent_dim : int
        Latent representation dimensionality.
    config : object
        Configuration object containing encoder parameters.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        config
    ):
        super().__init__(input_dim, latent_dim * 2, config)  # *2 for mean/logvar
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for GraphSAGE encoders. "
                "Install with: pip install torch_geometric"
            )
        
        self.latent_dim = latent_dim
        self.aggregator = getattr(config, 'aggregator', 'mean')
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        self.use_batch_norm = getattr(config, 'batch_norm', True)
        
        # Build GraphSAGE layers
        hidden_dims = getattr(config, 'hidden_dims', (256, 128, 64))
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [latent_dim * 2]  # *2 for mean/logvar
        
        for i in range(len(dims) - 1):
            self.layers.append(SAGEConv(dims[i], dims[i+1], aggr=self.aggregator))
        
        # Batch normalization layers (exclude final layer)
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(dims[i+1]) for i in range(len(dims) - 2)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Initialize weights
        self.apply(lambda m: initialize_weights(m, "xavier_uniform"))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GraphSAGE encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (n_nodes, input_dim).
        edge_index : torch.Tensor
            Graph edge indices of shape (2, n_edges).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance of latent representation, each of shape (n_nodes, latent_dim).
        """
        h = x
        
        # Forward through GraphSAGE layers (exclude final layer)
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            
            if self.use_batch_norm:
                h = self.batch_norms[i](h)
            
            h = F.relu(h)
            h = self.dropout(h)
        
        # Final layer for latent representation
        h = self.layers[-1](h, edge_index)
        
        # Split into mean and log variance
        mean, log_var = h.chunk(2, dim=1)
        
        # Ensure log_var is not too negative for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        return mean, log_var


class SpatialGraphEncoder(GraphSAGEEncoder):
    """
    GraphSAGE encoder for spatial graphs using coordinate information.
    
    Processes k-NN graphs built from spatial coordinates (x_pixel, y_pixel)
    to learn spatial neighborhood representations for cells.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimensionality (gene expression features).
    latent_dim : int
        Latent representation dimensionality.
    config : object
        Configuration object containing encoder parameters.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        config
    ):
        # Account for optional spatial coordinates (+2 for x_pixel, y_pixel)
        spatial_feature_dim = getattr(config, 'spatial_feature_dim', 2)
        effective_input_dim = input_dim + spatial_feature_dim
        
        super().__init__(effective_input_dim, latent_dim, config)
        self.spatial_feature_dim = spatial_feature_dim
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        spatial_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through spatial graph encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (n_nodes, input_dim).
        edge_index : torch.Tensor
            Graph edge indices of shape (2, n_edges).
        spatial_coords : torch.Tensor, optional
            Spatial coordinates of shape (n_nodes, spatial_feature_dim).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance of spatial latent representation.
        """
        # Incorporate spatial coordinates if provided
        if spatial_coords is not None:
            if spatial_coords.shape[1] != self.spatial_feature_dim:
                raise ValueError(
                    f"Expected spatial_coords with {self.spatial_feature_dim} features, "
                    f"got {spatial_coords.shape[1]}"
                )
            x = torch.cat([x, spatial_coords], dim=1)
        else:
            # Pad with zeros if no spatial coordinates provided
            batch_size = x.shape[0]
            device = x.device
            spatial_padding = torch.zeros(batch_size, self.spatial_feature_dim, device=device)
            x = torch.cat([x, spatial_padding], dim=1)
        
        return super().forward(x, edge_index)


class ExpressionGraphEncoder(GraphSAGEEncoder):
    """
    GraphSAGE encoder for expression similarity graphs.
    
    Processes k-NN graphs built from gene expression similarity
    to learn expression-based cellular representations.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimensionality (gene expression features).
    latent_dim : int
        Latent representation dimensionality.
    config : object
        Configuration object containing encoder parameters.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        config
    ):
        super().__init__(input_dim, latent_dim, config)
        
        # Expression-specific optimizations could be added here
        # For now, use the base GraphSAGE implementation
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through expression graph encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features (gene expression) of shape (n_nodes, input_dim).
        edge_index : torch.Tensor
            Graph edge indices of shape (2, n_edges).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance of expression latent representation.
        """
        return super().forward(x, edge_index)


class FusionModule(nn.Module):
    """
    Flexible fusion module for combining spatial and expression latent representations.
    
    Supports multiple fusion strategies: sum, concatenation, and attention-based
    fusion for variational latent variables.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimensionality (same for both spatial and expression).
    fusion_method : str, default "sum"
        Fusion method: "sum", "concat", or "attention".
    """
    
    def __init__(
        self,
        latent_dim: int,
        fusion_method: str = "sum"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_method = fusion_method
        
        if fusion_method not in ["sum", "concat", "attention"]:
            raise ValueError(
                f"Unknown fusion method: {fusion_method}. "
                "Supported methods: 'sum', 'concat', 'attention'"
            )
        
        # Initialize fusion-specific layers
        if fusion_method == "concat":
            self.projection = nn.Linear(2 * latent_dim, latent_dim)
        elif fusion_method == "attention":
            self.attention_net = nn.Sequential(
                nn.Linear(2 * latent_dim, latent_dim),
                nn.Tanh(),
                nn.Linear(latent_dim, 2),
                nn.Softmax(dim=1)
            )
        
        # Initialize weights
        self.apply(lambda m: initialize_weights(m, "xavier_uniform"))
    
    def forward(
        self,
        spatial_mean: torch.Tensor,
        spatial_log_var: torch.Tensor,
        expr_mean: torch.Tensor,
        expr_log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse spatial and expression latent representations.
        
        Parameters
        ----------
        spatial_mean : torch.Tensor
            Spatial latent means of shape (n_cells, latent_dim).
        spatial_log_var : torch.Tensor
            Spatial latent log-variances of shape (n_cells, latent_dim).
        expr_mean : torch.Tensor
            Expression latent means of shape (n_cells, latent_dim).
        expr_log_var : torch.Tensor
            Expression latent log-variances of shape (n_cells, latent_dim).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Fused latent means and log-variances, each of shape (n_cells, latent_dim).
        """
        # Validate input shapes
        if spatial_mean.shape != expr_mean.shape:
            raise ValueError(
                f"Spatial and expression means must have same shape. "
                f"Got {spatial_mean.shape} and {expr_mean.shape}"
            )
        
        if spatial_log_var.shape != expr_log_var.shape:
            raise ValueError(
                f"Spatial and expression log-variances must have same shape. "
                f"Got {spatial_log_var.shape} and {expr_log_var.shape}"
            )
        
        if self.fusion_method == "sum":
            # Simple additive fusion
            fused_mean = spatial_mean + expr_mean
            fused_log_var = spatial_log_var + expr_log_var
            
        elif self.fusion_method == "concat":
            # Concatenation with projection
            concat_mean = torch.cat([spatial_mean, expr_mean], dim=1)
            concat_log_var = torch.cat([spatial_log_var, expr_log_var], dim=1)
            
            fused_mean = self.projection(concat_mean)
            fused_log_var = self.projection(concat_log_var)
            
        elif self.fusion_method == "attention":
            # Attention-based weighted fusion
            combined_features = torch.cat([spatial_mean, expr_mean], dim=1)
            attention_weights = self.attention_net(combined_features)
            
            # Apply attention weights
            w_spatial = attention_weights[:, 0:1]  # Shape: (n_cells, 1)
            w_expr = attention_weights[:, 1:2]     # Shape: (n_cells, 1)
            
            fused_mean = w_spatial * spatial_mean + w_expr * expr_mean
            fused_log_var = w_spatial * spatial_log_var + w_expr * expr_log_var
        
        # Ensure log_var is numerically stable
        fused_log_var = torch.clamp(fused_log_var, min=-10.0, max=10.0)
        
        return fused_mean, fused_log_var
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for variational inference.
        
        Parameters
        ----------
        mean : torch.Tensor
            Latent means of shape (n_cells, latent_dim).
        log_var : torch.Tensor
            Latent log-variances of shape (n_cells, latent_dim).
            
        Returns
        -------
        torch.Tensor
            Sampled latent variables of shape (n_cells, latent_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std