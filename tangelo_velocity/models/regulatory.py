"""Regulatory network components for Stage 1 models."""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseEncoder, MLP, safe_log, initialize_weights, setup_gradient_masking


class SigmoidFeatureModule(nn.Module):
    """
    Learnable sigmoid feature transformation for RNA expression data.
    
    Implements a double sigmoid function that can be pre-trained on data CDF
    to provide smooth, monotonic feature transformations suitable for 
    regulatory network modeling.
    
    The transformation is: y = 0.5 * (sigmoid(a1*x + b1) + sigmoid(a2*x + b2))
    
    Parameters
    ----------
    n_genes : int
        Number of genes (features).
    n_components : int, default 10
        Number of sigmoid components per gene.
    init_a : float, default 1.0
        Initial value for slope parameters.
    init_b : float, default 0.0
        Initial value for bias parameters.
    """
    
    def __init__(
        self,
        n_genes: int,
        n_components: int = 10,
        init_a: float = 1.0,
        init_b: float = 0.0
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_components = n_components
        
        # Learnable parameters for each component and gene
        # Shape: (n_genes, n_components)
        self.slopes = nn.Parameter(
            torch.full((n_genes, n_components), init_a) + 
            0.1 * torch.randn(n_genes, n_components)
        )
        self.biases = nn.Parameter(
            torch.full((n_genes, n_components), init_b) +
            0.1 * torch.randn(n_genes, n_components)
        )
        
        # Mixing weights for components
        self.weights = nn.Parameter(
            torch.ones(n_genes, n_components) / n_components
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sigmoid feature transformation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_genes).
            
        Returns
        -------
        torch.Tensor
            Transformed features of shape (batch_size, n_genes).
        """
        # x: (batch_size, n_genes)
        # slopes, biases: (n_genes, n_components)
        
        # Expand x for broadcasting: (batch_size, n_genes, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute sigmoid components: (batch_size, n_genes, n_components)
        sigmoid_components = torch.sigmoid(
            self.slopes * x_expanded + self.biases
        )
        
        # Apply mixing weights and sum over components
        weights_normalized = F.softmax(self.weights, dim=-1)
        features = torch.sum(
            weights_normalized * sigmoid_components, 
            dim=-1
        )
        
        return features
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current parameter values."""
        return {
            'slopes': self.slopes.data.clone(),
            'biases': self.biases.data.clone(),
            'weights': self.weights.data.clone()
        }
    
    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """Set parameters from dictionary."""
        with torch.no_grad():
            if 'slopes' in params:
                self.slopes.data.copy_(params['slopes'])
            if 'biases' in params:
                self.biases.data.copy_(params['biases'])
            if 'weights' in params:
                self.weights.data.copy_(params['weights'])
    
    def pretrain_on_cdf(
        self,
        data: torch.Tensor,
        n_epochs: int = 100,
        learning_rate: float = 0.01
    ) -> None:
        """
        Pre-train sigmoid features to match data CDF.
        
        Parameters
        ----------
        data : torch.Tensor
            Gene expression data of shape (n_cells, n_genes).
        n_epochs : int, default 100
            Number of training epochs.
        learning_rate : float, default 0.01
            Learning rate for optimization.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Compute empirical CDF for each gene
        cdf_x, cdf_y = self._compute_cdf(data)
        
        self.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Predict sigmoid features
            pred_y = self.forward(cdf_x)
            
            # Compute loss
            loss = criterion(pred_y, cdf_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Sigmoid pretraining epoch {epoch}, loss: {loss.item():.6f}")
    
    def _compute_cdf(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute empirical CDF for each gene.
        
        Parameters
        ----------
        data : torch.Tensor
            Expression data of shape (n_cells, n_genes).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            x-values and y-values for CDF.
        """
        n_cells, n_genes = data.shape
        
        # For each gene, compute sorted values and CDF
        cdf_x_list = []
        cdf_y_list = []
        
        for gene_idx in range(n_genes):
            gene_data = data[:, gene_idx]
            sorted_data = torch.sort(gene_data)[0]
            
            # CDF y-values (uniform spacing)
            cdf_y = torch.linspace(0, 1, n_cells, device=data.device)
            
            cdf_x_list.append(sorted_data)
            cdf_y_list.append(cdf_y)
        
        # Stack into tensors
        cdf_x = torch.stack(cdf_x_list, dim=1)  # (n_cells, n_genes)
        cdf_y = torch.stack(cdf_y_list, dim=1)  # (n_cells, n_genes)
        
        return cdf_x, cdf_y


class LinearInteractionNetwork(nn.Module):
    """
    Linear gene-gene interaction network with ATAC masking and RegVelo-style constraints.
    
    Implements a linear transformation W @ features where W is masked by chromatin 
    accessibility data and constrained using RegVelo-style gradient masking.
    
    Parameters
    ----------
    n_genes : int
        Number of genes.
    use_bias : bool, default False
        Whether to include bias terms.
    interaction_strength : float, default 1.0
        Global scaling factor for interactions.
    soft_constraint : bool, default True
        Whether to use soft constraints (L1/L2 penalties) or hard constraints (gradient masking).
    """
    
    def __init__(
        self,
        n_genes: int,
        use_bias: bool = False,
        interaction_strength: float = 1.0,
        soft_constraint: bool = True
    ):
        super().__init__()
        self.n_genes = n_genes
        self.use_bias = use_bias
        self.interaction_strength = interaction_strength
        self.soft_constraint = soft_constraint
        
        # Interaction matrix (will be masked by ATAC data)
        self.interaction_matrix = nn.Parameter(
            torch.randn(n_genes, n_genes) * 0.1
        )
        
        # Bias terms (optional)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_genes))
        else:
            self.register_parameter('bias', None)
        
        # Buffer for ATAC mask (set during training)
        self.register_buffer('atac_mask', torch.ones(n_genes, n_genes))
        
        # RegVelo-style gradient masking hooks
        self.gradient_hooks = []
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize interaction matrix weights."""
        # Xavier initialization scaled by interaction strength
        nn.init.xavier_uniform_(self.interaction_matrix)
        self.interaction_matrix.data *= self.interaction_strength
    
    def set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """
        Set the ATAC-derived regulatory mask and update gradient masking.
        
        Parameters
        ----------
        atac_mask : torch.Tensor
            Binary mask of shape (n_genes, n_genes) where 1 indicates
            accessible chromatin and 0 indicates closed chromatin.
        """
        if atac_mask.shape != (self.n_genes, self.n_genes):
            raise ValueError(
                f"ATAC mask shape {atac_mask.shape} does not match "
                f"expected shape ({self.n_genes}, {self.n_genes})"
            )
        
        self.atac_mask.copy_(atac_mask)
        
        # Update gradient masking if using hard constraints
        if not self.soft_constraint:
            self._setup_gradient_masking()
    
    def _setup_gradient_masking(self) -> None:
        """
        Set up RegVelo-style gradient masking hooks.
        
        This method clears existing hooks and sets up new ones based on
        the current ATAC mask, following the RegVelo approach.
        """
        # Clear existing hooks
        self._clear_gradient_hooks()
        
        # Set up new gradient masking
        setup_gradient_masking(
            self.interaction_matrix, 
            self.atac_mask, 
            self.gradient_hooks
        )
    
    def _clear_gradient_hooks(self) -> None:
        """Clear all registered gradient hooks."""
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks.clear()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply linear interactions with ATAC masking.
        
        Parameters
        ----------
        features : torch.Tensor
            Input features of shape (batch_size, n_genes).
            
        Returns
        -------
        torch.Tensor
            Interaction outputs of shape (batch_size, n_genes).
        """
        # Apply ATAC mask to interaction matrix
        masked_matrix = self.interaction_matrix * self.atac_mask
        
        # Linear transformation
        output = torch.matmul(features, masked_matrix.T)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def get_interaction_matrix(self) -> torch.Tensor:
        """
        Get the current interaction matrix (with ATAC masking applied).
        
        Returns
        -------
        torch.Tensor
            Masked interaction matrix of shape (n_genes, n_genes).
        """
        return self.interaction_matrix * self.atac_mask
    
    def get_sparsity_loss(self, lambda_l1: float = 1.0, lambda_l2: float = 0.0) -> torch.Tensor:
        """
        Compute regularization loss on interaction matrix.
        
        For soft constraints, returns L1/L2 penalty on non-masked interactions
        (RegVelo approach). For hard constraints, returns L1/L2 penalty on 
        all interactions since gradient masking handles structure.
        
        Parameters
        ----------
        lambda_l1 : float, default 1.0
            Weight for L1 regularization.
        lambda_l2 : float, default 0.0
            Weight for L2 regularization.
            
        Returns
        -------
        torch.Tensor
            Combined regularization loss.
        """
        if self.soft_constraint:
            # RegVelo approach: penalize interactions violating ATAC mask
            non_mask = 1.0 - self.atac_mask
            masked_violations = self.interaction_matrix * non_mask
            
            l1_loss = lambda_l1 * torch.sum(torch.abs(masked_violations))
            l2_loss = lambda_l2 * torch.sum(masked_violations ** 2)
        else:
            # Hard constraints: regular sparsity penalty (mask enforced by gradients)
            l1_loss = lambda_l1 * torch.sum(torch.abs(self.interaction_matrix))
            l2_loss = lambda_l2 * torch.sum(self.interaction_matrix ** 2)
        
        return l1_loss + l2_loss


class RegulatoryNetwork(nn.Module):
    """
    Combined regulatory network integrating sigmoid features and linear interactions.
    
    This module combines the SigmoidFeatureModule and LinearInteractionNetwork
    to produce transcription rates for the ODE system, with RegVelo-style
    constraint handling.
    
    Parameters
    ----------
    n_genes : int
        Number of genes.
    config : TangeloConfig
        Configuration object containing regulatory network parameters.
    """
    
    def __init__(self, n_genes: int, config):
        super().__init__()
        self.n_genes = n_genes
        self.config = config
        
        # Store constraint parameters
        self.soft_constraint = getattr(config.regulatory, 'soft_constraint', True)
        self.lambda_l1 = getattr(config.regulatory, 'lambda_l1', 1.0)
        self.lambda_l2 = getattr(config.regulatory, 'lambda_l2', 0.0)
        
        # Sigmoid feature transformation
        self.sigmoid_features = SigmoidFeatureModule(
            n_genes=n_genes,
            n_components=config.regulatory.n_sigmoid_components
        )
        
        # Linear interaction network with constraint mode
        self.interaction_network = LinearInteractionNetwork(
            n_genes=n_genes,
            use_bias=config.regulatory.use_bias,
            interaction_strength=config.regulatory.interaction_strength,
            soft_constraint=self.soft_constraint
        )
        
        # Base transcription rate (learnable parameters)
        self.base_transcription = nn.Parameter(
            torch.ones(n_genes) * 0.1
        )
    
    def forward(self, spliced: torch.Tensor) -> torch.Tensor:
        """
        Compute transcription rates from spliced RNA.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
            
        Returns
        -------
        torch.Tensor
            Transcription rates of shape (batch_size, n_genes).
        """
        # Apply sigmoid feature transformation
        sigmoid_features = self.sigmoid_features(spliced)
        
        # Apply linear interactions
        interaction_output = self.interaction_network(sigmoid_features)
        
        # Combine with base transcription rate
        transcription_rates = (
            F.softplus(interaction_output + self.base_transcription)
        )
        
        return transcription_rates
    
    def set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """Set ATAC mask for regulatory constraints."""
        self.interaction_network.set_atac_mask(atac_mask)
    
    def pretrain_sigmoid(
        self,
        spliced_data: torch.Tensor,
        **kwargs
    ) -> None:
        """Pre-train sigmoid features on spliced RNA data."""
        self.sigmoid_features.pretrain_on_cdf(spliced_data, **kwargs)
    
    def get_interaction_matrix(self) -> torch.Tensor:
        """Get the learned interaction matrix."""
        return self.interaction_network.get_interaction_matrix()
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get regularization loss combining sparsity penalties.
        
        Returns
        -------
        torch.Tensor
            Combined regularization loss from interaction network.
        """
        return self.interaction_network.get_sparsity_loss(
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2
        )
    
    def compute_jacobian(self, spliced: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix of transcription rates w.r.t. spliced RNA.
        
        This follows the RegVelo approach for computing sensitivity of
        transcription rates to changes in gene expression.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
            
        Returns
        -------
        torch.Tensor
            Jacobian matrix of shape (batch_size, n_genes, n_genes).
        """
        batch_size, n_genes = spliced.shape
        
        # Compute sigmoid features and their derivatives
        sigmoid_features = self.sigmoid_features(spliced)
        
        # For numerical stability, use finite differences approximation
        eps = 1e-6
        jacobian = torch.zeros(batch_size, n_genes, n_genes, device=spliced.device)
        
        for i in range(n_genes):
            # Perturb the i-th gene
            spliced_perturbed = spliced.clone()
            spliced_perturbed[:, i] += eps
            
            # Compute transcription rates
            rates_original = self.forward(spliced)
            rates_perturbed = self.forward(spliced_perturbed)
            
            # Finite difference approximation
            jacobian[:, :, i] = (rates_perturbed - rates_original) / eps
        
        return jacobian
    
    def get_jacobian_regularization_loss(
        self, 
        spliced: torch.Tensor,
        lambda_jacobian: float = 0.1
    ) -> torch.Tensor:
        """
        Compute RegVelo-style Jacobian regularization loss.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
        lambda_jacobian : float, default 0.1
            Regularization weight for Jacobian penalty.
            
        Returns
        -------
        torch.Tensor
            L1 norm of the Jacobian matrix.
        """
        jacobian = self.compute_jacobian(spliced)
        return lambda_jacobian * torch.sum(torch.abs(jacobian))