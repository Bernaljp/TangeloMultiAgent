"""Base classes for Tangelo Velocity models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BaseVelocityModel(nn.Module, ABC):
    """
    Abstract base class for all Tangelo Velocity models.
    
    This class defines the common interface and functionality shared across
    all development stages of the Tangelo Velocity project.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object containing model hyperparameters.
    gene_dim : int
        Number of genes in the dataset.
    atac_dim : int, optional
        Number of ATAC features. Required for stages >= 1.
    """
    
    def __init__(
        self,
        config,
        gene_dim: int,
        atac_dim: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.gene_dim = gene_dim
        self.atac_dim = atac_dim
        self.development_stage = config.development_stage
        
        # Initialize common model components
        self._initialize_components()
        
    @abstractmethod
    def _initialize_components(self) -> None:
        """Initialize stage-specific model components."""
        pass
    
    @abstractmethod
    def forward(
        self,
        spliced: torch.Tensor,
        unspliced: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the velocity model.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (n_cells, n_genes).
        unspliced : torch.Tensor
            Unspliced RNA counts of shape (n_cells, n_genes).
        **kwargs
            Additional model-specific inputs.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing model outputs including 'velocity'.
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the total model loss.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from forward pass.
        targets : Dict[str, torch.Tensor]
            Target values for loss computation.
            
        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        pass
    
    def fit(
        self,
        training_data: Dict[str, torch.Tensor],
        n_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Train the velocity model.
        
        Parameters
        ----------
        training_data : Dict[str, torch.Tensor]
            Dictionary containing training data tensors.
        n_epochs : int, optional
            Number of training epochs. Uses config default if None.
        learning_rate : float, optional
            Learning rate for optimizer. Uses config default if None.
        **kwargs
            Additional training arguments.
        """
        # Use config defaults if not specified
        n_epochs = n_epochs or self.config.training.n_epochs
        learning_rate = learning_rate or self.config.training.learning_rate
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Training loop
        self.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(**training_data)
            
            # Compute loss
            loss = self.compute_loss(outputs, training_data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Logging
            if epoch % self.config.training.log_interval == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.6f}")
    
    def predict_velocity(self) -> torch.Tensor:
        """
        Predict RNA velocities for the current model state.
        
        Returns
        -------
        torch.Tensor
            Predicted velocity matrix of shape (n_cells, n_genes).
        """
        self.eval()
        if not hasattr(self, '_last_outputs'):
            raise RuntimeError("Model must be run through forward pass before predicting velocity.")
        return self._last_outputs['velocity']
    
    def get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get ODE parameters (beta, gamma, time) for each cell.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing ODE parameters.
        """
        if self.development_stage == 0:
            raise NotImplementedError("ODE parameters not available in Stage 0.")
        return self._get_ode_parameters()
    
    @abstractmethod
    def _get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """Stage-specific implementation of ODE parameter extraction."""
        pass
    
    def get_interaction_network(self) -> torch.Tensor:
        """
        Get the learned gene-gene interaction network.
        
        Returns
        -------
        torch.Tensor
            Interaction matrix of shape (n_genes, n_genes).
        """
        if self.development_stage < 1:
            raise NotImplementedError("Interaction network not available in Stage 0.")
        return self._get_interaction_network()
    
    @abstractmethod
    def _get_interaction_network(self) -> torch.Tensor:
        """Stage-specific implementation of interaction network extraction."""
        pass
    
    def get_transcription_rates(
        self,
        spliced: torch.Tensor
    ) -> torch.Tensor:
        """
        Get transcription rates from the regulatory network.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (n_cells, n_genes).
            
        Returns
        -------
        torch.Tensor
            Transcription rates of shape (n_cells, n_genes).
        """
        if self.development_stage < 1:
            raise NotImplementedError("Transcription rates not available in Stage 0.")
        return self._get_transcription_rates(spliced)
    
    @abstractmethod
    def _get_transcription_rates(self, spliced: torch.Tensor) -> torch.Tensor:
        """Stage-specific implementation of transcription rate computation."""
        pass
    
    def set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """
        Set ATAC-seq derived regulatory mask for interaction network.
        
        Parameters
        ----------
        atac_mask : torch.Tensor
            Binary mask of shape (n_genes, n_genes) indicating accessible
            chromatin regions for gene-gene interactions.
        """
        if self.development_stage < 1:
            raise NotImplementedError("ATAC masking not available in Stage 0.")
        self._set_atac_mask(atac_mask)
    
    @abstractmethod
    def _set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """Stage-specific implementation of ATAC mask setting."""
        pass
    
    def get_regulatory_loss(self) -> torch.Tensor:
        """
        Get regularization loss from the regulatory network.
        
        Returns
        -------
        torch.Tensor
            Regularization loss value (sparsity, gradient constraints, etc.).
        """
        if self.development_stage < 1:
            raise NotImplementedError("Regulatory loss not available in Stage 0.")
        return self._get_regulatory_loss()
    
    @abstractmethod
    def _get_regulatory_loss(self) -> torch.Tensor:
        """Stage-specific implementation of regulatory loss computation."""
        pass


class BaseStage1Model(BaseVelocityModel):
    """
    Base class for Stage 1+ models with regulatory network functionality.
    
    This class provides common implementations for regulatory network operations
    that are shared across Stage 1 and higher development stages.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object containing model hyperparameters.
    gene_dim : int
        Number of genes in the dataset.
    atac_dim : int
        Number of ATAC features.
    """
    
    def __init__(
        self,
        config,
        gene_dim: int,
        atac_dim: int
    ):
        super().__init__(config, gene_dim, atac_dim)
        
        if self.development_stage < 1:
            raise ValueError("BaseStage1Model requires development_stage >= 1")
        
        # Initialize regulatory network (to be set by subclasses)
        self.regulatory_network = None
    
    def _get_transcription_rates(self, spliced: torch.Tensor) -> torch.Tensor:
        """Common implementation for transcription rate computation."""
        if self.regulatory_network is None:
            raise RuntimeError("Regulatory network not initialized. "
                             "Subclass must set self.regulatory_network in _initialize_components().")
        return self.regulatory_network(spliced)
    
    def _set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """Common implementation for ATAC mask setting."""
        if self.regulatory_network is None:
            raise RuntimeError("Regulatory network not initialized.")
        self.regulatory_network.set_atac_mask(atac_mask)
    
    def _get_regulatory_loss(self) -> torch.Tensor:
        """Common implementation for regulatory loss computation."""
        if self.regulatory_network is None:
            raise RuntimeError("Regulatory network not initialized.")
        return self.regulatory_network.get_regularization_loss()
    
    def _get_interaction_network(self) -> torch.Tensor:
        """Common implementation for interaction network extraction."""
        if self.regulatory_network is None:
            raise RuntimeError("Regulatory network not initialized.")
        return self.regulatory_network.get_interaction_matrix()


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for encoder networks.
    
    This class provides a common interface for different types of encoders
    used in the Tangelo Velocity architecture.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    output_dim : int
        Dimensionality of output features.
    config : TangeloConfig
        Configuration object.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        **kwargs
            Additional encoder-specific inputs.
            
        Returns
        -------
        torch.Tensor
            Encoded representation.
        """
        pass


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.
    
    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    hidden_dims : tuple of int
        Hidden layer dimensions.
    output_dim : int
        Output dimensionality.
    activation : str, default "relu"
        Activation function name.
    dropout : float, default 0.0
        Dropout rate.
    batch_norm : bool, default False
        Whether to use batch normalization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super().__init__()
        
        # Activation function mapping
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        
        act_fn = activation_map[activation]
        
        # Build layers
        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Skip activation and normalization for output layer
            if i < len(dims) - 2:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                
                # Activation
                layers.append(act_fn())
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.network(x)


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable logarithm.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    eps : float, default 1e-8
        Small constant to prevent log(0).
        
    Returns
    -------
    torch.Tensor
        log(max(x, eps))
    """
    return torch.log(torch.clamp(x, min=eps))


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable inverse of the softplus function.
    
    For large values, uses approximation to avoid numerical issues.
    
    Parameters
    ----------
    x : torch.Tensor
        Input values (must be positive).
        
    Returns
    -------
    torch.Tensor
        Inverse softplus applied element-wise.
    """
    return torch.where(x > 20, x, torch.log(torch.expm1(x)))


def initialize_weights(module: nn.Module, init_type: str = "xavier_uniform") -> None:
    """
    Initialize module weights using specified method.
    
    Parameters
    ----------
    module : nn.Module
        Module to initialize.
    init_type : str, default "xavier_uniform"
        Initialization method name.
    """
    if isinstance(module, nn.Linear):
        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Unsupported initialization: {init_type}")
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def setup_gradient_masking(
    parameter: nn.Parameter, 
    mask: torch.Tensor,
    store_hooks: Optional[list] = None
) -> None:
    """
    Set up RegVelo-style gradient masking for a parameter.
    
    This function registers a backward hook that masks gradients according
    to a binary mask tensor, following the RegVelo approach for enforcing
    regulatory network structure constraints.
    
    Parameters
    ----------
    parameter : nn.Parameter
        Parameter to apply gradient masking to (typically interaction matrix weights).
    mask : torch.Tensor
        Binary mask tensor of same shape as parameter. Gradients are multiplied
        element-wise by this mask during backpropagation.
    store_hooks : list, optional
        List to store the registered hook for later removal. If None, hook
        cannot be removed.
        
    Examples
    --------
    >>> interaction_matrix = nn.Parameter(torch.randn(n_genes, n_genes))
    >>> atac_mask = torch.ones(n_genes, n_genes)
    >>> atac_mask[non_accessible_pairs] = 0
    >>> hooks = []
    >>> setup_gradient_masking(interaction_matrix, atac_mask, hooks)
    """
    if parameter.shape != mask.shape:
        raise ValueError(
            f"Parameter shape {parameter.shape} does not match mask shape {mask.shape}"
        )
    
    def _gradient_mask_hook(grad: torch.Tensor) -> torch.Tensor:
        """Apply element-wise gradient masking."""
        return grad * mask
    
    # Register the hook
    hook = parameter.register_hook(_gradient_mask_hook)
    
    # Store hook for potential removal
    if store_hooks is not None:
        store_hooks.append(hook)