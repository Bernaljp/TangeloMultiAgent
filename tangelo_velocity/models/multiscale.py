"""
Multiscale Integration Module for Stage 4 Advanced Features.

This module implements hierarchical batch sampling and multiscale loss computation
to help the model learn at multiple granularities (full batch → individual cells).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import warnings
import math
from dataclasses import dataclass


class MultiscaleSampler:
    """
    Hierarchical batch sampler that creates subsamples at different scales.
    
    Creates scales: full batch → half batch → quarter batch → ... → individual cells
    """
    
    def __init__(
        self, 
        min_scale_size: int = 1,
        max_scales: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize multiscale sampler.
        
        Parameters
        ----------
        min_scale_size : int, default 1
            Minimum batch size for sampling.
        max_scales : int, optional
            Maximum number of scales to use. If None, uses all possible scales.
        seed : int, optional
            Random seed for reproducible sampling.
        """
        self.min_scale_size = min_scale_size
        self.max_scales = max_scales
        if seed is not None:
            torch.manual_seed(seed)
    
    def generate_scales(self, batch_size: int) -> List[int]:
        """
        Generate hierarchical scale sizes for a given batch size.
        
        Parameters
        ----------
        batch_size : int
            Original batch size.
            
        Returns
        -------
        List[int]
            List of scale sizes in descending order.
        """
        scales = []
        current_size = batch_size
        
        while current_size >= self.min_scale_size:
            scales.append(current_size)
            current_size = max(self.min_scale_size, current_size // 2)
            
            # Stop if we've reached max scales
            if self.max_scales is not None and len(scales) >= self.max_scales:
                break
                
            # Break if we've reached minimum size
            if current_size == self.min_scale_size and self.min_scale_size in scales:
                break
        
        return scales
    
    def sample_batch(
        self, 
        data: Dict[str, torch.Tensor], 
        scale_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a subset of the batch at given scale size.
        
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Batch data with tensors of shape (batch_size, ...).
        scale_size : int
            Target size for sampling.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Sampled batch data.
        """
        batch_size = next(iter(data.values())).shape[0]
        
        if scale_size >= batch_size:
            return data
        
        # Random sampling without replacement
        indices = torch.randperm(batch_size)[:scale_size]
        
        sampled_data = {}
        for key, tensor in data.items():
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == batch_size:
                sampled_data[key] = tensor[indices]
            else:
                # Keep non-batch tensors as is
                sampled_data[key] = tensor
        
        return sampled_data
    
    def generate_multiscale_batches(
        self, 
        data: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate all multiscale batches for the given data.
        
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Original batch data.
            
        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of sampled batches at different scales.
        """
        batch_size = next(iter(data.values())).shape[0]
        scales = self.generate_scales(batch_size)
        
        multiscale_batches = []
        for scale_size in scales:
            sampled_batch = self.sample_batch(data, scale_size)
            multiscale_batches.append(sampled_batch)
        
        return multiscale_batches


class MultiscaleLoss(nn.Module):
    """
    Multiscale loss computation with weighted averaging across scales.
    """
    
    def __init__(
        self,
        scale_weights: Optional[List[float]] = None,
        weight_decay: float = 0.8,
        normalize_weights: bool = True
    ):
        """
        Initialize multiscale loss.
        
        Parameters
        ----------
        scale_weights : List[float], optional
            Weights for different scales. If None, uses exponential decay.
        weight_decay : float, default 0.8
            Decay factor for automatic weight generation (larger scales get higher weight).
        normalize_weights : bool, default True
            Whether to normalize weights to sum to 1.
        """
        super().__init__()
        self.scale_weights = scale_weights
        self.weight_decay = weight_decay
        self.normalize_weights = normalize_weights
    
    def generate_scale_weights(self, n_scales: int) -> torch.Tensor:
        """
        Generate weights for different scales using exponential decay.
        
        Parameters
        ----------
        n_scales : int
            Number of scales.
            
        Returns
        -------
        torch.Tensor
            Scale weights.
        """
        if self.scale_weights is not None:
            weights = torch.tensor(self.scale_weights[:n_scales], dtype=torch.float32)
        else:
            # Exponential decay: larger scales get higher weights
            weights = torch.tensor([
                self.weight_decay ** i for i in range(n_scales)
            ], dtype=torch.float32)
        
        if self.normalize_weights:
            weights = weights / weights.sum()
        
        return weights
    
    def forward(
        self,
        model: nn.Module,
        multiscale_batches: List[Dict[str, torch.Tensor]],
        targets_list: List[Dict[str, torch.Tensor]],
        loss_function: callable
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute multiscale loss across all scales.
        
        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        multiscale_batches : List[Dict[str, torch.Tensor]]
            List of batches at different scales.
        targets_list : List[Dict[str, torch.Tensor]]
            List of targets corresponding to each scale.
        loss_function : callable
            Loss function that takes (outputs, targets) and returns loss.
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, Any]]
            Total weighted loss and detailed loss information.
        """
        n_scales = len(multiscale_batches)
        scale_weights = self.generate_scale_weights(n_scales)
        scale_losses = []
        scale_info = {}
        
        total_loss = 0.0
        
        for i, (batch_data, targets) in enumerate(zip(multiscale_batches, targets_list)):
            try:
                # Forward pass at this scale
                outputs = model(**batch_data)
                
                # Compute loss at this scale
                scale_loss = loss_function(outputs, targets)
                scale_losses.append(scale_loss)
                
                # Add weighted contribution to total loss
                weighted_loss = scale_weights[i] * scale_loss
                total_loss = total_loss + weighted_loss
                
                # Store scale information
                scale_info[f'scale_{i}'] = {
                    'batch_size': next(iter(batch_data.values())).shape[0],
                    'loss': scale_loss.item(),
                    'weight': scale_weights[i].item(),
                    'weighted_loss': weighted_loss.item()
                }
                
            except Exception as e:
                warnings.warn(f"Error computing loss at scale {i}: {e}")
                # Skip this scale if computation fails
                continue
        
        if len(scale_losses) == 0:
            raise RuntimeError("All multiscale loss computations failed")
        
        loss_info = {
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'n_scales': len(scale_losses),
            'scale_weights': scale_weights.tolist(),
            'scales': scale_info
        }
        
        return total_loss, loss_info


class MultiscaleTrainer:
    """
    Integrated multiscale training coordinator.
    """
    
    def __init__(
        self,
        enable_multiscale: bool = True,
        min_scale_size: int = 1,
        max_scales: Optional[int] = None,
        scale_weights: Optional[List[float]] = None,
        weight_decay: float = 0.8,
        multiscale_probability: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize multiscale trainer.
        
        Parameters
        ----------
        enable_multiscale : bool, default True
            Whether to enable multiscale training.
        min_scale_size : int, default 1
            Minimum batch size for multiscale sampling.
        max_scales : int, optional
            Maximum number of scales to use.
        scale_weights : List[float], optional
            Custom weights for different scales.
        weight_decay : float, default 0.8
            Weight decay for automatic scale weighting.
        multiscale_probability : float, default 1.0
            Probability of using multiscale training (vs. regular training).
        seed : int, optional
            Random seed for reproducible sampling.
        """
        self.enable_multiscale = enable_multiscale
        self.multiscale_probability = multiscale_probability
        
        if self.enable_multiscale:
            self.sampler = MultiscaleSampler(
                min_scale_size=min_scale_size,
                max_scales=max_scales,
                seed=seed
            )
            self.multiscale_loss = MultiscaleLoss(
                scale_weights=scale_weights,
                weight_decay=weight_decay
            )
        else:
            self.sampler = None
            self.multiscale_loss = None
    
    def should_use_multiscale(self) -> bool:
        """
        Decide whether to use multiscale training for this batch.
        
        Returns
        -------
        bool
            Whether to use multiscale training.
        """
        if not self.enable_multiscale:
            return False
        
        return torch.rand(1).item() < self.multiscale_probability
    
    def compute_loss(
        self,
        model: nn.Module,
        batch_data: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_function: callable
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss with optional multiscale training.
        
        Parameters
        ----------
        model : nn.Module
            Model to train.
        batch_data : Dict[str, torch.Tensor]
            Input batch data.
        targets : Dict[str, torch.Tensor]
            Target data.
        loss_function : callable
            Loss function.
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, Any]]
            Loss and detailed information.
        """
        if self.should_use_multiscale():
            return self._compute_multiscale_loss(model, batch_data, targets, loss_function)
        else:
            return self._compute_regular_loss(model, batch_data, targets, loss_function)
    
    def _compute_multiscale_loss(
        self,
        model: nn.Module,
        batch_data: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_function: callable
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute multiscale loss."""
        # Generate multiscale batches
        multiscale_batches = self.sampler.generate_multiscale_batches(batch_data)
        
        # Generate corresponding targets
        targets_list = []
        for batch in multiscale_batches:
            # Sample targets to match batch size
            batch_size = next(iter(batch.values())).shape[0]
            original_batch_size = next(iter(targets.values())).shape[0]
            
            if batch_size >= original_batch_size:
                targets_list.append(targets)
            else:
                # Sample targets to match the sampled batch
                sampled_targets = self.sampler.sample_batch(targets, batch_size)
                targets_list.append(sampled_targets)
        
        # Compute multiscale loss
        total_loss, loss_info = self.multiscale_loss(
            model, multiscale_batches, targets_list, loss_function
        )
        
        loss_info['multiscale_used'] = True
        return total_loss, loss_info
    
    def _compute_regular_loss(
        self,
        model: nn.Module,
        batch_data: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_function: callable
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute regular single-scale loss."""
        outputs = model(**batch_data)
        loss = loss_function(outputs, targets)
        
        loss_info = {
            'total_loss': loss.item(),
            'multiscale_used': False,
            'batch_size': next(iter(batch_data.values())).shape[0]
        }
        
        return loss, loss_info
    
    def get_multiscale_info(self, batch_size: int) -> Dict[str, Any]:
        """
        Get information about multiscale scales for a given batch size.
        
        Parameters
        ----------
        batch_size : int
            Batch size to analyze.
            
        Returns
        -------
        Dict[str, Any]
            Information about multiscale scales.
        """
        if not self.enable_multiscale:
            return {'enabled': False}
        
        scales = self.sampler.generate_scales(batch_size)
        scale_weights = self.multiscale_loss.generate_scale_weights(len(scales))
        
        return {
            'enabled': True,
            'scales': scales,
            'scale_weights': scale_weights.tolist(),
            'n_scales': len(scales),
            'min_scale_size': self.sampler.min_scale_size,
            'max_scales': self.sampler.max_scales
        }


def test_multiscale_module():
    """Test the multiscale module implementation."""
    print("Testing Multiscale Integration Module...")
    
    # Test data
    batch_size = 16
    n_genes = 100
    
    batch_data = {
        'spliced': torch.randn(batch_size, n_genes),
        'unspliced': torch.randn(batch_size, n_genes)
    }
    
    targets = {
        'spliced': torch.randn(batch_size, n_genes),
        'unspliced': torch.randn(batch_size, n_genes)
    }
    
    # Test sampler
    print("\n1. Testing MultiscaleSampler...")
    sampler = MultiscaleSampler(min_scale_size=1, max_scales=5)
    
    scales = sampler.generate_scales(batch_size)
    print(f"Generated scales: {scales}")
    
    multiscale_batches = sampler.generate_multiscale_batches(batch_data)
    print(f"Number of multiscale batches: {len(multiscale_batches)}")
    
    for i, batch in enumerate(multiscale_batches):
        batch_size_i = next(iter(batch.values())).shape[0]
        print(f"  Scale {i}: batch size = {batch_size_i}")
    
    # Test trainer
    print("\n2. Testing MultiscaleTrainer...")
    trainer = MultiscaleTrainer(
        enable_multiscale=True,
        min_scale_size=2,
        max_scales=4,
        multiscale_probability=1.0
    )
    
    # Mock model and loss function
    class MockModel(nn.Module):
        def forward(self, **kwargs):
            return kwargs  # Return inputs as outputs
    
    def mock_loss_function(outputs, targets):
        return torch.tensor(1.0)  # Return constant loss
    
    model = MockModel()
    
    # Test multiscale loss computation
    loss, loss_info = trainer.compute_loss(model, batch_data, targets, mock_loss_function)
    
    print(f"Multiscale loss: {loss}")
    print(f"Loss info: {loss_info}")
    
    # Test multiscale info
    info = trainer.get_multiscale_info(batch_size)
    print(f"\nMultiscale info: {info}")
    
    print("\n✓ Multiscale module test completed successfully!")


@dataclass
class MultiscaleConfig:
    """Configuration for multiscale training."""
    enable_multiscale: bool = True
    min_scale_size: int = 1
    max_scales: Optional[int] = None
    scale_weights: Optional[List[float]] = None
    weight_decay: float = 0.8
    multiscale_probability: float = 1.0
    normalize_weights: bool = True
    seed: Optional[int] = None


def create_multiscale_config(
    enable_multiscale: bool = True,
    min_scale_size: int = 1,
    max_scales: Optional[int] = None,
    scale_weights: Optional[List[float]] = None,
    weight_decay: float = 0.8,
    multiscale_probability: float = 1.0,
    normalize_weights: bool = True,
    seed: Optional[int] = None
) -> MultiscaleConfig:
    """
    Create a multiscale configuration.
    
    Parameters
    ----------
    enable_multiscale : bool, default True
        Whether to enable multiscale training.
    min_scale_size : int, default 1
        Minimum batch size for multiscale sampling.
    max_scales : int, optional
        Maximum number of scales to use.
    scale_weights : List[float], optional
        Custom weights for different scales.
    weight_decay : float, default 0.8
        Weight decay for automatic scale weighting.
    multiscale_probability : float, default 1.0
        Probability of using multiscale training.
    normalize_weights : bool, default True
        Whether to normalize weights to sum to 1.
    seed : int, optional
        Random seed for reproducible sampling.
        
    Returns
    -------
    MultiscaleConfig
        Configuration object for multiscale training.
    """
    return MultiscaleConfig(
        enable_multiscale=enable_multiscale,
        min_scale_size=min_scale_size,
        max_scales=max_scales,
        scale_weights=scale_weights,
        weight_decay=weight_decay,
        multiscale_probability=multiscale_probability,
        normalize_weights=normalize_weights,
        seed=seed
    )


if __name__ == "__main__":
    test_multiscale_module()