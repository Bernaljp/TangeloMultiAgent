"""High-level API for Tangelo Velocity."""

from typing import Optional, Union, Dict, Any
import muon as mu
import scanpy as sc
import torch

from .config import TangeloConfig, get_stage_config
from .preprocessing import MuDataProcessor
from .models import get_velocity_model


class TangeloVelocity:
    """
    High-level interface for Tangelo Velocity estimation.
    
    This class provides a simple API for multi-modal velocity estimation
    that integrates spatial transcriptomics, RNA velocity, and ATAC-seq data.
    
    Parameters
    ----------
    config : TangeloConfig, optional
        Configuration object. If None, will use stage-appropriate defaults.
    stage : int, optional
        Development stage (0-4). Only used if config is None.
    device : str, optional
        Device for computation ("auto", "cpu", "cuda").
    """
    
    def __init__(
        self, 
        config: Optional[TangeloConfig] = None,
        stage: int = 3,
        device: str = "auto"
    ):
        # Set up configuration
        if config is None:
            config = get_stage_config(stage)
        
        if device != "auto":
            config.device = device
        elif config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.processor = MuDataProcessor(config)
        self.model = None
        self.is_fitted = False
        
        # Store processed data
        self._adata = None
        self._spatial_graph = None
        self._expression_graph = None
        
    def fit(
        self,
        adata: mu.MuData,
        copy: bool = False,
        **kwargs
    ) -> Optional[mu.MuData]:
        """
        Fit the Tangelo Velocity model to multi-modal data.
        
        Parameters
        ----------
        adata : mu.MuData
            Multi-modal annotated data object with 'rna' and 'atac' modalities.
        copy : bool, default False
            If True, return a copy of the data with results added.
        **kwargs
            Additional arguments passed to the model training.
            
        Returns
        -------
        mu.MuData or None
            If copy=True, returns annotated data with velocity estimates.
            Otherwise, modifies adata in-place and returns None.
        """
        # Copy data if requested
        if copy:
            adata = adata.copy()
            
        # Validate input data
        self._validate_input(adata)
        
        # Process data and build graphs
        print("Processing multi-modal data...")
        processed_data = self.processor.process_mudata(adata)
        
        # Store processed components
        self._adata = adata
        self._spatial_graph = processed_data["spatial_graph"]
        self._expression_graph = processed_data["expression_graph"]
        
        # Initialize model
        print(f"Initializing Stage {self.config.development_stage} model...")
        self.model = get_velocity_model(
            config=self.config,
            gene_dim=adata['rna'].n_vars,
            atac_dim=adata['atac'].n_vars if 'atac' in adata.mod else None,
        ).to(self.device)
        
        # Train model
        print("Training model...")
        self._train_model(processed_data, **kwargs)
        
        # Generate predictions
        print("Computing velocity estimates...")
        self._predict_velocity(adata)
        
        self.is_fitted = True
        
        if copy:
            return adata
            
    def predict(
        self,
        adata: Optional[mu.MuData] = None,
        copy: bool = False
    ) -> Optional[mu.MuData]:
        """
        Predict velocities for new data or re-predict for fitted data.
        
        Parameters
        ----------
        adata : mu.MuData, optional
            Multi-modal data. If None, uses fitted data.
        copy : bool, default False
            If True, return a copy with predictions.
            
        Returns
        -------
        mu.MuData or None
            Data with velocity predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
            
        if adata is None:
            adata = self._adata
            
        if copy:
            adata = adata.copy()
            
        self._predict_velocity(adata)
        
        if copy:
            return adata
    
    def compute_velocity_graph(
        self,
        adata: Optional[mu.MuData] = None,
        n_neighbors: int = 30,
        **kwargs
    ) -> None:
        """
        Compute velocity graph for downstream analysis.
        
        Parameters
        ----------
        adata : mu.MuData, optional
            Data with velocity estimates. If None, uses fitted data.
        n_neighbors : int, default 30
            Number of neighbors for velocity graph.
        **kwargs
            Additional arguments for graph construction.
        """
        if adata is None:
            adata = self._adata
            
        # Use scanpy's velocity graph computation
        sc.tl.velocity_graph(
            adata['rna'], 
            n_neighbors=n_neighbors,
            **kwargs
        )
    
    def compute_velocity_embedding(
        self,
        adata: Optional[mu.MuData] = None,
        basis: str = "umap",
        **kwargs
    ) -> None:
        """
        Compute velocity embedding for visualization.
        
        Parameters
        ----------
        adata : mu.MuData, optional
            Data with velocity graph. If None, uses fitted data.
        basis : str, default "umap"
            Embedding basis to use.
        **kwargs
            Additional arguments for embedding computation.
        """
        if adata is None:
            adata = self._adata
            
        # Compute velocity embedding
        sc.tl.velocity_embedding(
            adata['rna'],
            basis=basis,
            **kwargs
        )
    
    def get_latent_representations(self) -> Dict[str, torch.Tensor]:
        """
        Get learned latent representations.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing latent representations:
            - "spatial_latent": Spatial graph encoder output
            - "expression_latent": Expression graph encoder output  
            - "combined_latent": Combined latent representation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get latent representations.")
            
        return self.model.get_latent_representations()
    
    def get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get predicted ODE parameters for each cell.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing cell-specific parameters:
            - "beta": Splicing rates
            - "gamma": Degradation rates
            - "time": Cell times
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get ODE parameters.")
            
        return self.model.get_ode_parameters()
    
    def get_interaction_network(self) -> torch.Tensor:
        """
        Get learned gene-gene interaction network.
        
        Returns
        -------
        torch.Tensor
            Gene interaction matrix of shape (n_genes, n_genes).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get interaction network.")
            
        return self.model.get_interaction_network()
    
    def _validate_input(self, adata: mu.MuData) -> None:
        """Validate input MuData object."""
        # Check required modalities
        if 'rna' not in adata.mod:
            raise ValueError("MuData object must contain 'rna' modality.")
            
        # Check required RNA layers
        rna_data = adata['rna']
        required_layers = ['spliced', 'unspliced']
        missing_layers = [layer for layer in required_layers if layer not in rna_data.layers]
        if missing_layers:
            raise ValueError(f"RNA modality missing required layers: {missing_layers}")
            
        # Check spatial coordinates
        required_obs = ['x_pixel', 'y_pixel']
        missing_obs = [col for col in required_obs if col not in adata.obs.columns]
        if missing_obs:
            raise ValueError(f"MuData missing required spatial coordinates: {missing_obs}")
            
        # Check ATAC data if using regulatory network
        if self.config.development_stage >= 1 and self.config.regulatory.use_atac_masking:
            if 'atac' not in adata.mod:
                raise ValueError("ATAC modality required for regulatory network.")
            if 'open_chromatin' not in rna_data.layers:
                raise ValueError("RNA modality missing 'open_chromatin' layer for regulatory masking.")
    
    def _train_model(self, processed_data: Dict[str, Any], **kwargs) -> None:
        """Train the velocity model."""
        # Extract training data
        training_data = {
            'spliced': processed_data['spliced'],
            'unspliced': processed_data['unspliced'],
            'spatial_graph': processed_data['spatial_graph'],
            'expression_graph': processed_data['expression_graph'],
        }
        
        if self.config.regulatory.use_atac_masking:
            training_data['atac_mask'] = processed_data['atac_mask']
            
        # Train model
        self.model.fit(training_data, **kwargs)
    
    def _predict_velocity(self, adata: mu.MuData) -> None:
        """Generate velocity predictions and add to adata."""
        # Get model predictions
        with torch.no_grad():
            velocity = self.model.predict_velocity()
            
        # Add velocity to RNA modality
        adata['rna'].layers['velocity'] = velocity.cpu().numpy()
        
        # Add model-specific outputs
        if hasattr(self.model, 'get_ode_parameters'):
            ode_params = self.model.get_ode_parameters()
            for param_name, param_values in ode_params.items():
                adata['rna'].obs[f'tangelo_{param_name}'] = param_values.cpu().numpy()


# Convenience functions
def estimate_velocity(
    adata: mu.MuData,
    stage: int = 3,
    config: Optional[TangeloConfig] = None,
    copy: bool = False,
    **kwargs
) -> Optional[mu.MuData]:
    """
    Convenience function for velocity estimation.
    
    Parameters
    ----------
    adata : mu.MuData
        Multi-modal annotated data.
    stage : int, default 3
        Development stage to use.
    config : TangeloConfig, optional
        Custom configuration.
    copy : bool, default False
        Return copy with results.
    **kwargs
        Additional training arguments.
        
    Returns
    -------
    mu.MuData or None
        Data with velocity estimates if copy=True.
    """
    tv = TangeloVelocity(config=config, stage=stage)
    return tv.fit(adata, copy=copy, **kwargs)


def compare_stages(
    adata: mu.MuData,
    stages: tuple = (1, 2, 3),
    **kwargs
) -> Dict[int, mu.MuData]:
    """
    Compare velocity estimates across different development stages.
    
    Parameters
    ----------
    adata : mu.MuData
        Multi-modal annotated data.
    stages : tuple, default (1, 2, 3)
        Stages to compare.
    **kwargs
        Additional training arguments.
        
    Returns
    -------
    Dict[int, mu.MuData]
        Dictionary mapping stage numbers to results.
    """
    results = {}
    
    for stage in stages:
        print(f"\n=== Training Stage {stage} ===")
        tv = TangeloVelocity(stage=stage)
        results[stage] = tv.fit(adata.copy(), copy=True, **kwargs)
        
    return results