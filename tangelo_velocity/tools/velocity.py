"""High-level velocity estimation functions."""

import muon as mu
from typing import Optional, Dict, Any

from ..config import TangeloConfig
from ..preprocessing import MuDataProcessor  
from ..models import get_velocity_model


def estimate_velocity(
    adata: mu.MuData,
    stage: int = 1,
    config: Optional[TangeloConfig] = None,
    copy: bool = False,
    **kwargs
) -> Optional[mu.MuData]:
    """
    Estimate RNA velocity using Tangelo Velocity.
    
    This is a high-level convenience function that handles the complete
    velocity estimation workflow including preprocessing, model training,
    and velocity calculation.
    
    Parameters
    ----------
    adata : mu.MuData
        Multi-modal annotated data object with 'rna' and optionally 'atac' modalities.
        Required layers for 'rna': 'spliced', 'unspliced'
        Required obs: 'x_pixel', 'y_pixel' for spatial coordinates
    stage : int, default 1
        Development stage to use:
        - 0: Foundation (preprocessing only)
        - 1: Regulatory model with ATAC masking
        - 2: Graph model with spatial encoding
    config : TangeloConfig, optional
        Custom configuration. If None, uses stage-appropriate defaults.
    copy : bool, default False
        If True, return a copy of the data with results added.
        Otherwise, modifies adata in-place.
    **kwargs
        Additional arguments passed to model training.
        
    Returns
    -------
    mu.MuData or None
        If copy=True, returns annotated data with velocity estimates.
        Otherwise, modifies adata in-place and returns None.
        
    Notes
    -----
    This function performs:
    1. Data validation and preprocessing
    2. Graph construction (if stage >= 2)
    3. Model initialization and training
    4. Velocity estimation
    5. Result storage in adata.mod['rna'].layers['velocity']
    
    Examples
    --------
    >>> import muon as mu
    >>> import tangelo_velocity as tv
    >>> 
    >>> # Load your multi-modal data
    >>> adata = mu.read_h5mu("data.h5mu")
    >>> 
    >>> # Stage 1: Regulatory model
    >>> tv.tools.estimate_velocity(adata, stage=1)
    >>> 
    >>> # Stage 2: Graph model with custom config
    >>> config = tv.TangeloConfig(development_stage=2)
    >>> config.encoder.fusion_method = "attention"
    >>> tv.tools.estimate_velocity(adata, config=config)
    """
    # Copy data if requested
    if copy:
        adata = adata.copy()
        
    # Validate input data
    _validate_input_data(adata, stage)
    
    # Get or create configuration
    if config is None:
        from ..config import get_stage_config
        config = get_stage_config(stage)
    else:
        if config.development_stage != stage:
            raise ValueError(f"Config stage ({config.development_stage}) does not match requested stage ({stage})")
    
    # Set data dimensions in config
    config.gene_dim = adata['rna'].n_vars
    if 'atac' in adata.mod:
        config.atac_dim = adata['atac'].n_vars
    
    # Preprocess data
    processor = MuDataProcessor(config)
    processed_data = processor.process_mudata(adata)
    
    # Initialize model
    model = get_velocity_model(config, config.gene_dim, config.atac_dim)
    
    # Train model
    training_data = _prepare_training_data(processed_data, stage)
    model.fit(training_data, **kwargs)
    
    # Compute velocity
    velocity = model.predict_velocity()
    
    # Store results
    adata['rna'].layers['velocity'] = velocity
    
    # Store model parameters if available
    if hasattr(model, 'get_ode_parameters'):
        ode_params = model.get_ode_parameters()
        for param_name, param_values in ode_params.items():
            adata['rna'].obs[f'tangelo_{param_name}'] = param_values
            
    # Store latent representations if available
    if hasattr(model, 'get_latent_representations'):
        latent_reps = model.get_latent_representations()
        for rep_name, rep_values in latent_reps.items():
            adata['rna'].obsm[f'X_tangelo_{rep_name}'] = rep_values
    
    if copy:
        return adata


def _validate_input_data(adata: mu.MuData, stage: int) -> None:
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
        
    # Check ATAC data if using regulatory network (stage >= 1)
    if stage >= 1:
        if 'atac' not in adata.mod:
            print("Warning: ATAC modality not found. Regulatory masking will be disabled.")
        elif 'open_chromatin' not in rna_data.layers:
            print("Warning: 'open_chromatin' layer not found in RNA modality. Creating from ATAC data.")


def _prepare_training_data(processed_data: Dict[str, Any], stage: int) -> Dict[str, Any]:
    """Prepare training data based on stage requirements."""
    training_data = {
        'spliced': processed_data['spliced'],
        'unspliced': processed_data['unspliced'],
    }
    
    # Add stage-specific data
    if stage >= 1 and 'atac_mask' in processed_data:
        training_data['atac_mask'] = processed_data['atac_mask']
        
    if stage >= 2:
        training_data.update({
            'spatial_graph': processed_data['spatial_graph'],
            'expression_graph': processed_data['expression_graph'],
        })
        
        if 'spatial_coords' in processed_data:
            training_data['spatial_coords'] = processed_data['spatial_coords']
    
    return training_data