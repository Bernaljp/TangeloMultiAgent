"""Stage 1: Core Regulatory Model (MVP) implementation."""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseStage1Model, MLP
from .regulatory import RegulatoryNetwork
from .ode_dynamics import ODEParameterPredictor, VelocityODE, ODESolver
from .loss_functions import Stage1TotalLoss


class Stage1RegulatoryModel(BaseStage1Model):
    """
    Stage 1 regulatory model implementing linear interaction networks.
    
    This model serves as the MVP (Minimum Viable Product) for regulatory
    velocity estimation, focusing on:
    - Sigmoid feature mapping of spliced RNA
    - Linear gene-gene interactions with ATAC masking
    - ODE-based velocity prediction
    - Reconstruction loss with proper numerical stability
    
    The model implements the simplified RNA velocity system:
    du/dt = α(s) - β * u
    ds/dt = β * u - γ * s
    
    where α(s) comes from a regulatory network that transforms spliced RNA
    through sigmoid features and applies linear interactions masked by
    chromatin accessibility.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object with Stage 1 settings.
    gene_dim : int
        Number of genes in the dataset.
    atac_dim : int
        Number of ATAC features (required for Stage 1).
    """
    
    def __init__(
        self,
        config,
        gene_dim: int,
        atac_dim: int
    ):
        super().__init__(config, gene_dim, atac_dim)
        
        if atac_dim is None:
            raise ValueError("ATAC dimension is required for Stage 1 model.")
            
        # Validate configuration
        if config.development_stage != 1:
            raise ValueError(f"Expected stage 1, got stage {config.development_stage}")
    
    def _initialize_components(self) -> None:
        """Initialize Stage 1 model components."""
        # Regulatory network (sigmoid features + linear interactions)
        self.regulatory_network = RegulatoryNetwork(
            n_genes=self.gene_dim,
            config=self.config
        )
        
        # Simple feature encoder for parameter prediction
        # In Stage 1, we use a simple MLP instead of graph encoders
        latent_dim = 64  # Simple latent representation
        self.feature_encoder = MLP(
            input_dim=2 * self.gene_dim,  # [unspliced, spliced] concatenated
            hidden_dims=(128, 64),
            output_dim=latent_dim,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # ODE parameter predictor
        self.ode_parameter_predictor = ODEParameterPredictor(
            input_dim=latent_dim,
            n_genes=self.gene_dim,
            config=self.config
        )
        
        # ODE system
        self.velocity_ode = VelocityODE(
            n_genes=self.gene_dim,
            regulatory_network=self.regulatory_network
        )
        
        # ODE solver
        self.ode_solver = ODESolver(self.config)
        
        # Loss function
        self.loss_fn = Stage1TotalLoss(
            config=self.config,
            n_genes=self.gene_dim
        )
        
        # Store ATAC mask (set during training)
        self.register_buffer(
            'atac_mask',
            torch.eye(self.gene_dim)  # Default to identity (no masking)
        )
    
    def set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """
        Set the ATAC-derived regulatory mask.
        
        Parameters
        ----------
        atac_mask : torch.Tensor
            Binary mask of shape (n_genes, n_genes) indicating which
            gene-gene interactions are permitted by chromatin accessibility.
        """
        if atac_mask.shape != (self.gene_dim, self.gene_dim):
            raise ValueError(
                f"ATAC mask shape {atac_mask.shape} does not match "
                f"expected shape ({self.gene_dim}, {self.gene_dim})"
            )
        
        self.atac_mask.copy_(atac_mask)
        self.regulatory_network.set_atac_mask(atac_mask)
    
    def forward(
        self,
        spliced: torch.Tensor,
        unspliced: torch.Tensor,
        atac_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stage 1 regulatory model.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
        unspliced : torch.Tensor
            Unspliced RNA counts of shape (batch_size, n_genes).
        atac_mask : torch.Tensor, optional
            ATAC mask for this batch. If provided, updates the model's mask.
        **kwargs
            Additional arguments (ignored in Stage 1).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'pred_unspliced': Predicted unspliced RNA
            - 'pred_spliced': Predicted spliced RNA  
            - 'velocity': Predicted velocity
            - 'ode_params': ODE parameters (beta, gamma, time)
            - 'transcription_rates': Regulatory network output
        """
        batch_size = spliced.shape[0]
        
        # Update ATAC mask if provided
        if atac_mask is not None:
            self.set_atac_mask(atac_mask)
        
        # Encode features for parameter prediction
        # In Stage 1, we use simple concatenation of unspliced and spliced
        features = torch.cat([unspliced, spliced], dim=1)
        encoded_features = self.feature_encoder(features)
        
        # Predict ODE parameters
        ode_params = self.ode_parameter_predictor(encoded_features)
        
        # Get transcription rates from regulatory network
        transcription_rates = self.regulatory_network(spliced)
        
        # Set up initial conditions for ODE
        y0 = torch.cat([unspliced, spliced], dim=1)  # [u, s]
        
        # Solve ODE system
        t_span = self.config.ode.t_span
        ode_solution = self.ode_solver.solve(
            ode_system=self.velocity_ode,
            y0=y0,
            t_span=t_span,
            ode_params=ode_params
        )
        
        # Extract final state (predicted RNA abundances)
        final_state = ode_solution['final_state']
        pred_unspliced = final_state[:, :self.gene_dim]
        pred_spliced = final_state[:, self.gene_dim:]
        
        # Compute velocity at current state
        # Set ODE parameters for velocity computation
        self.velocity_ode.set_parameters(
            beta=ode_params['beta'],
            gamma=ode_params['gamma']
        )
        
        # Compute velocity vector
        velocity_vector = self.velocity_ode(0.0, y0)
        velocity = velocity_vector[:, self.gene_dim:]  # ds/dt (spliced velocity)
        
        # Store outputs for prediction
        outputs = {
            'pred_unspliced': pred_unspliced,
            'pred_spliced': pred_spliced,
            'velocity': velocity,
            'ode_params': ode_params,
            'transcription_rates': transcription_rates
        }
        
        self._last_outputs = outputs
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        similarity_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute comprehensive Stage 1 model loss.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from forward pass.
        targets : Dict[str, torch.Tensor]
            Target values including 'spliced' and 'unspliced'.
        similarity_matrix : torch.Tensor, optional
            Cell-cell similarity matrix for velocity coherence loss.
            
        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        # Compute comprehensive loss with ATAC mask and similarity matrix
        loss_dict = self.loss_fn(
            outputs=outputs,
            targets=targets,
            model=self,
            atac_mask=self.atac_mask,
            similarity_matrix=similarity_matrix
        )
        
        # Store loss components for logging
        self._last_loss_dict = loss_dict
        
        return loss_dict['total']
    
    def get_loss_dict(self) -> Dict[str, torch.Tensor]:
        """Get detailed loss breakdown from last computation."""
        if not hasattr(self, '_last_loss_dict'):
            raise RuntimeError("Loss must be computed before accessing loss dictionary.")
        return self._last_loss_dict
    
    def _get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """Get ODE parameters from last forward pass."""
        if not hasattr(self, '_last_outputs'):
            raise RuntimeError("Model must be run through forward pass first.")
        return self._last_outputs['ode_params']
    
    def _get_interaction_network(self) -> torch.Tensor:
        """Get the learned gene-gene interaction matrix."""
        return self.regulatory_network.get_interaction_matrix()
    
    def _get_transcription_rates(self, spliced: torch.Tensor) -> torch.Tensor:
        """Get transcription rates from regulatory network."""
        return self.regulatory_network(spliced)
    
    def _set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """Set ATAC mask for regulatory constraints."""
        self.set_atac_mask(atac_mask)  # Use the existing method
    
    def _get_regulatory_loss(self) -> torch.Tensor:
        """Get regulatory network regularization loss."""
        return self.regulatory_network.get_regularization_loss()
    
    def pretrain_sigmoid_features(
        self,
        spliced_data: torch.Tensor,
        n_epochs: int = 100,
        learning_rate: float = 0.01
    ) -> None:
        """
        Pre-train sigmoid features on spliced RNA data.
        
        This optional pre-training step fits the sigmoid transformation
        to match the empirical CDF of the spliced RNA data, providing
        better initialization for the regulatory network.
        
        Parameters
        ----------
        spliced_data : torch.Tensor
            Spliced RNA expression data of shape (n_cells, n_genes).
        n_epochs : int, default 100
            Number of pre-training epochs.
        learning_rate : float, default 0.01
            Learning rate for pre-training.
        """
        print("Pre-training sigmoid features...")
        self.regulatory_network.pretrain_sigmoid(
            spliced_data,
            n_epochs=n_epochs,
            learning_rate=learning_rate
        )
        print("Sigmoid pre-training complete.")
    
    def predict_transcription_rates(
        self,
        spliced: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict transcription rates for given spliced RNA.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
            
        Returns
        -------
        torch.Tensor
            Transcription rates of shape (batch_size, n_genes).
        """
        self.eval()
        with torch.no_grad():
            return self.regulatory_network(spliced)
    
    def get_regulatory_network_info(self) -> Dict[str, Any]:
        """
        Get information about the regulatory network structure.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing network information.
        """
        interaction_matrix = self.get_interaction_matrix()
        
        # Compute network statistics
        n_interactions = torch.sum(self.atac_mask).item()
        n_active_interactions = torch.sum(
            (torch.abs(interaction_matrix) > 0.01) & (self.atac_mask > 0)
        ).item()
        
        sparsity = 1.0 - (n_active_interactions / max(n_interactions, 1))
        
        return {
            'n_genes': self.gene_dim,
            'n_possible_interactions': n_interactions,
            'n_active_interactions': n_active_interactions,
            'sparsity': sparsity,
            'interaction_strength_mean': torch.mean(torch.abs(interaction_matrix)).item(),
            'interaction_strength_std': torch.std(torch.abs(interaction_matrix)).item(),
            'atac_mask_density': torch.mean(self.atac_mask).item()
        }
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the Stage 1 model architecture.
        
        Returns
        -------
        str
            Model summary string.
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        reg_info = self.get_regulatory_network_info()
        
        summary = f"""
Stage 1 Regulatory Model Summary
===============================
Development Stage: {self.development_stage}
Genes: {self.gene_dim}
ATAC Features: {self.atac_dim}

Parameters:
- Total: {n_params:,}
- Trainable: {n_trainable:,}

Regulatory Network:
- Possible interactions: {reg_info['n_possible_interactions']:,}
- Active interactions: {reg_info['n_active_interactions']:,}
- Sparsity: {reg_info['sparsity']:.3f}
- ATAC mask density: {reg_info['atac_mask_density']:.3f}

Components:
- Sigmoid feature components: {self.config.regulatory.n_sigmoid_components}
- ODE solver: {self.config.ode.solver}
- Loss distribution: Negative Binomial
"""
        return summary.strip()
    
    def get_velocity_decomposition(
        self,
        spliced: torch.Tensor,
        unspliced: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed velocity decomposition for analysis.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
        unspliced : torch.Tensor
            Unspliced RNA counts of shape (batch_size, n_genes).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing velocity components.
        """
        self.eval()
        with torch.no_grad():
            # Get current model outputs
            outputs = self.forward(spliced, unspliced)
            ode_params = outputs['ode_params']
            
            # Set ODE parameters
            self.velocity_ode.set_parameters(
                beta=ode_params['beta'],
                gamma=ode_params['gamma']
            )
            
            # Get detailed velocity components
            components = self.velocity_ode.get_velocity_components(spliced, unspliced)
            
            return {
                **components,
                'total_velocity': outputs['velocity'],
                'transcription_rates': outputs['transcription_rates'],
                'beta': ode_params['beta'],
                'gamma': ode_params['gamma']
            }
    
    def analyze_regulatory_interactions(
        self,
        gene_names: Optional[list] = None,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Analyze the learned regulatory interactions.
        
        Parameters
        ----------
        gene_names : list, optional
            Names of genes. If None, uses indices.
        threshold : float, default 0.01
            Threshold for considering an interaction significant.
            
        Returns
        -------
        Dict[str, Any]
            Analysis of regulatory interactions.
        """
        interaction_matrix = self.get_interaction_matrix()
        masked_interactions = interaction_matrix * self.atac_mask
        
        # Find significant interactions
        significant_mask = torch.abs(masked_interactions) > threshold
        significant_indices = torch.nonzero(significant_mask)
        
        interactions_list = []
        for idx in significant_indices:
            regulator_idx, target_idx = idx[0].item(), idx[1].item()
            strength = masked_interactions[regulator_idx, target_idx].item()
            
            regulator_name = gene_names[regulator_idx] if gene_names else f"Gene_{regulator_idx}"
            target_name = gene_names[target_idx] if gene_names else f"Gene_{target_idx}"
            
            interactions_list.append({
                'regulator': regulator_name,
                'target': target_name,
                'strength': strength,
                'regulator_idx': regulator_idx,
                'target_idx': target_idx
            })
        
        # Sort by strength
        interactions_list.sort(key=lambda x: abs(x['strength']), reverse=True)
        
        return {
            'n_significant_interactions': len(interactions_list),
            'interactions': interactions_list,
            'interaction_matrix': masked_interactions.detach().cpu().numpy(),
            'atac_mask': self.atac_mask.detach().cpu().numpy(),
            'analysis_threshold': threshold
        }
    
    def validate_model_setup(self) -> Dict[str, bool]:
        """
        Validate that the model is properly set up.
        
        Returns
        -------
        Dict[str, bool]
            Validation results for different components.
        """
        results = {}
        
        try:
            # Check regulatory network
            self.regulatory_network.get_interaction_matrix()
            results['regulatory_network'] = True
        except Exception:
            results['regulatory_network'] = False
        
        try:
            # Check ODE system
            self.velocity_ode.validate_setup()
            results['ode_system'] = True
        except Exception:
            results['ode_system'] = False
        
        try:
            # Check ODE solver
            self.ode_solver.validate_solver_config()
            results['ode_solver'] = True
        except Exception:
            results['ode_solver'] = False
        
        try:
            # Check ATAC mask
            assert self.atac_mask.shape == (self.gene_dim, self.gene_dim)
            results['atac_mask'] = True
        except Exception:
            results['atac_mask'] = False
        
        results['overall'] = all(results.values())
        
        return results