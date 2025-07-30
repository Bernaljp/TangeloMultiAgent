"""Stage 4: Advanced features model with temporal dynamics, uncertainty quantification, and interpretability."""

import warnings
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import numpy as np

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("torch_geometric not available. Stage 4 functionality will be limited.")

from .base import BaseVelocityModel, MLP
from .regulatory import RegulatoryNetwork
from .encoders import SpatialGraphEncoder, ExpressionGraphEncoder, FusionModule
from .ode_dynamics import ODEParameterPredictor, VelocityODE, ODESolver
from .stage3 import AttentionFusion, IntegratedODE
from .loss_functions import Stage1TotalLoss, Stage2TotalLoss, ELBOLoss, TangentSpaceLoss
from .multiscale import MultiscaleTrainer, MultiscaleLoss, MultiscaleConfig, create_multiscale_config


class TemporalDynamicsModule(nn.Module):
    """
    Temporal dynamics module for time-resolved velocity prediction.
    
    This module extends the base velocity model to handle temporal trajectories
    and predict velocity evolution over time.
    """
    
    def __init__(self, n_genes: int, latent_dim: int, n_time_points: int = 10):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_time_points = n_time_points
        
        # Temporal embedding network
        self.time_encoder = MLP(
            input_dim=1,  # Time as single scalar
            hidden_dims=(64, 32),
            output_dim=32,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Temporal-spatial feature fusion
        self.temporal_fusion = MLP(
            input_dim=latent_dim + 32,  # Latent features + time embedding
            hidden_dims=(128, 64),
            output_dim=latent_dim,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Temporal velocity predictor
        self.temporal_velocity_net = MLP(
            input_dim=latent_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.1,
            batch_norm=False
        )
        
        # Temporal regularization parameters
        self.register_parameter('temporal_smoothness', nn.Parameter(torch.tensor(1.0)))
    
    def forward(
        self, 
        latent_features: torch.Tensor, 
        time_points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict velocity evolution over time.
        
        Parameters
        ----------
        latent_features : torch.Tensor
            Latent features of shape (batch_size, latent_dim).
        time_points : torch.Tensor
            Time points of shape (batch_size, n_time_points).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing temporal velocity predictions.
        """
        batch_size = latent_features.shape[0]
        
        # Encode time points
        time_embedded = self.time_encoder(time_points.unsqueeze(-1))  # (batch, n_time, 32)
        
        # Expand latent features for each time point
        latent_expanded = latent_features.unsqueeze(1).expand(
            batch_size, self.n_time_points, -1
        )  # (batch, n_time, latent_dim)
        
        # Fuse temporal and latent information
        temporal_features = torch.cat([latent_expanded, time_embedded], dim=-1)
        fused_features = self.temporal_fusion(temporal_features)  # (batch, n_time, latent_dim)
        
        # Predict temporal velocities
        temporal_velocities = self.temporal_velocity_net(fused_features)  # (batch, n_time, n_genes)
        
        # Compute temporal consistency metrics
        velocity_differences = torch.diff(temporal_velocities, dim=1)
        temporal_smoothness = torch.mean(torch.sum(velocity_differences ** 2, dim=-1))
        
        return {
            'temporal_velocities': temporal_velocities,
            'temporal_features': fused_features,
            'temporal_smoothness': temporal_smoothness,
            'time_embedding': time_embedded
        }
    
    def get_temporal_regularization_loss(self, temporal_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal regularization loss for smooth velocity evolution.
        
        Parameters
        ----------
        temporal_outputs : Dict[str, torch.Tensor]
            Outputs from temporal dynamics module.
            
        Returns
        -------
        torch.Tensor
            Temporal regularization loss.
        """
        smoothness = temporal_outputs['temporal_smoothness']
        return self.temporal_smoothness * smoothness


class UncertaintyQuantificationModule(nn.Module):
    """
    Bayesian uncertainty quantification module for prediction confidence.
    
    This module provides uncertainty estimates for velocity predictions using
    variational inference and Monte Carlo dropout.
    """
    
    def __init__(self, n_genes: int, latent_dim: int, n_samples: int = 100):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        
        # Variational parameters for uncertainty estimation
        self.mean_predictor = MLP(
            input_dim=latent_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.2,  # Higher dropout for uncertainty
            batch_norm=True
        )
        
        self.variance_predictor = MLP(
            input_dim=latent_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.2,
            batch_norm=True
        )
        
        # Aleatoric uncertainty (data noise)
        self.aleatoric_net = MLP(
            input_dim=latent_dim,
            hidden_dims=(64, 32),
            output_dim=n_genes,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Epistemic uncertainty parameters
        self.register_parameter('epistemic_scale', nn.Parameter(torch.tensor(0.1)))
    
    def forward(self, latent_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict velocity with uncertainty quantification.
        
        Parameters
        ----------
        latent_features : torch.Tensor
            Latent features of shape (batch_size, latent_dim).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing velocity predictions and uncertainty estimates.
        """
        # Predict mean and variance
        velocity_mean = self.mean_predictor(latent_features)
        velocity_log_var = self.variance_predictor(latent_features)
        velocity_var = F.softplus(velocity_log_var)
        
        # Aleatoric uncertainty (inherent data noise)
        aleatoric_uncertainty = F.softplus(self.aleatoric_net(latent_features))
        
        # Monte Carlo dropout for epistemic uncertainty
        epistemic_samples = []
        self.train()  # Enable dropout during inference
        for _ in range(self.n_samples):
            sample = self.mean_predictor(latent_features)
            epistemic_samples.append(sample)
        self.eval()
        
        epistemic_samples = torch.stack(epistemic_samples, dim=0)  # (n_samples, batch, genes)
        epistemic_mean = torch.mean(epistemic_samples, dim=0)
        epistemic_uncertainty = torch.var(epistemic_samples, dim=0)
        
        # Total uncertainty = aleatoric + epistemic
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return {
            'velocity_mean': velocity_mean,
            'velocity_var': velocity_var,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'epistemic_samples': epistemic_samples,
            'confidence_intervals': self._compute_confidence_intervals(velocity_mean, total_uncertainty)
        }
    
    def _compute_confidence_intervals(
        self, 
        mean: torch.Tensor, 
        variance: torch.Tensor, 
        confidence: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence intervals for predictions.
        
        Parameters
        ----------
        mean : torch.Tensor
            Predicted means.
        variance : torch.Tensor
            Predicted variances.
        confidence : float, default 0.95
            Confidence level.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Upper and lower confidence bounds.
        """
        std = torch.sqrt(variance)
        z_score = torch.tensor(1.96)  # 95% confidence interval
        
        return {
            'lower_bound': mean - z_score * std,
            'upper_bound': mean + z_score * std
        }


class MultiScaleIntegrationModule(nn.Module):
    """
    Multi-scale integration module for cell-type and tissue-level coherence.
    
    This module ensures velocity predictions are coherent across different
    biological scales (single cells, cell types, tissues).
    """
    
    def __init__(self, n_genes: int, latent_dim: int, n_cell_types: int = 10):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_cell_types = n_cell_types
        
        # Cell-type-specific velocity modulation
        self.cell_type_embedding = nn.Embedding(n_cell_types, latent_dim)
        
        # Hierarchical attention for multi-scale integration
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Scale-specific velocity predictors
        self.cell_scale_predictor = MLP(
            input_dim=latent_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        self.tissue_scale_predictor = MLP(
            input_dim=latent_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Scale integration weights
        self.scale_weights = nn.Parameter(torch.ones(2))  # [cell, tissue]
    
    def forward(
        self, 
        latent_features: torch.Tensor, 
        cell_type_labels: torch.Tensor,
        tissue_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate velocity predictions across multiple biological scales.
        
        Parameters
        ----------
        latent_features : torch.Tensor
            Latent features of shape (batch_size, latent_dim).
        cell_type_labels : torch.Tensor
            Cell type labels of shape (batch_size,).
        tissue_context : torch.Tensor, optional
            Tissue-level context features.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Multi-scale velocity predictions.
        """
        batch_size = latent_features.shape[0]
        
        # Get cell type embeddings
        cell_type_emb = self.cell_type_embedding(cell_type_labels)  # (batch, latent_dim)
        
        # Combine with latent features
        cell_context = latent_features + cell_type_emb
        
        # Tissue context (use mean if not provided)
        if tissue_context is None:
            tissue_context = torch.mean(latent_features, dim=0, keepdim=True).expand(batch_size, -1)
        
        # Multi-head attention for scale integration
        scale_features = torch.stack([cell_context, tissue_context], dim=1)  # (batch, 2, latent_dim)
        attended_features, attention_weights = self.scale_attention(
            scale_features, scale_features, scale_features
        )
        
        # Scale-specific predictions
        cell_velocities = self.cell_scale_predictor(attended_features[:, 0, :])
        tissue_velocities = self.tissue_scale_predictor(attended_features[:, 1, :])
        
        # Weighted integration
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        integrated_velocity = (
            scale_weights_norm[0] * cell_velocities +
            scale_weights_norm[1] * tissue_velocities
        )
        
        return {
            'integrated_velocity': integrated_velocity,
            'cell_velocity': cell_velocities,
            'tissue_velocity': tissue_velocities,
            'scale_weights': scale_weights_norm,
            'attention_weights': attention_weights,
            'cell_type_embedding': cell_type_emb
        }


class AdvancedRegularizationModule(nn.Module):
    """
    Advanced regularization module with sparsity and biological constraints.
    
    This module implements sophisticated regularization techniques including
    pathway-aware sparsity, biological feasibility constraints, and
    multi-objective optimization.
    """
    
    def __init__(self, n_genes: int, pathway_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_genes = n_genes
        
        # Pathway-aware regularization
        if pathway_matrix is not None:
            self.register_buffer('pathway_matrix', pathway_matrix)
        else:
            # Default to identity if no pathway information
            self.register_buffer('pathway_matrix', torch.eye(n_genes))
        
        # Biological constraint parameters
        self.register_parameter('sparsity_weight', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('pathway_weight', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('smoothness_weight', nn.Parameter(torch.tensor(0.1)))
        
        # Adaptive regularization strengths
        self.adaptive_weights = nn.Parameter(torch.ones(n_genes))
    
    def compute_sparsity_loss(self, interaction_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive sparsity loss with gene-specific weights.
        
        Parameters
        ----------
        interaction_matrix : torch.Tensor
            Gene interaction matrix of shape (n_genes, n_genes).
            
        Returns
        -------
        torch.Tensor
            Sparsity regularization loss.
        """
        # Gene-specific adaptive weights
        adaptive_weights_norm = F.softplus(self.adaptive_weights)
        
        # Weighted L1 regularization
        weighted_interactions = interaction_matrix * adaptive_weights_norm.unsqueeze(1)
        sparsity_loss = torch.sum(torch.abs(weighted_interactions))
        
        return self.sparsity_weight * sparsity_loss
    
    def compute_pathway_loss(self, interaction_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute pathway-aware regularization loss.
        
        Parameters
        ----------
        interaction_matrix : torch.Tensor
            Gene interaction matrix of shape (n_genes, n_genes).
            
        Returns
        -------
        torch.Tensor
            Pathway regularization loss.
        """
        # Encourage interactions within pathways, discourage across pathways
        within_pathway = interaction_matrix * self.pathway_matrix
        across_pathway = interaction_matrix * (1.0 - self.pathway_matrix)
        
        # Reward within-pathway interactions, penalize across-pathway
        pathway_loss = (
            -0.1 * torch.sum(torch.abs(within_pathway)) +  # Encourage
            1.0 * torch.sum(torch.abs(across_pathway))     # Discourage
        )
        
        return self.pathway_weight * pathway_loss
    
    def compute_smoothness_loss(self, velocity_predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial/temporal smoothness loss.
        
        Parameters
        ----------
        velocity_predictions : torch.Tensor
            Velocity predictions of shape (batch_size, n_genes).
            
        Returns
        -------
        torch.Tensor
            Smoothness regularization loss.
        """
        # Total variation regularization
        if velocity_predictions.dim() == 2:
            # Simple case: encourage similar velocities for similar cells
            velocity_diffs = torch.diff(velocity_predictions, dim=0)
            smoothness_loss = torch.sum(torch.abs(velocity_diffs))
        else:
            smoothness_loss = torch.tensor(0.0, device=velocity_predictions.device)
        
        return self.smoothness_weight * smoothness_loss
    
    def forward(
        self, 
        interaction_matrix: torch.Tensor, 
        velocity_predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive regularization losses.
        
        Parameters
        ----------
        interaction_matrix : torch.Tensor
            Gene interaction matrix.
        velocity_predictions : torch.Tensor
            Velocity predictions.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of regularization losses.
        """
        sparsity_loss = self.compute_sparsity_loss(interaction_matrix)
        pathway_loss = self.compute_pathway_loss(interaction_matrix)
        smoothness_loss = self.compute_smoothness_loss(velocity_predictions)
        
        total_regularization = sparsity_loss + pathway_loss + smoothness_loss
        
        return {
            'sparsity_loss': sparsity_loss,
            'pathway_loss': pathway_loss,
            'smoothness_loss': smoothness_loss,
            'total_regularization': total_regularization
        }


class InterpretabilityToolsModule(nn.Module):
    """
    Interpretability tools module for feature importance and pathway analysis.
    
    This module provides methods to interpret model predictions, identify
    important genes and pathways, and generate biological insights.
    """
    
    def __init__(self, n_genes: int, gene_names: Optional[List[str]] = None):
        super().__init__()
        self.n_genes = n_genes
        self.gene_names = gene_names or [f"Gene_{i}" for i in range(n_genes)]
        
        # Attention mechanism for feature importance
        self.importance_attention = nn.MultiheadAttention(
            embed_dim=n_genes,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature importance scorer
        self.importance_scorer = MLP(
            input_dim=n_genes,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
    
    def compute_feature_importance(
        self, 
        velocity_predictions: torch.Tensor,
        spliced_expression: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance scores for velocity predictions.
        
        Parameters
        ----------
        velocity_predictions : torch.Tensor
            Velocity predictions of shape (batch_size, n_genes).
        spliced_expression : torch.Tensor
            Spliced expression of shape (batch_size, n_genes).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Feature importance scores and rankings.
        """
        # Attention-based importance
        attended_features, attention_weights = self.importance_attention(
            velocity_predictions.unsqueeze(1),
            spliced_expression.unsqueeze(1),
            spliced_expression.unsqueeze(1)
        )
        
        # Importance scores
        importance_scores = self.importance_scorer(attended_features.squeeze(1))
        importance_scores = F.softmax(importance_scores, dim=-1)
        
        # Gene rankings
        gene_rankings = torch.argsort(importance_scores, dim=-1, descending=True)
        
        return {
            'importance_scores': importance_scores,
            'attention_weights': attention_weights,
            'gene_rankings': gene_rankings,
            'top_genes': self._get_top_genes(importance_scores, gene_rankings)
        }
    
    def _get_top_genes(
        self, 
        importance_scores: torch.Tensor, 
        gene_rankings: torch.Tensor, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top-k most important genes with their scores.
        
        Parameters
        ----------
        importance_scores : torch.Tensor
            Importance scores for genes.
        gene_rankings : torch.Tensor
            Gene rankings by importance.
        top_k : int, default 10
            Number of top genes to return.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of top genes with names and scores.
        """
        batch_size = importance_scores.shape[0]
        top_genes_list = []
        
        for batch_idx in range(batch_size):
            batch_top_genes = []
            for rank in range(min(top_k, self.n_genes)):
                gene_idx = gene_rankings[batch_idx, rank].item()
                score = importance_scores[batch_idx, gene_idx].item()
                
                batch_top_genes.append({
                    'gene_name': self.gene_names[gene_idx],
                    'gene_index': gene_idx,
                    'importance_score': score,
                    'rank': rank
                })
            
            top_genes_list.append(batch_top_genes)
        
        return top_genes_list
    
    def analyze_pathway_activity(
        self,
        interaction_matrix: torch.Tensor,
        pathway_matrix: torch.Tensor,
        gene_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze pathway-level activity and interactions.
        
        Parameters
        ----------
        interaction_matrix : torch.Tensor
            Gene interaction matrix.
        pathway_matrix : torch.Tensor
            Pathway membership matrix.
        gene_names : List[str], optional
            Gene names for interpretation.
            
        Returns
        -------
        Dict[str, Any]
            Pathway analysis results.
        """
        # Pathway-level interaction strengths
        pathway_interactions = interaction_matrix * pathway_matrix
        pathway_strengths = torch.sum(torch.abs(pathway_interactions), dim=1)
        
        # Most active pathways
        top_pathways = torch.argsort(pathway_strengths, descending=True)
        
        return {
            'pathway_strengths': pathway_strengths,
            'top_pathways': top_pathways,
            'pathway_interactions': pathway_interactions
        }


class Stage4AdvancedModel(BaseVelocityModel):
    """
    Stage 4 advanced model with temporal dynamics, uncertainty quantification,
    multi-scale integration, advanced regularization, and interpretability tools.
    
    This model represents the complete implementation of the Tangelo Velocity
    framework with all advanced features:
    
    1. **Temporal Dynamics**: Time-resolved velocity prediction
    2. **Uncertainty Quantification**: Bayesian approaches for prediction confidence  
    3. **Multi-Scale Integration**: Cell-type and tissue-level coherence
    4. **Advanced Regularization**: Sparsity and biological constraints
    5. **Interpretability Tools**: Feature importance and pathway analysis
    
    The model maintains the critical mathematical formulation α = W @ sigmoid(s)
    and implements the sigmoid pretraining protocol across all components.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object with Stage 4 settings.
    gene_dim : int
        Number of genes in the dataset.
    atac_dim : int
        Number of ATAC features (required for regulatory masking).
    """
    
    def __init__(
        self,
        config,
        gene_dim: int,
        atac_dim: int
    ):
        # Validate requirements
        if atac_dim is None:
            raise ValueError("ATAC dimension is required for Stage 4 advanced model.")
            
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for Stage 4. "
                "Install with: pip install torch_geometric"
            )
        
        super().__init__(config, gene_dim, atac_dim)
        
        # Validate configuration
        if config.development_stage != 4:
            raise ValueError(f"Expected stage 4, got stage {config.development_stage}")
    
    def _initialize_components(self) -> None:
        """Initialize Stage 4 advanced model components."""
        # Get configuration parameters
        encoder_config = getattr(self.config, 'encoder', None)
        if encoder_config is None:
            from types import SimpleNamespace
            encoder_config = SimpleNamespace(
                hidden_dims=(512, 256, 128),
                latent_dim=128,  # Larger for Stage 4
                dropout=0.1,
                batch_norm=True,
                aggregator='mean',
                fusion_method='attention',
                spatial_feature_dim=2
            )
        
        latent_dim = getattr(encoder_config, 'latent_dim', 128)
        
        # === STAGE 1-3 FOUNDATION COMPONENTS ===
        
        # Regulatory Network (Stage 1)
        self.regulatory_network = RegulatoryNetwork(
            n_genes=self.gene_dim,
            config=self.config
        )
        
        # Graph Encoders (Stage 2)
        self.spatial_encoder = SpatialGraphEncoder(
            input_dim=self.gene_dim,
            latent_dim=latent_dim,
            config=encoder_config
        )
        
        self.expression_encoder = ExpressionGraphEncoder(
            input_dim=self.gene_dim,
            latent_dim=latent_dim,
            config=encoder_config
        )
        
        # Graph fusion (Stage 2)
        self.graph_fusion = FusionModule(
            latent_dim=latent_dim,
            fusion_method=getattr(encoder_config, 'fusion_method', 'sum')
        )
        
        # Integration Architecture (Stage 3)
        self.rna_encoder = MLP(
            input_dim=2 * self.gene_dim,  # [unspliced, spliced]
            hidden_dims=(256, 128),
            output_dim=128,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        self.attention_fusion = AttentionFusion(
            regulatory_dim=128,
            graph_dim=latent_dim,
            output_dim=latent_dim
        )
        
        self.integrated_ode = IntegratedODE(
            n_genes=self.gene_dim,
            regulatory_network=self.regulatory_network,
            config=self.config
        )
        
        # === STAGE 4 ADVANCED COMPONENTS ===
        
        # 1. Temporal Dynamics
        self.temporal_dynamics = TemporalDynamicsModule(
            n_genes=self.gene_dim,
            latent_dim=latent_dim,
            n_time_points=getattr(self.config, 'n_time_points', 10)
        )
        
        # 2. Uncertainty Quantification
        self.uncertainty_module = UncertaintyQuantificationModule(
            n_genes=self.gene_dim,
            latent_dim=latent_dim,
            n_samples=getattr(self.config, 'uncertainty_samples', 100)
        )
        
        # 3. Multi-Scale Integration
        self.multiscale_module = MultiScaleIntegrationModule(
            n_genes=self.gene_dim,
            latent_dim=latent_dim,
            n_cell_types=getattr(self.config, 'n_cell_types', 10)
        )
        
        # 4. Advanced Regularization
        pathway_matrix = getattr(self.config, 'pathway_matrix', None)
        self.regularization_module = AdvancedRegularizationModule(
            n_genes=self.gene_dim,
            pathway_matrix=pathway_matrix
        )
        
        # 5. Interpretability Tools
        gene_names = getattr(self.config, 'gene_names', None)
        self.interpretability_module = InterpretabilityToolsModule(
            n_genes=self.gene_dim,
            gene_names=gene_names
        )
        
        # === ENHANCED ODE SYSTEM ===
        
        self.ode_parameter_predictor = ODEParameterPredictor(
            input_dim=latent_dim,
            n_genes=self.gene_dim,
            config=self.config
        )
        
        self.ode_solver = ODESolver(self.config)
        
        # === LOSS FUNCTIONS ===
        
        self.stage1_loss = Stage1TotalLoss(
            config=self.config,
            n_genes=self.gene_dim
        )
        self.stage2_loss = Stage2TotalLoss(self.config)
        self.elbo_loss = ELBOLoss()
        self.tangent_space_loss = TangentSpaceLoss()
        
        # === MULTISCALE TRAINING INTEGRATION ===
        
        # Initialize multiscale configuration
        multiscale_config = getattr(self.config, 'multiscale', None)
        if multiscale_config is None:
            # Create default multiscale config if not provided
            multiscale_config = create_multiscale_config(
                enable=False,  # Disabled by default for backward compatibility
                max_scales=4,
                min_scale_size=1,
                scale_strategy="geometric"
            )
        
        # Store multiscale configuration
        self.multiscale_config = multiscale_config
        
        # Initialize multiscale loss (will be used if enabled)
        # Note: We'll set the base_loss_fn later to avoid circular dependency
        self.multiscale_loss = None
        
        # Multiscale trainer (optional, for advanced training routines)
        self._multiscale_trainer = None
        
        # === BUFFERS AND TRACKING ===
        
        # ATAC mask storage
        self.register_buffer(
            'atac_mask',
            torch.eye(self.gene_dim)  # Default to identity
        )
        
        # Tracking variables
        self._last_outputs = None
        self._last_loss_dict = None
    
    def set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """
        Set the ATAC-derived regulatory mask for Stage 4.
        
        Parameters
        ----------
        atac_mask : torch.Tensor
            Binary mask of shape (n_genes, n_genes) indicating permitted interactions.
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
        spatial_graph: Optional[Data] = None,
        expression_graph: Optional[Data] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        atac_mask: Optional[torch.Tensor] = None,
        time_points: Optional[torch.Tensor] = None,
        cell_type_labels: Optional[torch.Tensor] = None,
        tissue_context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stage 4 advanced model.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
        unspliced : torch.Tensor
            Unspliced RNA counts of shape (batch_size, n_genes).
        spatial_graph : torch_geometric.Data, optional
            Spatial k-NN graph with edge_index.
        expression_graph : torch_geometric.Data, optional
            Expression similarity graph with edge_index.
        spatial_coords : torch.Tensor, optional
            Spatial coordinates of shape (batch_size, 2).
        atac_mask : torch.Tensor, optional
            ATAC mask for regulatory constraints.
        time_points : torch.Tensor, optional
            Time points for temporal dynamics.
        cell_type_labels : torch.Tensor, optional
            Cell type labels for multi-scale integration.
        tissue_context : torch.Tensor, optional
            Tissue-level context features.
        **kwargs
            Additional arguments.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Comprehensive model outputs with advanced features.
        """
        batch_size = spliced.shape[0]
        device = spliced.device
        
        # Validate inputs
        if spliced.shape != unspliced.shape:
            raise ValueError(
                f"Spliced and unspliced shapes must match. "
                f"Got {spliced.shape} and {unspliced.shape}"
            )
        
        # Update ATAC mask if provided
        if atac_mask is not None:
            self.set_atac_mask(atac_mask)
        
        # Handle missing graphs
        if spatial_graph is None:
            warnings.warn("No spatial graph provided. Creating identity graph.")
            spatial_graph = self._create_identity_graph(batch_size, device)
        
        if expression_graph is None:
            warnings.warn("No expression graph provided. Creating identity graph.")
            expression_graph = self._create_identity_graph(batch_size, device)
        
        # === STAGE 1-3 FOUNDATION PROCESSING ===
        
        # RNA feature encoding
        rna_features = torch.cat([unspliced, spliced], dim=1)
        encoded_rna = self.rna_encoder(rna_features)
        
        # Graph-based feature extraction
        try:
            # Spatial graph encoding
            spatial_mean, spatial_log_var = self.spatial_encoder(
                spliced, spatial_graph.edge_index, spatial_coords
            )
            
            # Expression graph encoding
            expr_mean, expr_log_var = self.expression_encoder(
                spliced, expression_graph.edge_index
            )
            
            # Fuse graph representations
            graph_mean, graph_log_var = self.graph_fusion(
                spatial_mean, spatial_log_var, expr_mean, expr_log_var
            )
            
            # Sample from latent distribution
            graph_latent = self.graph_fusion.reparameterize(graph_mean, graph_log_var)
            
        except Exception as e:
            raise RuntimeError(f"Error in graph processing: {e}")
        
        # Integration through attention fusion
        try:
            integrated_features, attention_weights = self.attention_fusion(
                encoded_rna, graph_latent
            )
        except Exception as e:
            raise RuntimeError(f"Error in attention fusion: {e}")
        
        # === STAGE 4 ADVANCED FEATURE PROCESSING ===
        
        outputs = {}
        
        # 1. Temporal Dynamics
        if time_points is not None:
            try:
                temporal_outputs = self.temporal_dynamics(integrated_features, time_points)
                outputs.update({f"temporal_{k}": v for k, v in temporal_outputs.items()})
            except Exception as e:
                warnings.warn(f"Temporal dynamics failed: {e}")
        
        # 2. Uncertainty Quantification
        try:
            uncertainty_outputs = self.uncertainty_module(integrated_features)
            outputs.update({f"uncertainty_{k}": v for k, v in uncertainty_outputs.items()})
            
            # Use uncertainty-aware velocity as primary prediction
            primary_velocity = uncertainty_outputs['velocity_mean']
            
        except Exception as e:
            warnings.warn(f"Uncertainty quantification failed: {e}")
            # Fallback to basic velocity prediction
            primary_velocity = self._compute_basic_velocity(spliced, unspliced, graph_latent)
        
        # 3. Multi-Scale Integration
        if cell_type_labels is not None:
            try:
                multiscale_outputs = self.multiscale_module(
                    integrated_features, cell_type_labels, tissue_context
                )
                outputs.update({f"multiscale_{k}": v for k, v in multiscale_outputs.items()})
                
                # Use multi-scale integrated velocity
                primary_velocity = multiscale_outputs['integrated_velocity']
                
            except Exception as e:
                warnings.warn(f"Multi-scale integration failed: {e}")
        
        # === INTEGRATED TRANSCRIPTION RATE COMPUTATION ===
        
        try:
            # Compute integrated transcription rates: α = W @ sigmoid(s) + G(graph)
            integrated_transcription_rates = self.integrated_ode.get_integrated_transcription_rates(
                spliced, graph_latent
            )
            outputs['transcription_rates'] = integrated_transcription_rates
            
        except Exception as e:
            warnings.warn(f"Transcription rate computation failed: {e}")
            # Fallback to regulatory network only
            outputs['transcription_rates'] = self.regulatory_network.compute_transcription_rates_direct(spliced)
        
        # === ODE INTEGRATION AND VELOCITY COMPUTATION ===
        
        try:
            # Predict ODE parameters
            ode_params = self.ode_parameter_predictor(integrated_features)
            
            # Set ODE parameters
            self.integrated_ode.set_parameters(
                beta=ode_params['beta'],
                gamma=ode_params['gamma']
            )
            
            # Compute velocity using integrated system
            y0 = torch.cat([unspliced, spliced], dim=1)
            velocity_vector = self.integrated_ode(0.0, y0, graph_latent)
            base_velocity = velocity_vector[:, self.gene_dim:]  # ds/dt
            
            # Use primary velocity if available, otherwise use base velocity
            if 'primary_velocity' not in locals():
                primary_velocity = base_velocity
            
            outputs.update({
                'velocity': primary_velocity,
                'base_velocity': base_velocity,
                'ode_params': ode_params
            })
            
        except Exception as e:
            warnings.warn(f"ODE integration failed: {e}")
            # Final fallback
            outputs['velocity'] = torch.zeros_like(spliced)
            outputs['ode_params'] = {
                'beta': torch.ones(self.gene_dim, device=device),
                'gamma': torch.ones(self.gene_dim, device=device)
            }
        
        # === ADVANCED ANALYSIS AND INTERPRETABILITY ===
        
        # 4. Feature Importance Analysis
        try:
            importance_outputs = self.interpretability_module.compute_feature_importance(
                outputs['velocity'], spliced
            )
            outputs.update({f"importance_{k}": v for k, v in importance_outputs.items()})
            
        except Exception as e:
            warnings.warn(f"Feature importance analysis failed: {e}")
        
        # === COMPREHENSIVE OUTPUT ASSEMBLY ===
        
        # Add foundation components
        outputs.update({
            # RNA and graph features
            'rna_features': encoded_rna,
            'spatial_latent': spatial_mean,
            'spatial_log_var': spatial_log_var,
            'expression_latent': expr_mean,
            'expression_log_var': expr_log_var,
            'graph_latent': graph_latent,
            'graph_mean': graph_mean,
            'graph_log_var': graph_log_var,
            
            # Integration components
            'integrated_features': integrated_features,
            'attention_weights': attention_weights,
            
            # For loss computation
            'latent_mean': graph_mean,
            'latent_log_var': graph_log_var,
            
            # Regulatory components
            'regulatory_rates': self.regulatory_network.compute_transcription_rates_direct(spliced),
            'interaction_matrix': self.regulatory_network.get_interaction_matrix()
        })
        
        # Store for later access
        self._last_outputs = outputs
        
        return outputs
    
    def _compute_basic_velocity(
        self, 
        spliced: torch.Tensor, 
        unspliced: torch.Tensor, 
        graph_latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute basic velocity as fallback when advanced modules fail.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts.
        unspliced : torch.Tensor
            Unspliced RNA counts.
        graph_latent : torch.Tensor
            Graph latent features.
            
        Returns
        -------
        torch.Tensor
            Basic velocity prediction.
        """
        try:
            transcription_rates = self.integrated_ode.get_integrated_transcription_rates(
                spliced, graph_latent
            )
            # Simple velocity: α - γ * s
            gamma = torch.ones(self.gene_dim, device=spliced.device)
            return transcription_rates - gamma.unsqueeze(0) * spliced
        except:
            return torch.zeros_like(spliced)
    
    def _create_identity_graph(self, n_nodes: int, device: torch.device) -> Data:
        """Create identity graph for missing graph inputs."""
        edge_index = torch.stack([
            torch.arange(n_nodes, device=device),
            torch.arange(n_nodes, device=device)
        ])
        return Data(edge_index=edge_index, num_nodes=n_nodes)
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        similarity_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute comprehensive Stage 4 loss with advanced regularization.
        
        Combines losses from all stages plus advanced regularization terms.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from forward pass.
        targets : Dict[str, torch.Tensor]
            Target values including 'spliced' and 'unspliced'.
        similarity_matrix : torch.Tensor, optional
            Cell-cell similarity matrix for coherence losses.
            
        Returns
        -------
        torch.Tensor
            Total comprehensive loss value.
        """
        device = outputs['velocity'].device
        loss_components = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # === FOUNDATION LOSSES (STAGES 1-3) ===
        
        # Stage 1 regulatory losses
        try:
            stage1_loss_dict = self.stage1_loss(
                outputs=outputs,
                targets=targets,
                model=self,
                atac_mask=self.atac_mask,
                similarity_matrix=similarity_matrix
            )
            
            regulatory_loss = stage1_loss_dict['total']
            loss_components.update({
                f"stage1_{k}": v for k, v in stage1_loss_dict.items()
            })
            
        except Exception as e:
            warnings.warn(f"Stage 1 loss computation failed: {e}")
            regulatory_loss = torch.tensor(0.0, device=device)
        
        # Stage 2 graph losses
        try:
            stage2_loss, stage2_components = self.stage2_loss(outputs, targets)
            graph_loss = stage2_loss
            loss_components.update({
                f"stage2_{k}": v for k, v in stage2_components.items()
            })
            
        except Exception as e:
            warnings.warn(f"Stage 2 loss computation failed: {e}")
            graph_loss = torch.tensor(0.0, device=device)
        
        # === STAGE 4 ADVANCED LOSSES ===
        
        # 1. Temporal regularization
        temporal_loss = torch.tensor(0.0, device=device)
        if 'temporal_smoothness' in outputs:
            temporal_loss = self.temporal_dynamics.get_temporal_regularization_loss({
                'temporal_smoothness': outputs['temporal_smoothness']
            })
            loss_components['temporal_loss'] = temporal_loss
        
        # 2. Uncertainty losses
        uncertainty_loss = torch.tensor(0.0, device=device)
        if 'uncertainty_velocity_var' in outputs:
            # Negative log-likelihood for uncertainty
            mean = outputs.get('uncertainty_velocity_mean', outputs['velocity'])
            var = outputs['uncertainty_velocity_var']
            target_velocity = targets.get('velocity', torch.zeros_like(mean))
            
            uncertainty_loss = 0.5 * torch.sum(
                torch.log(var) + (target_velocity - mean) ** 2 / var
            )
            loss_components['uncertainty_loss'] = uncertainty_loss
        
        # 3. Multi-scale consistency
        multiscale_loss = torch.tensor(0.0, device=device)
        if 'multiscale_cell_velocity' in outputs and 'multiscale_tissue_velocity' in outputs:
            cell_vel = outputs['multiscale_cell_velocity']
            tissue_vel = outputs['multiscale_tissue_velocity']
            multiscale_loss = 0.1 * F.mse_loss(cell_vel, tissue_vel)
            loss_components['multiscale_loss'] = multiscale_loss
        
        # 4. Advanced regularization
        advanced_reg_loss = torch.tensor(0.0, device=device)
        if 'interaction_matrix' in outputs:
            try:
                reg_outputs = self.regularization_module(
                    outputs['interaction_matrix'],
                    outputs['velocity']
                )
                advanced_reg_loss = reg_outputs['total_regularization']
                loss_components.update({
                    f"reg_{k}": v for k, v in reg_outputs.items()
                })
                
            except Exception as e:
                warnings.warn(f"Advanced regularization failed: {e}")
        
        # === LOSS COMBINATION ===
        
        # Stage-specific weights
        loss_weights = {
            'regulatory': 0.3,      # Stage 1
            'graph': 0.25,          # Stage 2  
            'integration': 0.15,    # Stage 3
            'temporal': 0.1,        # Stage 4 temporal
            'uncertainty': 0.05,    # Stage 4 uncertainty
            'multiscale': 0.05,     # Stage 4 multi-scale
            'regularization': 0.1   # Stage 4 regularization
        }
        
        total_loss = (
            loss_weights['regulatory'] * regulatory_loss +
            loss_weights['graph'] * graph_loss +
            loss_weights['temporal'] * temporal_loss +
            loss_weights['uncertainty'] * uncertainty_loss +
            loss_weights['multiscale'] * multiscale_loss +
            loss_weights['regularization'] * advanced_reg_loss
        )
        
        # Store comprehensive loss breakdown
        loss_components.update({
            'regulatory_total': regulatory_loss,
            'graph_total': graph_loss,
            'temporal_total': temporal_loss,
            'uncertainty_total': uncertainty_loss,
            'multiscale_total': multiscale_loss,
            'regularization_total': advanced_reg_loss,
            'total': total_loss
        })
        
        self._last_loss_dict = loss_components
        
        return total_loss
    
    def pretrain_sigmoid_features(
        self,
        spliced_data: torch.Tensor,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        freeze_after_pretraining: bool = True
    ) -> None:
        """
        Pre-train sigmoid features for Stage 4 advanced model.
        
        This implements the sigmoid pretraining protocol ensuring the
        correct α = W @ sigmoid(s) formulation across all components.
        
        Parameters
        ----------
        spliced_data : torch.Tensor
            Spliced RNA expression data of shape (n_cells, n_genes).
        n_epochs : int, default 100
            Number of pre-training epochs.
        learning_rate : float, default 0.01
            Learning rate for pre-training.
        freeze_after_pretraining : bool, default True
            Whether to freeze sigmoid parameters after pretraining.
        """
        print("=== STAGE 4 ADVANCED SIGMOID PRETRAINING PROTOCOL ===")
        print("Pre-training sigmoid features for all regulatory components...")
        
        # Pretrain regulatory network sigmoid features
        self.regulatory_network.pretrain_sigmoid(
            spliced_data,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            freeze_after_pretraining=freeze_after_pretraining
        )
        
        print("=== STAGE 4 SIGMOID PRETRAINING COMPLETE ===")
        if freeze_after_pretraining:
            print("Sigmoid parameters FROZEN across all components.")
            print("Only interaction networks and advanced modules will train.")
        else:
            print("All parameters remain trainable.")
    
    def is_sigmoid_frozen(self) -> bool:
        """
        Check if sigmoid features are currently frozen.
        
        Returns
        -------
        bool
            True if sigmoid features are frozen, False otherwise.
        """
        return self.regulatory_network.is_sigmoid_frozen()
    
    # === MULTISCALE TRAINING METHODS ===
    
    def _initialize_multiscale_loss(self) -> None:
        """Initialize multiscale loss with proper base loss function."""
        def base_loss_fn(outputs, targets, similarity_matrix=None, **kwargs):
            return self.compute_loss(outputs, targets, similarity_matrix, **kwargs)
        
        self.multiscale_loss = MultiscaleLoss(
            base_loss_fn=base_loss_fn,
            config=self.multiscale_config
        )
    
    def enable_multiscale_training(
        self,
        max_scales: int = 4,
        min_scale_size: int = 1,
        scale_strategy: str = "geometric",
        weights: Optional[List[float]] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Enable multiscale training for the Stage 4 model.
        
        Parameters
        ----------
        max_scales : int, default 4
            Maximum number of scales to use.
        min_scale_size : int, default 1
            Minimum size for the smallest scale.
        scale_strategy : str, default "geometric"
            Strategy for determining scale sizes.
        weights : List[float], optional
            Custom weights for each scale.
        random_seed : int, optional
            Random seed for reproducible sampling.
        """
        # Update multiscale configuration
        self.multiscale_config.enable_multiscale = True
        self.multiscale_config.max_scales = max_scales
        self.multiscale_config.min_scale_size = min_scale_size
        self.multiscale_config.scale_strategy = scale_strategy
        if weights is not None:
            self.multiscale_config.multiscale_weights = weights
        if random_seed is not None:
            self.multiscale_config.random_seed = random_seed
        
        # Reinitialize multiscale loss with updated config
        self._initialize_multiscale_loss()
        
        print(f"=== MULTISCALE TRAINING ENABLED ===")
        print(f"Max scales: {max_scales}")
        print(f"Min scale size: {min_scale_size}")
        print(f"Scale strategy: {scale_strategy}")
        print(f"Scale weights: {self.multiscale_config.multiscale_weights}")
    
    def disable_multiscale_training(self) -> None:
        """Disable multiscale training and revert to standard training."""
        self.multiscale_config.enable_multiscale = False
        
        # Reinitialize multiscale loss with disabled config
        self._initialize_multiscale_loss()
        
        print("=== MULTISCALE TRAINING DISABLED ===")
    
    def is_multiscale_enabled(self) -> bool:
        """
        Check if multiscale training is currently enabled.
        
        Returns
        -------
        bool
            True if multiscale training is enabled, False otherwise.
        """
        return self.multiscale_config.enable_multiscale
    
    def get_multiscale_trainer(self) -> MultiscaleTrainer:
        """
        Get or create a multiscale trainer for this model.
        
        Returns
        -------
        MultiscaleTrainer
            Trainer with multiscale capabilities.
        """
        if self._multiscale_trainer is None:
            self._multiscale_trainer = MultiscaleTrainer(
                model=self,
                base_loss_fn=self.compute_loss,
                config=self.multiscale_config
            )
        return self._multiscale_trainer
    
    def compute_multiscale_loss(
        self,
        batch_data: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        similarity_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss using multiscale training if enabled.
        
        Parameters
        ----------
        batch_data : Dict[str, torch.Tensor]
            Input batch data containing 'spliced', 'unspliced', etc.
        targets : Dict[str, torch.Tensor]
            Target data.
        similarity_matrix : torch.Tensor, optional
            Cell-cell similarity matrix.
        **kwargs
            Additional arguments for loss computation.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Loss dictionary with multiscale components if enabled.
        """
        if not self.multiscale_config.enable_multiscale:
            # Standard single-scale loss computation
            outputs = self.forward(**batch_data)
            return self.compute_loss(outputs, targets, similarity_matrix, **kwargs)
        
        # Initialize multiscale loss if not already done
        if self.multiscale_loss is None:
            self._initialize_multiscale_loss()
        
        # Prepare batch data for multiscale loss
        # The multiscale loss expects the model to be callable
        def model_fn(**inputs):
            return self.forward(**inputs)
        
        # Use multiscale loss computation
        return self.multiscale_loss(
            model=model_fn,
            batch_data=batch_data,
            targets=targets,
            similarity_matrix=similarity_matrix,
            **kwargs
        )
    
    # === ABSTRACT METHOD IMPLEMENTATIONS ===
    
    def _get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """Get ODE parameters from last forward pass."""
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError("Model must be run through forward pass first.")
        return self._last_outputs.get('ode_params', {})
    
    def _get_interaction_network(self) -> torch.Tensor:
        """Get the learned gene-gene interaction matrix."""
        return self.regulatory_network.get_interaction_matrix()
    
    def _get_transcription_rates(self, spliced: torch.Tensor) -> torch.Tensor:
        """Get integrated transcription rates."""
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError("Model must be run through forward pass first.")
        return self._last_outputs.get('transcription_rates', self.regulatory_network.compute_transcription_rates_direct(spliced))
    
    def _set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """Set ATAC mask for regulatory constraints."""
        self.set_atac_mask(atac_mask)
    
    def _get_regulatory_loss(self) -> torch.Tensor:
        """Get regulatory network regularization loss."""
        return self.regulatory_network.get_regularization_loss()
    
    def get_loss_dict(self) -> Dict[str, torch.Tensor]:
        """Get detailed loss breakdown from last computation."""
        if not hasattr(self, '_last_loss_dict') or self._last_loss_dict is None:
            raise RuntimeError("Loss must be computed before accessing loss dictionary.")
        return self._last_loss_dict
    
    def get_advanced_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of Stage 4 advanced features.
        
        Returns
        -------
        Dict[str, Any]
            Analysis of all advanced components and their performance.
        """
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError("Model must be run through forward pass first.")
        
        outputs = self._last_outputs
        analysis = {}
        
        # Temporal analysis
        if 'temporal_smoothness' in outputs:
            analysis['temporal'] = {
                'smoothness_score': outputs['temporal_smoothness'].item(),
                'has_temporal_predictions': 'temporal_velocities' in outputs
            }
        
        # Uncertainty analysis
        if 'uncertainty_total_uncertainty' in outputs:
            total_unc = outputs['uncertainty_total_uncertainty']
            analysis['uncertainty'] = {
                'mean_uncertainty': torch.mean(total_unc).item(),
                'uncertainty_range': (torch.min(total_unc).item(), torch.max(total_unc).item()),
                'high_uncertainty_genes': torch.sum(total_unc > torch.median(total_unc)).item()
            }
        
        # Multi-scale analysis
        if 'multiscale_scale_weights' in outputs:
            scale_weights = outputs['multiscale_scale_weights']
            analysis['multiscale'] = {
                'cell_weight': scale_weights[0].item(),
                'tissue_weight': scale_weights[1].item(),
                'scale_balance': abs(scale_weights[0] - scale_weights[1]).item()
            }
        
        # Feature importance analysis
        if 'importance_scores' in outputs:
            importance = outputs['importance_scores']
            analysis['interpretability'] = {
                'top_gene_importance': torch.max(importance, dim=1)[0].mean().item(),
                'importance_entropy': -torch.sum(
                    importance * torch.log(importance + 1e-8), dim=1
                ).mean().item()
            }
        
        return analysis
    
    def get_model_summary(self) -> str:
        """
        Get a comprehensive summary of the Stage 4 advanced model.
        
        Returns
        -------
        str
            Detailed model summary with all advanced features.
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Component parameter counts
        component_params = {}
        for name, module in self.named_children():
            component_params[name] = sum(p.numel() for p in module.parameters())
        
        summary = f"""
Stage 4 Advanced Model Summary
=============================
Development Stage: {self.development_stage}
Genes: {self.gene_dim}
ATAC Features: {self.atac_dim}

Parameters:
- Total: {n_params:,}
- Trainable: {n_trainable:,}

Component Breakdown:
- Regulatory Network: {component_params.get('regulatory_network', 0):,}
- Spatial Encoder: {component_params.get('spatial_encoder', 0):,}
- Expression Encoder: {component_params.get('expression_encoder', 0):,}
- Temporal Dynamics: {component_params.get('temporal_dynamics', 0):,}
- Uncertainty Module: {component_params.get('uncertainty_module', 0):,}
- Multi-Scale Module: {component_params.get('multiscale_module', 0):,}
- Regularization Module: {component_params.get('regularization_module', 0):,}
- Interpretability Module: {component_params.get('interpretability_module', 0):,}

Advanced Features:
✓ Temporal Dynamics: Time-resolved velocity prediction
✓ Uncertainty Quantification: Bayesian prediction confidence
✓ Multi-Scale Integration: Cell-type and tissue coherence
✓ Advanced Regularization: Pathway-aware sparsity constraints
✓ Interpretability Tools: Feature importance and pathway analysis
✓ Multiscale Training: {'ENABLED' if self.is_multiscale_enabled() else 'DISABLED'}

Mathematical Foundation:
- Velocity Formulation: α = W @ sigmoid(s) (CORRECTED)
- Sigmoid Pretraining: {'FROZEN' if self.is_sigmoid_frozen() else 'ACTIVE'}
- Loss Integration: All stages (1-4) with advanced regularization
- ODE Solver: {self.config.ode.solver}

Model Capabilities:
- Time-resolved velocity trajectories
- Uncertainty-aware predictions with confidence intervals
- Multi-scale biological coherence
- Pathway-informed sparsity regularization
- Comprehensive model interpretability
- Hierarchical multiscale training: {'ACTIVE' if self.is_multiscale_enabled() else 'INACTIVE'}

Multiscale Configuration:
- Max Scales: {self.multiscale_config.max_scales}
- Min Scale Size: {self.multiscale_config.min_scale_size}
- Scale Strategy: {self.multiscale_config.scale_strategy}
- Scale Weights: {self.multiscale_config.multiscale_weights}
"""
        return summary.strip()