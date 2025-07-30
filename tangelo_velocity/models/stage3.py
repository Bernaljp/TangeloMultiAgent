"""Stage 3: Integrated model combining regulatory networks and graph neural networks."""

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
    warnings.warn("torch_geometric not available. Stage 3 functionality will be limited.")

from .base import BaseVelocityModel, MLP
from .regulatory import RegulatoryNetwork
from .encoders import SpatialGraphEncoder, ExpressionGraphEncoder, FusionModule
from .ode_dynamics import ODEParameterPredictor, VelocityODE, ODESolver
from .loss_functions import Stage1TotalLoss, Stage2TotalLoss, ELBOLoss, TangentSpaceLoss


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining regulatory and graph-based predictions.
    
    This module learns to dynamically weight the contributions of regulatory networks
    and graph neural networks for optimal velocity prediction integration.
    """
    
    def __init__(self, regulatory_dim: int, graph_dim: int, output_dim: int):
        super().__init__()
        self.regulatory_dim = regulatory_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        
        # Attention mechanism for weighing regulatory vs graph contributions
        self.attention_net = MLP(
            input_dim=regulatory_dim + graph_dim,
            hidden_dims=(128, 64),
            output_dim=2,  # Weights for [regulatory, graph]
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Projection layers
        self.regulatory_proj = nn.Linear(regulatory_dim, output_dim)
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        
        # Final fusion layer
        self.fusion_layer = MLP(
            input_dim=output_dim,
            hidden_dims=(output_dim // 2,),
            output_dim=output_dim,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
    
    def forward(
        self, 
        regulatory_features: torch.Tensor, 
        graph_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse regulatory and graph features using attention.
        
        Parameters
        ----------
        regulatory_features : torch.Tensor
            Features from regulatory network of shape (batch_size, regulatory_dim).
        graph_features : torch.Tensor
            Features from graph encoders of shape (batch_size, graph_dim).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Fused features and attention weights.
        """
        # Compute attention weights
        combined = torch.cat([regulatory_features, graph_features], dim=-1)
        attention_logits = self.attention_net(combined)
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Project features to common space
        reg_proj = self.regulatory_proj(regulatory_features)
        graph_proj = self.graph_proj(graph_features)
        
        # Apply attention weights
        weighted_reg = attention_weights[:, 0:1] * reg_proj
        weighted_graph = attention_weights[:, 1:2] * graph_proj
        
        # Fuse weighted features
        fused = weighted_reg + weighted_graph
        final_features = self.fusion_layer(fused)
        
        return final_features, attention_weights


class IntegratedODE(nn.Module):
    """
    Enhanced ODE system that integrates regulatory and graph-based transcription rates.
    
    Implements the integrated formulation: α = W @ sigmoid(s) + G(graph_features)
    where regulatory and graph components are combined for transcription rate prediction.
    """
    
    def __init__(self, n_genes: int, regulatory_network: RegulatoryNetwork, config):
        super().__init__()
        self.n_genes = n_genes
        self.regulatory_network = regulatory_network
        self.config = config
        
        # Graph-based transcription rate modulation
        latent_dim = getattr(config.encoder, 'latent_dim', 64)
        self.graph_transcription_net = MLP(
            input_dim=latent_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Integration weights for combining regulatory and graph components
        self.integration_weights = nn.Parameter(torch.ones(2))  # [regulatory_weight, graph_weight]
        
        # ODE parameters (will be set dynamically)
        self.beta = None
        self.gamma = None
    
    def set_parameters(self, beta: torch.Tensor, gamma: torch.Tensor) -> None:
        """Set ODE parameters for integration."""
        self.beta = beta
        self.gamma = gamma
    
    def get_integrated_transcription_rates(
        self, 
        spliced: torch.Tensor, 
        graph_latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute integrated transcription rates combining regulatory and graph components.
        
        α = w₁ * W @ sigmoid(s) + w₂ * G(graph_features)
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
        graph_latent : torch.Tensor
            Graph-derived latent features of shape (batch_size, latent_dim).
            
        Returns
        -------
        torch.Tensor
            Integrated transcription rates of shape (batch_size, n_genes).
        """
        # Get regulatory component using proper matrix multiplication: W @ sigmoid(s)
        regulatory_rates = self.regulatory_network.compute_transcription_rates_direct(spliced)
        
        # Get graph component: G(graph_features)
        graph_rates = self.graph_transcription_net(graph_latent)
        
        # Apply softmax to integration weights for proper weighting
        weights = F.softmax(self.integration_weights, dim=0)
        
        # Combine with learned weights
        integrated_rates = (
            weights[0] * regulatory_rates + 
            weights[1] * graph_rates
        )
        
        # Ensure positive transcription rates
        return F.softplus(integrated_rates)
    
    def forward(self, t: float, y: torch.Tensor, graph_latent: torch.Tensor) -> torch.Tensor:
        """
        Compute ODE derivatives with integrated transcription rates.
        
        System: du/dt = α(s, graph) - β * u
                ds/dt = β * u - γ * s
        
        Parameters
        ----------
        t : float
            Current time (not used in autonomous system).
        y : torch.Tensor
            Current state [u, s] of shape (batch_size, 2 * n_genes).
        graph_latent : torch.Tensor
            Graph-derived latent features.
            
        Returns
        -------
        torch.Tensor
            Time derivatives [du/dt, ds/dt].
        """
        batch_size = y.shape[0]
        
        # Split state into unspliced and spliced
        unspliced = y[:, :self.n_genes]
        spliced = y[:, self.n_genes:]
        
        # Get integrated transcription rates
        alpha = self.get_integrated_transcription_rates(spliced, graph_latent)
        
        # Ensure beta and gamma are set
        if self.beta is None or self.gamma is None:
            raise RuntimeError("ODE parameters must be set before forward pass")
        
        # Expand parameters to match batch size if needed
        if self.beta.dim() == 1:
            beta = self.beta.unsqueeze(0).expand(batch_size, -1)
        else:
            beta = self.beta
            
        if self.gamma.dim() == 1:
            gamma = self.gamma.unsqueeze(0).expand(batch_size, -1)
        else:
            gamma = self.gamma
        
        # Compute derivatives
        du_dt = alpha - beta * unspliced
        ds_dt = beta * unspliced - gamma * spliced
        
        return torch.cat([du_dt, ds_dt], dim=1)


class Stage3IntegratedModel(BaseVelocityModel):
    """
    Stage 3 integrated model combining regulatory networks and graph neural networks.
    
    This model implements the complete integration of:
    - Stage 1 regulatory networks with sigmoid features and ATAC masking
    - Stage 2 dual GraphSAGE encoders for spatial and expression graphs
    - Novel attention-based fusion mechanism
    - Enhanced ODE system with integrated transcription rates
    
    The model processes multi-modal inputs (RNA + ATAC + spatial graphs) to predict
    cellular velocities using a comprehensive integration of regulatory and graph-based
    dynamics.
    
    Mathematical Formulation:
    α = w₁ * W(sigmoid(s)) + w₂ * G(graph_features)
    
    where W represents the regulatory network and G represents graph-derived
    transcription rate modulation.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object with Stage 3 settings.
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
            raise ValueError("ATAC dimension is required for Stage 3 integrated model.")
            
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for Stage 3. "
                "Install with: pip install torch_geometric"
            )
        
        super().__init__(config, gene_dim, atac_dim)
        
        # Validate configuration
        if config.development_stage != 3:
            raise ValueError(f"Expected stage 3, got stage {config.development_stage}")
    
    def _initialize_components(self) -> None:
        """Initialize Stage 3 integrated model components."""
        # Get configuration parameters
        encoder_config = getattr(self.config, 'encoder', None)
        if encoder_config is None:
            from types import SimpleNamespace
            encoder_config = SimpleNamespace(
                hidden_dims=(512, 256, 128),
                latent_dim=64,
                dropout=0.1,
                batch_norm=True,
                aggregator='mean',
                fusion_method='attention',
                spatial_feature_dim=2
            )
        
        latent_dim = getattr(encoder_config, 'latent_dim', 64)
        
        # 1. Stage 1 Components: Regulatory Network
        self.regulatory_network = RegulatoryNetwork(
            n_genes=self.gene_dim,
            config=self.config
        )
        
        # 2. Stage 2 Components: Graph Encoders
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
        
        # Graph fusion module
        self.graph_fusion = FusionModule(
            latent_dim=latent_dim,
            fusion_method=getattr(encoder_config, 'fusion_method', 'sum')
        )
        
        # 3. Stage 3 Novel Components: Integration Architecture
        
        # Feature encoders for different modalities
        self.rna_encoder = MLP(
            input_dim=2 * self.gene_dim,  # [unspliced, spliced]
            hidden_dims=(256, 128),
            output_dim=128,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Attention-based fusion of regulatory and graph features
        self.attention_fusion = AttentionFusion(
            regulatory_dim=128,  # From RNA encoder
            graph_dim=latent_dim,  # From graph fusion
            output_dim=latent_dim
        )
        
        # Enhanced ODE parameter predictor using integrated features
        self.ode_parameter_predictor = ODEParameterPredictor(
            input_dim=latent_dim,
            n_genes=self.gene_dim,
            config=self.config
        )
        
        # Integrated ODE system
        self.integrated_ode = IntegratedODE(
            n_genes=self.gene_dim,
            regulatory_network=self.regulatory_network,
            config=self.config
        )
        
        # ODE solver
        self.ode_solver = ODESolver(self.config)
        
        # Combined loss functions
        self.stage1_loss = Stage1TotalLoss(
            config=self.config,
            n_genes=self.gene_dim
        )
        self.stage2_loss = Stage2TotalLoss(self.config)
        
        # Integration-specific losses
        self.elbo_loss = ELBOLoss()
        self.tangent_space_loss = TangentSpaceLoss()
        
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
        Set the ATAC-derived regulatory mask for Stage 3 integration.
        
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
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stage 3 integrated model.
        
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
        **kwargs
            Additional arguments.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Comprehensive model outputs including integrated velocity predictions.
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
        
        # === MULTI-MODAL FEATURE EXTRACTION ===
        
        # 1. RNA feature encoding
        rna_features = torch.cat([unspliced, spliced], dim=1)
        encoded_rna = self.rna_encoder(rna_features)
        
        # 2. Graph-based feature extraction
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
        
        # === INTEGRATION THROUGH ATTENTION FUSION ===
        
        # Combine regulatory (RNA) and graph features using attention
        try:
            integrated_features, attention_weights = self.attention_fusion(
                encoded_rna, graph_latent
            )
        except Exception as e:
            raise RuntimeError(f"Error in attention fusion: {e}")
        
        # === ODE PARAMETER PREDICTION ===
        
        # Predict ODE parameters from integrated features
        try:
            ode_params = self.ode_parameter_predictor(integrated_features)
        except Exception as e:
            raise RuntimeError(f"Error in ODE parameter prediction: {e}")
        
        # === INTEGRATED TRANSCRIPTION RATE COMPUTATION ===
        
        # Get integrated transcription rates: α = w₁ * W @ sigmoid(s) + w₂ * G(graph)
        try:
            integrated_transcription_rates = self.integrated_ode.get_integrated_transcription_rates(
                spliced, graph_latent
            )
        except Exception as e:
            raise RuntimeError(f"Error in transcription rate integration: {e}")
        
        # === ENHANCED ODE INTEGRATION ===
        
        # Set ODE parameters
        self.integrated_ode.set_parameters(
            beta=ode_params['beta'],
            gamma=ode_params['gamma']
        )
        
        # Solve integrated ODE system
        try:
            # Set up initial conditions
            y0 = torch.cat([unspliced, spliced], dim=1)
            t_span = self.config.ode.t_span
            
            # Create a wrapper function for ODE solver that includes graph_latent
            def ode_func(t, y):
                return self.integrated_ode(t, y, graph_latent)
            
            # Solve ODE - simplified approach for now
            ode_solution = self.ode_solver.solve(
                ode_system=self.integrated_ode,
                y0=y0,
                t_span=t_span,
                ode_params=ode_params,
                graph_latent=graph_latent  # Pass as additional parameter
            )
            
            # Extract final state
            final_state = ode_solution['final_state']
            pred_unspliced = final_state[:, :self.gene_dim]
            pred_spliced = final_state[:, self.gene_dim:]
            
        except Exception as e:
            # Fallback: use current state as prediction
            warnings.warn(f"ODE integration failed: {e}. Using current state.")
            pred_unspliced = unspliced
            pred_spliced = spliced
        
        # === INTEGRATED VELOCITY COMPUTATION ===
        
        # Compute velocity using integrated transcription rates
        try:
            # Enhanced velocity: ds/dt using integrated dynamics
            velocity_vector = self.integrated_ode(0.0, y0, graph_latent)
            velocity = velocity_vector[:, self.gene_dim:]  # ds/dt (spliced velocity)
        except Exception as e:
            # Fallback velocity computation
            warnings.warn(f"Integrated velocity computation failed: {e}. Using fallback.")
            velocity = integrated_transcription_rates - ode_params['gamma'].unsqueeze(1) * spliced
        
        # === COMPREHENSIVE OUTPUT ASSEMBLY ===
        
        outputs = {
            # Primary outputs
            'velocity': velocity,
            'pred_unspliced': pred_unspliced,
            'pred_spliced': pred_spliced,
            
            # Transcription rates
            'transcription_rates': integrated_transcription_rates,
            'regulatory_rates': self.regulatory_network.compute_transcription_rates_direct(spliced),
            'graph_transcription_rates': self.integrated_ode.graph_transcription_net(graph_latent),
            
            # Integration components
            'integrated_features': integrated_features,
            'attention_weights': attention_weights,
            'integration_weights': F.softmax(self.integrated_ode.integration_weights, dim=0),
            
            # RNA features
            'rna_features': encoded_rna,
            
            # Graph features
            'spatial_latent': spatial_mean,
            'spatial_log_var': spatial_log_var,
            'expression_latent': expr_mean,
            'expression_log_var': expr_log_var,
            'graph_latent': graph_latent,
            'graph_mean': graph_mean,
            'graph_log_var': graph_log_var,
            
            # ODE parameters
            'ode_params': ode_params,
            
            # For loss computation
            'latent_mean': graph_mean,
            'latent_log_var': graph_log_var,
        }
        
        # Store for later access
        self._last_outputs = outputs
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        similarity_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute comprehensive Stage 3 integrated loss.
        
        Combines losses from:
        - Stage 1: Regulatory network reconstruction and consistency
        - Stage 2: Graph-based ELBO and tangent space losses  
        - Stage 3: Integration-specific attention and consistency losses
        
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
            Total integrated loss value.
        """
        device = outputs['velocity'].device
        
        # Initialize loss components
        loss_components = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # === STAGE 1 REGULATORY LOSSES ===
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
                f"regulatory_{k}": v for k, v in stage1_loss_dict.items()
            })
            
        except Exception as e:
            warnings.warn(f"Stage 1 loss computation failed: {e}. Using zero loss.")
            regulatory_loss = torch.tensor(0.0, device=device)
        
        # === STAGE 2 GRAPH LOSSES ===
        try:
            stage2_loss, stage2_components = self.stage2_loss(outputs, targets)
            
            graph_loss = stage2_loss
            loss_components.update({
                f"graph_{k}": v for k, v in stage2_components.items()
            })
            
        except Exception as e:
            warnings.warn(f"Stage 2 loss computation failed: {e}. Using zero loss.")
            graph_loss = torch.tensor(0.0, device=device)
        
        # === STAGE 3 INTEGRATION LOSSES ===
        
        # 1. Attention regularization (encourage balanced weighting)
        if 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']
            # Encourage attention weights to be not too extreme (entropy regularization)
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), dim=1
            ).mean()
            attention_loss = -0.1 * attention_entropy  # Negative because we want high entropy
            loss_components['attention_regularization'] = attention_loss
        else:
            attention_loss = torch.tensor(0.0, device=device)
        
        # 2. Integration weight regularization (prevent one component from dominating)
        if hasattr(self.integrated_ode, 'integration_weights'):
            integration_weights = F.softmax(self.integrated_ode.integration_weights, dim=0)
            # Encourage balanced integration (entropy regularization)
            integration_entropy = -torch.sum(
                integration_weights * torch.log(integration_weights + 1e-8)
            )
            integration_loss = -0.05 * integration_entropy
            loss_components['integration_regularization'] = integration_loss
        else:
            integration_loss = torch.tensor(0.0, device=device)
        
        # 3. Transcription rate consistency loss
        if 'regulatory_rates' in outputs and 'graph_transcription_rates' in outputs:
            reg_rates = outputs['regulatory_rates']
            graph_rates = outputs['graph_transcription_rates']
            
            # Encourage some similarity between regulatory and graph-derived rates
            consistency_loss = 0.01 * F.mse_loss(reg_rates, graph_rates)
            loss_components['transcription_consistency'] = consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=device)
        
        # === COMBINE ALL LOSSES ===
        
        # Weight the different loss components
        loss_weights = {
            'regulatory': 0.4,  # Stage 1 contribution
            'graph': 0.4,       # Stage 2 contribution  
            'integration': 0.2  # Stage 3 integration contribution
        }
        
        total_loss = (
            loss_weights['regulatory'] * regulatory_loss +
            loss_weights['graph'] * graph_loss +
            loss_weights['integration'] * (
                attention_loss + integration_loss + consistency_loss
            )
        )
        
        # Store comprehensive loss breakdown
        loss_components.update({
            'regulatory_total': regulatory_loss,
            'graph_total': graph_loss,
            'attention_loss': attention_loss,
            'integration_loss': integration_loss,
            'consistency_loss': consistency_loss,
            'total': total_loss
        })
        
        self._last_loss_dict = loss_components
        
        return total_loss
    
    def _get_ode_parameters(self) -> Dict[str, torch.Tensor]:
        """Get ODE parameters from last forward pass."""
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError("Model must be run through forward pass first.")
        return self._last_outputs['ode_params']
    
    def _get_interaction_network(self) -> torch.Tensor:
        """Get the learned gene-gene interaction matrix from regulatory network."""
        return self.regulatory_network.get_interaction_matrix()
    
    def _get_transcription_rates(self, spliced: torch.Tensor) -> torch.Tensor:
        """Get integrated transcription rates."""
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError("Model must be run through forward pass first.")
        return self._last_outputs['transcription_rates']
    
    def _set_atac_mask(self, atac_mask: torch.Tensor) -> None:
        """Set ATAC mask for regulatory constraints."""
        self.set_atac_mask(atac_mask)
    
    def _get_regulatory_loss(self) -> torch.Tensor:
        """Get regulatory network regularization loss."""
        return self.regulatory_network.get_regularization_loss()
    
    def _create_identity_graph(self, n_nodes: int, device: torch.device) -> Data:
        """Create identity graph for missing graph inputs."""
        edge_index = torch.stack([
            torch.arange(n_nodes, device=device),
            torch.arange(n_nodes, device=device)
        ])
        return Data(edge_index=edge_index, num_nodes=n_nodes)
    
    def get_loss_dict(self) -> Dict[str, torch.Tensor]:
        """Get detailed loss breakdown from last computation."""
        if not hasattr(self, '_last_loss_dict') or self._last_loss_dict is None:
            raise RuntimeError("Loss must be computed before accessing loss dictionary.")
        return self._last_loss_dict
    
    def get_integration_analysis(self) -> Dict[str, Any]:
        """
        Analyze the integration between regulatory and graph components.
        
        Returns
        -------
        Dict[str, Any]
            Analysis of integration effectiveness and component contributions.
        """
        if not hasattr(self, '_last_outputs') or self._last_outputs is None:
            raise RuntimeError("Model must be run through forward pass first.")
        
        outputs = self._last_outputs
        analysis = {}
        
        # Attention weight analysis
        if 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']
            analysis['attention'] = {
                'regulatory_weight_mean': attention_weights[:, 0].mean().item(),
                'graph_weight_mean': attention_weights[:, 1].mean().item(),
                'attention_entropy': -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), dim=1
                ).mean().item()
            }
        
        # Integration weight analysis
        if hasattr(self.integrated_ode, 'integration_weights'):
            integration_weights = F.softmax(self.integrated_ode.integration_weights, dim=0)
            analysis['integration_weights'] = {
                'regulatory': integration_weights[0].item(),
                'graph': integration_weights[1].item()
            }
        
        # Component correlation analysis
        if 'regulatory_rates' in outputs and 'graph_transcription_rates' in outputs:
            reg_rates = outputs['regulatory_rates'].detach()
            graph_rates = outputs['graph_transcription_rates'].detach()
            
            # Compute correlation between regulatory and graph components
            correlation = torch.corrcoef(torch.stack([
                reg_rates.flatten(),
                graph_rates.flatten()
            ]))[0, 1]
            
            analysis['component_correlation'] = correlation.item()
        
        return analysis
    
    def get_model_summary(self) -> str:
        """
        Get a comprehensive summary of the Stage 3 integrated model.
        
        Returns
        -------
        str
            Detailed model summary.
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Get component parameter counts
        reg_params = sum(p.numel() for p in self.regulatory_network.parameters())
        spatial_params = sum(p.numel() for p in self.spatial_encoder.parameters())
        expr_params = sum(p.numel() for p in self.expression_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.attention_fusion.parameters())
        ode_params = sum(p.numel() for p in self.integrated_ode.parameters())
        
        summary = f"""
Stage 3 Integrated Model Summary
==============================
Development Stage: {self.development_stage}
Genes: {self.gene_dim}
ATAC Features: {self.atac_dim}

Parameters:
- Total: {n_params:,}
- Trainable: {n_trainable:,}

Component Breakdown:
- Regulatory Network: {reg_params:,}
- Spatial Encoder: {spatial_params:,}
- Expression Encoder: {expr_params:,}
- Attention Fusion: {fusion_params:,}
- Integrated ODE: {ode_params:,}

Architecture:
- Latent Dimension: {getattr(self.config.encoder, 'latent_dim', 64)}
- Fusion Method: {getattr(self.config.encoder, 'fusion_method', 'attention')}
- Sigmoid Components: {self.config.regulatory.n_sigmoid_components}
- ODE Solver: {self.config.ode.solver}

Integration Features:
- Multi-modal input processing (RNA + ATAC + Graphs)
- Attention-based regulatory-graph fusion
- Enhanced transcription rate: α = W @ sigmoid(s) + G(graph)
- Comprehensive loss combining all stages
"""
        return summary.strip()
    
    def pretrain_sigmoid_features(
        self,
        spliced_data: torch.Tensor,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        freeze_after_pretraining: bool = True
    ) -> None:
        """
        Pre-train sigmoid features for Stage 3 integrated model.
        
        This implements the sigmoid pretraining protocol for the regulatory component
        of the integrated Stage 3 model, ensuring proper α = W @ sigmoid(s) formulation.
        
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
        print("=== STAGE 3 SIGMOID PRETRAINING PROTOCOL ===")
        print("Pre-training sigmoid features for regulatory component...")
        self.regulatory_network.pretrain_sigmoid(
            spliced_data,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            freeze_after_pretraining=freeze_after_pretraining
        )
        print("=== STAGE 3 SIGMOID PRETRAINING COMPLETE ===")
        if freeze_after_pretraining:
            print("Sigmoid parameters FROZEN. Only interaction network W will train.")
            print("Graph components remain fully trainable.")
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