"""Loss functions for Tangelo Velocity models."""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, NegativeBinomial, Poisson
from torch.distributions.kl import kl_divergence as kl

from .base import safe_log, MLP, initialize_weights


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for RNA abundance matching using KL divergence.
    
    This loss compares predicted RNA abundances with observed counts using
    appropriate distributions (Negative Binomial or Poisson) and computes
    KL divergence for robust reconstruction.
    
    Parameters
    ----------
    distribution : str, default "nb"
        Distribution assumption ("nb" for Negative Binomial, "poisson" for Poisson).
    theta_init : float, default 10.0
        Initial overdispersion parameter for Negative Binomial.
    eps : float, default 1e-8
        Small constant for numerical stability.
    """
    
    def __init__(
        self,
        distribution: str = "nb",
        theta_init: float = 10.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.distribution = distribution.lower()
        self.eps = eps
        
        if self.distribution not in ["nb", "poisson", "normal"]:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        # Learnable overdispersion parameter for Negative Binomial
        if self.distribution == "nb":
            self.log_theta = nn.Parameter(
                torch.log(torch.tensor(theta_init))
            )
    
    def forward(
        self,
        pred_unspliced: torch.Tensor,
        pred_spliced: torch.Tensor,
        obs_unspliced: torch.Tensor,
        obs_spliced: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.
        
        Parameters
        ----------
        pred_unspliced : torch.Tensor
            Predicted unspliced counts of shape (batch, n_genes).
        pred_spliced : torch.Tensor
            Predicted spliced counts of shape (batch, n_genes).
        obs_unspliced : torch.Tensor
            Observed unspliced counts of shape (batch, n_genes).
        obs_spliced : torch.Tensor
            Observed spliced counts of shape (batch, n_genes).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing loss components.
        """
        if self.distribution == "nb":
            return self._negative_binomial_loss(
                pred_unspliced, pred_spliced,
                obs_unspliced, obs_spliced
            )
        elif self.distribution == "poisson":
            return self._poisson_loss(
                pred_unspliced, pred_spliced,
                obs_unspliced, obs_spliced
            )
        elif self.distribution == "normal":
            return self._normal_loss(
                pred_unspliced, pred_spliced,
                obs_unspliced, obs_spliced
            )
    
    def _negative_binomial_loss(
        self,
        pred_u: torch.Tensor,
        pred_s: torch.Tensor,
        obs_u: torch.Tensor,
        obs_s: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute Negative Binomial reconstruction loss."""
        theta = torch.exp(self.log_theta)
        
        # Ensure predictions are positive
        pred_u = torch.clamp(pred_u, min=self.eps)
        pred_s = torch.clamp(pred_s, min=self.eps)
        
        # Create distributions
        dist_u = NegativeBinomial(
            total_count=theta,
            probs=theta / (theta + pred_u)
        )
        dist_s = NegativeBinomial(
            total_count=theta,
            probs=theta / (theta + pred_s)
        )
        
        # Compute negative log-likelihood
        nll_u = -dist_u.log_prob(obs_u).sum(dim=-1)
        nll_s = -dist_s.log_prob(obs_s).sum(dim=-1)
        
        total_loss = nll_u + nll_s
        
        return {
            'total': total_loss.mean(),
            'unspliced': nll_u.mean(),
            'spliced': nll_s.mean(),
            'theta': theta.mean()
        }
    
    def _poisson_loss(
        self,
        pred_u: torch.Tensor,
        pred_s: torch.Tensor,
        obs_u: torch.Tensor,
        obs_s: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute Poisson reconstruction loss."""
        # Ensure predictions are positive
        pred_u = torch.clamp(pred_u, min=self.eps)
        pred_s = torch.clamp(pred_s, min=self.eps)
        
        # Create distributions
        dist_u = Poisson(pred_u)
        dist_s = Poisson(pred_s)
        
        # Compute negative log-likelihood
        nll_u = -dist_u.log_prob(obs_u).sum(dim=-1)
        nll_s = -dist_s.log_prob(obs_s).sum(dim=-1)
        
        total_loss = nll_u + nll_s
        
        return {
            'total': total_loss.mean(),
            'unspliced': nll_u.mean(),
            'spliced': nll_s.mean()
        }
    
    def _normal_loss(
        self,
        pred_u: torch.Tensor,
        pred_s: torch.Tensor,
        obs_u: torch.Tensor,
        obs_s: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute Normal (MSE) reconstruction loss."""
        mse_u = F.mse_loss(pred_u, obs_u, reduction='none').sum(dim=-1)
        mse_s = F.mse_loss(pred_s, obs_s, reduction='none').sum(dim=-1)
        
        total_loss = mse_u + mse_s
        
        return {
            'total': total_loss.mean(),
            'unspliced': mse_u.mean(),
            'spliced': mse_s.mean()
        }


class VelocityConsistencyLoss(nn.Module):
    """
    Loss for ensuring velocity consistency with observed dynamics.
    
    This loss encourages predicted velocities to be consistent with
    the local neighborhood structure in gene expression space.
    
    Parameters
    ----------
    consistency_weight : float, default 1.0
        Weight for consistency loss component.
    """
    
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        predicted_velocity: torch.Tensor,
        expression_graph: torch.Tensor,
        current_expression: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity consistency loss.
        
        Parameters
        ----------
        predicted_velocity : torch.Tensor
            Predicted velocities of shape (batch, n_genes).
        expression_graph : torch.Tensor
            Expression similarity graph adjacency matrix.
        current_expression : torch.Tensor
            Current expression state of shape (batch, n_genes).
            
        Returns
        -------
        torch.Tensor
            Consistency loss value.
        """
        # Compute expected next state
        next_state = current_expression + predicted_velocity
        
        # Compute consistency with graph neighbors
        # This encourages similar cells to have similar velocity predictions
        
        # Weight matrix from graph (should be normalized)
        weights = F.normalize(expression_graph, p=1, dim=1)
        
        # Expected velocity based on neighbors
        neighbor_velocity = torch.matmul(weights, predicted_velocity)
        
        # Consistency loss
        consistency_loss = F.mse_loss(predicted_velocity, neighbor_velocity)
        
        return self.consistency_weight * consistency_loss


class RegularizationLoss(nn.Module):
    """
    Regularization losses for model parameters.
    
    This module computes various regularization terms to encourage
    sparse, biologically plausible models.
    
    Parameters
    ----------
    l1_weight : float, default 0.01
        Weight for L1 regularization.
    l2_weight : float, default 0.001
        Weight for L2 regularization.
    interaction_sparsity_weight : float, default 0.1
        Weight for encouraging sparse interaction networks.
    """
    
    def __init__(
        self,
        l1_weight: float = 0.01,
        l2_weight: float = 0.001,
        interaction_sparsity_weight: float = 0.1
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.interaction_sparsity_weight = interaction_sparsity_weight
    
    def forward(self, model) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.
        
        Parameters
        ----------
        model : nn.Module
            Model to compute regularization for.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of regularization loss components.
        """
        l1_loss = 0.0
        l2_loss = 0.0
        interaction_sparsity = 0.0
        
        for name, param in model.named_parameters():
            # L1 and L2 regularization on all parameters
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
            
            # Special handling for interaction matrices
            if 'interaction_matrix' in name:
                # Encourage sparsity in interaction networks
                interaction_sparsity += torch.sum(torch.abs(param))
        
        return {
            'l1': self.l1_weight * l1_loss,
            'l2': self.l2_weight * l2_loss,
            'interaction_sparsity': self.interaction_sparsity_weight * interaction_sparsity,
            'total': (
                self.l1_weight * l1_loss +
                self.l2_weight * l2_loss +
                self.interaction_sparsity_weight * interaction_sparsity
            )
        }


class VelocityCoherenceLoss(nn.Module):
    """
    Loss for encouraging coherent velocity patterns.
    
    This loss promotes velocity consistency by penalizing rapid changes
    in velocity direction and magnitude across similar cells.
    
    Parameters
    ----------
    magnitude_weight : float, default 1.0
        Weight for velocity magnitude consistency.
    direction_weight : float, default 1.0
        Weight for velocity direction consistency.
    """
    
    def __init__(
        self,
        magnitude_weight: float = 1.0,
        direction_weight: float = 1.0
    ):
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
    
    def forward(
        self,
        velocities: torch.Tensor,
        similarity_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute velocity coherence loss.
        
        Parameters
        ----------
        velocities : torch.Tensor
            Predicted velocities of shape (batch_size, n_genes).
        similarity_matrix : torch.Tensor, optional
            Cell-cell similarity matrix. If None, uses uniform weights.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing coherence loss components.
        """
        batch_size, n_genes = velocities.shape
        
        if similarity_matrix is None:
            # Use uniform similarity (all cells equally similar)
            similarity_matrix = torch.ones(batch_size, batch_size, device=velocities.device)
            similarity_matrix = similarity_matrix / batch_size
        
        # Magnitude coherence: similar cells should have similar velocity magnitudes
        velocity_magnitudes = torch.norm(velocities, dim=1, keepdim=True)  # (batch, 1)
        magnitude_diff = velocity_magnitudes - velocity_magnitudes.T  # (batch, batch)
        magnitude_loss = torch.sum(similarity_matrix * magnitude_diff ** 2)
        
        # Direction coherence: similar cells should have similar velocity directions
        velocity_normalized = F.normalize(velocities, p=2, dim=1)  # (batch, genes)
        cosine_similarity = torch.matmul(velocity_normalized, velocity_normalized.T)  # (batch, batch)
        direction_loss = torch.sum(similarity_matrix * (1 - cosine_similarity))
        
        # Combine losses
        total_loss = (
            self.magnitude_weight * magnitude_loss +
            self.direction_weight * direction_loss
        )
        
        return {
            'total': total_loss,
            'magnitude': magnitude_loss,
            'direction': direction_loss
        }


class RegulatoryNetworkLoss(nn.Module):
    """
    Loss functions specific to regulatory network constraints.
    
    Implements RegVelo-style losses including sparsity penalties,
    Jacobian regularization, and ATAC constraint violations.
    
    Parameters
    ----------
    sparsity_weight : float, default 1.0
        Weight for interaction sparsity penalty.
    jacobian_weight : float, default 0.1
        Weight for Jacobian regularization.
    atac_violation_weight : float, default 10.0
        Weight for ATAC constraint violation penalty.
    """
    
    def __init__(
        self,
        sparsity_weight: float = 1.0,
        jacobian_weight: float = 0.1,
        atac_violation_weight: float = 10.0
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.jacobian_weight = jacobian_weight
        self.atac_violation_weight = atac_violation_weight
    
    def forward(
        self,
        regulatory_network,
        spliced: torch.Tensor,
        atac_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regulatory network losses.
        
        Parameters
        ----------
        regulatory_network : RegulatoryNetwork
            The regulatory network module.
        spliced : torch.Tensor
            Spliced RNA counts for Jacobian computation.
        atac_mask : torch.Tensor, optional
            ATAC-seq accessibility mask.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing regulatory loss components.
        """
        losses = {}
        
        # Sparsity loss from regulatory network
        sparsity_loss = regulatory_network.get_regularization_loss()
        losses['sparsity'] = self.sparsity_weight * sparsity_loss
        
        # Jacobian regularization loss
        jacobian_loss = regulatory_network.get_jacobian_regularization_loss(
            spliced, lambda_jacobian=self.jacobian_weight
        )
        losses['jacobian'] = jacobian_loss
        
        # ATAC constraint violation loss (for soft constraints)
        if atac_mask is not None and regulatory_network.soft_constraint:
            interaction_matrix = regulatory_network.get_interaction_matrix()
            violation_mask = 1.0 - atac_mask
            violations = interaction_matrix * violation_mask
            atac_loss = self.atac_violation_weight * torch.sum(torch.abs(violations))
            losses['atac_violation'] = atac_loss
        else:
            losses['atac_violation'] = torch.tensor(0.0, device=spliced.device)
        
        # Total regulatory loss
        losses['total'] = sum(losses.values())
        
        return losses


class Stage1TotalLoss(nn.Module):
    """
    Comprehensive loss function for Stage 1 regulatory model.
    
    Combines reconstruction, regulatory network, velocity coherence,
    and general regularization losses according to configuration weights.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object containing loss weights.
    n_genes : int
        Number of genes (for loss normalization).
    """
    
    def __init__(self, config, n_genes: int):
        super().__init__()
        self.config = config
        self.n_genes = n_genes
        
        # Initialize loss components
        self.reconstruction_loss = ReconstructionLoss(
            distribution=getattr(config.loss, 'distribution', 'nb'),
            theta_init=getattr(config.loss, 'theta_init', 10.0)
        )
        
        self.regulatory_loss = RegulatoryNetworkLoss(
            sparsity_weight=getattr(config.loss, 'regulatory_sparsity_weight', 1.0),
            jacobian_weight=getattr(config.loss, 'jacobian_weight', 0.1),
            atac_violation_weight=getattr(config.loss, 'atac_violation_weight', 10.0)
        )
        
        self.velocity_coherence_loss = VelocityCoherenceLoss(
            magnitude_weight=getattr(config.loss, 'velocity_magnitude_weight', 1.0),
            direction_weight=getattr(config.loss, 'velocity_direction_weight', 1.0)
        )
        
        self.regularization_loss = RegularizationLoss(
            l1_weight=getattr(config.loss, 'l1_reg', 0.01),
            l2_weight=getattr(config.loss, 'l2_reg', 0.001),
            interaction_sparsity_weight=getattr(config.loss, 'interaction_sparsity_weight', 0.01)
        )
        
        # Loss weights from config
        self.reconstruction_weight = getattr(config.loss, 'reconstruction_weight', 1.0)
        self.regulatory_weight = getattr(config.loss, 'regulatory_weight', 0.1)
        self.velocity_coherence_weight = getattr(config.loss, 'velocity_coherence_weight', 0.1)
        self.regularization_weight = getattr(config.loss, 'regularization_weight', 0.01)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model,
        atac_mask: Optional[torch.Tensor] = None,
        similarity_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for Stage 1 model.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs containing predicted values. Expected keys:
            - 'pred_unspliced': Predicted unspliced RNA
            - 'pred_spliced': Predicted spliced RNA  
            - 'velocity': Predicted RNA velocities
            - 'transcription_rates': Transcription rates from regulatory network
        targets : Dict[str, torch.Tensor]
            Target values for supervision. Expected keys:
            - 'unspliced': Observed unspliced RNA
            - 'spliced': Observed spliced RNA
        model : nn.Module
            Model instance for regularization.
        atac_mask : torch.Tensor, optional
            ATAC-seq accessibility mask for regulatory constraints.
        similarity_matrix : torch.Tensor, optional
            Cell-cell similarity matrix for velocity coherence.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing all loss components.
        """
        all_losses = {}
        
        # 1. Reconstruction loss
        recon_losses = self.reconstruction_loss(
            pred_unspliced=outputs['pred_unspliced'],
            pred_spliced=outputs['pred_spliced'],
            obs_unspliced=targets['unspliced'],
            obs_spliced=targets['spliced']
        )
        all_losses.update({
            'reconstruction_total': recon_losses['total'],
            'reconstruction_unspliced': recon_losses['unspliced'],
            'reconstruction_spliced': recon_losses['spliced']
        })
        if 'theta' in recon_losses:
            all_losses['theta'] = recon_losses['theta']
        
        # 2. Regulatory network losses
        if hasattr(model, 'regulatory_network') and model.regulatory_network is not None:
            regulatory_losses = self.regulatory_loss(
                regulatory_network=model.regulatory_network,
                spliced=targets['spliced'],
                atac_mask=atac_mask
            )
            all_losses.update({
                'regulatory_total': regulatory_losses['total'],
                'regulatory_sparsity': regulatory_losses['sparsity'],
                'regulatory_jacobian': regulatory_losses['jacobian'],
                'regulatory_atac_violation': regulatory_losses['atac_violation']
            })
        else:
            # Zero regulatory losses if no regulatory network
            device = targets['spliced'].device
            all_losses.update({
                'regulatory_total': torch.tensor(0.0, device=device),
                'regulatory_sparsity': torch.tensor(0.0, device=device),
                'regulatory_jacobian': torch.tensor(0.0, device=device),
                'regulatory_atac_violation': torch.tensor(0.0, device=device)
            })
        
        # 3. Velocity coherence loss
        if 'velocity' in outputs:
            coherence_losses = self.velocity_coherence_loss(
                velocities=outputs['velocity'],
                similarity_matrix=similarity_matrix
            )
            all_losses.update({
                'velocity_coherence_total': coherence_losses['total'],
                'velocity_coherence_magnitude': coherence_losses['magnitude'],
                'velocity_coherence_direction': coherence_losses['direction']
            })
        else:
            # Zero coherence losses if no velocity output
            device = targets['spliced'].device
            all_losses.update({
                'velocity_coherence_total': torch.tensor(0.0, device=device),
                'velocity_coherence_magnitude': torch.tensor(0.0, device=device),
                'velocity_coherence_direction': torch.tensor(0.0, device=device)
            })
        
        # 4. General regularization losses
        reg_losses = self.regularization_loss(model)
        all_losses.update({
            'regularization_total': reg_losses['total'],
            'regularization_l1': reg_losses['l1'],
            'regularization_l2': reg_losses['l2'],
            'regularization_interaction_sparsity': reg_losses['interaction_sparsity']
        })
        
        # 5. Compute weighted total loss
        total_loss = (
            self.reconstruction_weight * all_losses['reconstruction_total'] +
            self.regulatory_weight * all_losses['regulatory_total'] +
            self.velocity_coherence_weight * all_losses['velocity_coherence_total'] +
            self.regularization_weight * all_losses['regularization_total']
        )
        
        all_losses['total'] = total_loss
        
        return all_losses


# Utility functions for loss computation

def compute_kl_divergence_nb(
    pred_mean: torch.Tensor,
    pred_theta: torch.Tensor,
    target_mean: torch.Tensor,
    target_theta: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute KL divergence between two Negative Binomial distributions.
    
    Parameters
    ----------
    pred_mean : torch.Tensor
        Predicted means.
    pred_theta : torch.Tensor
        Predicted overdispersion parameters.
    target_mean : torch.Tensor
        Target means.
    target_theta : torch.Tensor
        Target overdispersion parameters.
    eps : float, default 1e-8
        Small constant for numerical stability.
        
    Returns
    -------
    torch.Tensor
        KL divergence values.
    """
    # Ensure positive parameters
    pred_mean = torch.clamp(pred_mean, min=eps)
    target_mean = torch.clamp(target_mean, min=eps)
    pred_theta = torch.clamp(pred_theta, min=eps)
    target_theta = torch.clamp(target_theta, min=eps)
    
    # Convert to success probability parameterization
    p1 = pred_theta / (pred_theta + pred_mean)
    p2 = target_theta / (target_theta + target_mean)
    
    # Create distributions
    dist1 = NegativeBinomial(total_count=pred_theta, probs=p1)
    dist2 = NegativeBinomial(total_count=target_theta, probs=p2)
    
    # Compute KL divergence using built-in function
    return kl(dist1, dist2)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked MSE loss.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values.
    mask : torch.Tensor
        Binary mask (1 for valid, 0 for invalid).
        
    Returns
    -------
    torch.Tensor
        Masked MSE loss.
    """
    squared_diff = (pred - target) ** 2
    masked_diff = squared_diff * mask
    
    # Average over valid entries only
    valid_count = torch.clamp(mask.sum(), min=1.0)
    return masked_diff.sum() / valid_count


# Stage 2 specific loss functions

class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for variational inference.
    
    Combines reconstruction loss with KL divergence regularization
    for proper variational learning of latent representations.
    
    Parameters
    ----------
    kl_weight : float, default 1.0
        Weight for KL divergence term (beta in beta-VAE).
    """
    
    def __init__(self, kl_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight
    
    def forward(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mean: torch.Tensor, 
        log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO loss.
        
        Parameters
        ----------
        recon_x : torch.Tensor
            Reconstructed data of shape (batch_size, features).
        x : torch.Tensor
            Original data of shape (batch_size, features).
        mean : torch.Tensor
            Latent means of shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Latent log-variances of shape (batch_size, latent_dim).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Total ELBO loss, reconstruction loss, and KL loss.
        """
        # Reconstruction loss (negative log-likelihood)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss (assume standard normal prior)
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        # Total ELBO (negative, since we minimize)
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


class TangentSpaceLoss(nn.Module):
    """
    Tangent space loss ensuring velocity vectors align with data manifold.
    
    Uses expression latent variables to define local manifold structure
    and constrains velocity predictions to be biologically plausible.
    
    Parameters
    ----------
    n_neighbors : int, default 30
        Number of neighbors for manifold structure computation.
    """
    
    def __init__(self, n_neighbors: int = 30):
        super().__init__()
        self.n_neighbors = n_neighbors
        
        # Network to derive manifold coefficients from expression latent
        # This network learns to map expression latent variables to 
        # coefficients that define the local tangent space
        self.coefficient_net = nn.Sequential(
            nn.Linear(64, 128),  # Assuming latent_dim=64, configurable
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_neighbors),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        self.apply(lambda m: initialize_weights(m, "xavier_uniform"))
    
    def forward(
        self, 
        velocity: torch.Tensor, 
        expression_latent: torch.Tensor,
        neighbor_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute tangent space alignment loss.
        
        Parameters
        ----------
        velocity : torch.Tensor
            Predicted velocity vectors of shape (n_cells, n_genes).
        expression_latent : torch.Tensor
            Expression latent variables of shape (n_cells, latent_dim).
        neighbor_indices : torch.Tensor, optional
            Indices of k-nearest neighbors for each cell.
            
        Returns
        -------
        torch.Tensor
            Tangent space alignment loss.
        """
        # Adapt coefficient network to expression latent dimension
        if expression_latent.shape[1] != 64:
            # Dynamically adjust first layer if needed
            if not hasattr(self, '_adapted_net'):
                latent_dim = expression_latent.shape[1]
                self.coefficient_net[0] = nn.Linear(latent_dim, 128)
                self._adapted_net = True
        
        # Compute manifold coefficients from expression latent
        phi_coeffs = self.coefficient_net(expression_latent)
        
        # Simplified tangent space loss implementation
        # In a full implementation, this would use neighbor dynamics
        # and local regression similar to GraphVelo
        
        # For now, implement a simplified version that encourages
        # velocity consistency with expression-based weights
        batch_size, n_genes = velocity.shape
        
        # Compute pairwise velocity differences
        velocity_expanded = velocity.unsqueeze(1)  # (batch, 1, genes)
        velocity_diff = velocity_expanded - velocity.unsqueeze(0)  # (batch, batch, genes)
        
        # Weight by expression-based coefficients
        # This encourages similar expression states to have similar velocities
        expression_similarity = torch.matmul(expression_latent, expression_latent.t())
        expression_similarity = F.softmax(expression_similarity, dim=1)
        
        # Weighted velocity consistency loss
        weighted_diff = velocity_diff.pow(2).sum(dim=2)  # (batch, batch)
        tangent_loss = torch.sum(expression_similarity * weighted_diff)
        
        # Normalize by batch size
        tangent_loss = tangent_loss / (batch_size * batch_size)
        
        return tangent_loss


class Stage2TotalLoss(nn.Module):
    """
    Combined loss function for Stage 2 model training.
    
    Integrates ELBO loss, tangent space loss, and existing velocity losses
    with configurable weights for multi-objective optimization.
    
    Parameters
    ----------
    config : object
        Configuration object containing loss weights and parameters.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize Stage 2 specific loss components
        self.elbo_loss = ELBOLoss(
            kl_weight=getattr(config.loss, 'elbo_weight', 0.01)
        )
        
        self.tangent_loss = TangentSpaceLoss(
            n_neighbors=getattr(config.graph, 'n_neighbors_expression', 30)
        )
        
        # Reuse existing loss components
        self.recon_loss = ReconstructionLoss(
            distribution=getattr(config.loss, 'distribution', 'nb')
        )
        
        self.velocity_consistency_loss = VelocityConsistencyLoss(
            consistency_weight=getattr(config.loss, 'velocity_consistency_weight', 0.1)
        )
        
        # Loss weights from config
        self.weights = {
            'elbo': getattr(config.loss, 'elbo_weight', 0.01),
            'tangent': getattr(config.loss, 'tangent_space_weight', 0.1),
            'reconstruction': getattr(config.loss, 'reconstruction_weight', 1.0),
            'velocity': getattr(config.loss, 'velocity_consistency_weight', 0.1)
        }
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Stage 2 total loss.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs containing:
            - 'velocity': Predicted velocities
            - 'latent_mean': Latent means
            - 'latent_log_var': Latent log-variances
            - 'expression_latent': Expression latent variables
        targets : Dict[str, torch.Tensor]
            Target values containing:
            - 'velocity': Target velocities (if available)
            - 'spliced': Observed spliced RNA
            - 'unspliced': Observed unspliced RNA
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Total loss and dictionary of loss components.
        """
        total_loss = 0
        loss_components = {}
        
        # 1. ELBO loss for variational inference
        if all(key in outputs for key in ['latent_mean', 'latent_log_var']):
            # Use velocity as reconstruction target
            velocity_pred = outputs['velocity']
            velocity_target = targets.get('velocity', velocity_pred.detach())
            
            elbo_total, recon_loss, kl_loss = self.elbo_loss(
                velocity_pred, velocity_target,
                outputs['latent_mean'], outputs['latent_log_var']
            )
            
            total_loss += self.weights['elbo'] * elbo_total
            loss_components.update({
                'elbo_total': elbo_total,
                'elbo_reconstruction': recon_loss,
                'elbo_kl': kl_loss
            })
        
        # 2. Tangent space loss for manifold alignment
        if 'expression_latent' in outputs:
            tangent_loss = self.tangent_loss(
                outputs['velocity'], outputs['expression_latent']
            )
            total_loss += self.weights['tangent'] * tangent_loss
            loss_components['tangent_space'] = tangent_loss
        
        # 3. Standard reconstruction loss on RNA counts
        if all(key in targets for key in ['spliced', 'unspliced']):
            # Reconstruct RNA from velocity predictions
            # This is a simplified reconstruction - full implementation
            # would integrate ODE dynamics
            recon_losses = self.recon_loss(
                pred_unspliced=targets['unspliced'] + outputs['velocity'],  # Simplified
                pred_spliced=targets['spliced'],
                obs_unspliced=targets['unspliced'],
                obs_spliced=targets['spliced']
            )
            total_loss += self.weights['reconstruction'] * recon_losses['total']
            loss_components['reconstruction'] = recon_losses['total']
        
        # 4. Velocity consistency loss (if available)
        if 'velocity' in outputs and 'expression_graph' in targets:
            velocity_loss = self.velocity_consistency_loss(
                predicted_velocity=outputs['velocity'],
                expression_graph=targets['expression_graph'],
                current_expression=targets['spliced']
            )
            total_loss += self.weights['velocity'] * velocity_loss
            loss_components['velocity_consistency'] = velocity_loss
        
        loss_components['total'] = total_loss
        
        return total_loss, loss_components