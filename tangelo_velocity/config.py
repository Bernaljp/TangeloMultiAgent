"""Configuration classes for Tangelo Velocity."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Tuple
import yaml
import json


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    # Spatial graph parameters
    n_neighbors_spatial: int = 8
    spatial_method: str = "knn"  # "knn" or "radius"
    spatial_radius: Optional[float] = None
    
    # Expression graph parameters  
    n_neighbors_expression: int = 15
    expression_method: str = "knn"
    expression_metric: str = "cosine"
    
    # Node2vec preprocessing (optional)
    use_node2vec: bool = False
    node2vec_dim: int = 128
    node2vec_walk_length: int = 80
    node2vec_num_walks: int = 10
    node2vec_p: float = 1.0
    node2vec_q: float = 1.0


@dataclass
class EncoderConfig:
    """Configuration for GraphSAGE encoders."""
    # Architecture
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    latent_dim: int = 32
    dropout: float = 0.1
    batch_norm: bool = True
    activation: str = "relu"
    
    # GraphSAGE specific
    aggregator: str = "mean"  # "mean", "max", "lstm"
    num_layers: int = 3
    
    # Fusion strategy
    fusion_method: str = "sum"  # "sum", "concat", "attention"
    
    # Spatial features
    spatial_feature_dim: int = 2


@dataclass
class RegulatoryConfig:
    """Configuration for regulatory network."""
    # ATAC masking
    use_atac_masking: bool = True
    atac_threshold: float = 0.1
    
    # Interaction network
    use_bias: bool = False
    interaction_strength: float = 1.0
    
    # Sigmoid features
    use_sigmoid_features: bool = True
    n_sigmoid_components: int = 10
    sigmoid_init_range: float = 1.0
    
    # Base transcription
    base_transcription: float = 0.1
    
    # Initialization parameters
    linear_init_std: float = 0.1
    
    # Regularization
    soft_constraint: bool = True
    lambda_l1: float = 0.0
    lambda_l2: float = 1e-4


@dataclass
class ODEConfig:
    """Configuration for ODE integration."""
    # Solver parameters
    solver: str = "dopri5"  # torchode solver
    rtol: float = 1e-5
    atol: float = 1e-7
    max_steps: int = 1000
    
    # Time parameters
    t_span: Tuple[float, float] = (0.0, 1.0)
    n_time_points: int = 50
    
    # Parameter initialization
    init_beta_range: Tuple[float, float] = (0.1, 2.0)
    init_gamma_range: Tuple[float, float] = (0.1, 1.0)
    init_time_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class MultiscaleConfig:
    """Configuration for multiscale training."""
    # Multiscale training settings
    enable_multiscale: bool = False
    multiscale_weights: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    min_scale_size: int = 1
    max_scales: int = 4
    scale_strategy: str = "geometric"  # "geometric", "linear", "custom"
    random_seed: Optional[int] = None


@dataclass
class Stage4Config:
    """Configuration for Stage 4 advanced features."""
    # Temporal dynamics
    temporal_n_time_points: int = 10
    temporal_prediction_horizon: float = 2.0
    
    # Uncertainty quantification
    uncertainty_samples: int = 100
    uncertainty_method: str = "dropout"  # "dropout", "ensemble", "bayesian"
    
    # Multi-scale integration
    n_cell_types: int = 10
    multiscale_method: str = "hierarchical"  # "hierarchical", "attention"
    
    # Interpretability
    interpretability_top_k: int = 50
    feature_importance_method: str = "integrated_gradients"  # "integrated_gradients", "attention"
    
    # Advanced regularization
    pathway_regularization: float = 0.01
    sparsity_regularization: float = 0.001


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # Loss weights
    reconstruction_weight: float = 1.0
    tangent_space_weight: float = 0.1
    elbo_weight: float = 0.01
    
    # Multi-level batching (legacy - replaced by multiscale)
    use_hierarchical_loss: bool = False
    batch_levels: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    level_weights: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 1e-4


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    
    # Training schedule
    n_epochs: int = 100
    batch_size: int = 512
    validation_split: float = 0.1
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Logging
    log_interval: int = 10
    save_interval: int = 25


@dataclass
class TangeloConfig:
    """Main configuration class for Tangelo Velocity."""
    
    # Stage configuration
    development_stage: int = 0  # 0, 1, 2, 3, 4
    
    # Component configurations
    graph: GraphConfig = field(default_factory=GraphConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    regulatory: RegulatoryConfig = field(default_factory=RegulatoryConfig)
    ode: ODEConfig = field(default_factory=ODEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    
    # Data configuration
    gene_dim: Optional[int] = None  # Inferred from data
    atac_dim: Optional[int] = None  # Inferred from data
    spatial_dim: int = 2
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TangeloConfig":
        """Create config from dictionary."""
        # Extract nested configurations
        graph_config = GraphConfig(**config_dict.get("graph", {}))
        encoder_config = EncoderConfig(**config_dict.get("encoder", {}))
        regulatory_config = RegulatoryConfig(**config_dict.get("regulatory", {}))
        ode_config = ODEConfig(**config_dict.get("ode", {}))
        loss_config = LossConfig(**config_dict.get("loss", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        multiscale_config = MultiscaleConfig(**config_dict.get("multiscale", {}))
        stage4_config = Stage4Config(**config_dict.get("stage4", {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["graph", "encoder", "regulatory", "ode", "loss", "training", "multiscale", "stage4"]}
        
        return cls(
            graph=graph_config,
            encoder=encoder_config,
            regulatory=regulatory_config,
            ode=ode_config,
            loss=loss_config,
            training=training_config,
            multiscale=multiscale_config,
            stage4=stage4_config,
            **main_config
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "TangeloConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> "TangeloConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "development_stage": self.development_stage,
            "gene_dim": self.gene_dim,
            "atac_dim": self.atac_dim,
            "spatial_dim": self.spatial_dim,
            "device": self.device,
            "graph": self.graph.__dict__,
            "encoder": self.encoder.__dict__,
            "regulatory": self.regulatory.__dict__,
            "ode": self.ode.__dict__,
            "loss": self.loss.__dict__,
            "training": self.training.__dict__,
            "multiscale": self.multiscale.__dict__,
        }
    
    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Stage-specific default configurations
def get_stage_config(stage: int) -> TangeloConfig:
    """Get default configuration for specific development stage."""
    
    if stage == 0:
        # Stage 0: Foundation - basic preprocessing and graph construction
        return TangeloConfig(
            development_stage=0,
            graph=GraphConfig(
                n_neighbors_spatial=8,
                n_neighbors_expression=15,
                use_node2vec=False,
            ),
            training=TrainingConfig(
                n_epochs=50,
                batch_size=256,
            )
        )
    
    elif stage == 1:
        # Stage 1: Regulatory model MVP
        return TangeloConfig(
            development_stage=1,
            regulatory=RegulatoryConfig(
                use_atac_masking=True,
                use_sigmoid_features=True,
            ),
            loss=LossConfig(
                reconstruction_weight=1.0,
                tangent_space_weight=0.0,  # Not used yet
                elbo_weight=0.0,  # Not used yet
            ),
            training=TrainingConfig(
                n_epochs=100,
                learning_rate=1e-3,
            )
        )
    
    elif stage == 2:
        # Stage 2: Graph model MVP
        return TangeloConfig(
            development_stage=2,
            encoder=EncoderConfig(
                hidden_dims=(256, 128, 64),
                latent_dim=32,
                fusion_method="sum",
            ),
            loss=LossConfig(
                reconstruction_weight=1.0,
                tangent_space_weight=0.1,
                elbo_weight=0.01,
            ),
            training=TrainingConfig(
                n_epochs=150,
                learning_rate=1e-3,
            )
        )
    
    elif stage == 3:
        # Stage 3: Integrated model
        return TangeloConfig(
            development_stage=3,
            encoder=EncoderConfig(
                hidden_dims=(512, 256, 128),
                latent_dim=64,
                fusion_method="attention",
            ),
            regulatory=RegulatoryConfig(
                use_atac_masking=True,
                use_sigmoid_features=True,
            ),
            loss=LossConfig(
                reconstruction_weight=1.0,
                tangent_space_weight=0.2,
                elbo_weight=0.05,
                use_hierarchical_loss=False,  # Not yet
            ),
            training=TrainingConfig(
                n_epochs=200,
                learning_rate=5e-4,
                batch_size=1024,
            )
        )
    
    elif stage == 4:
        # Stage 4: Advanced features
        return TangeloConfig(
            development_stage=4,
            encoder=EncoderConfig(
                hidden_dims=(512, 256, 128),
                latent_dim=64,
                fusion_method="attention",
            ),
            regulatory=RegulatoryConfig(
                use_atac_masking=True,
                use_sigmoid_features=True,
            ),
            loss=LossConfig(
                reconstruction_weight=1.0,
                tangent_space_weight=0.2,
                elbo_weight=0.05,
                use_hierarchical_loss=True,  # Legacy - kept for compatibility
                batch_levels=(1.0, 0.5, 0.25, 0.125),
                level_weights=(0.4, 0.3, 0.2, 0.1),
            ),
            multiscale=MultiscaleConfig(
                enable_multiscale=True,
                multiscale_weights=(0.4, 0.3, 0.2, 0.1),
                min_scale_size=1,
                max_scales=4,
                scale_strategy="geometric",
                random_seed=42,
            ),
            training=TrainingConfig(
                n_epochs=300,
                learning_rate=1e-4,
                batch_size=1024,
            )
        )
    
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 0, 1, 2, 3, or 4.")