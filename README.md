# Tangelo Velocity
a
A novel computational method for multi-modal single-cell RNA velocity estimation that integrates spatial transcriptomics, RNA velocity, and ATAC-seq data using graph neural networks and ordinary differential equation modeling.

## Overview

Tangelo Velocity addresses key limitations in current velocity estimation methods by:

- **Multi-modal Integration**: Seamlessly combines RNA-seq, ATAC-seq, and spatial transcriptomics data
- **Graph-based Architecture**: Uses dual GraphSAGE encoders for spatial and expression relationships
- **Regulatory Networks**: Incorporates chromatin accessibility to constrain gene-gene interactions
- **Cell-specific Dynamics**: Predicts individual cell parameters rather than global averages
- **Biologically Plausible Velocities**: Ensures velocity vectors lie in the data manifold tangent space

## Key Features

### Dual GraphSAGE Architecture
- **Spatial Encoder**: Learns from spatial neighborhood relationships
- **Expression Encoder**: Captures gene expression similarities
- **Fusion Strategies**: Multiple ways to combine spatial and expression information

### Regulatory Network Integration
- **ATAC-seq Informed**: Uses chromatin accessibility to mask gene interactions
- **Interaction Networks**: Learns regulatory relationships between genes
- **Sigmoid Features**: Learnable feature transformations for complex dynamics

### Advanced ODE Modeling
- **TorchODE Integration**: High-performance differential equation solving
- **Cell-specific Parameters**: Individual splicing/degradation rates per cell
- **Hierarchical Loss**: Multi-level parameter estimation for robustness

### Comprehensive Analysis Tools
- **Velocity Metrics**: Quantitative evaluation of velocity quality
- **Trajectory Analysis**: Cell fate and transition analysis
- **Perturbation Studies**: In-silico gene knockdown experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tangelo-velocity.git
cd tangelo-velocity

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,docs,examples]"
```

## Quick Start

```python
import tangelo_velocity as tv
import muon as mu

# Load your multi-modal data (MuData with 'rna' and 'atac' modalities)
adata = mu.read_h5mu("your_data.h5mu")

# Basic velocity estimation (Stage 3 - Integrated Model)
tv.estimate_velocity(adata)

# Or use the full API for more control
model = tv.TangeloVelocity(stage=3)
model.fit(adata)

# Compute velocity graph and embedding
model.compute_velocity_graph(adata)
model.compute_velocity_embedding(adata, basis="umap")

# Visualize results
tv.plotting.plot_velocity_embedding(adata, basis="umap")
tv.plotting.plot_spatial_velocity(adata)
```

## Input Data Format

Tangelo Velocity expects a `MuData` object with the following structure:

```python
MuData object with n_obs × n_vars
  obs: 'x_pixel', 'y_pixel', 'x_position', 'y_position'
  2 modalities:
    rna: n_cells x n_genes
      layers: 'spliced', 'unspliced', 'open_chromatin'
      obsm: 'X_pca' (optional)
    atac: n_cells x n_peaks  
      layers: 'counts', 'X_tfidf'
      obsm: 'X_lsi' (optional)
```

## Development Stages

Tangelo Velocity is designed with a staged development approach:

### Stage 0: Foundation
- Data preprocessing and graph construction
- Basic utilities and configuration system

### Stage 1: Regulatory Model (MVP)
- Linear interaction network with ATAC masking
- Basic ODE integration with TorchODE
- Reconstruction loss optimization

### Stage 2: Graph Model (MVP)  
- Dual GraphSAGE encoders
- Latent space fusion strategies
- Tangent space loss for manifold constraints

### Stage 3: Integrated Model
- Combined regulatory and graph architectures
- Cell-specific ODE parameter prediction
- Multi-component loss optimization

### Stage 4: Advanced Features
- Hierarchical batch parameter estimation
- Cell-cell interaction terms
- Dynamic regulatory networks
- Advanced trajectory analysis

## Configuration

Use YAML or programmatic configuration for reproducible experiments:

```python
# Programmatic configuration
config = tv.TangeloConfig(
    development_stage=3,
    graph=tv.config.GraphConfig(
        n_neighbors_spatial=8,
        n_neighbors_expression=15,
    ),
    encoder=tv.config.EncoderConfig(
        latent_dim=64,
        fusion_method="attention",
    ),
    training=tv.config.TrainingConfig(
        n_epochs=200,
        learning_rate=1e-3,
    )
)

# YAML configuration
config = tv.TangeloConfig.from_yaml("config.yaml")
```

## Advanced Usage

### Compare Different Stages
```python
# Compare velocity estimates across development stages
results = tv.compare_stages(adata, stages=(1, 2, 3))

# Analyze differences
for stage, result in results.items():
    print(f"Stage {stage} velocity quality:")
    metrics = tv.analysis.VelocityMetrics(result)
    print(metrics.summary())
```

### Extract Model Components
```python
# Get learned representations
latent_reps = model.get_latent_representations()
spatial_latent = latent_reps["spatial_latent"]
expression_latent = latent_reps["expression_latent"]

# Get cell-specific ODE parameters
ode_params = model.get_ode_parameters()
splicing_rates = ode_params["beta"]
degradation_rates = ode_params["gamma"]

# Get gene interaction network
interaction_matrix = model.get_interaction_network()
```

### Perturbation Analysis
```python
# Simulate gene knockdown
perturbation = tv.analysis.PerturbationAnalysis(model)
knockdown_result = perturbation.simulate_knockdown(
    gene="target_gene",
    knockdown_strength=0.8
)

# Analyze cell fate changes
fate_changes = perturbation.analyze_fate_changes(knockdown_result)
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 1.12.0
- torch-geometric ≥ 2.0.0
- torchode ≥ 0.2.0
- scanpy ≥ 1.8.0
- muon ≥ 0.1.0
- anndata ≥ 0.8.0

## Documentation

- [API Reference](docs/api/)
- [User Guide](docs/user_guide/)
- [Tutorials](docs/tutorials/)
- [Development Guide](docs/development/)

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## Citation

If you use Tangelo Velocity in your research, please cite:

```bibtex
@software{tangelo_velocity2024,
  title={Tangelo Velocity: Multi-modal Single-Cell Velocity Estimation},
  author={Tangelo Velocity Team},
  year={2024},
  url={https://github.com/yourusername/tangelo-velocity}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
