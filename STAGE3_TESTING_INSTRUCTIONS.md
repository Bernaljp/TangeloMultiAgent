# Stage 3 Integrated Model Testing Instructions

## Overview
This document provides comprehensive instructions for testing the Stage 3 integrated model that combines regulatory networks (Stage 1) and graph neural networks (Stage 2) into a unified multi-modal velocity estimation system.

## Prerequisites

### Required Dependencies
```bash
# Core dependencies
pip install torch torch-geometric
pip install scanpy muon
pip install numpy pandas matplotlib seaborn
pip install torchdiffeq  # For ODE solving
```

### Data Requirements
The Stage 3 model expects MuData objects with the following structure:
```python
adata = mu.MuData({
    'rna': adata_rna,    # Shape: (n_cells, n_genes)
    'atac': adata_atac   # Shape: (n_cells, n_atac_features)
})

# Required RNA layers
adata['rna'].layers['spliced']    # Spliced RNA counts
adata['rna'].layers['unspliced']  # Unspliced RNA counts

# Required spatial coordinates
adata.obs['x_pixel']  # X spatial coordinates
adata.obs['y_pixel']  # Y spatial coordinates
```

## Testing Scenarios

### 1. Basic Model Initialization
```python
import tangelo_velocity as tv

# Test Stage 3 model creation
config = tv.TangeloConfig(development_stage=3)
config.gene_dim = 1000  # Adjust to your data
config.atac_dim = 5000  # Adjust to your data

model = tv.models.get_velocity_model(config, gene_dim=1000, atac_dim=5000)
print(f"Model type: {type(model)}")
print(f"Stage: {model.development_stage}")
```

### 2. Data Loading and Validation
```python
# Load your data
adata = mu.read_h5mu("your_data.h5mu")

# Validate data structure
def validate_stage3_data(adata):
    """Validate data meets Stage 3 requirements."""
    errors = []
    
    # Check modalities
    if 'rna' not in adata.mod:
        errors.append("Missing 'rna' modality")
    if 'atac' not in adata.mod:
        errors.append("Missing 'atac' modality")
    
    # Check RNA layers
    if 'rna' in adata.mod:
        rna = adata['rna']
        if 'spliced' not in rna.layers:
            errors.append("Missing 'spliced' layer in RNA")
        if 'unspliced' not in rna.layers:
            errors.append("Missing 'unspliced' layer in RNA")
    
    # Check spatial coordinates
    if 'x_pixel' not in adata.obs.columns:
        errors.append("Missing 'x_pixel' coordinates")
    if 'y_pixel' not in adata.obs.columns:
        errors.append("Missing 'y_pixel' coordinates")
    
    return errors

errors = validate_stage3_data(adata)
if errors:
    print("Data validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Data validation passed!")
```

### 3. Preprocessing Test
```python
# Test preprocessing pipeline
from tangelo_velocity.preprocessing import MuDataProcessor

config = tv.TangeloConfig(development_stage=3)
processor = MuDataProcessor(config)

# Process data
processed_data = processor.process_mudata(adata)

# Validate processed outputs
expected_keys = [
    'spliced', 'unspliced', 'atac_mask',
    'spatial_graph', 'expression_graph', 'spatial_coords'
]

for key in expected_keys:
    if key in processed_data:
        print(f"✓ {key}: {processed_data[key].shape}")
    else:
        print(f"✗ Missing: {key}")
```

### 4. Model Forward Pass Test
```python
import torch

# Create test tensors
n_cells, n_genes = 100, 500
n_atac = 1000

spliced = torch.randn(n_cells, n_genes)
unspliced = torch.randn(n_cells, n_genes)
spatial_coords = torch.randn(n_cells, 2)

# Create dummy graphs (for testing)
from torch_geometric.data import Data
edge_index = torch.randint(0, n_cells, (2, n_cells * 5))  # 5 edges per node
spatial_graph = Data(edge_index=edge_index, num_nodes=n_cells)
expression_graph = Data(edge_index=edge_index, num_nodes=n_cells)

# Test forward pass
model.eval()
with torch.no_grad():
    outputs = model(
        spliced=spliced,
        unspliced=unspliced,
        spatial_graph=spatial_graph,
        expression_graph=expression_graph,
        spatial_coords=spatial_coords
    )

# Validate outputs
expected_outputs = [
    'velocity', 'regulatory_alpha', 'graph_features',
    'fused_alpha', 'ode_parameters'
]

print("Forward pass outputs:")
for key in expected_outputs:
    if key in outputs:
        print(f"✓ {key}: {outputs[key].shape}")
    else:
        print(f"✗ Missing: {key}")
```

### 5. Training Test
```python
# Test training loop
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create training data
training_data = {
    'spliced': spliced,
    'unspliced': unspliced,
    'spatial_graph': spatial_graph,
    'expression_graph': expression_graph,
    'spatial_coords': spatial_coords
}

targets = {
    'spliced': spliced,
    'unspliced': unspliced
}

# Training step
def training_step(model, training_data, targets):
    optimizer.zero_grad()
    
    outputs = model(**training_data)
    loss = model.compute_loss(outputs, targets)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Test a few training steps
print("Training test:")
for epoch in range(5):
    loss = training_step(model, training_data, targets)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
```

### 6. Integration Analysis Test
```python
# Test component integration analysis
def test_integration_analysis(model, training_data):
    """Test the integration of regulatory and graph components."""
    model.eval()
    with torch.no_grad():
        outputs = model(**training_data)
    
    # Check regulatory component
    if hasattr(model, 'get_regulatory_contribution'):
        reg_contrib = model.get_regulatory_contribution()
        print(f"Regulatory contribution shape: {reg_contrib.shape}")
    
    # Check graph component
    if hasattr(model, 'get_graph_contribution'):
        graph_contrib = model.get_graph_contribution()
        print(f"Graph contribution shape: {graph_contrib.shape}")
    
    # Check fusion weights
    if hasattr(model, 'get_fusion_weights'):
        fusion_weights = model.get_fusion_weights()
        print(f"Fusion weights: {fusion_weights}")
    
    return outputs

outputs = test_integration_analysis(model, training_data)
```

### 7. High-Level API Test
```python
# Test the high-level convenience function
import tangelo_velocity as tv

# Load data
adata = mu.read_h5mu("your_data.h5mu")

# Run Stage 3 velocity estimation
try:
    tv.tools.estimate_velocity(adata, stage=3, copy=False)
    print("✓ High-level API test passed")
    
    # Check results
    if 'velocity' in adata['rna'].layers:
        velocity_shape = adata['rna'].layers['velocity'].shape
        print(f"✓ Velocity computed: {velocity_shape}")
    else:
        print("✗ Velocity layer not found")
        
except Exception as e:
    print(f"✗ High-level API test failed: {e}")
```

### 8. Comparison with Previous Stages
```python
# Compare Stage 3 with Stages 1 and 2
def compare_stages(adata_subset):
    """Compare velocity estimates across stages."""
    results = {}
    
    for stage in [1, 2, 3]:
        try:
            adata_copy = adata_subset.copy()
            tv.tools.estimate_velocity(adata_copy, stage=stage)
            results[f'stage_{stage}'] = adata_copy['rna'].layers['velocity']
            print(f"✓ Stage {stage} completed")
        except Exception as e:
            print(f"✗ Stage {stage} failed: {e}")
            results[f'stage_{stage}'] = None
    
    return results

# Use subset for faster testing
adata_subset = adata[:100, :].copy()  # First 100 cells
stage_results = compare_stages(adata_subset)
```

## Expected Outputs

### Model Components
- **Regulatory Network**: Sigmoid feature transformation with ATAC masking
- **Graph Encoders**: Spatial and expression GraphSAGE encoders
- **Fusion Module**: Integration of regulatory and graph predictions
- **ODE System**: Enhanced velocity computation using integrated features

### Velocity Outputs
- **Shape**: (n_cells, n_genes) matching input RNA data
- **Range**: Biologically plausible velocity values
- **Spatial Coherence**: Nearby cells should have similar velocities
- **Integration Balance**: Contribution from both regulatory and graph components

### Loss Components
- **Reconstruction Loss**: RNA abundance prediction accuracy
- **Regulatory Loss**: ATAC-constrained interaction penalties
- **Graph Loss**: ELBO and tangent space losses
- **Fusion Loss**: Integration coherence penalties

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or use gradient checkpointing
2. **Graph Construction Failures**: Check spatial coordinates and k-NN parameters
3. **ATAC Masking Issues**: Verify ATAC-RNA feature mapping
4. **Convergence Problems**: Adjust learning rates and loss weights

### Performance Benchmarks
- **Training Time**: ~2-5 minutes per epoch for 1000 cells, 500 genes
- **Memory Usage**: ~2-4 GB for typical datasets
- **Convergence**: Loss should stabilize within 50-100 epochs

### Validation Metrics
- **Reconstruction R²**: Should be > 0.7 for spliced RNA
- **Velocity Coherence**: Spatial correlation > 0.5
- **Integration Score**: Balanced contribution (0.3-0.7 each component)

## Success Criteria

A successful Stage 3 implementation should:
1. ✅ Initialize without errors
2. ✅ Process multi-modal data correctly
3. ✅ Generate biologically plausible velocity estimates
4. ✅ Show improved performance over individual stages
5. ✅ Demonstrate proper integration of regulatory and graph components
6. ✅ Maintain computational efficiency
7. ✅ Provide interpretable component contributions

## Next Steps

After successful testing:
1. Run on your full dataset
2. Compare with existing velocity methods
3. Analyze biological interpretability
4. Optimize hyperparameters for your specific use case
5. Consider Stage 4 advanced features if needed