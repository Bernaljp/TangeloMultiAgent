# Stage 4 Advanced Features Testing Instructions

## Overview
This document provides comprehensive testing instructions for Stage 4 advanced features, with critical focus on the correct mathematical formulation and sigmoid pretraining protocol.

## Critical Mathematical Requirements

### Velocity Model Formulation
**ESSENTIAL**: α = W @ sigmoid(s) where:
- W is the interaction network matrix (gene-gene interactions)
- sigmoid(s) is the sigmoid-transformed spliced counts
- @ denotes matrix multiplication
- This formulation must be consistent across ALL stages

### Sigmoid Pretraining Protocol
**ESSENTIAL**: 
- Sigmoid function parameters are pretrained first
- After pretraining, sigmoid parameters are FROZEN (no gradient updates)
- Only interaction network W continues training
- This ensures stable feature representations

## Prerequisites

### Required Dependencies
```bash
# Core dependencies
pip install torch torch-geometric
pip install scanpy muon
pip install numpy pandas matplotlib seaborn
pip install torchdiffeq
pip install scikit-learn  # For uncertainty quantification
pip install scipy  # For statistical tests
```

## Testing Framework

### 1. Mathematical Formulation Validation
```python
import torch
import tangelo_velocity as tv

def test_alpha_formulation():
    """Test that α = W @ sigmoid(s) is correctly implemented."""
    
    # Create test data
    n_cells, n_genes = 100, 50
    spliced = torch.randn(n_cells, n_genes)
    
    # Test across all stages
    for stage in [1, 2, 3, 4]:
        print(f"\n=== Testing Stage {stage} ===")
        
        config = tv.TangeloConfig(development_stage=stage)
        model = tv.models.get_velocity_model(config, gene_dim=n_genes, atac_dim=100)
        
        # Access internal components
        if hasattr(model, 'regulatory_network'):
            reg_net = model.regulatory_network
            
            # Get sigmoid transformation
            sigmoid_s = reg_net.sigmoid_features(spliced)  # Should be (n_cells, n_genes)
            print(f"Sigmoid features shape: {sigmoid_s.shape}")
            
            # Get interaction matrix
            W = reg_net.get_interaction_matrix()  # Should be (n_genes, n_genes)
            print(f"Interaction matrix W shape: {W.shape}")
            
            # Compute α = W @ sigmoid(s)^T, then transpose back
            # Note: pytorch uses @ for matrix multiplication
            alpha = (W @ sigmoid_s.T).T  # Shape: (n_cells, n_genes)
            print(f"Alpha shape: {alpha.shape}")
            
            # Verify this matches model output
            model_alpha = reg_net(spliced)
            print(f"Model alpha shape: {model_alpha.shape}")
            
            # Test mathematical consistency
            diff = torch.abs(alpha - model_alpha).max()
            print(f"Max difference between manual and model alpha: {diff:.6f}")
            
            if diff < 1e-5:
                print("✓ Mathematical formulation CORRECT: α = W @ sigmoid(s)")
            else:
                print("✗ Mathematical formulation INCORRECT")
                print("  Expected: α = W @ sigmoid(s)")
                print("  Found different implementation")
        else:
            print(f"✗ Stage {stage} missing regulatory_network")

test_alpha_formulation()
```

### 2. Sigmoid Pretraining Test
```python
def test_sigmoid_pretraining():
    """Test sigmoid pretraining and freezing mechanism."""
    
    print("=== Testing Sigmoid Pretraining ===")
    
    # Create synthetic data
    n_cells, n_genes = 500, 100
    spliced = torch.randn(n_cells, n_genes)
    
    # Test on Stage 4 model
    config = tv.TangeloConfig(development_stage=4)
    model = tv.models.get_velocity_model(config, gene_dim=n_genes, atac_dim=200)
    
    # Test pretraining functionality
    if hasattr(model, 'pretrain_sigmoid_features'):
        print("✓ Pretrain method found")
        
        # Get initial sigmoid parameters
        initial_params = {}
        for name, param in model.regulatory_network.sigmoid_features.named_parameters():
            initial_params[name] = param.clone()
            print(f"Initial {name}: requires_grad={param.requires_grad}")
        
        # Run pretraining
        print("\nRunning sigmoid pretraining...")
        model.pretrain_sigmoid_features(spliced, n_epochs=10, learning_rate=0.01)
        
        # Check parameters after pretraining
        print("\nAfter pretraining:")
        params_changed = False
        for name, param in model.regulatory_network.sigmoid_features.named_parameters():
            param_change = torch.abs(param - initial_params[name]).max()
            print(f"{name}: requires_grad={param.requires_grad}, max_change={param_change:.6f}")
            if param_change > 1e-6:
                params_changed = True
        
        if params_changed:
            print("✓ Sigmoid parameters updated during pretraining")
        else:
            print("✗ Sigmoid parameters NOT updated during pretraining")
        
        # Test freezing
        print("\nTesting parameter freezing...")
        frozen_params = {}
        for name, param in model.regulatory_network.sigmoid_features.named_parameters():
            frozen_params[name] = param.clone()
            if not param.requires_grad:
                print(f"✓ {name} is frozen (requires_grad=False)")
            else:
                print(f"✗ {name} is NOT frozen (requires_grad=True)")
        
        # Test that frozen parameters don't change during training
        print("\nTesting training with frozen sigmoid...")
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
        
        # Training step
        model.train()
        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(spliced, torch.randn(n_cells, n_genes))
            targets = {'spliced': spliced, 'unspliced': torch.randn(n_cells, n_genes)}
            loss = model.compute_loss(outputs, targets)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        # Verify sigmoid parameters didn't change
        print("\nVerifying sigmoid parameters remained frozen:")
        sigmoid_changed = False
        for name, param in model.regulatory_network.sigmoid_features.named_parameters():
            param_change = torch.abs(param - frozen_params[name]).max()
            print(f"{name}: max_change={param_change:.6f}")
            if param_change > 1e-6:
                sigmoid_changed = True
        
        if not sigmoid_changed:
            print("✓ Sigmoid parameters remained FROZEN during training")
        else:
            print("✗ Sigmoid parameters CHANGED during training (should be frozen)")
            
    else:
        print("✗ Pretrain method NOT found")

test_sigmoid_pretraining()
```

### 3. Stage 4 Advanced Features Test
```python
def test_stage4_features():
    """Test Stage 4 advanced features."""
    
    print("=== Testing Stage 4 Advanced Features ===")
    
    # Create test data
    n_cells, n_genes = 200, 75
    spliced = torch.randn(n_cells, n_genes)
    unspliced = torch.randn(n_cells, n_genes)
    
    config = tv.TangeloConfig(development_stage=4)
    model = tv.models.get_velocity_model(config, gene_dim=n_genes, atac_dim=150)
    
    # Test 1: Temporal Dynamics
    print("\n1. Testing Temporal Dynamics...")
    if hasattr(model, 'predict_temporal_velocity'):
        time_points = torch.linspace(0, 2.0, 5)  # 5 time points
        temporal_velocities = model.predict_temporal_velocity(spliced, unspliced, time_points)
        print(f"✓ Temporal velocities shape: {temporal_velocities.shape}")
        # Expected shape: (n_cells, n_genes, n_timepoints)
    else:
        print("✗ Temporal dynamics not implemented")
    
    # Test 2: Uncertainty Quantification
    print("\n2. Testing Uncertainty Quantification...")
    if hasattr(model, 'predict_with_uncertainty'):
        velocity_mean, velocity_std = model.predict_with_uncertainty(spliced, unspliced, n_samples=10)
        print(f"✓ Velocity mean shape: {velocity_mean.shape}")
        print(f"✓ Velocity std shape: {velocity_std.shape}")
        print(f"✓ Mean uncertainty: {velocity_std.mean().item():.4f}")
    else:
        print("✗ Uncertainty quantification not implemented")
    
    # Test 3: Multi-Scale Integration
    print("\n3. Testing Multi-Scale Integration...")
    if hasattr(model, 'get_multiscale_features'):
        # Provide cell type labels for testing
        cell_types = torch.randint(0, 3, (n_cells,))  # 3 cell types
        multiscale_features = model.get_multiscale_features(spliced, unspliced, cell_types)
        print(f"✓ Multi-scale features: {list(multiscale_features.keys())}")
        for key, value in multiscale_features.items():
            print(f"  {key}: {value.shape}")
    else:
        print("✗ Multi-scale integration not implemented")
    
    # Test 4: Advanced Regularization
    print("\n4. Testing Advanced Regularization...")
    if hasattr(model, 'get_regularization_terms'):
        reg_terms = model.get_regularization_terms()
        print(f"✓ Regularization terms: {list(reg_terms.keys())}")
        for key, value in reg_terms.items():
            print(f"  {key}: {value.item():.6f}")
    else:
        print("✗ Advanced regularization not implemented")
    
    # Test 5: Interpretability Tools
    print("\n5. Testing Interpretability Tools...")
    if hasattr(model, 'get_feature_importance'):
        importance_scores = model.get_feature_importance(spliced, unspliced)
        print(f"✓ Feature importance shape: {importance_scores.shape}")
        print(f"✓ Top 5 important features: {torch.topk(importance_scores.mean(0), 5).indices}")
    else:
        print("✗ Interpretability tools not implemented")

test_stage4_features()
```

### 4. Performance and Scalability Test
```python
def test_stage4_performance():
    """Test Stage 4 performance and scalability."""
    
    print("=== Testing Stage 4 Performance ===")
    
    import time
    
    # Test different data sizes
    test_sizes = [(100, 50), (500, 100), (1000, 200)]
    
    for n_cells, n_genes in test_sizes:
        print(f"\nTesting size: {n_cells} cells, {n_genes} genes")
        
        # Create data
        spliced = torch.randn(n_cells, n_genes)
        unspliced = torch.randn(n_cells, n_genes)
        
        config = tv.TangeloConfig(development_stage=4)
        model = tv.models.get_velocity_model(config, gene_dim=n_genes, atac_dim=n_genes*2)
        
        # Time forward pass
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            outputs = model(spliced, unspliced)
        forward_time = time.time() - start_time
        
        # Time training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        
        start_time = time.time()
        optimizer.zero_grad()
        outputs = model(spliced, unspliced)
        targets = {'spliced': spliced, 'unspliced': unspliced}
        loss = model.compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        training_time = time.time() - start_time
        
        print(f"  Forward pass: {forward_time:.3f}s")
        print(f"  Training step: {training_time:.3f}s")
        print(f"  Loss value: {loss.item():.4f}")
        
        # Memory usage (rough estimate)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

test_stage4_performance()
```

### 5. Integration with High-Level API
```python
def test_stage4_integration():
    """Test Stage 4 integration with high-level API."""
    
    print("=== Testing Stage 4 Integration ===")
    
    # Create mock MuData
    import muon as mu
    import scanpy as sc
    import numpy as np
    
    # Create synthetic data
    n_cells, n_genes, n_atac = 300, 100, 200
    
    # RNA data
    rna_data = np.random.negative_binomial(20, 0.3, (n_cells, n_genes))
    adata_rna = sc.AnnData(rna_data)
    adata_rna.layers['spliced'] = np.random.negative_binomial(15, 0.3, (n_cells, n_genes))
    adata_rna.layers['unspliced'] = np.random.negative_binomial(10, 0.3, (n_cells, n_genes))
    
    # ATAC data
    atac_data = np.random.binomial(1, 0.1, (n_cells, n_atac))
    adata_atac = sc.AnnData(atac_data)
    
    # Spatial coordinates
    adata_rna.obs['x_pixel'] = np.random.uniform(0, 100, n_cells)
    adata_rna.obs['y_pixel'] = np.random.uniform(0, 100, n_cells)
    
    # Create MuData
    mdata = mu.MuData({'rna': adata_rna, 'atac': adata_atac})
    mdata.obs['x_pixel'] = adata_rna.obs['x_pixel']
    mdata.obs['y_pixel'] = adata_rna.obs['y_pixel']
    
    print(f"Created test data: {mdata}")
    
    # Test Stage 4 estimation
    try:
        # Use high-level API
        tv.tools.estimate_velocity(mdata, stage=4)
        
        print("✓ Stage 4 velocity estimation completed")
        
        # Check outputs
        if 'velocity' in mdata['rna'].layers:
            velocity = mdata['rna'].layers['velocity']
            print(f"✓ Velocity computed: {velocity.shape}")
            print(f"✓ Velocity range: [{velocity.min():.3f}, {velocity.max():.3f}]")
        else:
            print("✗ Velocity layer not found")
        
        # Check Stage 4 specific outputs
        stage4_keys = [key for key in mdata['rna'].obsm.keys() if 'tangelo' in key]
        if stage4_keys:
            print(f"✓ Stage 4 outputs: {stage4_keys}")
        else:
            print("✗ No Stage 4 specific outputs found")
            
    except Exception as e:
        print(f"✗ Stage 4 estimation failed: {e}")
        import traceback
        traceback.print_exc()

test_stage4_integration()
```

### 6. Comparison Across Stages
```python
def test_cross_stage_comparison():
    """Compare velocity estimates across all stages."""
    
    print("=== Cross-Stage Comparison ===")
    
    # Create consistent test data
    n_cells, n_genes = 150, 80
    np.random.seed(42)  # For reproducibility
    torch.manual_seed(42)
    
    spliced = torch.randn(n_cells, n_genes)
    unspliced = torch.randn(n_cells, n_genes)
    
    results = {}
    
    for stage in [1, 2, 3, 4]:
        print(f"\nTesting Stage {stage}...")
        
        try:
            config = tv.TangeloConfig(development_stage=stage)
            model = tv.models.get_velocity_model(config, gene_dim=n_genes, atac_dim=100)
            
            # Test mathematical formulation
            if hasattr(model, 'regulatory_network'):
                # Test α = W @ sigmoid(s)
                reg_net = model.regulatory_network
                sigmoid_s = reg_net.sigmoid_features(spliced)
                W = reg_net.get_interaction_matrix()
                manual_alpha = (W @ sigmoid_s.T).T
                model_alpha = reg_net(spliced)
                
                alpha_diff = torch.abs(manual_alpha - model_alpha).max()
                print(f"  α formulation error: {alpha_diff:.6f}")
                
                if alpha_diff < 1e-5:
                    print(f"  ✓ Stage {stage}: Correct α = W @ sigmoid(s)")
                else:
                    print(f"  ✗ Stage {stage}: Incorrect α formulation")
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                if stage <= 2:
                    outputs = model(spliced, unspliced)
                else:
                    # Stages 3+ may need additional inputs
                    from torch_geometric.data import Data
                    edge_index = torch.randint(0, n_cells, (2, n_cells * 3))
                    spatial_graph = Data(edge_index=edge_index, num_nodes=n_cells)
                    outputs = model(spliced, unspliced, spatial_graph=spatial_graph, expression_graph=spatial_graph)
            
            velocity = outputs['velocity']
            results[f'stage_{stage}'] = {
                'velocity_shape': velocity.shape,
                'velocity_mean': velocity.mean().item(),
                'velocity_std': velocity.std().item(),
                'velocity_range': (velocity.min().item(), velocity.max().item())
            }
            
            print(f"  ✓ Stage {stage} completed successfully")
            
        except Exception as e:
            print(f"  ✗ Stage {stage} failed: {e}")
            results[f'stage_{stage}'] = None
    
    # Compare results
    print(f"\n=== Comparison Summary ===")
    for stage_key, result in results.items():
        if result:
            print(f"{stage_key}:")
            print(f"  Shape: {result['velocity_shape']}")
            print(f"  Mean: {result['velocity_mean']:.4f}")
            print(f"  Std: {result['velocity_std']:.4f}")
            print(f"  Range: [{result['velocity_range'][0]:.4f}, {result['velocity_range'][1]:.4f}]")

test_cross_stage_comparison()
```

## Success Criteria

### Mathematical Correctness
- ✅ α = W @ sigmoid(s) implemented correctly across all stages
- ✅ Sigmoid parameters frozen after pretraining
- ✅ Matrix dimensions consistent and valid

### Stage 4 Features
- ✅ Temporal dynamics functionality
- ✅ Uncertainty quantification working
- ✅ Multi-scale integration implemented
- ✅ Advanced regularization active
- ✅ Interpretability tools functional

### Performance Benchmarks
- ✅ Forward pass < 1s for 1000 cells, 200 genes
- ✅ Training step < 2s for same size
- ✅ Memory usage reasonable (< 4GB for typical datasets)
- ✅ Loss convergence within 100 epochs

### Integration Quality
- ✅ High-level API works with Stage 4
- ✅ Results stored correctly in MuData
- ✅ Consistent with previous stages where applicable
- ✅ Error handling robust

## Critical Validation Points

1. **Mathematical Formula**: MUST be α = W @ sigmoid(s), not W(sigmoid(s))
2. **Sigmoid Freezing**: Parameters must NOT update after pretraining
3. **Advanced Features**: All Stage 4 features must be functional
4. **Performance**: Must scale reasonably with data size
5. **Integration**: Must work with existing Tangelo Velocity ecosystem

## Troubleshooting

### Common Issues
- **Mathematical errors**: Check matrix dimensions and operations
- **Memory issues**: Reduce batch size or use gradient checkpointing
- **Convergence problems**: Verify sigmoid pretraining completed
- **Integration failures**: Check MuData format and required layers

Use these tests systematically to validate the Stage 4 implementation before deployment.